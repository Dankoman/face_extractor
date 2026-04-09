import os
import sqlite3
import requests
import time
from typing import Optional, List, Dict, Tuple
from pathlib import Path

STASHDB_ENDPOINT = "https://stashdb.org/graphql"
TPDB_ENDPOINT = "https://theporndb.net/graphql"
PMVSTASH_ENDPOINT = "https://pmvstash.org/graphql"
FANSDB_ENDPOINT = "https://fansdb.cc/graphql"

CACHE_DB = Path(__file__).parent / "identity_cache.db"

class ExternalIdentityResolver:
    def __init__(self, stashdb_key: str = None, tpdb_key: str = None, pmvstash_key: str = None, fansdb_key: str = None):
        self.stashdb_key = stashdb_key or os.environ.get("STASHDB_API_KEY")
        self.tpdb_key = tpdb_key or os.environ.get("TPDB_API_KEY")
        self.pmvstash_key = pmvstash_key or os.environ.get("PMVSTASH_API_KEY")
        self.fansdb_key = fansdb_key or os.environ.get("FANSDB_API_KEY")
        self._init_db()
        self.session = requests.Session()

    def _init_db(self):
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS external_names (
                    alias TEXT PRIMARY KEY,
                    canonical_name TEXT,
                    source TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _get_from_cache(self, alias: str) -> Optional[Tuple[str, str]]:
        with sqlite3.connect(CACHE_DB) as conn:
            row = conn.execute(
                "SELECT canonical_name, source FROM external_names WHERE LOWER(alias) = LOWER(?)",
                (alias,)
            ).fetchone()
            return row if row else None

    def _save_to_cache(self, alias: str, canonical_name: str, source: str):
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO external_names (alias, canonical_name, source) VALUES (?, ?, ?)",
                (alias, canonical_name, source)
            )

    def resolve(self, name: str) -> Optional[Tuple[str, str]]:
        """
        Resolve a name to its canonical version.
        Returns: (canonical_name, source)
        """
        # 1. Check Cache
        cached = self._get_from_cache(name)
        if cached:
            return cached

        # 2. Try in priority order
        sources = [
            (self.stashdb_key, STASHDB_ENDPOINT, "StashDB"),
            (self.tpdb_key, TPDB_ENDPOINT, "ThePornDB"),
            (self.pmvstash_key, PMVSTASH_ENDPOINT, "PMVStash"),
            (self.fansdb_key, FANSDB_ENDPOINT, "FansDB"),
        ]

        for key, endpoint, source_name in sources:
            if not key:
                continue
            
            res = None
            if source_name == "ThePornDB":
                res = self._query_tpdb(name)
            else:
                # All others are Stash-based
                res = self._query_stash_generic(name, endpoint, key, source_name)
            
            if res:
                self._save_to_cache(name, res, source_name)
                return res, source_name

        return None

    def _query_stash_generic(self, name: str, endpoint: str, api_key: str, source_label: str) -> Optional[str]:
        query = """
        query QueryPerformers($input: PerformerQueryInput!) {
          queryPerformers(input: $input) {
            count
            performers {
              name
              aliases
            }
          }
        }
        """
        headers = {"ApiKey": api_key}
        lname = name.lower()
        search_fields = ["name", "names"]
        
        for field in search_fields:
            variables = {"input": {field: name, "per_page": 10}}
            try:
                time.sleep(0.3)
                resp = self.session.post(endpoint, json={"query": query, "variables": variables}, headers=headers, timeout=10)
                if resp.status_code == 422:
                    continue
                resp.raise_for_status()
                data = resp.json()
                performers = data.get("data", {}).get("queryPerformers", {}).get("performers", [])
                
                for p in performers:
                    p_name = p.get("name")
                    if p_name and p_name.lower() == lname:
                        return p_name
                    for alias in p.get("aliases", []):
                        if alias.lower() == lname:
                            return p_name
            except Exception as e:
                print(f"{source_label} error for {name} ({field}): {e}")
                continue
        return None

    def _query_tpdb(self, name: str) -> Optional[str]:
        query = """
        query SearchPerformers($q: String!) {
          searchPerformers(q: $q) {
            name
            alias
          }
        }
        """
        headers = {"Authorization": f"Bearer {self.tpdb_key}"}
        try:
            time.sleep(0.3)
            resp = self.session.post(TPDB_ENDPOINT, json={"query": query, "variables": {"q": name}}, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            performers = data.get("data", {}).get("searchPerformers", [])
            lname = name.lower()
            for p in performers:
                p_name = p.get("name")
                if p_name and p_name.lower() == lname:
                    return p_name
                aliases = p.get("alias")
                if isinstance(aliases, str) and aliases.lower() == lname:
                    return p_name
                elif isinstance(aliases, list):
                    for a in aliases:
                        if a.lower() == lname:
                            return p_name
            return None
        except Exception as e:
            print(f"ThePornDB error for {name}: {e}")
            return None

if __name__ == "__main__":
    resolver = ExternalIdentityResolver()
    print(f"Resolving 'Riley Reid': {resolver.resolve('Riley Reid')}")
