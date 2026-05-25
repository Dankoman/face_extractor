import os
import sys
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from pyswip import Prolog

class IdentityResolver:
    def __init__(self, exclusions_file: Path, db_path: Optional[Path] = None):
        self.prolog = Prolog()
        self.exclusions_file = Path(exclusions_file) if exclusions_file else None
        
        if db_path is None:
            self.db_path = Path(__file__).parent / "arcface_work-ppic" / "processed.db"
        else:
            self.db_path = Path(db_path)
            
        self.prolog_file = Path(__file__).parent / "identity.pl"
        
        # Load the Prolog rules
        self.prolog.consult(str(self.prolog_file))
        
        self._resolution_cache = {}
        self.load_data()

    def load_data(self):
        """Parse text files and database, then assert facts into Prolog."""
        seen_names = {} # name -> line_number
        
        # 1. Aliases from database (processed.db)
        if self.db_path and self.db_path.exists():
            try:
                conn = sqlite3.connect(str(self.db_path))
                # Hämta alias och primärnamn
                cur = conn.execute("SELECT alias, primary_name FROM aliases")
                db_aliases = cur.fetchall()
                
                primaries = set()
                for alias, primary in db_aliases:
                    if alias == primary:
                        continue # Skip self-aliases
                    alias_esc = alias.replace("'", "\\'")
                    primary_esc = primary.replace("'", "\\'")
                    
                    self.prolog.assertz(f"link('{alias_esc}', '{primary_esc}')")
                    primaries.add(primary_esc)
                
                # Sätt alla unika primärnamn som primary()
                for p in primaries:
                    self.prolog.assertz(f"primary('{p}')")
                    
                conn.close()
            except Exception as e:
                print(f"⚠️  WARNING: Could not load aliases from DB {self.db_path}: {e}", file=sys.stderr)

        # 2. Exclusions from similar_exclusions.txt
        # Format: Name1|Name2
        if self.exclusions_file.exists():
            with open(self.exclusions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2:
                        p1 = parts[0].replace("'", "\\'")
                        p2 = parts[1].replace("'", "\\'")
                        self.prolog.assertz(f"exclude('{p1}', '{p2}')")

        # 2. Exclusions from similar_exclusions.txt

    def add_external_truth(self, alias: str, canonical: str, source: str):
        """Provide external truth (e.g. from StashDB)."""
        alias_esc = alias.replace("'", "\\'")
        canonical_esc = canonical.replace("'", "\\'")
        self.prolog.assertz(f"external_truth('{alias_esc}', '{canonical_esc}', '{source}')")

    def resolve(self, name: str) -> str:
        """Resolve a name to its best canonical form."""
        if name in self._resolution_cache:
            return self._resolution_cache[name]
            
        name_esc = name.replace("'", "\\'")
        query = f"resolved_canonical('{name_esc}', FinalName)"
        results = list(self.prolog.query(query))
        
        resolved = results[0]["FinalName"] if results else name
        self._resolution_cache[name] = resolved
        return resolved

    def resolve_many(self, names: list[str]) -> dict[str, str]:
        """Bulk resolve a list of names. Returns {original: canonical} mapping."""
        unique_names = sorted(list(set(names)))
        mapping = {}
        for name in unique_names:
            mapping[name] = self.resolve(name)
        return mapping

    def check_conflicts(self) -> list[tuple[str, str]]:
        """Find all logical conflicts (linked names that are excluded)."""
        results = list(self.prolog.query("conflict(X, Y)"))
        # Prolog returns both (A, B) and (B, A), filter for unique pairs
        conflicts = set()
        for res in results:
            pair = tuple(sorted([res["X"], res["Y"]]))
            conflicts.add(pair)
        return list(conflicts)

    def get_group(self, name: str) -> list[str]:
        """Get all names associated with the same person."""
        name_esc = name.replace("'", "\\'")
        results = list(self.prolog.query(f"group_members('{name_esc}', Members)"))
        if results:
            return results[0]["Members"]
        return [name]

    def get_scrape_decision(self, name: str, samples: int, variance: float) -> dict:
        """Get a structured scraping decision for a person."""
        name_esc = name.replace("'", "\\'")
        query = f"scraping_decision('{name_esc}', {samples}, {variance}, FinalName, Action, Reason)"
        results = list(self.prolog.query(query))
        if results:
            res = results[0]
            return {
                "name": name,
                "canonical": res["FinalName"],
                "action": res["Action"],
                "reason": res["Reason"]
            }
        return {
            "name": name,
            "canonical": name,
            "action": "none",
            "reason": "Kunde inte fastställa beslut"
        }

if __name__ == "__main__":
    # Quick test
    resolver = IdentityResolver("similar_exclusions.txt")
    print(f"Conflicts found: {resolver.check_conflicts()}")
