#!/usr/bin/env python3
"""Gemensamt data-access-lager för processed-databasen (SQLite)."""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed (
    path    TEXT PRIMARY KEY,
    ok      INTEGER NOT NULL,
    reason  TEXT DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_processed_ok ON processed(ok);

CREATE TABLE IF NOT EXISTS aliases (
    alias        TEXT PRIMARY KEY,
    primary_name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_aliases_primary ON aliases(primary_name);
"""


def open_db(db_path: str | Path) -> sqlite3.Connection:
    """Öppna (eller skapa) databasen med WAL-mode för bättre concurrency."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


# --------------- Läsa ---------------

def load_processed_set(conn: sqlite3.Connection) -> Set[str]:
    """Returnera set med alla paths (motsvarar gamla load_processed)."""
    cur = conn.execute("SELECT path FROM processed")
    return {row[0] for row in cur}


def get_alias_map(conn: sqlite3.Connection) -> Dict[str, str]:
    """Hämta alla alias och primärnamn som en {alias -> primär} dict."""
    cur = conn.execute("SELECT alias, primary_name FROM aliases")
    return {row[0]: row[1] for row in cur}


def resolve_name(conn: sqlite3.Connection, name: str) -> str:
    """Slå upp ett namn och returnera dess primärnamn (eller namnet självt)."""
    cur = conn.execute("SELECT primary_name FROM aliases WHERE alias = ?", (name,))
    row = cur.fetchone()
    return row[0] if row else name



def is_processed(conn: sqlite3.Connection, path: str) -> bool:
    """Kolla om en specifik path redan är processad."""
    cur = conn.execute("SELECT 1 FROM processed WHERE path = ?", (path,))
    return cur.fetchone() is not None


def iter_all(conn: sqlite3.Connection):
    """Yield alla rader som (path, ok, reason)."""
    cur = conn.execute("SELECT path, ok, reason FROM processed")
    for row in cur:
        yield row[0], bool(row[1]), row[2]


def count(conn: sqlite3.Connection) -> int:
    """Returnera totalt antal rader."""
    cur = conn.execute("SELECT COUNT(*) FROM processed")
    return cur.fetchone()[0]


# --------------- Skriva ---------------

def add_processed(conn: sqlite3.Connection, path: str, ok: bool, reason: str = "ok") -> None:
    """Lägg till en processad bild. Ignorera om path redan finns."""
    conn.execute(
        "INSERT OR IGNORE INTO processed (path, ok, reason) VALUES (?, ?, ?)",
        (path, int(ok), reason),
    )
    conn.commit()


def add_aliases_batch(conn: sqlite3.Connection, mapping: Dict[str, str]) -> None:
    """Spara en batch med alias-mappningar {alias: primary}. Skriver över befintliga."""
    data = [(alias, primary) for alias, primary in mapping.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO aliases (alias, primary_name) VALUES (?, ?)",
        data,
    )
    conn.commit()


def clear_aliases(conn: sqlite3.Connection) -> None:
    """Töm hela alias-tabellen."""
    conn.execute("DELETE FROM aliases")
    conn.commit()



def add_processed_batch(conn: sqlite3.Connection, records: Iterable[Tuple[str, bool, str]]) -> int:
    """Batch-insert. Returnerar antal insertade rader."""
    inserted = 0
    batch: List[Tuple[str, int, str]] = []
    for path, ok, reason in records:
        batch.append((path, int(ok), reason))
        if len(batch) >= 10_000:
            conn.executemany(
                "INSERT OR IGNORE INTO processed (path, ok, reason) VALUES (?, ?, ?)",
                batch,
            )
            inserted += len(batch)
            batch.clear()
    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO processed (path, ok, reason) VALUES (?, ?, ?)",
            batch,
        )
        inserted += len(batch)
    conn.commit()
    return inserted


# --------------- Radera ---------------

def remove_by_paths(conn: sqlite3.Connection, paths: Iterable[str]) -> int:
    """Ta bort specifika paths. Returnerar antal borttagna."""
    path_list = list(paths)
    if not path_list:
        return 0
    # SQLite har gräns på antal parametrar, chunka vid behov
    removed = 0
    for chunk in _chunks(path_list, 500):
        placeholders = ",".join("?" for _ in chunk)
        cur = conn.execute(f"DELETE FROM processed WHERE path IN ({placeholders})", chunk)
        removed += cur.rowcount
    conn.commit()
    return removed


def remove_by_persons(conn: sqlite3.Connection, person_labels: Iterable[str]) -> int:
    """Ta bort alla poster vars path innehåller en av de angivna person-mapparna.

    Person extraheras som den näst sista komponenten i path
    (t.ex. /data/pBook/PersonName/file.jpg → PersonName).
    """
    labels = set(person_labels)
    if not labels:
        return 0
    # Eftersom person inte är en kolumn måste vi hämta alla paths och filtrera i Python.
    # Fortfarande snabbare än att skriva om hela filen.
    to_delete: List[str] = []
    for path, _ok, _reason in iter_all(conn):
        person = Path(path).parent.name
        if person in labels:
            to_delete.append(path)
    return remove_by_paths(conn, to_delete)


# --------------- Uppdatera ---------------

def update_path(conn: sqlite3.Connection, old_path: str, new_path: str) -> bool:
    """Uppdatera en path. Returnerar True om raden hittades."""
    cur = conn.execute(
        "UPDATE OR REPLACE processed SET path = ? WHERE path = ?",
        (new_path, old_path),
    )
    conn.commit()
    return cur.rowcount > 0


def update_paths_batch(conn: sqlite3.Connection, mapping: Dict[str, str]) -> int:
    """Uppdatera flera paths baserat på {old_path: new_path}. Returnerar antal uppdaterade."""
    if not mapping:
        return 0
    updated = 0
    for old, new in mapping.items():
        cur = conn.execute("UPDATE OR REPLACE processed SET path = ? WHERE path = ?", (new, old))
        updated += cur.rowcount
    conn.commit()
    return updated


# --------------- Statistik ---------------

def get_stats_by_person(conn: sqlite3.Connection) -> Dict[str, Dict]:
    """Returnera per-person-statistik: {person: {total, ok, fail, reasons: {reason: count}}}.

    Person extraheras från path (parent directory name).
    """
    stats: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "ok": 0, "fail": 0, "reasons": defaultdict(int)}
    )
    for path, ok, reason in iter_all(conn):
        person = Path(path).parent.name
        stats[person]["total"] += 1
        if ok:
            stats[person]["ok"] += 1
        else:
            stats[person]["fail"] += 1
            stats[person]["reasons"][reason] += 1
    return stats


# --------------- Hjälpfunktion ---------------

def find_missing_paths(conn: sqlite3.Connection) -> Tuple[List[str], Dict[str, List[str]]]:
    """Hitta paths i DB som inte finns på disk.

    Returns:
        missing_paths: lista av saknade fil-paths
        person_missing: {person_label: [missing_paths]}
    """
    missing_paths: List[str] = []
    person_missing: Dict[str, List[str]] = defaultdict(list)

    for path, _ok, _reason in iter_all(conn):
        if not Path(path).exists():
            person = Path(path).parent.name
            missing_paths.append(path)
            person_missing[person].append(path)

    return missing_paths, dict(person_missing)


# --------------- Intern ---------------

def _chunks(lst: List, n: int):
    """Yield chunks av storlek n."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
