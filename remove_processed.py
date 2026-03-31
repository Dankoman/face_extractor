#!/usr/bin/env python3
"""Filtrera bort poster ur processed-databasen baserat på remove.txt och merge.txt."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import processed_db


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# load_alias_map ersatt av processed_db.get_alias_map()


def expand_remove_set(remove_names: Iterable[str], alias_to_primary: Dict[str, str],
                      primary_to_aliases: Dict[str, Set[str]]):
    primary_counts = {alias_to_primary.get(name, name): 0 for name in remove_names}
    expanded: Set[str] = set()
    for primary in primary_counts:
        expanded.update(primary_to_aliases.get(primary, {primary}))
    return expanded, primary_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ta bort poster ur processed-databasen")
    parser.add_argument("--db", default="arcface_work-ppic/processed.db", help="SQLite-databas")
    parser.add_argument("--remove", default="remove.txt", help="Textfil med namn att ta bort")
    parser.add_argument("--no-alias", action="store_true", help="Utöka inte alias – ta bara bort exakt angivna namn")
    args = parser.parse_args()

    db_path = Path(args.db)
    remove_path = Path(args.remove)

    if not remove_path.exists():
        raise SystemExit(f"Hittar inte remove-filen: {remove_path}")

    remove_names = read_lines(remove_path)
    conn = processed_db.open_db(db_path)
    
    if args.no_alias:
        expanded_remove = set(remove_names)
    else:
        alias_to_primary = processed_db.get_alias_map(conn)
        # Re-construct primary_to_aliases for expand_remove_set
        primary_to_aliases = defaultdict(set)
        for alias, primary in alias_to_primary.items():
            primary_to_aliases[primary].add(alias)
            
        expanded_remove, _ = expand_remove_set(remove_names, alias_to_primary, primary_to_aliases)

    total_before = processed_db.count(conn)

    removed = processed_db.remove_by_persons(conn, expanded_remove)
    kept = processed_db.count(conn)

    conn.close()

    print(f"Totalrader (före): {total_before}")
    print(f"Borttagna: {removed}")
    print(f"Kvar: {kept}")


if __name__ == "__main__":
    main()
