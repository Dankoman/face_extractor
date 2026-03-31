#!/usr/bin/env python3
"""One-shot migrering: merge.txt → processed.db (SQLite)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import processed_db


def migrate_merge(merge_path: Path, db_path: Path) -> None:
    if not merge_path.exists():
        print(f"❌ merge.txt hittades inte: {merge_path}", file=sys.stderr)
        sys.exit(1)

    conn = processed_db.open_db(db_path)
    
    # Läs merge.txt
    mapping = {}
    print(f"📖 Läser {merge_path} ...")
    with merge_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            names = [segment.strip() for segment in line.split("|") if segment.strip()]
            if not names:
                continue
            
            primary = names[0]
            # Mappa primärnamnet till sig självt
            mapping[primary] = primary
            # Mappa alla alias till primärnamnet
            for alias in names[1:]:
                mapping[alias] = primary

    print(f"   Hittade {len(mapping)} unika namn (inkl. alias).")

    print(f"💾 Skriver till aliases-tabellen i {db_path} ...")
    processed_db.add_aliases_batch(conn, mapping)

    total_in_db = len(processed_db.get_alias_map(conn))
    print(f"\n✅ Migrering av alias klar!")
    print(f"   Totalt i DB: {total_in_db}")

    conn.close()
    print(f"\n📝 merge.txt har INTE tagits bort: {merge_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrera merge.txt till SQLite")
    parser.add_argument(
        "--merge",
        default="merge.txt",
        help="Käll-merge.txt-fil",
    )
    parser.add_argument(
        "--db",
        default="arcface_work-ppic/processed.db",
        help="Mål-SQLite-databas",
    )
    args = parser.parse_args()
    migrate_merge(Path(args.merge), Path(args.db))


if __name__ == "__main__":
    main()
