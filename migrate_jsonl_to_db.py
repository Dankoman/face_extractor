#!/usr/bin/env python3
"""One-shot migrering: processed-ppic.jsonl → processed.db (SQLite)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import processed_db


def migrate(jsonl_path: Path, db_path: Path) -> None:
    if not jsonl_path.exists():
        print(f"❌ JSONL-filen hittades inte: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    if db_path.exists():
        print(f"⚠️  Databasen finns redan: {db_path}")
        print("   Nya poster läggs till (INSERT OR IGNORE), dubbletter hoppas över.")

    conn = processed_db.open_db(db_path)
    count_before = processed_db.count(conn)

    print(f"📖 Läser {jsonl_path} ...")
    records = []
    errors = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Rad {lineno}: ogiltigt JSON – {e}", file=sys.stderr)
                errors += 1
                continue
            path = rec.get("path", "")
            ok = rec.get("ok", True)
            reason = rec.get("reason", "ok")
            records.append((path, ok, reason))

    print(f"   Läste {len(records)} poster ({errors} felaktiga rader hoppade över)")

    print(f"💾 Skriver till {db_path} ...")
    inserted = processed_db.add_processed_batch(conn, records)

    count_after = processed_db.count(conn)
    new_rows = count_after - count_before

    print(f"\n✅ Migrering klar!")
    print(f"   JSONL-rader:      {len(records)}")
    print(f"   Redan i DB:       {count_before}")
    print(f"   Nya rader:        {new_rows}")
    print(f"   Totalt i DB:      {count_after}")

    if count_after != len(records) and count_before == 0:
        print(f"\n⚠️  Antal i DB ({count_after}) matchar inte JSONL ({len(records)}).")
        print("   Det kan bero på duplicerade paths i JSONL-filen.")

    conn.close()
    print(f"\n📝 JSONL-filen har INTE tagits bort: {jsonl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrera processed JSONL till SQLite")
    parser.add_argument(
        "--jsonl",
        default="arcface_work-ppic/processed-ppic.jsonl",
        help="Käll-JSONL-fil",
    )
    parser.add_argument(
        "--db",
        default="arcface_work-ppic/processed.db",
        help="Mål-SQLite-databas",
    )
    args = parser.parse_args()
    migrate(Path(args.jsonl), Path(args.db))


if __name__ == "__main__":
    main()
