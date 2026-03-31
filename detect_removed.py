#!/usr/bin/env python3
"""Detektera borttagna bilder och rensa processed + embeddings för berörda personer."""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import processed_db


# load_alias_map ersatt av processed_db.get_alias_map()


def resolve_affected_persons(
    person_missing: Dict[str, List[str]],
    alias_to_primary: Dict[str, str],
    data_root: Path,
) -> Set[str]:
    """Bestäm vilka person-labels som behöver rensas.

    Om en alias-katalog saknar bilder men primär-katalogen fortfarande finns
    inkluderas aliaset ändå (vi rensar alla dess embeddings för re-encoding).
    Returnerar alla labels (inklusive alias-varianter) som ska rensas.
    """
    affected_primaries: Set[str] = set()
    for person in person_missing:
        primary = alias_to_primary.get(person, person)
        affected_primaries.add(primary)

    # Expandera till alla alias-varianter av berörda primärer
    primary_to_aliases: Dict[str, Set[str]] = defaultdict(set)
    for alias, primary in alias_to_primary.items():
        primary_to_aliases[primary].add(alias)
        primary_to_aliases[primary].add(primary)

    affected_labels: Set[str] = set()
    for primary in affected_primaries:
        affected_labels.update(primary_to_aliases.get(primary, {primary}))

    return affected_labels


def prune_embeddings(embeddings_path: Path, affected_labels: Set[str]) -> Tuple[int, int]:
    """Ta bort embeddings för berörda personer.

    Returns:
        (removed, kept)
    """
    if not embeddings_path.exists():
        return 0, 0

    with embeddings_path.open("rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]

    keep_idx = [i for i, label in enumerate(y) if label not in affected_labels]
    removed = len(y) - len(keep_idx)

    X_new = [X[i] for i in keep_idx]
    y_new = [y[i] for i in keep_idx]

    with embeddings_path.open("wb") as f:
        pickle.dump({"X": X_new, "y": y_new}, f)

    return removed, len(keep_idx)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detektera borttagna bilder och rensa processed + embeddings"
    )
    parser.add_argument(
        "--data-root",
        default="/home/marqs/Bilder/pBook",
        help="Rot-katalog med person-mappar",
    )
    parser.add_argument(
        "--embeddings",
        default="arcface_work-ppic/embeddings_ppic.pkl",
        help="Pickle-fil med embeddings",
    )
    parser.add_argument(
        "--db",
        default="arcface_work-ppic/processed.db",
        help="SQLite-databas med processade bilder",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    embeddings_path = Path(args.embeddings)
    data_root = Path(args.data_root)

    conn = processed_db.open_db(db_path)

    # 1. Hitta bilder som saknas på disk
    missing_paths, person_missing = processed_db.find_missing_paths(conn)

    if not missing_paths:
        print("✅ Inga borttagna bilder hittades – inget att rensa.")
        conn.close()
        return

    print(f"🔍 Hittade {len(missing_paths)} borttagna bild(er) för {len(person_missing)} person(er):")
    for person, paths in sorted(person_missing.items()):
        print(f"  {person}: {len(paths)} borttagna")

    # 2. Lös alias och bestäm vilka labels som berörs
    alias_to_primary = processed_db.get_alias_map(conn)
    affected_labels = resolve_affected_persons(person_missing, alias_to_primary, data_root)

    print(f"\n🏷️  Berörda labels (inkl. alias): {sorted(affected_labels)}")

    # 3. Rensa processed (DB)
    proc_removed = processed_db.remove_by_persons(conn, affected_labels)
    proc_kept = processed_db.count(conn)
    print(f"\n📋 Processed: {proc_removed} poster borttagna, {proc_kept} kvar")

    # 4. Rensa embeddings
    emb_removed, emb_kept = prune_embeddings(embeddings_path, affected_labels)
    print(f"🧠 Embeddings: {emb_removed} borttagna, {emb_kept} kvar")

    conn.close()
    print(f"\n♻️  Kör encode-steget för att re-bearbeta kvarvarande bilder för: {sorted(person_missing.keys())}")


if __name__ == "__main__":
    main()
