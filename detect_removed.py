#!/usr/bin/env python3
"""Detektera borttagna bilder och rensa processed + embeddings för berörda personer."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_alias_map(merge_path: Path) -> Dict[str, str]:
    """Läs merge.txt och returnera {alias -> primary}."""
    alias_to_primary: Dict[str, str] = {}
    if not merge_path.exists():
        return alias_to_primary
    for raw in merge_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        names = [part.strip() for part in raw.split("|") if part.strip()]
        if not names:
            continue
        primary = names[0]
        for name in names:
            alias_to_primary[name] = primary
    return alias_to_primary


def find_missing_images(processed_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Läs processed JSONL och hitta paths som inte finns på disk.

    Returns:
        missing_paths: lista av saknade fil-paths
        person_missing: {person_label: [missing_paths]}
    """
    missing_paths: List[str] = []
    person_missing: Dict[str, List[str]] = defaultdict(list)

    if not processed_path.exists():
        return missing_paths, person_missing

    with processed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            path_str = rec.get("path", "")
            if not path_str:
                continue
            if not Path(path_str).exists():
                person_label = Path(path_str).parent.name
                missing_paths.append(path_str)
                person_missing[person_label].append(path_str)

    return missing_paths, dict(person_missing)


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


def prune_processed(processed_path: Path, affected_labels: Set[str]) -> Tuple[int, int]:
    """Ta bort alla processed-poster för berörda personer.

    Returns:
        (removed, kept)
    """
    if not processed_path.exists():
        return 0, 0

    tmp_path = processed_path.with_suffix(processed_path.suffix + ".tmp")
    removed = 0
    kept = 0

    with processed_path.open("r", encoding="utf-8") as src, \
         tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                dst.write(line)
                kept += 1
                continue
            path_str = rec.get("path", "")
            person = Path(path_str).parent.name if path_str else ""
            if person in affected_labels:
                removed += 1
            else:
                dst.write(line)
                kept += 1

    tmp_path.replace(processed_path)
    return removed, kept


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
        "--processed",
        default="arcface_work-ppic/processed-ppic.jsonl",
        help="JSONL-fil med processade bilder",
    )
    parser.add_argument(
        "--merge",
        default="merge.txt",
        help="Alias-fil (pipe-separerad)",
    )
    args = parser.parse_args()

    processed_path = Path(args.processed)
    embeddings_path = Path(args.embeddings)
    merge_path = Path(args.merge)
    data_root = Path(args.data_root)

    # 1. Hitta bilder som saknas på disk
    missing_paths, person_missing = find_missing_images(processed_path)

    if not missing_paths:
        print("✅ Inga borttagna bilder hittades – inget att rensa.")
        return

    print(f"🔍 Hittade {len(missing_paths)} borttagna bild(er) för {len(person_missing)} person(er):")
    for person, paths in sorted(person_missing.items()):
        print(f"  {person}: {len(paths)} borttagna")

    # 2. Lös alias och bestäm vilka labels som berörs
    alias_to_primary = load_alias_map(merge_path)
    affected_labels = resolve_affected_persons(person_missing, alias_to_primary, data_root)

    print(f"\n🏷️  Berörda labels (inkl. alias): {sorted(affected_labels)}")

    # 3. Rensa processed
    proc_removed, proc_kept = prune_processed(processed_path, affected_labels)
    print(f"\n📋 Processed: {proc_removed} poster borttagna, {proc_kept} kvar")

    # 4. Rensa embeddings
    emb_removed, emb_kept = prune_embeddings(embeddings_path, affected_labels)
    print(f"🧠 Embeddings: {emb_removed} borttagna, {emb_kept} kvar")

    print(f"\n♻️  Kör encode-steget för att re-bearbeta kvarvarande bilder för: {sorted(person_missing.keys())}")


if __name__ == "__main__":
    main()
