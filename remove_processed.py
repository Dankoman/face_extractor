#!/usr/bin/env python3
"""Filtrera bort poster ur processed-ppic.jsonl baserat på remove.txt och merge.txt."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_alias_map(merge_path: Path) -> tuple[Dict[str, str], Dict[str, Set[str]]]:
    alias_to_primary: Dict[str, str] = {}
    primary_to_aliases: Dict[str, Set[str]] = defaultdict(set)
    if not merge_path.exists():
        return alias_to_primary, primary_to_aliases
    for raw in merge_path.read_text(encoding="utf-8").splitlines():
        names = [part.strip() for part in raw.strip().split("|") if part.strip()]
        if not names:
            continue
        primary = names[0]
        for name in names:
            alias_to_primary[name] = primary
            primary_to_aliases[primary].add(name)
    return alias_to_primary, primary_to_aliases


def expand_remove_set(remove_names: Iterable[str], alias_to_primary: Dict[str, str],
                      primary_to_aliases: Dict[str, Set[str]]):
    primary_counts = {alias_to_primary.get(name, name): 0 for name in remove_names}
    expanded: Set[str] = set()
    for primary in primary_counts:
        expanded.update(primary_to_aliases.get(primary, {primary}))
    return expanded, primary_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ta bort poster ur processed JSONL")
    parser.add_argument("--processed", default="arcface_work-ppic/processed-ppic.jsonl", help="JSONL-fil att filtrera")
    parser.add_argument("--remove", default="remove.txt", help="Textfil med namn att ta bort")
    parser.add_argument("--merge", default="merge.txt", help="Aliasfil (pipe-separerad)")
    args = parser.parse_args()

    processed_path = Path(args.processed)
    remove_path = Path(args.remove)
    merge_path = Path(args.merge)

    if not processed_path.exists():
        raise SystemExit(f"Hittar inte processed-filen: {processed_path}")
    if not remove_path.exists():
        raise SystemExit(f"Hittar inte remove-filen: {remove_path}")

    remove_names = read_lines(remove_path)
    alias_to_primary, primary_to_aliases = load_alias_map(merge_path)
    expanded_remove, counters = expand_remove_set(remove_names, alias_to_primary, primary_to_aliases)

    tmp_path = processed_path.with_suffix(processed_path.suffix + ".tmp")

    total = kept = removed = 0
    with processed_path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for raw in src:
            total += 1
            line = raw.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dst.write(raw)
                kept += 1
                continue
            path = rec.get("path", "")
            performer = Path(path).parent.name if path else ""
            primary = alias_to_primary.get(performer, performer)
            if performer in expanded_remove or primary in counters:
                counters[primary] = counters.get(primary, 0) + 1
                removed += 1
                continue
            dst.write(raw)
            kept += 1

    tmp_path.replace(processed_path)

    print(f"Totalrader: {total}")
    print(f"Borttagna: {removed}")
    print(f"Kvar: {kept}")
    missing = [name for name, count in counters.items() if count == 0]
    if missing:
        print("Hittade inte:")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
