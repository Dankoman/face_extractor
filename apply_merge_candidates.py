#!/usr/bin/env python3
"""Apply merge?.csv suggestions to merge.txt without creating duplicates."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def parse_pipe_line(raw: str) -> List[str]:
    return [segment.strip() for segment in raw.strip().split("|") if segment.strip()]


def is_score(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_merge(merge_path: Path) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str]]:
    entries: Dict[str, List[str]] = {}
    order: List[str] = []
    name_to_primary: Dict[str, str] = {}
    if not merge_path.exists():
        return entries, order, name_to_primary
    for raw in merge_path.read_text(encoding="utf-8").splitlines():
        names = parse_pipe_line(raw)
        if not names:
            continue
        primary, *aliases = names
        entries[primary] = aliases
        order.append(primary)
        for name in names:
            name_to_primary[name] = primary
    return entries, order, name_to_primary


def remove_primary(
    primary: str,
    entries: Dict[str, List[str]],
    order: List[str],
    name_to_primary: Dict[str, str],
) -> None:
    names = [primary] + entries.get(primary, [])
    entries.pop(primary, None)
    if primary in order:
        order.remove(primary)
    for name in names:
        name_to_primary.pop(name, None)


def apply_candidates(
    merge_path: Path,
    csv_path: Path,
) -> Tuple[bool, int]:
    entries, order, name_to_primary = load_merge(merge_path)
    if not csv_path.exists():
        return False, len(order)

    changed = False
    lines = [
        line
        for line in csv_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for raw in lines:
        names = [
            token
            for token in parse_pipe_line(raw)
            if token and not is_score(token)
        ]
        if not names:
            continue
        primary_candidate = names[0]
        queue = list(names)
        collected_order: List[str] = []
        collected_set = set()
        primaries_seen: List[str] = []
        insert_index = len(order)

        while queue:
            name = queue.pop(0).strip()
            if not name or name in collected_set:
                continue
            collected_set.add(name)
            collected_order.append(name)
            old_primary = name_to_primary.get(name)
            if old_primary and old_primary not in primaries_seen:
                primaries_seen.append(old_primary)
                if old_primary in order:
                    idx = order.index(old_primary)
                    if idx < insert_index:
                        insert_index = idx
                full_names = [old_primary] + entries.get(old_primary, [])
                for alias in full_names:
                    if alias not in collected_set:
                        queue.append(alias)

        if not collected_set:
            continue

        # Ensure the desired primary stays left-most.
        ordered_unique: List[str] = []
        for name in names + collected_order:
            if name in collected_set and name not in ordered_unique:
                ordered_unique.append(name)
        if not ordered_unique:
            continue

        primary = primary_candidate
        if primary not in ordered_unique:
            ordered_unique.insert(0, primary)

        aliases = [name for name in ordered_unique if name != primary]

        # Skip if nothing would change.
        current_primary = name_to_primary.get(primary)
        if (
            current_primary == primary
            and set(entries.get(primary, [])) == set(aliases)
            and not primaries_seen
        ):
            continue

        for old_primary in primaries_seen:
            remove_primary(old_primary, entries, order, name_to_primary)

        if primary in entries and primary not in primaries_seen:
            remove_primary(primary, entries, order, name_to_primary)
            insert_index = min(insert_index, len(order))

        insert_index = min(insert_index, len(order))
        order.insert(insert_index, primary)
        entries[primary] = aliases
        for name in [primary] + aliases:
            name_to_primary[name] = primary
        changed = True

    if changed:
        lines_out = ["|".join([primary] + entries[primary]) for primary in order]
        merge_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")

    return changed, len(order)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply merge?.csv suggestions to merge.txt")
    ap.add_argument("--merge", type=Path, default=Path("merge.txt"), help="Path to merge.txt")
    ap.add_argument(
        "--candidates",
        type=Path,
        default=Path("merge?.csv"),
        help="Candidate alias file (default: merge?.csv)",
    )
    args = ap.parse_args()

    changed, total = apply_candidates(args.merge, args.candidates)
    if not args.candidates.exists():
        print(f"No candidate file found at {args.candidates}; nothing to do.")
        return
    if changed:
        print(f"Updated {args.merge} using {args.candidates}. Total entries: {total}.")
    else:
        print(f"No changes applied. {args.merge} already up to date (entries: {total}).")


if __name__ == "__main__":
    main()
