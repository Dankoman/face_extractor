#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import processed_db



def parse_pipe_line(raw: str) -> List[str]:
    return [segment.strip() for segment in raw.strip().split("|") if segment.strip()]


def is_score(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


# load_merge och remove_primary ersatta av processed_db-metoder


def apply_candidates(
    db_path: Path,
    csv_path: Path,
) -> bool:
    conn = processed_db.open_db(db_path)
    # alias_map: {name: primary}
    alias_map = processed_db.get_alias_map(conn)
    
    # Re-construct primary_to_aliases: {primary: [alias1, alias2...]}
    primary_to_aliases = defaultdict(list)
    for name, primary in alias_map.items():
        if name != primary:
            primary_to_aliases[primary].append(name)
            
    if not csv_path.exists():
        conn.close()
        return False

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
            old_primary = alias_map.get(name)
            if old_primary and old_primary not in primaries_seen:
                primaries_seen.append(old_primary)
                full_names = [old_primary] + primary_to_aliases.get(old_primary, [])
                for alias in full_names:
                    if alias not in collected_set:
                        queue.append(alias)

        if not collected_set:
            continue

        # Determine new primary and group
        new_primary = primary_candidate
        all_members = set(collected_set)
        all_members.add(new_primary)
        
        # Check for changes
        is_changed = False
        for member in all_members:
            if alias_map.get(member) != new_primary:
                is_changed = True
                break
        
        if not is_changed:
            continue

        # Apply update to local structures
        for member in all_members:
            # If this member was a primary of something else, merge those too
            if member in primary_to_aliases:
                for sub_alias in primary_to_aliases[member]:
                    alias_map[sub_alias] = new_primary
                    if sub_alias not in all_members:
                        # This shouldn't normally happen with the while-loop above, but just in case
                        pass 
            alias_map[member] = new_primary
            
        # Refresh primary_to_aliases for next candidate row
        primary_to_aliases.clear()
        for name, primary in alias_map.items():
            if name != primary:
                primary_to_aliases[primary].append(name)
                
        changed = True

    if changed:
        processed_db.add_aliases_batch(conn, alias_map)

    conn.close()
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply to_be_merged.csv suggestions to SQLite DB")
    ap.add_argument("--db", type=Path, default=Path("arcface_work-ppic/processed.db"), help="Path to processed.db")
    ap.add_argument(
        "--candidates",
        type=Path,
        default=Path("to_be_merged.csv"),
        help="Candidate alias file (default: to_be_merged.csv)",
    )
    args = ap.parse_args()

    if not args.candidates.exists():
        print(f"No candidate file found at {args.candidates}; nothing to do.")
        return

    changed = apply_candidates(args.db, args.candidates)
    if changed:
        print(f"Updated aliases in {args.db} using {args.candidates}.")
    else:
        print(f"No changes applied. {args.db} already up to date.")


if __name__ == "__main__":
    main()
