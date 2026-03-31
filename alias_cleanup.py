#!/usr/bin/env python3
"""
Normalize alias/main folders, move alias files, and keep processed DB in sync.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import processed_db


@dataclass
class Stats:
    alias_dirs_found: int = 0
    alias_dirs_normalized: int = 0
    main_dirs_normalized: int = 0
    files_renamed: int = 0
    files_moved: int = 0
    conflicts: int = 0
    created_main_dirs: int = 0
    removed_missing_entries: int = 0
    removed_empty_dirs: int = 0
    db_updates: int = 0

    def as_dict(self) -> Dict[str, int]:
        return self.__dict__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize alias/main folders, move alias files, and update processed DB."
    )
    parser.add_argument("--data-root", required=True, type=Path, help="Root directory that contains performer folders.")
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="SQLite-databas (processed.db) som ska uppdateras med nya filplatser.",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Remove DB entries whose files are missing after the move.",
    )
    parser.add_argument(
        "--missing-log",
        type=Path,
        default=None,
        help="Optional path where missing DB entries should be logged (and removed if --prune-missing is set).",
    )
    return parser.parse_args()


# load_alias_map ersatt av processed_db.get_alias_map()


def normalize_directory(dir_path: Path, prefix: str, rename_map: Dict[str, str], stats: Stats) -> None:
    files = sorted([p for p in dir_path.iterdir() if p.is_file()], key=lambda p: p.name.casefold())
    if not files:
        return
    width = max(3, len(str(len(files))))
    temp_moves: List[Tuple[Path, Path, str]] = []
    for idx, file_path in enumerate(files, 1):
        new_name = f"{prefix}-{idx:0{width}d}{file_path.suffix}"
        if file_path.name == new_name:
            continue
        tmp = file_path.with_name(f".__tmp__rename__{uuid.uuid4().hex}__{file_path.name}")
        file_path.rename(tmp)
        dest = file_path.with_name(new_name)
        temp_moves.append((tmp, dest, str(file_path)))
    if not temp_moves:
        return
    for tmp, dest, original in temp_moves:
        if dest.exists():
            raise RuntimeError(f"Destination already exists: {dest}")
        tmp.rename(dest)
        rename_map[original] = str(dest)
        stats.files_renamed += 1


def move_alias_files(
    alias_dir: Path,
    main_dir: Path,
    move_map: Dict[str, str],
    stats: Stats,
) -> None:
    files = sorted([p for p in alias_dir.iterdir() if p.is_file()], key=lambda p: p.name.casefold())
    if not files:
        try:
            alias_dir.rmdir()
        except OSError:
            pass
        return
    if not main_dir.exists():
        main_dir.mkdir(parents=True)
        stats.created_main_dirs += 1
    for file_path in files:
        dest = main_dir / file_path.name
        if dest.exists():
            stem, suffix = file_path.stem, file_path.suffix
            candidate = main_dir / f"{stem}-konflikt{suffix}"
            counter = 2
            while candidate.exists():
                candidate = main_dir / f"{stem}-konflikt{counter}{suffix}"
                counter += 1
            dest = candidate
            stats.conflicts += 1
        move_map[str(file_path)] = str(dest)
        file_path.rename(dest)
        stats.files_moved += 1
    try:
        alias_dir.rmdir()
    except OSError:
        pass


def remove_empty_dirs(root: Path) -> int:
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        path = Path(dirpath)
        if path == root:
            continue
        if not dirnames and not filenames:
            try:
                path.rmdir()
                removed += 1
            except OSError:
                pass
    return removed


def update_db(
    conn,
    rename_map: Dict[str, str],
    move_map: Dict[str, str],
    stats: Stats,
) -> None:
    """Uppdatera databasrader med nya filplatser."""
    if not rename_map and not move_map:
        return
    # Bygg en samlad mapping: old_path -> new_path
    combined: Dict[str, str] = {}
    for old, new in rename_map.items():
        combined[old] = new
    for old, new in move_map.items():
        # Om gamla pathen redan renamedats, koppla vidare
        actual_old = old
        for orig, renamed in rename_map.items():
            if renamed == old:
                actual_old = orig
                break
        combined[actual_old] = new

    stats.db_updates = processed_db.update_paths_batch(conn, combined)


def prune_missing_entries(
    conn,
    stats: Stats,
    missing_log: Path | None,
) -> None:
    missing_paths, _ = processed_db.find_missing_paths(conn)

    if missing_log:
        if missing_paths:
            missing_log.write_text("\n".join(missing_paths))
        elif missing_log.exists():
            missing_log.unlink()

    if not missing_paths:
        return

    removed = processed_db.remove_by_paths(conn, missing_paths)
    stats.removed_missing_entries += removed


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    db_path = args.db

    conn = processed_db.open_db(db_path)
    alias_map = processed_db.get_alias_map(conn)
    existing_aliases = sorted([alias for alias in alias_map if (data_root / alias).is_dir()], key=str.casefold)
    stats = Stats(alias_dirs_found=len(existing_aliases))

    rename_map: Dict[str, str] = {}
    for alias in existing_aliases:
        before = stats.files_renamed
        normalize_directory(data_root / alias, alias, rename_map, stats)
        if stats.files_renamed > before:
            stats.alias_dirs_normalized += 1

    primary_dirs = sorted({alias_map[alias] for alias in existing_aliases}, key=str.casefold)
    for primary in primary_dirs:
        dir_path = data_root / primary
        if not dir_path.exists():
            continue
        before = stats.files_renamed
        normalize_directory(dir_path, primary, rename_map, stats)
        if stats.files_renamed > before:
            stats.main_dirs_normalized += 1

    move_map: Dict[str, str] = {}
    for alias in existing_aliases:
        main = alias_map.get(alias)
        if not main:
            continue
        move_alias_files(data_root / alias, data_root / main, move_map, stats)

    conn = processed_db.open_db(db_path)
    update_db(conn, rename_map, move_map, stats)

    stats.removed_empty_dirs = remove_empty_dirs(data_root)

    if args.prune_missing or args.missing_log:
        prune_missing_entries(conn, stats, args.missing_log)

    conn.close()
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
