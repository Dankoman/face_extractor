#!/usr/bin/env python3
"""
Normalize alias/main folders, move alias files, and keep processed JSONL in sync.
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
    json_updates: int = 0

    def as_dict(self) -> Dict[str, int]:
        return self.__dict__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize alias/main folders, move alias files, and update processed JSON."
    )
    parser.add_argument("--data-root", required=True, type=Path, help="Root directory that contains performer folders.")
    parser.add_argument("--merge", required=True, type=Path, help="merge.txt file with alias mapping.")
    parser.add_argument(
        "--processed",
        required=True,
        type=Path,
        help="processed-ppic.jsonl that should be updated with the new file locations.",
    )
    parser.add_argument(
        "--backup-suffix",
        default="alias_cleanup",
        help="Suffix for the processed JSONL backup file (default: alias_cleanup).",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Remove JSON entries whose files are missing after the move.",
    )
    parser.add_argument(
        "--missing-log",
        type=Path,
        default=None,
        help="Optional path where missing JSON entries should be logged (and removed if --prune-missing is set).",
    )
    return parser.parse_args()


def load_alias_map(merge_path: Path) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    if not merge_path.exists():
        return alias_map
    with merge_path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            names = [part.strip() for part in line.split("|") if part.strip()]
            if len(names) < 2:
                continue
            primary = names[0]
            for alias in names[1:]:
                alias_map[alias] = primary
    return alias_map


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


def update_json(
    processed_path: Path,
    rename_map: Dict[str, str],
    move_map: Dict[str, str],
    stats: Stats,
    backup_suffix: str,
) -> None:
    if not rename_map and not move_map:
        return
    backup_path = processed_path.with_name(f"{processed_path.name}.{backup_suffix}.{int(time.time())}.bak")
    shutil.copy2(processed_path, backup_path)
    tmp_path = processed_path.with_suffix(".tmp_alias_cleanup")
    updated = 0
    with processed_path.open() as src, tmp_path.open("w") as dst:
        for line in src:
            data = json.loads(line)
            path = data.get("path")
            new_path = rename_map.get(path, path)
            new_path = move_map.get(new_path, new_path)
            if new_path != path:
                data["path"] = new_path
                updated += 1
            dst.write(json.dumps(data, ensure_ascii=False) + "\n")
    tmp_path.replace(processed_path)
    stats.json_updates += updated


def prune_missing_entries(
    processed_path: Path,
    stats: Stats,
    missing_log: Path | None,
) -> None:
    missing_rows: List[str] = []
    with processed_path.open() as fh:
        for idx, line in enumerate(fh, 1):
            data = json.loads(line)
            if not Path(data["path"]).exists():
                missing_rows.append(f"{idx}:{data['path']}")
    if missing_log:
        if missing_rows:
            missing_log.write_text("\n".join(missing_rows))
        elif missing_log.exists():
            missing_log.unlink()
    if not missing_rows:
        return
    missing_set = {row.split(":", 1)[1] for row in missing_rows}
    tmp_path = processed_path.with_suffix(".tmp_alias_clean_missing")
    removed = 0
    with processed_path.open() as src, tmp_path.open("w") as dst:
        for line in src:
            data = json.loads(line)
            if data["path"] in missing_set:
                removed += 1
                continue
            dst.write(line)
    tmp_path.replace(processed_path)
    stats.removed_missing_entries += removed


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    processed_json = args.processed
    merge_file = args.merge

    alias_map = load_alias_map(merge_file)
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

    update_json(processed_json, rename_map, move_map, stats, args.backup_suffix)

    stats.removed_empty_dirs = remove_empty_dirs(data_root)

    if args.prune_missing or args.missing_log:
        prune_missing_entries(processed_json, stats, args.missing_log)

    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
