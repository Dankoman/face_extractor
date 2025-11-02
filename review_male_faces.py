#!/usr/bin/env python3
"""Interactively review male-classified faces and optionally delete the images.

Reads the `male_faces.csv` (or any compatible CSV) produced by
`find_male_faces.py`, filters rows on a configurable male probability,
sorts them descending, and lets you inspect/remove each hit. When you
confirm removal it will delete the image file from disk **and** append the
label to `remove.txt` (unless it already exists there).

Example usage:

    python3 review_male_faces.py male_faces.csv \
        --threshold 0.5 --remove-file remove.txt

Use `--dry-run` to preview actions without deleting or modifying files.
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

Row = Tuple[str, float, float, Path]
StateMap = Dict[str, str]


def load_rows(csv_path: Path, threshold: float) -> List[Row]:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows: List[Row] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="|")
        for idx, row in enumerate(reader, 1):
            if not row or row[0].lower() == "label":
                continue
            if len(row) < 4:
                print(f"[warn] Skipping malformed row {idx}: {row}")
                continue
            label = row[0]
            try:
                male_val = float(row[1])
            except ValueError:
                male_val = 0.0
            try:
                female_val = float(row[2])
            except ValueError:
                female_val = 0.0
            path = Path(row[3])
            if male_val >= threshold:
                rows.append((label, male_val, female_val, path))

    if not rows:
        raise SystemExit(
            f"No rows meet threshold {threshold}. Did you run find_male_faces.py?"
        )

    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def load_remove_list(remove_file: Path) -> set[str]:
    if not remove_file.exists():
        return set()
    return {line.strip() for line in remove_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_to_remove(remove_file: Path, name: str) -> None:
    remove_file.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if remove_file.exists():
        if remove_file.stat().st_size > 0:
            with remove_file.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    needs_newline = True
    with remove_file.open("a", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        fh.write(name + "\n")


def load_state(state_file: Path) -> StateMap:
    if not state_file.exists():
        return {}
    mapping: StateMap = {}
    with state_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or "|" not in line:
                continue
            path, action = line.split("|", 1)
            mapping[path] = action
    return mapping


def append_state(state_handle, path: Path, action: str) -> None:
    state_handle.write(f"{path}|{action}\n")
    state_handle.flush()


def open_with_pillow(image_path: Path) -> bool:
    if Image is None:
        return False
    try:
        img = Image.open(image_path)
        img.show()
        return True
    except Exception as exc:
        print(f"  [warn] PIL failed to open image: {exc}")
        return False


def open_with_command(command: str, image_path: Path) -> bool:
    try:
        parts = shlex.split(command)
        subprocess.run(parts + [str(image_path)], check=False)
        return True
    except Exception as exc:
        print(f"  [warn] viewer command failed ({command}): {exc}")
        return False


def open_with_xdg(image_path: Path) -> bool:
    try:
        subprocess.run(["xdg-open", str(image_path)], check=False)
        return True
    except Exception as exc:
        print(f"  [warn] xdg-open failed: {exc}")
        return False


def view_image(image_path: Path, prefer_xdg: bool, custom_viewer: str | None) -> None:
    if custom_viewer:
        if open_with_command(custom_viewer, image_path):
            return
    if prefer_xdg:
        if open_with_xdg(image_path):
            return
        if open_with_pillow(image_path):
            return
    else:
        if open_with_pillow(image_path):
            return
        if open_with_xdg(image_path):
            return
    print("  [warn] Could not open image automatically.")


def review_rows(
    rows: Iterable[Row],
    remove_file: Path,
    dry_run: bool,
    auto_yes: bool,
    auto_view: bool,
    prefer_xdg: bool,
    viewer_cmd: str | None,
    state_map: StateMap,
    state_handle,
) -> None:
    existing_names = load_remove_list(remove_file)
    removed = 0
    skipped = 0
    missing = 0
    resumed = 0

    print(f"Loaded {len(existing_names)} existing names in {remove_file}")
    print("Starting review – type 'y' to delete, Enter or 'n' to skip.\n")

    total_rows = len(rows)
    for idx, (label, male_prob, female_prob, path) in enumerate(rows, 1):
        print(f"[{idx}/{total_rows}] {label}")
        print(f"  male_prob : {male_prob:.3f}")
        print(f"  female_prob : {female_prob:.3f}")
        print(f"  path : {path}")

        path_str = str(path)
        if state_map and path_str in state_map:
            print(f"  -> Skipping (already processed: {state_map[path_str]})")
            resumed += 1
            continue

        if not path.exists():
            print("  -> File missing on disk. Skipping.")
            missing += 1
            if state_handle and not dry_run:
                append_state(state_handle, path, "missing")
                state_map[path_str] = "missing"
            continue

        if auto_view:
            view_image(path, prefer_xdg, viewer_cmd)

        if auto_yes:
            decision = "y"
        else:
            decision = ""
            while True:
                decision = input("  Action [y=delete / n=skip / v=view / q=quit] (default n): ").strip().lower()
                if decision == "":
                    decision = "n"
                if decision == "v":
                    view_image(path, prefer_xdg, viewer_cmd)
                    continue
                if decision in {"y", "n", "q"}:
                    break
                print("  [info] Please answer y, n, v or q.")
            if decision == "q":
                print("  -> Abort requested. Stopping review.")
                break

        do_remove = decision == "y"
        if not do_remove:
            skipped += 1
            if state_handle and not dry_run:
                append_state(state_handle, path, "skip")
                state_map[path_str] = "skip"
            continue

        if dry_run:
            print("  DRY-RUN: would delete and append to remove.txt")
            removed += 1
            continue

        try:
            path.unlink()
            print("  Deleted image from disk.")
        except OSError as exc:
            print(f"  ERROR deleting file: {exc}")
            if state_handle and not dry_run:
                append_state(state_handle, path, f"error:{exc}")
                state_map[path_str] = f"error:{exc}"
            continue

        if label not in existing_names:
            append_to_remove(remove_file, label)
            existing_names.add(label)
            print("  Added label to remove.txt")
        else:
            print("  Label already in remove.txt")

        removed += 1
        if state_handle and not dry_run:
            append_state(state_handle, path, "deleted")
            state_map[path_str] = "deleted"

    print("\nReview finished.")
    print(f"  Removed/deleted : {removed}")
    print(f"  Skipped : {skipped}")
    print(f"  Missing files : {missing}")
    if resumed:
        print(f"  Previously processed (resume): {resumed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive reviewer for male_faces.csv entries")
    parser.add_argument("csv", type=Path, nargs="?", default=Path("male_faces.csv"),
                        help="CSV exported by find_male_faces.py (pipe-delimited)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum male probability to review (default 0.5)")
    parser.add_argument("--remove-file", type=Path, default=Path("remove.txt"),
                        help="File where labels should be appended")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not delete files or modify remove.txt")
    parser.add_argument("--auto-yes", action="store_true",
                        help="Delete all hits without prompting (careful!)")
    parser.add_argument("--auto-view", action="store_true",
                        help="Automatically open each image before prompting")
    parser.add_argument("--prefer-xdg", action="store_true",
                        help="Prefer xdg-open over PIL.Image.show when viewing")
    parser.add_argument("--viewer", help="Custom viewer command (e.g. 'firefox')")
    parser.add_argument("--state-file", type=Path, default=Path("review_state.log"),
                        help="File used to store resume state (default review_state.log)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore state file and review all rows anew")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv, args.threshold)
    state_map: StateMap = {}
    state_handle = None
    if not args.no_resume and args.state_file:
        state_map = load_state(args.state_file)
    if not args.dry_run and args.state_file:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        state_handle = args.state_file.open("a", encoding="utf-8")
    review_rows(
        rows,
        args.remove_file,
        dry_run=args.dry_run,
        auto_yes=args.auto_yes,
        auto_view=args.auto_view,
        prefer_xdg=args.prefer_xdg,
        viewer_cmd=args.viewer,
        state_map=state_map,
        state_handle=state_handle,
    )
    if state_handle:
        state_handle.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
