#!/usr/bin/env python3
"""Flytta felaktiga bilder till top_label och uppdatera embeddings/processed.

Typisk körning:
    python fix_mismatches.py --csv low_confidence_images.csv --run-pipeline

Förväntar sig en per-bild-CSV från train_confidence.py med kolumnerna
path,label,confidence,top_label,top_confidence,delta,mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/home/marqs/Bilder/pBook")
DEFAULT_WORKDIR = SCRIPT_DIR / "arcface_work-ppic"
DEFAULT_EMBEDDINGS = DEFAULT_WORKDIR / "embeddings_ppic.pkl"
DEFAULT_PROCESSED = DEFAULT_WORKDIR / "processed-ppic.jsonl"
DEFAULT_MERGED = DEFAULT_WORKDIR / "embeddings_ppic_merged.pkl"
DEFAULT_MODEL_OUT = DEFAULT_WORKDIR / "face_knn_arcface_ppic.pkl"


@dataclass
class Row:
    idx: int
    path: str
    label: str
    top_label: str
    mismatch: bool
    delta: float


def read_rows(csv_path: Path) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append(
                Row(
                    idx=idx,
                    path=row.get("path", "").strip(),
                    label=row.get("label", "").strip(),
                    top_label=row.get("top_label", "").strip(),
                    mismatch=row.get("mismatch", "").strip().lower() == "yes",
                    delta=float(row.get("delta", "0") or 0.0),
                )
            )
    return rows


def move_files(rows: Iterable[Row], data_root: Path, dry_run: bool = False) -> List[tuple[Row, Path]]:
    performed: List[tuple[Row, Path]] = []
    for row in rows:
        if not row.path:
            print(f"Skippar rad {row.idx}: saknar path")
            continue
        src = Path(row.path)
        if not src.exists():
            print(f"Skippar rad {row.idx}: hittar inte {src}")
            continue
        if not row.top_label:
            print(f"Skippar rad {row.idx}: saknar top_label")
            continue
        dest_dir = data_root / row.top_label
        dest = dest_dir / src.name
        if dest.resolve() == src.resolve():
            print(f"Skippar rad {row.idx}: redan i korrekt mapp ({dest})")
            continue
        if dest.exists():
            print(f"Skippar rad {row.idx}: målfil finns redan {dest}")
            continue
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
        performed.append((row, dest))
    return performed


def rewrite_embeddings(emb_path: Path, total_rows: int, drop_indices: Set[int]) -> None:
    if not drop_indices:
        return
    with emb_path.open("rb") as f:
        data = pickle.load(f)
    X: Sequence = data["X"]
    y: Sequence[str] = data["y"]
    if len(X) != total_rows or len(y) != total_rows:
        raise SystemExit(
            f"Embeddings-längd ({len(X)}) matchar inte CSV-rader ({total_rows}). "
            "Generera om CSV med train_confidence.py --csv-per-image först."
        )
    keep_idx = [i for i in range(total_rows) if i not in drop_indices]
    X_new = [X[i] for i in keep_idx]
    y_new = [y[i] for i in keep_idx]
    tmp = emb_path.with_suffix(emb_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"X": X_new, "y": y_new}, f)
    tmp.replace(emb_path)


def rewrite_processed(proc_path: Path, drop_paths: Set[str]) -> None:
    if not drop_paths or not proc_path.exists():
        return
    tmp = proc_path.with_suffix(proc_path.suffix + ".tmp")
    removed = 0
    with proc_path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for raw in src:
            line = raw.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dst.write(raw)
                continue
            path = rec.get("path", "")
            if path in drop_paths:
                removed += 1
                continue
            dst.write(raw)
    tmp.replace(proc_path)
    print(f"Tog bort {removed} rader ur processed.")


def run_pipeline(data_root: Path, workdir: Path) -> None:
    python = sys.executable
    merged = workdir / "embeddings_ppic_merged.pkl"
    model_out = workdir / "face_knn_arcface_ppic.pkl"
    steps = [
        [python, "face_arc_pipeline.py", "--mode", "encode", "--data-root", str(data_root), "--workdir", str(workdir), "--allow-upsample", "--max-yaw", "40"],
        [python, "merge.py"],
        [python, "face_arc_pipeline.py", "--mode", "train", "--embeddings", str(merged), "--model-out", str(model_out)],
    ]
    for cmd in steps:
        print(f"Kör: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flytta mismatchade bilder och uppdatera embeddings/processed.")
    ap.add_argument("--csv", required=True, type=Path, help="Per-bild-CSV från train_confidence.py")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Rotmapp med personmappar")
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="Workdir för embeddings/processed")
    ap.add_argument("--min-delta", type=float, default=0.0, help="Flytta endast rader med delta >= detta värde")
    ap.add_argument("--dry-run", action="store_true", help="Visa vad som skulle flyttas utan att skriva")
    ap.add_argument("--run-pipeline", action="store_true", help="Kör encode+merge+train efter flytt")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    candidates = [r for r in rows if r.mismatch and r.delta >= args.min_delta]
    print(f"Mismatch-rader: {len(candidates)} (av {len(rows)})")
    performed = move_files(candidates, args.data_root, dry_run=args.dry_run)
    if args.dry_run:
        for row, dest in performed:
            print(f"[DRY-RUN] Skulle flytta {row.path} -> {dest}")
        return

    if not performed:
        print("Inget att flytta.")
        return

    moved_indices = {row.idx for row, _ in performed}
    moved_paths = {row.path for row, _ in performed}

    emb_path = args.workdir / DEFAULT_EMBEDDINGS.name if args.workdir else DEFAULT_EMBEDDINGS
    proc_path = args.workdir / DEFAULT_PROCESSED.name if args.workdir else DEFAULT_PROCESSED

    rewrite_embeddings(emb_path, len(rows), moved_indices)
    rewrite_processed(proc_path, moved_paths)

    print(f"Flyttade {len(performed)} filer.")

    if args.run_pipeline:
        run_pipeline(args.data_root, args.workdir)


if __name__ == "__main__":
    main()
