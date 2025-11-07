#!/usr/bin/env python3
"""Beräkna "konfidens" per person baserat på ArcFace-embeddings.

Skriptet tar embeddings-pickeln (som `face_arc_pipeline.py` producerar),
beräknar centroiden för varje label och mäter cosinus-similaritet mellan
varje embedding och centroiden. Låg likhet brukar indikera att personen
har spridda/otydliga ansikten och bör granskas.

Exempel:

    python train_confidence.py --top 50
    python train_confidence.py --threshold 0.55 --csv low_confidence.csv

`--top` visar de labels som har lägst `centroid_norm`. `--threshold` filtrerar
istället på en absolut nivå. Med `--csv` får du (per default) samma urval
som skrivs i konsolen; lägg till `--csv-full` om du vill dumpa hela tabellen.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_embeddings(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = pickle.loads(path.read_bytes())
    X = np.vstack(data["X"]).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    y = np.array(data["y"], dtype=object)
    return X, y


def load_aliases(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    primary_to_aliases: Dict[str, set] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if "|" in raw:
            names = [segment.strip() for segment in raw.split("|") if segment.strip()]
        elif ":" in raw:
            label, members = raw.split(":", 1)
            names = [label.strip()] + [m.strip() for m in members.split(",") if m.strip()]
        else:
            names = [raw]
        if not names:
            continue
        primary = names[0]
        aliases = primary_to_aliases.setdefault(primary, set())
        aliases.update(names)
    alias_map: Dict[str, List[str]] = {}
    for primary, names in primary_to_aliases.items():
        cleaned = sorted(name for name in names if name != primary)
        if cleaned:
            alias_map[primary] = cleaned
    return alias_map


def cosine_similarity_to_centroid(vectors: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        # om samtliga embeddings är noll (defekt data)
        return np.zeros(len(vectors), dtype=np.float32), centroid_norm
    centroid_unit = centroid / centroid_norm
    sims = (vectors @ centroid_unit).astype(np.float32)
    return sims, centroid_norm


def aggregate_stats(X: np.ndarray, y: np.ndarray) -> List[Dict[str, float]]:
    stats: List[Dict[str, float]] = []
    for label in np.unique(y):
        mask = y == label
        vectors = X[mask]
        sims, centroid_norm = cosine_similarity_to_centroid(vectors)
        stats.append(
            {
                "label": label,
                "count": int(vectors.shape[0]),
                "min": float(sims.min()),
                "median": float(np.median(sims)),
                "mean": float(sims.mean()),
                "max": float(sims.max()),
                "std": float(sims.std(ddof=0)),
                "centroid_norm": float(centroid_norm),
            }
        )
    stats.sort(key=lambda row: row["centroid_norm"])  # lägst centroid först
    return stats


def write_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = ["label", "count", "min", "median", "mean", "max", "std", "centroid_norm", "aliases"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lista labels med låg centroid-likhet")
    ap.add_argument(
        "--embeddings",
        default=Path("arcface_work-ppic/embeddings_ppic.pkl"),
        type=Path,
        help="Pickle-fil från träningen",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=50,
        help="Visa N rader med lägst centroid_norm",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Filtrera på min-likhet under denna nivå",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        help="Skriv statistik till CSV-fil (standard: bara visade rader)",
    )
    ap.add_argument(
        "--csv-full",
        action="store_true",
        help="Skriv hela tabellen till CSV i stället för filtrerat urval",
    )
    ap.add_argument(
        "--merge",
        type=Path,
        default=Path("merge.txt"),
        help="Merge-fil för att lista alias (default: merge.txt)",
    )
    ap.add_argument(
        "--missing-aliases-only",
        action="store_true",
        help="Visa bara labels som saknar alias i merge-filen",
    )
    args = ap.parse_args()

    X, y = load_embeddings(args.embeddings)
    stats = aggregate_stats(X, y)
    alias_lookup = load_aliases(args.merge)
    for row in stats:
        aliases = alias_lookup.get(row["label"], [])
        row["aliases"] = ", ".join(aliases)

    if args.missing_aliases_only:
        stats = [row for row in stats if not alias_lookup.get(row["label"])]

    if args.threshold is not None:
        stats_to_show = [row for row in stats if row["min"] < args.threshold]
    else:
        stats_to_show = stats[: args.top]

    header = f"{'label':30s}  {'n':>5s}  {'min':>6s}  {'median':>6s}  {'mean':>6s}  {'std':>6s}  aliases"
    print(header)
    print("-" * len(header))
    for row in stats_to_show:
        alias_str = row.get("aliases", "")
        print(
            f"{row['label'][:30]:30s}  {row['count']:5d}  "
            f"{row['min']:.3f}  {row['median']:.3f}  {row['mean']:.3f}  {row['std']:.3f}  {alias_str}"
        )

    if args.csv:
        rows_for_csv = stats if args.csv_full else stats_to_show
        write_csv(args.csv, rows_for_csv)
        print(f"\nSkrev {len(rows_for_csv)} rader till {args.csv}")


if __name__ == "__main__":
    main()
