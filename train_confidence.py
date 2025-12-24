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
from typing import Dict, Iterable, List, Optional, Tuple

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


def compute_label_stats(
    X: np.ndarray, y: np.ndarray
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    """
    Beräknar statistik per label och returnerar även centroider för vidare beräkningar.

    Returns:
        stats: list med label-statistik (sorterad på centroid_norm)
        labels: ndarray med labels i samma ordning som centroids
        centroids: ndarray med normaliserade centroidvektorer (shape: n_labels x dim)
    """
    stats: List[Dict[str, float]] = []
    centroids: List[np.ndarray] = []
    labels = np.unique(y)
    for label in labels:
        mask = y == label
        vectors = X[mask]
        sims, centroid_norm = cosine_similarity_to_centroid(vectors)
        if centroid_norm == 0:
            centroid_unit = np.zeros_like(vectors[0]) if len(vectors) else np.array([], dtype=np.float32)
        else:
            centroid_unit = (vectors.mean(axis=0) / centroid_norm).astype(np.float32)
        centroids.append(centroid_unit)
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
    return stats, labels, np.vstack(centroids) if centroids else np.empty((0, X.shape[1]), dtype=np.float32)


def write_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = ["label", "count", "min", "median", "mean", "max", "std", "centroid_norm", "aliases"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_paths_from_processed_json(proc_path: Path) -> List[str]:
    """Ren JSON-variant av loader (säker och rak)."""
    if not proc_path.exists():
        print(f"⚠️ Hittade ingen processed-fil: {proc_path}")
        return []
    ok_paths: List[str] = []
    import json

    for raw in proc_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        if rec.get("ok"):
            ok_paths.append(rec["path"])
    return ok_paths


def write_per_image_csv(
    path: Path, rows: List[Dict[str, object]], include_header: Optional[List[str]] = None
) -> None:
    if not rows:
        return
    keys = include_header or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_per_image_rows(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    paths: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """
    Returnerar en rad per embedding med:
    - path (om känd)
    - label
    - confidence mot egen centroid
    - top_label och top_confidence (närmsta centroid)
    - delta och mismatch-flagga
    """
    if labels.size == 0 or centroids.size == 0:
        return []
    label_to_idx = {label: i for i, label in enumerate(labels)}
    sims = X @ centroids.T  # shape: n_samples x n_labels
    own_idx = np.array([label_to_idx[label] for label in y])
    own_conf = sims[np.arange(len(X)), own_idx]
    best_idx = sims.argmax(axis=1)
    top_labels = labels[best_idx]
    top_conf = sims[np.arange(len(X)), best_idx]

    rows: List[Dict[str, object]] = []
    for i in range(len(X)):
        path = paths[i] if paths and i < len(paths) else ""
        folder_name = Path(path).parent.name if path else ""
        rows.append(
            {
                "path": path,
                "folder_name": folder_name,
                "label": y[i],
                "confidence": float(own_conf[i]),
                "top_label": top_labels[i],
                "top_confidence": float(top_conf[i]),
                "delta": float(top_conf[i] - own_conf[i]),
                "mismatch": "yes" if top_labels[i] != y[i] else "",
            }
        )
    return rows


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
    ap.add_argument(
        "--csv-per-image",
        type=Path,
        help="Skriv per-bild confidence till CSV (kolumner: path, label, confidence, top_label, top_confidence, delta, mismatch)",
    )
    ap.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="processed-*.jsonl för att hämta paths (default: samma katalog som embeddings, processed-ppic.jsonl)",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/marqs/Bilder/pBook"),
        help="Rotkatalog för bilder (används för att filtrera bort namn utan mapp)",
    )
    args = ap.parse_args()

    X, y = load_embeddings(args.embeddings)
    stats, labels, centroids = compute_label_stats(X, y)
    alias_lookup = load_aliases(args.merge)
    
    # Filtrera bort labels som inte har en fysisk mapp
    if args.data_root and args.data_root.exists():
        existing_labels = {
            label for label in labels 
            if (args.data_root / label).is_dir()
        }
        # Vi behåller stats bara för de som finns
        stats = [s for s in stats if s["label"] in existing_labels]
        # Uppdatera även labels referensen om vi ska använda den senare, 
        # men stats_to_show baseras på stats
    else:
        existing_labels = set(labels)

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

    if args.csv_per_image:
        default_processed = args.embeddings.parent / "processed-ppic.jsonl"
        proc_path = args.processed or default_processed
        paths = load_paths_from_processed_json(proc_path)
        if len(paths) != len(y):
            print(
                f"⚠️ Antal paths i {proc_path} ({len(paths)}) matchar inte embeddings ({len(y)}). "
                "Fortsätter ändå (saknade paths lämnas tomma)."
            )
        rows = build_per_image_rows(X, y, labels, centroids, paths)
        
        # Filtrera även per-bild-rader på existerande mappar
        if args.data_root and args.data_root.exists():
             rows = [r for r in rows if r["label"] in existing_labels]
             
        write_per_image_csv(args.csv_per_image, rows, ["path", "folder_name", "label", "confidence", "top_label", "top_confidence", "delta", "mismatch"])
        print(f"Skrev {len(rows)} per-bildrader till {args.csv_per_image}")


if __name__ == "__main__":
    main()
