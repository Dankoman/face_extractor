#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import cv2
from insightface.app import FaceAnalysis
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_JSON_DEFAULT = SCRIPT_DIR / "arcface_work-ppic" / "processed-ppic.jsonl"
GENDER_PROTO_DEFAULT = SCRIPT_DIR / "models" / "deploy_gender.prototxt"
GENDER_MODEL_DEFAULT = SCRIPT_DIR / "models" / "gender_net.caffemodel"


def load_gender_net(proto: Path, model: Path):
    if not proto.exists() or not model.exists():
        raise SystemExit(f"Missing gender model files: {proto} / {model}")
    return cv2.dnn.readNetFromCaffe(str(proto), str(model))


def iter_processed(processed_json: Path) -> Iterable[Path]:
    if not processed_json.exists():
        raise SystemExit(f"Processed file missing: {processed_json}")
    with processed_json.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = rec.get("path")
            if not rec.get("ok") or not path:
                continue
            yield Path(path)


def detect_gender(app: FaceAnalysis, net, image_path: Path, verbose: bool = False):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        if verbose:
            print(f"[skip] Could not read image: {image_path}")
        return None
    faces = app.get(bgr)
    if not faces:
        if verbose:
            print(f"[skip] No face detected: {image_path}")
        return None
    face = max(faces, key=lambda f: f.det_score)
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, bgr.shape[1]), min(y2, bgr.shape[0])
    if x2 <= x1 or y2 <= y1:
        if verbose:
            print(f"[skip] Invalid bbox for {image_path}")
        return None
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        if verbose:
            print(f"[skip] Empty crop for {image_path}")
        return None
    try:
        blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
    except Exception:
        if verbose:
            print(f"[skip] Failed to create blob for {image_path}")
        return None
    net.setInput(blob)
    preds = net.forward()[0]
    male = float(preds[0])
    female = float(preds[1]) if len(preds) > 1 else 0.0
    return male, female


def main():
    parser = argparse.ArgumentParser(description="Lista etiketter vars ansikten klassas som manliga")
    parser.add_argument("--processed", type=Path, default=PROCESSED_JSON_DEFAULT,
                        help="Path till processed-ppic.jsonl")
    parser.add_argument("--gender-proto", type=Path, default=GENDER_PROTO_DEFAULT,
                        help="deploy_gender.prototxt")
    parser.add_argument("--gender-model", type=Path, default=GENDER_MODEL_DEFAULT,
                        help="gender_net.caffemodel")
    parser.add_argument("--csv", type=Path,
                        help="Skriv resultat till CSV (format: label|male_prob|female_prob|path)")
    parser.add_argument("--verbose", action="store_true", help="Visa detaljerad logg under körning")
    args = parser.parse_args()

    print("Loading insightface detection…")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)
    print("Loading gender net…")
    gender_net = load_gender_net(args.gender_proto, args.gender_model)

    male_hits: Dict[str, int] = defaultdict(int)
    total = 0

    csv_writer = None
    csv_handle = None
    existing_paths = set()
    existing_hits: Dict[str, int] = defaultdict(int)
    existing_records = 0
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = True
        if args.csv.exists():
            with args.csv.open("r", encoding="utf-8", newline="") as existing:
                reader = csv.reader(existing, delimiter='|')
                for row in reader:
                    if not row:
                        continue
                    if row[0] == "label" and len(row) >= 4:
                        write_header = False
                        continue
                    if len(row) < 4:
                        continue
                    label, male_prob_str, female_prob_str, path = row[0], row[1], row[2], row[3]
                    existing_paths.add(path)
                    try:
                        male_val = float(male_prob_str)
                        female_val = float(female_prob_str)
                    except ValueError:
                        male_val = 0.0
                        female_val = 0.0
                    if male_val >= female_val:
                        existing_hits[label] += 1
                    existing_records += 1
        csv_handle = args.csv.open("a", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_handle, delimiter='|')
        if write_header:
            csv_writer.writerow(["label", "male_prob", "female_prob", "path"])

    processed_paths = list(iter_processed(args.processed))
    iterator = processed_paths if args.verbose else tqdm(processed_paths, desc="Scanning", unit="img")

    for image_path in iterator:
        label = image_path.parent.name
        image_path_str = str(image_path)
        if csv_writer and image_path_str in existing_paths:
            if args.verbose:
                print(f"[skip] Already recorded: {image_path_str}")
            continue
        result = detect_gender(app, gender_net, image_path, verbose=args.verbose)
        if result is None:
            continue
        male_prob, female_prob = result
        if args.verbose:
            print(f"[hit] {label} -> male={male_prob:.3f} female={female_prob:.3f}")
        total += 1
        if csv_writer:
            csv_writer.writerow([label, f"{male_prob:.3f}", f"{female_prob:.3f}", str(image_path)])
            csv_handle.flush()
        if male_prob >= female_prob:
            male_hits[label] += 1

    if csv_handle:
        csv_handle.close()
        print(f"Skrev CSV: {args.csv}")

    # Merge existing hit counts with newly found ones
    for label, count in existing_hits.items():
        male_hits[label] += count

    if not male_hits:
        print("No male faces detected.")
        return

    print("Labels with at least one male-classified face:")
    for label, cnt in sorted(male_hits.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{label}: {cnt}")
    total_all = total + existing_records
    print(f"\nNew faces checked this run: {total}")
    if existing_records:
        print(f"Existing records reused: {existing_records}")
    print(f"Total faces accounted for: {total_all}")
    print(f"Total labels flagged: {len(male_hits)}")


if __name__ == "__main__":
    main()
