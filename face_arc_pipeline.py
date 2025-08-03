#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArcFace-pipeline med progressbar & checkpoints.

- Läser data_root/person/*.jpg|png|...
- Detekterar ansikten med InsightFace (SCRFD), beräknar embeddings (ArcFace)
- Upsamplar små bilder innan detektion (valbart)
- Fallback: om ingen detektion -> direkt embedding på 112x112 (valbart)
- Sparar embeddings + etiketter i embeddings.pkl
- Tränar KNN (cosine) och sparar modell i face_knn_arcface.pkl
- Checkpoints: processed.jsonl + embeddings.pkl
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from insightface.app import FaceAnalysis

# Dynamisk sökväg till script-katalogen för modeller
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# ------------------ Default ------------------
DEFAULT_DATA_ROOT   = "/home/marqs/Bilder/pBook"
DEFAULT_WORKDIR     = "./arcface_work-ppic"
EMB_PKL             = "embeddings_ppic.pkl"
PROC_JSONL          = "processed-ppic.jsonl"
MODEL_PKL           = "face_knn_arcface_ppic.pkl"

MIN_WIDTH           = 40
MIN_HEIGHT          = 40
UPSAMPLE_TARGET_MIN = 180   # minsta sida efter upsampling-försök

K_DEFAULT           = 3
MIN_PER_CLASS_DEF   = 2
# ---------------------------------------------

# ---------------- Gender-CNN ------------------
GENDER_PROTO = str(MODELS_DIR / "deploy_gender.prototxt")
GENDER_MODEL = str(MODELS_DIR / "gender_net.caffemodel")
# Läs in köns-CNN först
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
GENDER_LIST  = ["Male", "Female"]
FEMALE_THRESH = 0.7
# ---------------------------------------------

# -------------- Hjälpfunktioner --------------

def init_app(providers: List[str]) -> FaceAnalysis:
    app = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
        allowed_modules=['detection','recognition','genderage']
    )
    app.prepare(ctx_id=0)
    return app


def load_processed(proc_path: Path) -> set:
    done = set()
    if proc_path.exists():
        with proc_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["path"])
                except Exception:
                    pass
    return done


def append_processed(proc_path: Path, img_path: str, ok: bool) -> None:
    with proc_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"path": img_path, "ok": ok}) + "\n")


def load_embeddings(emb_path: Path) -> Tuple[List[np.ndarray], List[str]]:
    if not emb_path.exists():
        return [], []
    with emb_path.open("rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"]


def save_embeddings(emb_path: Path, X, y):
    emb_path = Path(emb_path)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = emb_path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    tmp.rename(emb_path)


def iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for person_dir in root.iterdir():
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        for p in person_dir.iterdir():
            if p.suffix.lower() in exts:
                yield str(p), label


def load_image_rgb(path: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    return np.array(img)


def rgb_to_bgr(arr: np.ndarray) -> np.ndarray:
    return arr[:, :, ::-1]


def upsample_if_needed(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    m = min(h, w)
    if m >= UPSAMPLE_TARGET_MIN:
        return img_rgb
    scale = UPSAMPLE_TARGET_MIN / m
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return np.array(Image.fromarray(img_rgb).resize((new_w, new_h), Image.BICUBIC))


def get_embedding_direct(rec_model, img_rgb: np.ndarray) -> Optional[np.ndarray]:
    if img_rgb is None:
        return None
    img_112 = Image.fromarray(img_rgb).resize((112, 112), Image.BICUBIC)
    bgr = rgb_to_bgr(np.array(img_112)).astype(np.uint8)
    emb = rec_model.get(bgr)
    if emb is None or emb.size == 0:
        return None
    return emb.astype(np.float32)


# ----------- Embedding-funktion -----------

def compute_embedding(app: FaceAnalysis,
                      rec_model,
                      img_path: str,
                      allow_fallback: bool,
                      allow_upsample: bool) -> tuple[Optional[np.ndarray], int, bool]:
    """
    Returnerar (embedding, antal_ansikten, fallback_användes).
    Endast om exakt 1 ansikte hittas och det klassificeras som kvinna returneras embedding.
    """
    img_rgb = load_image_rgb(img_path)
    if img_rgb is None:
        return None, 0, False

    h, w = img_rgb.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return None, 0, False

    if allow_upsample:
        img_rgb = upsample_if_needed(img_rgb)

    bgr = rgb_to_bgr(img_rgb)
    faces = app.get(bgr)
    if len(faces) != 1:
        return None, len(faces), False

    face = faces[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    face_crop = bgr[y1:y2, x1:x2]

    blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()[0]
    female_prob = float(preds[1])
    if female_prob < FEMALE_THRESH:
        return None, 1, False

    emb = face.embedding
    if emb is None or emb.size == 0:
        return None, 1, False

    return emb.astype(np.float32), 1, False


# ----------------- Pipeline -----------------

def encode(args) -> None:
    data_root = Path(args.data_root)
    workdir   = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    emb_path  = workdir / EMB_PKL
    proc_path = workdir / PROC_JSONL

    X, y      = load_embeddings(emb_path)
    processed = load_processed(proc_path)

    all_imgs = list(iter_images(data_root))
    todo = [(p, lbl) for p, lbl in all_imgs if p not in processed]

    if not todo:
        print("✅ Inget att göra – allt är redan processat.")
        return

    print(f"🔍 Att processa: {len(todo)} bilder (skippar {len(processed)})")

    app = init_app(["CPUExecutionProvider"])
    rec_model = app.models["recognition"]

    try:
        for path, label in tqdm(todo, unit="img"):
            ok = False
            try:
                emb, n, fb = compute_embedding(
                    app,
                    rec_model,
                    path,
                    allow_fallback=args.allow_fallback,
                    allow_upsample=args.allow_upsample
                )
                if emb is not None:
                    X.append(emb)
                    y.append(label)
                    ok = True
            except Exception as e:
                print(f"❌ {path}: {e}", file=sys.stderr)

            append_processed(proc_path, path, ok)
            if ok and (len(X) % args.flush_every == 0):
                save_embeddings(emb_path, X, y)

    except KeyboardInterrupt:
        print("\n⏹️ Avbrutet. Sparar state...")

    finally:
        save_embeddings(emb_path, X, y)
        print(f"💾 Sparade embeddings ({len(X)}) till {emb_path}")


def train(args) -> None:
    workdir   = Path(args.workdir)
    emb_path  = workdir / EMB_PKL
    model_out = Path(args.model_out)

    X_list, y_list = load_embeddings(emb_path)
    if not X_list:
        print("❌ Inga embeddings – kör encode först.")
        sys.exit(1)

    X = np.vstack(X_list)
    y = np.array(y_list)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    counts = {cls: np.sum(y == cls) for cls in np.unique(y)}
    min_count = min(counts.values())
    k = 1 if min_count < args.min_per_class else args.k

    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine")
    clf.fit(X, y_enc)

    with model_out.open("wb") as f:
        pickle.dump({"model": clf, "label_encoder": le}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Modell sparad: {model_out} (k={k}, klasser={len(np.unique(y))})")


def main():
    global EMB_PKL

    ap = argparse.ArgumentParser(description="ArcFace-pipeline (progressbar + checkpoints, fallback & upsampling)")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Rotmapp med personmappar")
    ap.add_argument("--workdir",   default=DEFAULT_WORKDIR,   help="Arbetsmapp/checkpoints")
    ap.add_argument("--embeddings", default=EMB_PKL,          help="Fil med embeddings (.pkl)")
    ap.add_argument("--model-out", default=MODEL_PKL,         help="Filnamn för sparad modell (.pkl)")
    ap.add_argument("--mode", choices=["encode", "train", "both"], default="both")
    ap.add_argument("--flush-every", type=int, default=200, help="Spara embeddings var N:e lyckade bild")
    ap.add_argument("-k", type=int, default=K_DEFAULT, help="k för KNN")
    ap.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS_DEF,
                    help="Om minsta klass < detta -> k=1")

    ap.add_argument("--allow-fallback", action="store_true",
                    help="Tillåt direkt embedding utan detektor om inga ansikten hittas")
    ap.add_argument("--allow-upsample", action="store_true",
                    help="Upsampla små bilder innan ny detektionskörning")

    args = ap.parse_args()
    EMB_PKL = args.embeddings

    if args.mode in ("encode", "both"):
        encode(args)
    if args.mode in ("train", "both"):
        train(args)

if __name__ == "__main__":
    main()
