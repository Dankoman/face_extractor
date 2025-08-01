#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from insightface.app import FaceAnalysis

# ---------- Standardvärden ----------
DEFAULT_MODEL       = "face_knn_arcface_interim.pkl"
DEFAULT_THRESHOLD   = 0.55
MIN_WIDTH           = 40
MIN_HEIGHT          = 40
UPSAMPLE_TARGET_MIN = 180
ROTATION_DEGREES    = [90, -90, 180]  # kan utökas med små vinklar om du vill
# ------------------------------------


# ---------- Bild/embedding-hjälp ----------
def load_image_rgb(path: Path) -> np.ndarray | None:
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        return None

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

def rotate_rgb(img_rgb: np.ndarray, deg: float) -> np.ndarray:
    return np.array(Image.fromarray(img_rgb).rotate(deg, expand=True, resample=Image.BICUBIC))

def get_embedding_direct(rec_model, img_rgb: np.ndarray) -> np.ndarray | None:
    """Direkt embedding (utan detektor). Resize -> 112x112."""
    img_112 = Image.fromarray(img_rgb).resize((112, 112), Image.BICUBIC)
    bgr = rgb_to_bgr(np.array(img_112)).astype(np.uint8)
    emb = rec_model.get(bgr)
    return emb.astype(np.float32) if emb is not None and emb.size else None


def detect_once(app: FaceAnalysis, img_rgb: np.ndarray):
    faces = app.get(rgb_to_bgr(img_rgb))
    return faces


def compute_embedding(app: FaceAnalysis,
                      rec_model,
                      img_path: Path,
                      allow_fallback: bool,
                      allow_upsample: bool,
                      try_rotate: bool):
    """
    Returnerar (embedding, n_faces, used_fallback, used_rotation_deg)
    - Accepterar exakt 1 ansikte. Om fallback används räknas det som 1.
    - Testar upsampling och rotationer innan fallback.
    """
    img_rgb = load_image_rgb(img_path)
    if img_rgb is None:
        return None, 0, False, None

    h, w = img_rgb.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        if allow_fallback:
            emb = get_embedding_direct(rec_model, img_rgb)
            return (emb, 1 if emb is not None else 0, True, None)
        return None, 0, False, None

    # 1) Försök detektera direkt
    faces = detect_once(app, img_rgb)
    if len(faces) == 1:
        emb = faces[0].embedding
        return (emb.astype(np.float32), 1, False, None) if emb is not None else (None, 1, False, None)

    # 2) Upsample om 0 och tillåtet
    if len(faces) == 0 and allow_upsample:
        img_up = upsample_if_needed(img_rgb)
        if img_up is not img_rgb:
            faces = detect_once(app, img_up)
            if len(faces) == 1:
                emb = faces[0].embedding
                return (emb.astype(np.float32), 1, False, None) if emb is not None else (None, 1, False, None)

    # 3) Rotationer
    if len(faces) == 0 and try_rotate:
        for deg in ROTATION_DEGREES:
            img_rot = rotate_rgb(img_rgb, deg)
            faces_r = detect_once(app, img_rot)
            if len(faces_r) == 1:
                emb = faces_r[0].embedding
                if emb is not None and emb.size:
                    return emb.astype(np.float32), 1, False, deg

    # 4) Fallback
    if len(faces) == 0 and allow_fallback:
        emb = get_embedding_direct(rec_model, img_rgb)
        return (emb, 1 if emb is not None else 0, True, None)

    # >1 eller fortfarande 0 (utan fallback)
    return None, len(faces), False, None


# ---------- Huvudprogram ----------
def main():
    ap = argparse.ArgumentParser(description="Testa ArcFace-KNN-modellen på en bild")
    ap.add_argument("image", type=Path, help="Sökväg till bilden som ska testas")
    ap.add_argument("--model", default=DEFAULT_MODEL, type=Path, help="Pickle-fil med KNN-modell")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Cosine-tröskel för UNKNOWN (sätt negativt för att stänga av)")
    ap.add_argument("--topk", type=int, default=3, help="Visa topp K kandidater")
    ap.add_argument("--allow-fallback", action="store_true", help="Direkt embedding om detektor missar")
    ap.add_argument("--allow-upsample", action="store_true", help="Upsampla små bilder innan ny detektion")
    ap.add_argument("--try-rotate", action="store_true", help="Prova rotera bilden (90/-90/180) om ingen detektion")
    ap.add_argument("--provider", nargs="+", default=["CPUExecutionProvider"],
                    help="ONNXRuntime providers, t.ex. CUDAExecutionProvider")
    args = ap.parse_args()

    # Ladda modell
    with args.model.open("rb") as f:
        bundle = pickle.load(f)
    if "model" not in bundle or "label_encoder" not in bundle:
        raise ValueError(f"{args.model} saknar 'model' och/eller 'label_encoder'")
    clf = bundle["model"]
    le  = bundle["label_encoder"]

    # Initiera InsightFace
    app = FaceAnalysis(name="buffalo_l", providers=args.provider)
    app.prepare(ctx_id=0)
    rec_model = app.models["recognition"]

    # Ladda bilden
    img_rgb = load_image_rgb(args.image)
    if img_rgb is None:
        print(f"❌ Kunde inte läsa bilden: {args.image}")
        return

    # Detektera ansikten
    faces = detect_once(app, img_rgb)
    if len(faces) == 0:
        print(f"⏭️ {args.image}: Inga ansikten hittades")
        return

    print(f"Bild: {args.image}")
    print(f"Antal ansikten: {len(faces)}")

    # Iterera över alla ansikten
    for i, face in enumerate(faces):
        emb = face.embedding
        if emb is None or emb.size == 0:
            print(f"  Ansikte {i + 1}: Ingen embedding tillgänglig")
            continue

        # Predicera
        probs = clf.predict_proba([emb])[0]
        pred_id = np.argmax(probs)
        prob = probs[pred_id]
        name = le.inverse_transform([pred_id])[0]

        # Cosine mot närmaste prototyp
        neigh_id = clf.kneighbors([emb], n_neighbors=1, return_distance=False)[0][0]
        proto    = clf._fit_X[neigh_id]
        cos_sim  = cosine_similarity([emb], [proto])[0, 0]

        unknown = False
        if args.threshold >= 0 and cos_sim < args.threshold:
            name = "UNKNOWN"
            unknown = True

        # Utskrift för varje ansikte
        print(f"  Ansikte {i + 1}:")
        print(f"    Prediktion:  {name}")
        print(f"    P(class):    {prob:.3f}")
        print(f"    CosSim:      {cos_sim:.3f}  (thr={args.threshold})")

        if not unknown and args.topk > 1:
            idxs = probs.argsort()[-args.topk:][::-1]
            print("    Top-K:")
            for j in idxs:
                print(f"      {le.inverse_transform([j])[0]:20s}  prob={probs[j]:.3f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)
