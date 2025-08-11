#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API-endpoint för Stash plugin integration
- Behåller befintliga /api/detect (multipart/form-data)
- Lägger till /recognize (råt kroppsinnehåll eller multipart) + stöd för topp-K kandidater
- Returnerar format som pluginet förväntar sig: [{ box:{x,y,w,h}, candidates:[{name, score}] }]
"""

import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, url_for
from flask_cors import CORS
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from test_model import load_image_rgb, detect_once, FaceAnalysis, cosine_similarity
import tempfile
from io import BytesIO

# Flask-app
app = Flask(__name__)
CORS(app)  # Aktivera CORS för plugin-integration

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initiera InsightFace
app_insight = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app_insight.prepare(ctx_id=0)

# Ladda KNN-modellen
MODEL_PATH = "arcface_work-ppic/face_knn_arcface_ppic.pkl"
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
le = bundle["label_encoder"]

# Standardinställningar
THRESHOLD = 0.2
FONT_PATH = "/nix/store/59p03gp3vzbrhd7xjiw3npgbdd68x3y0-dejavu-fonts-2.37/share/fonts/truetype/DejaVuSans-Bold.ttf"


# ------------------------- Hjälpfunktioner -------------------------

def _save_temp_image(file_storage=None, raw_bytes: bytes | None = None) -> str:
    """Spara inkommande bild (multipart eller rå body) till en tempfil och returnera sökvägen."""
    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if file_storage is not None:
            file_storage.save(tmp.name)
        else:
            tmp.write(raw_bytes)
        return tmp.name


def _predict_top_k(emb: np.ndarray, k: int) -> list[dict]:
    """Returnera topp-K kandidater som [{name, score}] där score är cosinus-likhet (0..1)."""
    # Hämta K närmaste grannar
    n_neighbors = max(1, int(k))
    neigh_idx = clf.kneighbors([emb], n_neighbors=n_neighbors, return_distance=False)[0]

    # Räkna ut cosinus-likhet mot respektive prototyp/lagrade vektor
    # clf._fit_X är (N_protos, D) – samma ordning som neigh_idx
    candidates = []
    seen = set()
    for idx in neigh_idx:
        proto = clf._fit_X[idx]
        sim = float(cosine_similarity([emb], [proto])[0, 0])  # 0..1
        name = str(le.inverse_transform([clf._y[idx]])[0]) if hasattr(clf, "_y") else str(le.inverse_transform([idx])[0])
        if name not in seen:
            candidates.append({"name": name, "score": sim})
            seen.add(name)
    return candidates


def _faces_to_plugin_boxes(faces) -> list[dict]:
    """Konvertera ansikten till {x,y,w,h}-boxar."""
    boxes = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        boxes.append({"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)})
    return boxes


# ------------------------- Nya API:t: /recognize -------------------------
@app.post("/recognize")
def recognize():
    """
    In:  rå bild (request.data) ELLER multipart 'image'
    QS:  top_k (default 3)
    Ut:  [ { box:{x,y,w,h}, candidates:[{name,score}, ...] }, ... ]
    """
    try:
        # Läs indata
        file = request.files.get("image") if "image" in request.files else None
        raw = request.get_data() if (file is None) else None
        if file is None and (raw is None or len(raw) == 0):
            return jsonify({"error": "Ingen bild skickad"}), 400

        tmp_path = _save_temp_image(file_storage=file, raw_bytes=raw)
        try:
            img_rgb = load_image_rgb(Path(tmp_path))
            if img_rgb is None:
                return jsonify({"error": "Kunde inte läsa bilden"}), 400

            faces = detect_once(app_insight, img_rgb)
            if len(faces) == 0:
                return jsonify([])

            top_k = int(request.args.get("top_k", 3))
            results = []
            boxes = _faces_to_plugin_boxes(faces)

            for face, box in zip(faces, boxes):
                emb = face.embedding
                if emb is None or emb.size == 0:
                    results.append({"box": box, "candidates": []})
                    continue

                candidates = _predict_top_k(emb, k=top_k)

                # Filtrera bort låga scores på serversidan (behåll allt; klienten filtrerar på min_confidence)
                results.append({"box": box, "candidates": candidates})

            return jsonify(results)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        return jsonify({"error": f"Serverfel: {str(e)}"}), 500


# ------------------------- Befintligt API: /api/detect -------------------------

def process_faces_for_api(faces, predictions, image_width, image_height):
    """Konverterar ansiktsdata till äldre API-format (kompat för befintlig UI)."""
    results = []
    for face, (name, cos_sim) in zip(faces, predictions):
        bbox = face.bbox.astype(int)
        confidence = cos_sim
        face_data = {
            "name": name,
            "confidence": float(confidence),
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        }
        results.append(face_data)
    return {
        "faces": results,
        "image_width": image_width,
        "image_height": image_height,
        "total_faces": len(results)
    }


@app.route("/api/detect", methods=["POST"])
def api_detect_faces():
    """Äldre endpoint som returnerar en bästa-kandidat per ansikte."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Ingen bild skickad"}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Ingen fil vald"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            img_rgb = load_image_rgb(Path(temp_path))
            if img_rgb is None:
                return jsonify({"error": "Kunde inte läsa bilden"}), 400

            faces = detect_once(app_insight, img_rgb)
            if len(faces) == 0:
                return jsonify({
                    "faces": [],
                    "image_width": img_rgb.shape[1],
                    "image_height": img_rgb.shape[0],
                    "total_faces": 0
                })

            predictions = []
            for face in faces:
                emb = face.embedding
                if emb is None or emb.size == 0:
                    predictions.append(("UNKNOWN", 0.0))
                    continue

                probs = clf.predict_proba([emb])[0]
                pred_id = np.argmax(probs)
                name = le.inverse_transform([pred_id])[0]

                neigh_id = clf.kneighbors([emb], n_neighbors=1, return_distance=False)[0][0]
                proto = clf._fit_X[neigh_id]
                cos_sim = cosine_similarity([emb], [proto])[0, 0]

                if cos_sim < THRESHOLD:
                    name = "UNKNOWN"

                predictions.append((name, float(cos_sim)))

            result = process_faces_for_api(faces, predictions, img_rgb.shape[1], img_rgb.shape[0])
            return jsonify(result)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    except Exception as e:
        return jsonify({"error": f"Serverfel: {str(e)}"}), 500


# ------------------------- Övriga endpoints -------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "service": "face_extractor",
        "version": "1.1.0",
        "model_loaded": clf is not None,
        "threshold": THRESHOLD
    })


@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify({
        "threshold": THRESHOLD,
        "model_path": MODEL_PATH,
        "classes": le.classes_.tolist() if hasattr(le, 'classes_') else []
    })


@app.route("/api/config", methods=["POST"])
def api_set_config():
    global THRESHOLD
    try:
        data = request.get_json()
        if 'threshold' in data:
            new_threshold = float(data['threshold'])
            if 0.0 <= new_threshold <= 1.0:
                THRESHOLD = new_threshold
                return jsonify({"message": "Threshold uppdaterad", "threshold": THRESHOLD})
            else:
                return jsonify({"error": "Threshold måste vara mellan 0.0 och 1.0"}), 400
        return jsonify({"error": "Ingen giltig konfiguration skickad"}), 400
    except Exception as e:
        return jsonify({"error": f"Fel vid uppdatering: {str(e)}"}), 500


# ------------------------- Webbgränssnitt (oförändrat) -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if not file:
            return "Ingen bild uppladdad", 400

        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        img_rgb = load_image_rgb(Path(image_path))
        faces = detect_once(app_insight, img_rgb)

        if len(faces) == 0:
            return "Inga ansikten hittades i bilden", 400

        predictions = []
        for face in faces:
            emb = face.embedding
            if emb is None or emb.size == 0:
                predictions.append(("UNKNOWN", 0.0))
                continue

            probs = clf.predict_proba([emb])[0]
            pred_id = np.argmax(probs)
            name = le.inverse_transform([pred_id])[0]

            neigh_id = clf.kneighbors([emb], n_neighbors=1, return_distance=False)[0][0]
            proto = clf._fit_X[neigh_id]
            cos_sim = cosine_similarity([emb], [proto])[0, 0]

            if cos_sim < THRESHOLD:
                name = "UNKNOWN"

            predictions.append((name, cos_sim))

        result_path = draw_bounding_boxes(image_path, faces, predictions)

        result_image_url = url_for('static', filename=f"results/{os.path.basename(image_path)}")
        return render_template("result.html", result_image=result_image_url)

    return render_template("index.html")


def draw_bounding_boxes(image_path, faces, predictions):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except OSError:
        print("⚠️  Kunde inte ladda fonten, använder standardfont.")
        font = ImageFont.load_default()

    for face, (name, cos_sim) in zip(faces, predictions):
        bbox = face.bbox.astype(int)
        if cos_sim < 0.3:
            color = "red"
        elif 0.3 <= cos_sim <= 0.7:
            color = "yellow"
        else:
            color = "green"
        percentage = cos_sim * 100
        label = f"{name} ({percentage:.1f}%)"
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=3)
        draw.text((bbox[0], bbox[1] - 20), label, fill=color, font=font)

    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    img.save(result_path)
    return result_path


@app.route("/clear_temp", methods=["POST"])
def clear_temp():
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"⚠️  Kunde inte ta bort {file_path}: {e}")
    return "Temporära filer rensade", 200


if __name__ == "__main__":
    print("🚀 Startar Face Extractor API...")
    print(f"📊 Modell: {MODEL_PATH}")
    print(f"🎯 Threshold: {THRESHOLD}")
    print(f"👥 Klasser: {len(le.classes_) if hasattr(le, 'classes_') else 'Okänt'}")
    print("🌐 API-endpoints:")
    print("   POST /recognize      - Ansiktsigenkänning (topp-K, rå eller multipart)")
    print("   POST /api/detect     - (legacy) Bästa-kandidat per ansikte")
    print("   GET  /api/health     - Hälsokontroll")
    print("   GET  /api/config     - Hämta konfiguration")
    print("   POST /api/config     - Uppdatera konfiguration")
    print("📱 Webbgränssnitt: http://localhost:5001")

    app.run(host='0.0.0.0', port=5000, debug=True)
