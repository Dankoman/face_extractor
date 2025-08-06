#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API-endpoint för Stash plugin integration
Utökar befintlig Flask-app med JSON API för ansiktsigenkänning
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
import base64
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


def process_faces_for_api(faces, predictions, image_width, image_height):
    """Konverterar ansiktsdata till API-format för Stash plugin."""
    results = []
    
    for face, (name, cos_sim) in zip(faces, predictions):
        bbox = face.bbox.astype(int)
        confidence = cos_sim
        
        # Konvertera bounding box till format som förväntas av plugin
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
    """API-endpoint för ansiktsigenkänning från Stash plugin."""
    try:
        # Kontrollera att en bild skickades
        if 'image' not in request.files:
            return jsonify({"error": "Ingen bild skickad"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Ingen fil vald"}), 400
        
        # Spara temporär fil
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Ladda bilden och detektera ansikten
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
            
            # Predicera för varje ansikte
            predictions = []
            for face in faces:
                emb = face.embedding
                if emb is None or emb.size == 0:
                    predictions.append(("UNKNOWN", 0.0))
                    continue
                
                probs = clf.predict_proba([emb])[0]
                pred_id = np.argmax(probs)
                name = le.inverse_transform([pred_id])[0]
                
                # Cosine-similaritet
                neigh_id = clf.kneighbors([emb], n_neighbors=1, return_distance=False)[0][0]
                proto = clf._fit_X[neigh_id]
                cos_sim = cosine_similarity([emb], [proto])[0, 0]
                
                if cos_sim < THRESHOLD:
                    name = "UNKNOWN"
                
                predictions.append((name, cos_sim))
            
            # Formatera resultat för API
            result = process_faces_for_api(faces, predictions, img_rgb.shape[1], img_rgb.shape[0])
            
            return jsonify(result)
            
        finally:
            # Rensa temporär fil
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        return jsonify({"error": f"Serverfel: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Hälsokontroll för API."""
    return jsonify({
        "status": "ok",
        "service": "face_extractor",
        "version": "1.0.0",
        "model_loaded": clf is not None,
        "threshold": THRESHOLD
    })


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Hämta aktuell konfiguration."""
    return jsonify({
        "threshold": THRESHOLD,
        "model_path": MODEL_PATH,
        "classes": le.classes_.tolist() if hasattr(le, 'classes_') else []
    })


@app.route("/api/config", methods=["POST"])
def api_set_config():
    """Uppdatera konfiguration."""
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


# Behåll befintliga routes för webbgränssnittet
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Ladda upp filen
        file = request.files["image"]
        if not file:
            return "Ingen bild uppladdad", 400

        # Spara uppladdad bild
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Ladda bilden och detektera ansikten
        img_rgb = load_image_rgb(Path(image_path))
        faces = detect_once(app_insight, img_rgb)

        if len(faces) == 0:
            return "Inga ansikten hittades i bilden", 400

        # Predicera för varje ansikte
        predictions = []
        for face in faces:
            emb = face.embedding
            if emb is None or emb.size == 0:
                predictions.append(("UNKNOWN", 0.0))
                continue

            probs = clf.predict_proba([emb])[0]
            pred_id = np.argmax(probs)
            name = le.inverse_transform([pred_id])[0]

            # Cosine-similaritet
            neigh_id = clf.kneighbors([emb], n_neighbors=1, return_distance=False)[0][0]
            proto = clf._fit_X[neigh_id]
            cos_sim = cosine_similarity([emb], [proto])[0, 0]

            if cos_sim < THRESHOLD:
                name = "UNKNOWN"

            predictions.append((name, cos_sim))

        # Rita bounding boxes och spara resultat
        result_path = draw_bounding_boxes(image_path, faces, predictions)

        # Returnera resultatet med en ny HTML-sida
        result_image_url = url_for('static', filename=f"results/{os.path.basename(image_path)}")
        return render_template("result.html", result_image=result_image_url)

    return render_template("index.html")


def draw_bounding_boxes(image_path, faces, predictions):
    """Ritar bounding boxes och namn på bilden med färger baserade på cos_sim."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except OSError:
        print("⚠️  Kunde inte ladda fonten, använder standardfont.")
        font = ImageFont.load_default()

    for face, (name, cos_sim) in zip(faces, predictions):
        bbox = face.bbox.astype(int)  # Bounding box

        # Välj färg baserat på cos_sim
        if cos_sim < 0.3:
            color = "red"
        elif 0.3 <= cos_sim <= 0.7:
            color = "yellow"
        else:
            color = "green"

        # Formatera värdet som procent
        percentage = cos_sim * 100
        label = f"{name} ({percentage:.1f}%)"

        # Rita bounding box och text
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=3)
        draw.text((bbox[0], bbox[1] - 20), label, fill=color, font=font)

    # Spara resultatbilden i RESULT_FOLDER
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    img.save(result_path)
    return result_path


@app.route("/clear_temp", methods=["POST"])
def clear_temp():
    """Rensar temporära filer."""
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Ta bort filen
            except Exception as e:
                print(f"⚠️  Kunde inte ta bort {file_path}: {e}")
    return "Temporära filer rensade", 200


if __name__ == "__main__":
    print("🚀 Startar Face Extractor API...")
    print(f"📊 Modell: {MODEL_PATH}")
    print(f"🎯 Threshold: {THRESHOLD}")
    print(f"👥 Klasser: {len(le.classes_) if hasattr(le, 'classes_') else 'Okänt'}")
    print("🌐 API-endpoints:")
    print("   POST /api/detect - Ansiktsigenkänning")
    print("   GET  /api/health - Hälsokontroll")
    print("   GET  /api/config - Hämta konfiguration")
    print("   POST /api/config - Uppdatera konfiguration")
    print("📱 Webbgränssnitt: http://localhost:5001")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

