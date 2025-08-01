import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, render_template, send_file, url_for
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from test_model import load_image_rgb, detect_once, FaceAnalysis, cosine_similarity

# Flask-app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initiera InsightFace
app_insight = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app_insight.prepare(ctx_id=0)

# Ladda KNN-modellen
MODEL_PATH = "face_knn_arcface_new.pkl"
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
le = bundle["label_encoder"]

# Standardinställningar
THRESHOLD = 0.2
FONT_PATH = "/nix/store/59p03gp3vzbrhd7xjiw3npgbdd68x3y0-dejavu-fonts-2.37/share/fonts/truetype/DejaVuSans-Bold.ttf"  # Ändra om nödvändigt


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
    app.run(host='0.0.0.0', port=5000, debug=True)