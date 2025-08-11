#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Extractor – API + enkelt webgränssnitt
-------------------------------------------------
Nyheter i denna version:
- /recognize stödjer topp-K kandidater och tar råbytes direkt (clipboard/drag & drop) utan att först skriva till disk
- /web är ett litet HTML-gränssnitt som kan:
  * Klistra in bilder direkt från urklipp (Ctrl+V)
  * Dra-och-släpp eller välja fil
  * Visa topp 3 kandidater per ansikte (utan Stash-uppslag)

Beroenden (som i ditt projekt): Flask, numpy, Pillow, OpenCV, insightface (via FaceAnalysis mm.)
"""
from __future__ import annotations

import os
import io
import pickle
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

# === Importer från din kodbas ===
# Antaganden utifrån tidigare versioner; anpassa importväg vid behov
from test_model import FaceAnalysis, detect_once, cosine_similarity

# -------------------------------------------------
# Flask-app
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# Modell- och runtimeinit
# -------------------------------------------------
MODEL_PATH = "arcface_work-ppic/face_knn_arcface_ppic.pkl"
FONT_PATH = "/nix/store/59p03gp3vzbrhd7xjiw3npgbdd68x3y0-dejavu-fonts-2.37/share/fonts/truetype/DejaVuSans-Bold.ttf"

# InsightFace
app_insight = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app_insight.prepare(ctx_id=0)

# KNN + label encoder
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
le = bundle["label_encoder"]

# Legacy tröskel – används inte för topp-K filtrering i UI:t, men finns kvar
THRESHOLD = 0.2

# -------------------------------------------------
# Hjälpfunktioner
# -------------------------------------------------

def _imread_bytes_to_rgb(b: bytes) -> np.ndarray | None:
    """Läs bytes till RGB-ndarray (H,W,3). Returnerar None vid fel."""
    arr = np.frombuffer(b, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def _pil_draw_boxes(img_rgb: np.ndarray, faces, per_face_topk: List[List[Tuple[str, float]]]) -> bytes:
    """Ritar bbox + topp-3 etiketter på en kopia av bilden. Returnerar JPEG-bytes."""
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(FONT_PATH, 18)
    except Exception:
        font = ImageFont.load_default()

    for face, topk in zip(faces, per_face_topk):
        x1, y1, x2, y2 = face.bbox.astype(int)
        # färg enligt högsta score
        best = topk[0][1] if topk else 0.0
        if best < 0.3:
            color = (220, 60, 60)
        elif best < 0.7:
            color = (230, 200, 70)
        else:
            color = (70, 220, 120)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # skriv max tre rader
        y_txt = y1 - 20
        for name, sc in topk[:3]:
            txt = f"{name} ({int(sc*100)}%)"
            draw.text((x1, y_txt), txt, fill=color, font=font)
            y_txt -= 20

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _faces_to_boxes(faces):
    boxes = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        boxes.append({"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)})
    return boxes


def _predict_top_k(emb: np.ndarray, k: int) -> list[dict]:
    """Returnera topp-K [{name, score}] med cosinuslikhet mot närmaste prototyper."""
    n_neighbors = max(1, int(k))
    neigh_idx = clf.kneighbors([emb], n_neighbors=n_neighbors, return_distance=False)[0]
    candidates = []
    seen = set()
    # clf._fit_X – prototypvektorer i samma indexordning
    for idx in neigh_idx:
        proto = clf._fit_X[idx]
        sim = float(cosine_similarity([emb], [proto])[0, 0])  # 0..1
        # Mappa index -> klassnamn
        if hasattr(clf, "_y"):
            cls_idx = clf._y[idx]
            name = str(le.inverse_transform([cls_idx])[0])
        else:
            name = str(le.inverse_transform([idx])[0])
        if name not in seen:
            candidates.append({"name": name, "score": sim})
            seen.add(name)
    # sortera nedåt för säkerhets skull
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# -------------------------------------------------
# API – topp-K och råbytesstöd (clipboard/drag&drop)
# -------------------------------------------------
@app.post("/recognize")
def recognize():
    """
    In:
      - rå bytes i body (t.ex. fetch(blob)) ELLER multipart med 'image'
      - querystring: top_k (default 3)
    Ut:
      [ { box:{x,y,w,h}, candidates:[{name,score}, ...] }, ... ]
    """
    try:
        top_k = int(request.args.get("top_k", 3))
        file = request.files.get("image") if "image" in request.files else None

        if file is not None:
            data = file.read()
        else:
            data = request.get_data()  # rå kropp

        if not data:
            return jsonify({"error": "Ingen bild mottagen"}), 400

        img_rgb = _imread_bytes_to_rgb(data)
        if img_rgb is None:
            return jsonify({"error": "Kunde inte läsa bilden"}), 400

        faces = detect_once(app_insight, img_rgb)
        if not faces:
            return jsonify([])

        result = []
        for face in faces:
            emb = face.embedding
            if emb is None or emb.size == 0:
                result.append({"box": _faces_to_boxes([face])[0], "candidates": []})
                continue
            candidates = _predict_top_k(emb, k=top_k)
            result.append({"box": _faces_to_boxes([face])[0], "candidates": candidates})

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Serverfel: {e}"}), 500


# -------------------------------------------------
# Enkel webbsida – topp 3, clipboard & drag/drop
# -------------------------------------------------
@app.get("/web")
def web_ui():
    html = """
<!doctype html>
<html lang="sv">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Face Extractor – Webbtest</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;margin:0;background:#0e0f12;color:#eaecef}
    header{padding:16px 20px;border-bottom:1px solid #23252b}
    main{display:grid;grid-template-columns:420px 1fr;gap:16px;padding:16px}
    .pane{background:#14161b;border:1px solid #23252b;border-radius:12px}
    .pane h2{margin:0;padding:12px 14px;border-bottom:1px solid #23252b;font-size:15px;color:#b6bbc7}
    .pane .content{padding:12px 14px}
    #drop{display:flex;align-items:center;justify-content:center;height:280px;border:2px dashed #3a3f4b;border-radius:12px;background:#0e1014;color:#9aa0ac;text-align:center}
    #drop.drag{background:#111726;border-color:#6b83ff;color:#cfd6ff}
    #preview{display:block;max-width:100%;height:auto;border-radius:10px;border:1px solid #23252b}
    #candidates{display:flex;flex-direction:column;gap:10px}
    .cand{display:flex;justify-content:space-between;gap:12px;padding:10px;border:1px solid #23252b;border-radius:10px;background:#101216}
    .cand b{color:#fff}
    .row{display:flex;align-items:center;gap:8px}
    .meta{color:#9aa0ac;font-size:12px}
    canvas{display:block;max-width:100%;background:#000;border-radius:10px;border:1px solid #23252b}
    .hint{color:#9aa0ac;font-size:12px;margin-top:8px}
    .actions{display:flex;gap:10px}
    button{background:#2a61ff;border:0;color:#fff;padding:8px 12px;border-radius:10px;cursor:pointer}
    button:disabled{opacity:.6;cursor:not-allowed}
    input[type=file]{display:none}
    label.btn{display:inline-block;background:#1d2027;border:1px solid #2b2f39;color:#cbd1dd;padding:8px 12px;border-radius:10px;cursor:pointer}
  </style>
</head>
<body>
  <header>
    <strong>Face Extractor – Webbtest</strong>
    <span class="meta">(klistra in bild med Ctrl+V, eller dra-och-släpp / välj fil)</span>
  </header>
  <main>
    <section class="pane">
      <h2>Bildinmatning</h2>
      <div class="content">
        <div id="drop">Släpp bild här eller <label class="btn" for="file">välj fil</label><input type="file" id="file" accept="image/*"></div>
        <div class="hint">Tips: markera en bild i clipboard och tryck <b>Ctrl+V</b> här i fönstret.</div>
        <div class="actions" style="margin-top:10px">
          <button id="run" disabled>Kör igenkänning (topp 3)</button>
        </div>
      </div>
    </section>
    <section class="pane">
      <h2>Resultat</h2>
      <div class="content">
        <canvas id="canvas"></canvas>
        <div id="candidates"></div>
      </div>
    </section>
  </main>
  <script>
  const drop = document.getElementById('drop');
  const fileInput = document.getElementById('file');
  const runBtn = document.getElementById('run');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const candidatesEl = document.getElementById('candidates');
  let imageBlob = null;
  let img = new Image();

  function setImageFromBlob(blob){
    imageBlob = blob;
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      canvas.width = img.width; canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      runBtn.disabled = false;
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }

  // Drag & drop
  ;['dragenter','dragover'].forEach(evt=>drop.addEventListener(evt, e=>{e.preventDefault(); drop.classList.add('drag');}))
  ;['dragleave','drop'].forEach(evt=>drop.addEventListener(evt, e=>{e.preventDefault(); drop.classList.remove('drag');}))
  drop.addEventListener('drop', e=>{
    const f = e.dataTransfer.files?.[0];
    if (f) setImageFromBlob(f);
  });

  // File picker
  fileInput.addEventListener('change', e=>{
    const f = fileInput.files?.[0];
    if (f) setImageFromBlob(f);
  });

  // Clipboard paste (Ctrl+V)
  window.addEventListener('paste', e=>{
    const items = e.clipboardData?.items || [];
    for (const it of items){
      if (it.type.startsWith('image/')){
        const blob = it.getAsFile();
        if (blob){ setImageFromBlob(blob); break; }
      }
    }
  });

  async function recognize(){
    runBtn.disabled = true;
    candidatesEl.innerHTML = '';
    const resp = await fetch('/recognize?top_k=3', {
      method: 'POST',
      body: imageBlob
    });
    if(!resp.ok){
      candidatesEl.textContent = 'Fel: ' + resp.status;
      runBtn.disabled = false; return;
    }
    const data = await resp.json();

    // Rita om bilden
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 3;

    data.forEach((d, i)=>{
      const {x,y,w,h} = d.box;
      const best = (d.candidates?.[0]?.score)||0;
      // färg
      let color = 'rgb(70,220,120)';
      if (best < 0.3) color = 'rgb(220,60,60)';
      else if (best < 0.7) color = 'rgb(230,200,70)';
      ctx.strokeStyle = color;
      ctx.strokeRect(x,y,w,h);

      const box = document.createElement('div');
      box.className = 'cand';
      const title = document.createElement('div');
      title.innerHTML = `<div class="row"><b>Face ${i+1}</b><span class="meta">${w}×${h}</span></div>`;
      box.appendChild(title);

      (d.candidates||[]).slice(0,3).forEach(c=>{
        const row = document.createElement('div');
        row.className = 'row';
        row.innerHTML = `<span>${c.name}</span><span class="meta">${Math.round(c.score*100)}%</span>`;
        box.appendChild(row);
      });
      candidatesEl.appendChild(box);
    });

    runBtn.disabled = false;
  }

  runBtn.addEventListener('click', recognize);
  </script>
</body>
</html>
    """
    return render_template_string(html)


# -------------------------------------------------
# Hälso- och infoendpoints
# -------------------------------------------------
@app.get("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "service": "face_extractor",
        "version": "1.2.0",
        "model_loaded": clf is not None,
        "classes": len(le.classes_) if hasattr(le, 'classes_') else None,
    })


@app.get("/api/config")
def api_get_config():
    return jsonify({
        "threshold": THRESHOLD,
        "model_path": MODEL_PATH,
        "classes": le.classes_.tolist() if hasattr(le, 'classes_') else []
    })


@app.post("/api/config")
def api_set_config():
    global THRESHOLD
    data = request.get_json(silent=True) or {}
    if 'threshold' in data:
        new_threshold = float(data['threshold'])
        if 0.0 <= new_threshold <= 1.0:
            THRESHOLD = new_threshold
            return jsonify({"message": "Threshold uppdaterad", "threshold": THRESHOLD})
        return jsonify({"error": "Threshold måste vara mellan 0.0 och 1.0"}), 400
    return jsonify({"error": "Ingen giltig konfiguration skickad"}), 400


# -------------------------------------------------
# Legacy index (valfritt kvar)
# -------------------------------------------------
@app.get("/")
def index_redirect():
    # Peka om till nya UI:t
    return ("<meta http-equiv='refresh' content='0;url=/web'>"), 302


if __name__ == "__main__":
    print("🚀 Startar Face Extractor API…")
    print("   /web           – web UI med clipboard & drag/drop")
    print("   POST /recognize?top_k=3  – topp-K API")
    print("   GET  /api/health         – hälsa")
    print("   GET/POST /api/config     – konfigurera")
    app.run(host="0.0.0.0", port=5000, debug=True)
