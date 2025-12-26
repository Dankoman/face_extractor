#!/usr/bin/env python3
"""
Scan Folder Outliers
--------------------
Analyserar mappar för att hitta "udda" bilder som inte passar in med resten av bilderna i samma mapp.
Till skillnad från `train_confidence.py` (som jämför med den 'globala' sanningen för etiketten),
jämför detta script bilden med **övriga bilder i samma mapp**.

Algoritm:
1. Gruppera alla embeddings per fysisk mapp.
2. För varje mapp:
   a. Beräkna en 'lokal centroid' (medelvärdet av alla ansikten i mappen).
   b. Beräkna avstånd (cosinus) för varje bild till denna centroid.
   c. Markera bilder som ligger långt ifrån centroiden (> threshold) som 'outliers'.
   d. För outliers: Använd den tränade KNN-modellen för att gissa vem det egentligen är.
3. Skriv ut rapport.

Användning:
  python3 scan_folder_outliers.py --threshold 0.4
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def load_data(emb_path: Path, proc_path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """Laddar embeddings och matchande filvägar."""
    if not emb_path.exists():
        sys.exit(f"❌ Embeddings file missing: {emb_path}")
    
    print(f"Loading embeddings from {emb_path}...")
    with emb_path.open("rb") as f:
        data = pickle.load(f)
    X = np.vstack(data["X"]).astype(np.float32)
    # Normera vektorerna för cosinus-beräkningar
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    
    # Ladda paths från processed.jsonl för att koppla vektor -> fil
    # OBS: Vi antar att embeddings ligger i samma ordning som processed.jsonl (om den genererats sekventiellt)
    # Men face_arc_pipeline sparar processed löpande och embeddings i batch. 
    # Det är säkrast om vi verifierar längden. I nuvarande pipeline är ordningen bevarad.
    paths = []
    import json
    if proc_path.exists():
        with proc_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("ok"):
                        paths.append(rec["path"])
                except:
                    pass
    
    # Trimma om antalet inte matchar (t.ex. om avbruten körning)
    if len(paths) > len(X):
        paths = paths[:len(X)]
    elif len(paths) < len(X):
        print(f"⚠️ Varning: Hittade {len(paths)} sökvägar men {len(X)} embeddings. Path-mappningen kan vara osäker.")
        # Vi kan inte garantera rätt mappning, men vi kör på det vi har.
    
    return X, paths

def load_model(model_path: Path):
    """Laddar den tränade KNN-modellen för predictions."""
    if not model_path.exists():
        return None, None
    with model_path.open("rb") as f:
        data = pickle.load(f)
    return data.get("model"), data.get("label_encoder")

def get_folder_groups(X: np.ndarray, paths: List[str]) -> Dict[str, List[int]]:
    """Grupperar index i X baserat på mappnamn."""
    groups = {}
    for idx, path in enumerate(paths):
        folder = Path(path).parent.name
        if folder not in groups:
            groups[folder] = []
        groups[folder].append(idx)
    return groups

def main():
    parser = argparse.ArgumentParser(description="Hitta outliers i mappar.")
    parser.add_argument("--workdir", type=Path, default=Path("arcface_work-ppic"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Avståndströskel (0.0-1.0). Högre = färre larm. (Default: 0.5)")
    parser.add_argument("--min-images", type=int, default=3, help="Ignorera mappar med färre än N bilder.")
    parser.add_argument("--min-suggestion-conf", type=float, default=0.0, help="Visa bara förslag med confidence över detta värde.")
    parser.add_argument("--csv", type=Path, help="Spara resultat till CSV-fil.")
    args = parser.parse_args()

    emb_path = args.workdir / "embeddings_ppic.pkl"
    proc_path = args.workdir / "processed-ppic.jsonl"
    model_path = args.workdir / "face_knn_arcface_ppic.pkl"

    X, paths = load_data(emb_path, proc_path)
    knn, le = load_model(model_path)
    
    if knn is None:
        print("⚠️ Ingen tränad modell hittades. Kan inte gissa namn på outliers, bara detektera dem.")

    groups = get_folder_groups(X, paths)
    
    print(f"Analyserar {len(groups)} mappar...\n")
    print(f"{'Folder':<30} | {'Outliers':<8} | {'Suggestion'}")
    print("-" * 80)

    total_outliers = 0
    csv_rows = []

    for folder in sorted(groups.keys()):
        indices = groups[folder]
        if len(indices) < args.min_images:
            continue
            
        # Hämta vektorer för denna mapp
        folder_vectors = X[indices]
        
        # Beräkna centroid för mappen
        centroid = folder_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid /= centroid_norm
        
        # Beräkna 'similarity' mot centroiden för varje bild (högre = bättre)
        sims = folder_vectors @ centroid
        
        outlier_local_indices = np.where(sims < args.threshold)[0]
        
        if len(outlier_local_indices) > 0:
            # Spara potentiella outliers temporärt
            potential_outliers = []
            
            for local_idx in outlier_local_indices:
                global_idx = indices[local_idx]
                score = sims[local_idx]
                path = paths[global_idx]
                filename = Path(path).name
                
                suggestion = "-"
                best_prob = 0.0
                pred_label = ""
                prob_str = "0.0"

                if knn:
                    # Prediktera vem det ser ut som
                    vec = X[global_idx].reshape(1, -1)
                    pred_idx = knn.predict(vec)[0]
                    pred_label = le.inverse_transform([pred_idx])[0]
                    
                    # Hämta prob/distance om möjligt (KNN stöder predict_proba)
                    probs = knn.predict_proba(vec)[0]
                    best_prob = probs.max()
                    
                    suggestion = f"{pred_label} ({best_prob:.2f})"
                    prob_str = f"{best_prob:.2f}"
                
                # Filtrera på suggestion confidence om användaren begärt det
                if best_prob >= args.min_suggestion_conf:
                    potential_outliers.append({
                        "filename": filename,
                        "score": score,
                        "suggestion": suggestion,
                        "path": path,
                        "pred_label": pred_label,
                        "prob_str": prob_str
                    })
            
            if potential_outliers:
                print(f"📂 {folder} ({len(indices)} images)")
                total_outliers += len(potential_outliers)
                for item in potential_outliers:
                     print(f"   ❌ {item['filename']:<20} (sim: {item['score']:.2f}) -> Ser ut som: {item['suggestion']}")
                     
                     if args.csv:
                        csv_rows.append({
                            "folder": folder,
                            "filename": item['filename'],
                            "path": item['path'],
                            "similarity_to_folder": item['score'],
                            "suggested_label": item['pred_label'],
                            "suggestion_confidence": item['prob_str']
                        })
                print("")

    if args.csv and csv_rows:
        import csv
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["folder", "filename", "path", "similarity_to_folder", "suggested_label", "suggestion_confidence"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"💾 Sparade {len(csv_rows)} rader till {args.csv}")

    print("-" * 80)
    print(f"Klar. Hittade totalt {total_outliers} outliers i {len(groups)} mappar.")

if __name__ == "__main__":
    main()
