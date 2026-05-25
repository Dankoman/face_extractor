#!/usr/bin/env python3
"""
Klustrar embeddings för varje modell och tar bort co-stars/outliers.
Hanterar även bimodal splits om mappen innehåller två stora kluster (namnkrock).
"""
import argparse
import pickle
import shutil
import sqlite3
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.cluster import DBSCAN

def open_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    return conn

def remove_from_processed(conn, path: str):
    cur = conn.cursor()
    cur.execute("DELETE FROM processed WHERE path = ?", (path,))
    conn.commit()

def update_processed_path(conn, old_path: str, new_path: str):
    cur = conn.cursor()
    cur.execute("UPDATE processed SET path = ? WHERE path = ?", (new_path, old_path))
    conn.commit()

def main():
    parser = argparse.ArgumentParser(description="DBSCAN Clustering för att rensa co-stars och klyva modeller")
    parser.add_argument("--data-root", required=True, type=str, help="Sökväg till pBook")
    parser.add_argument("--embeddings", required=True, type=str, help="Sökväg till embeddings_ppic.pkl")
    parser.add_argument("--db", required=True, type=str, help="Sökväg till processed.db")
    parser.add_argument("--eps", type=float, default=0.3, help="Cosinus-avstånd för DBSCAN")
    parser.add_argument("--min-samples", type=int, default=3, help="Minsta antalet bilder för ett giltigt kluster")
    parser.add_argument("--dry-run", action="store_true", help="Kör utan att radera filer")
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        print(f"❌ Hittade inte {emb_path}")
        return

    with open(emb_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    paths = data.get("paths", [])
    
    # Pad paths if missing (older encodings)
    if len(paths) < len(X):
        paths = paths + [None] * (len(X) - len(paths))

    # Gruppera indices per person
    person_indices = defaultdict(list)
    for i, label in enumerate(y):
        person_indices[label].append(i)

    to_remove_indices = set()
    to_update = [] # list of (idx, new_label, old_path, new_path)

    conn = open_db(Path(args.db))

    for label, indices in person_indices.items():
        if len(indices) < args.min_samples:
            continue

        person_X = np.vstack([X[i] for i in indices])
        
        # DBSCAN med cosine (avstånd) -> precomputed eller metric="cosine"
        clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine").fit(person_X)
        labels = clustering.labels_
        
        labels_list = list(labels)
        cluster_counts = {l: labels_list.count(l) for l in set(labels_list) if l != -1}
        
        if not cluster_counts:
            # Allt är brus? Konstigt, behåll som det är eller radera? Vi behåller.
            continue
            
        # Sortera kluster efter storlek
        sorted_clusters = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)
        main_cluster = sorted_clusters[0][0]
        
        # Klyvning / Namnkrock (Bimodal Split)
        bimodal_cluster = None
        if len(sorted_clusters) > 1:
            second_size = sorted_clusters[1][1]
            # Om näst största klustret har minst 5 bilder och är mer än 30% av huvudklustret
            if second_size >= 5 and second_size >= (sorted_clusters[0][1] * 0.3):
                bimodal_cluster = sorted_clusters[1][0]
                new_label = f"{label}_1"
                new_dir = Path(args.data_root) / new_label
                if not args.dry_run:
                    new_dir.mkdir(exist_ok=True, parents=True)
                print(f"⚠️ [SPLIT] {label} verkar ha en namnkrock! Delar ut ett kluster på {second_size} bilder till {new_label}.")

        removed_count = 0
        for i, cluster_label in enumerate(labels):
            global_idx = indices[i]
            img_path = paths[global_idx]
            
            if cluster_label == main_cluster:
                continue # Huvudmodellen
                
            elif bimodal_cluster is not None and cluster_label == bimodal_cluster:
                # Klyv detta ansikte till ny mapp
                if img_path and Path(img_path).exists():
                    old_path_obj = Path(img_path)
                    new_path_obj = new_dir / old_path_obj.name
                    if not args.dry_run:
                        shutil.move(str(old_path_obj), str(new_path_obj))
                        update_processed_path(conn, str(old_path_obj), str(new_path_obj))
                    to_update.append((global_idx, new_label, img_path, str(new_path_obj)))
                else:
                    # Saknar path (gammal embedding), byter bara label
                    to_update.append((global_idx, new_label, None, None))
                    
            else:
                # Outlier / brus / mindre co-star kluster
                if img_path and Path(img_path).exists():
                    if not args.dry_run:
                        Path(img_path).unlink()
                        remove_from_processed(conn, img_path)
                    removed_count += 1
                to_remove_indices.add(global_idx)

        if removed_count > 0:
            print(f"🧹 [CLEAN] {label}: Rensade {removed_count} felaktiga bilder/outliers. Behåller {sorted_clusters[0][1]}.")

    # Genomför ändringar i minnet och spara
    if to_remove_indices or to_update:
        if args.dry_run:
            print(f"DRY-RUN: Skulle ha raderat {len(to_remove_indices)} embeddings och uppdaterat {len(to_update)}.")
        else:
            # Uppdatera splits
            for idx, new_label, old_p, new_p in to_update:
                y[idx] = new_label
                if new_p:
                    paths[idx] = new_p

            # Ta bort raderade
            keep_idx = [i for i in range(len(X)) if i not in to_remove_indices]
            X_new = [X[i] for i in keep_idx]
            y_new = [y[i] for i in keep_idx]
            paths_new = [paths[i] for i in keep_idx]

            # Spara
            tmp = emb_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump({"X": X_new, "y": y_new, "paths": paths_new}, f)
            tmp.rename(emb_path)
            print(f"✅ Sparade tvättade embeddings: {len(X_new)} kvar (tog bort {len(to_remove_indices)}).")
    else:
        print("✅ Inga outliers hittades som behövde rensas.")

    conn.close()

if __name__ == "__main__":
    main()
