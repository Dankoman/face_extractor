#!/usr/bin/env python3
import pickle
from pathlib import Path
import processed_db

# Filvägar
EMBEDDINGS_FILE = "arcface_work-ppic/embeddings_ppic.pkl"
OUTPUT_FILE = "arcface_work-ppic/embeddings_ppic_merged.pkl"

# load_alias_map ersatt av processed_db.get_alias_map()

def merge_embeddings(embeddings_file: str, alias_map: dict, output_file: str):
    """Slår ihop embeddings baserat på alias-mappningen."""
    with open(embeddings_file, "rb") as f:
        data = pickle.load(f)

    X = data["X"]  # Embeddings
    y = data["y"]  # Etiketter

    # Uppdatera etiketter baserat på alias-mappningen
    updated_y = [alias_map.get(label, label) for label in y]

    # Skapa en ny lista med unika etiketter och deras embeddings
    merged_X = []
    merged_y = []
    seen_labels = {}

    for embedding, label in zip(X, updated_y):
        if label not in seen_labels:
            seen_labels[label] = len(merged_X)
            merged_X.append(embedding)
            merged_y.append(label)
        else:
            # Om etiketten redan finns, slå ihop embeddings (medelvärde)
            idx = seen_labels[label]
            merged_X[idx] = (merged_X[idx] + embedding) / 2

    # Spara den nya embeddings-filen
    with open(output_file, "wb") as f:
        pickle.dump({"X": merged_X, "y": merged_y}, f)

    print(f"✅ Slutfört! Sparade sammanslagna embeddings till {output_file}")

if __name__ == "__main__":
    db_path = "arcface_work-ppic/processed.db"
    conn = processed_db.open_db(db_path)
    alias_map = processed_db.get_alias_map(conn)
    conn.close()
    
    print(f"Alias-mappning laddad från DB ({len(alias_map)} namn)")

    # Slå ihop embeddings
    merge_embeddings(EMBEDDINGS_FILE, alias_map, OUTPUT_FILE)
