#!/usr/bin/env python3
import pickle
from pathlib import Path

# Filvägar
MERGE_FILE = "merge.txt"
EMBEDDINGS_FILE = "arcface_work/embeddings.pkl"
OUTPUT_FILE = "arcface_work/embeddings_merged.pkl"

def load_alias_map(merge_file: str) -> dict:
    """Läser merge.txt och skapar en alias-mappning."""
    alias_map = {}
    with open(merge_file, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                names = line.strip().split("|")
                primary_name = names[0]  # Namnet längst till vänster
                for alias in names[1:]:
                    if alias:  # Ignorera tomma alias
                        alias_map[alias] = primary_name
    return alias_map

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
    # Ladda alias-mappning från merge.txt
    alias_map = load_alias_map(MERGE_FILE)
    print(f"Alias-mappning: {alias_map}")

    # Slå ihop embeddings
    merge_embeddings(EMBEDDINGS_FILE, alias_map, OUTPUT_FILE)
