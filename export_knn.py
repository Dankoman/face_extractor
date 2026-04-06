#!/usr/bin/env python3
import pickle
import numpy as np
import json
import argparse
from pathlib import Path

def export_model(pkl_path, out_dir):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f"Error: {pkl_path} not found")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pkl_path}...")
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    clf = data["model"]
    le = data["label_encoder"]

    # Extract embeddings (X) and labels (y indices)
    # KNeighborsClassifier stores the training samples in _fit_X
    # For newer versions of sklearn, it might be in different places but usually _fit_X
    if hasattr(clf, "_fit_X"):
        X = clf._fit_X
    else:
        # Fallback if scikit-learn version is different
        print("Attribute _fit_X not found, trying fallback...")
        X = clf.X_fit_ if hasattr(clf, "X_fit_") else None
    
    if X is None:
        print("Could not find training data in KNN model")
        return

    # If X is sparse or not a numpy array, convert it
    if not isinstance(X, np.ndarray):
        X = X.toarray() if hasattr(X, "toarray") else np.array(X)
    
    X = X.astype(np.float32)

    # Get labels as strings
    # clf.classes_ are the integer indices, le.inverse_transform gives the names
    labels = le.inverse_transform(clf.classes_)
    
    # We also need the actual mapping for each sample in X
    # clf._y is the index of the class for each sample
    y_indices = clf._y
    sample_labels = [str(labels[idx]) for idx in y_indices]

    # Save binary embeddings
    bin_path = out_dir / "embeddings.bin"
    X.tofile(bin_path)
    
    # Save labels list
    json_path = out_dir / "labels.json"
    with json_path.open("w") as f:
        json.dump(sample_labels, f, ensure_ascii=False, indent=2)

    print(f"Exported {X.shape[0]} embeddings to {bin_path}")
    print(f"Exported labels to {json_path}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="face_knn_arcface_ppic.pkl", help="Path to the .pkl model")
    parser.add_argument("--out", default="arcface_work-ppic/export", help="Output directory")
    args = parser.parse_args()
    
    export_model(args.pkl, args.out)
