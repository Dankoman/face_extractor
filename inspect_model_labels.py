import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def inspect_pkl(path: Path):
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"--- Inspecting {path} ---")
    with path.open("rb") as f:
        data = pickle.load(f)
    
    if "y" in data:
        # It's the embeddings file
        labels = data["y"]
        unique_labels = sorted(list(set(labels)))
        print(f"Total samples: {len(labels)}")
        print(f"Unique labels: {len(unique_labels)}")
        if "Nella Jones" in unique_labels:
            print("ALERT: 'Nella Jones' found in labels!")
        elif "Kendall Rae" in unique_labels:
            print("'Kendall Rae' found in labels.")
        else:
            print("Neither Nella Jones nor Kendall Rae found.")
            
    elif "label_encoder" in data:
        # It's the model file
        le = data["label_encoder"]
        classes = le.classes_
        print(f"Model classes: {len(classes)}")
        if "Nella Jones" in classes:
            print("ALERT: 'Nella Jones' found in model classes!")
        elif "Kendall Rae" in classes:
            print("'Kendall Rae' found in model classes.")
        
    print("")

def main():
    workdir = Path("arcface_work-ppic")
    inspect_pkl(workdir / "embeddings_ppic.pkl")
    inspect_pkl(workdir / "face_knn_arcface_ppic.pkl")

if __name__ == "__main__":
    main()
