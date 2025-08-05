import pickle

fn = "arcface_work-ppic/embeddings_ppic_merged.pkl"
with open(fn, "rb") as f:
    data = pickle.load(f)
X, y = data["X"], data["y"]

keep_idx = [i for i, lbl in enumerate(y)
            if lbl not in ("Avena Segal", "Tea")]
X_new = [X[i] for i in keep_idx]
y_new = [y[i] for i in keep_idx]

with open(fn, "wb") as f:
    pickle.dump({"X": X_new, "y": y_new}, f)
