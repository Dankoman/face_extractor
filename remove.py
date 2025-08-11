import pickle

# Filvägar
fn = "arcface_work-ppic/embeddings_ppic.pkl"
remove_file = "remove.txt"

# Läs in listan med namn som ska tas bort
with open(remove_file, "r", encoding="utf-8") as f:
    remove_names = {line.strip() for line in f if line.strip()}

# Läs pickle-filen
with open(fn, "rb") as f:
    data = pickle.load(f)

X, y = data["X"], data["y"]

# Filtrera bort alla namn som finns i remove.txt
keep_idx = [i for i, lbl in enumerate(y) if lbl not in remove_names]

X_new = [X[i] for i in keep_idx]
y_new = [y[i] for i in keep_idx]

# Spara tillbaka
with open(fn, "wb") as f:
    pickle.dump({"X": X_new, "y": y_new}, f)

print(f"Tog bort {len(y) - len(y_new)} poster från '{fn}' baserat på '{remove_file}'.")
