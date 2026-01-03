import pickle, numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

data = pickle.load(open("arcface_work-ppic/embeddings_ppic_merged.pkl","rb"))
X = np.vstack(data["X"])              # (N, 512)
y = np.array(data["y"])               # etiketter (mappnamn)

# 1. Centroids
centroids = {}
for lab in np.unique(y):
    centroids[lab] = X[y==lab].mean(axis=0)

labs = list(centroids.keys())
C = np.vstack([centroids[l] for l in labs])
S = cosine_similarity(C)              # (L, L), L = antal mappar


# 1.5 Load exclusions
exclusions = set()
exclusion_path = Path("similar_exclusions.txt")
if exclusion_path.exists():
    with exclusion_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            if len(parts) >= 2:
                # Store as frozenset for order-independence
                exclusions.add(frozenset([parts[0].strip(), parts[1].strip()]))

THR = 0.49
MAX_ROWS = 50
pairs = []
for i in range(len(labs)):
    for j in range(i+1, len(labs)):
        if S[i,j] >= THR:
            # Check exclusion
            if frozenset([labs[i], labs[j]]) not in exclusions:
                pairs.append((labs[i], labs[j], float(S[i,j])))

pairs.sort(key=lambda t: -t[2])
top_pairs = pairs[:MAX_ROWS]

output_path = Path("merge?.csv")
with output_path.open("w", encoding="utf-8") as fp:
    for a, b, s in top_pairs:
        fp.write(f"{a}|{b}||{s:.3f}\n")

for a, b, s in top_pairs:
    print(f"{a}  <->  {b}   cos_sim={s:.3f}")

print(f"Wrote {len(top_pairs)} pairs to {output_path}")
