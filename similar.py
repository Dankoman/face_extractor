import pickle, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

data = pickle.load(open("/home/marqs/Programmering/Python/3.11/doppelganger/face_extractor/arcface_work/embeddings_new.pkl","rb"))
X = np.vstack(data["X"])              # (N, 512)
y = np.array(data["y"])               # etiketter (mappnamn)

# 1. Centroids
centroids = {}
for lab in np.unique(y):
    centroids[lab] = X[y==lab].mean(axis=0)

labs = list(centroids.keys())
C = np.vstack([centroids[l] for l in labs])
S = cosine_similarity(C)              # (L, L), L = antal mappar

THR = 0.65
pairs = []
for i in range(len(labs)):
    for j in range(i+1, len(labs)):
        if S[i,j] >= THR:
            pairs.append((labs[i], labs[j], float(S[i,j])))

pairs.sort(key=lambda t: -t[2])
for a,b,s in pairs[:50]:
    print(f"{a}  <->  {b}   cos_sim={s:.3f}")
