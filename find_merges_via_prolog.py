import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from identity_resolver import IdentityResolver
from external_resolver import ExternalIdentityResolver

def main():
    print("Laddar prolog-databas och befintliga merge/exclusions...")
    db_path = Path("arcface_work-ppic/processed.db")
    prolog_resolver = IdentityResolver("merge.txt", "similar_exclusions.txt", db_path=db_path)
    ext_resolver = ExternalIdentityResolver()
    
    print("Laddar embeddings...")
    embeddings_path = Path("arcface_work-ppic/embeddings_ppic.pkl")
    if not embeddings_path.exists():
        print(f"Hittade inte {embeddings_path}")
        return
        
    with embeddings_path.open("rb") as f:
        data = pickle.load(f)
        
    X = np.vstack(data["X"])
    y = np.array(data["y"])
    
    centroids = {}
    print("Filtrerar personer (minst 5 bilder, max varians 0.3)...")
    for lab in np.unique(y):
        mask = (y == lab)
        embs = X[mask]
        samples = len(embs)
        
        if samples < 5:
            continue
            
        centroid = embs.mean(axis=0)
        
        dists = cosine_distances(embs, centroid.reshape(1, -1)).flatten()
        intra_variance = float(dists.mean())
        
        if intra_variance > 0.3:
            continue
            
        centroids[lab] = centroid
        
    labs = list(centroids.keys())
    print(f"Hittade {len(labs)} godkända personer av {len(np.unique(y))} totalt.")
    C = np.vstack([centroids[l] for l in labs])
    S = cosine_similarity(C)
    
    THR = 0.40
    print(f"Hittar par med similarity >= {THR}...")
    
    pairs = []
    for i in range(len(labs)):
        for j in range(i+1, len(labs)):
            if S[i,j] >= THR:
                pairs.append((labs[i], labs[j], float(S[i,j])))
                
    pairs.sort(key=lambda t: -t[2])
    
    names_to_lookup = set()
    for a, b, sim in pairs:
        a_esc = a.replace("'", "\\'")
        b_esc = b.replace("'", "\\'")
        prolog_resolver.prolog.assertz(f"similar('{a_esc}', '{b_esc}', {sim:.3f})")
        prolog_resolver.prolog.assertz(f"similar('{b_esc}', '{a_esc}', {sim:.3f})")
        names_to_lookup.add(a)
        names_to_lookup.add(b)
        
    print(f"Slår upp {len(names_to_lookup)} unika namn mot externa källor (StashDB etc)...")
    total = len(names_to_lookup)
    for i, name in enumerate(names_to_lookup, 1):
        # Skriv ut på samma rad för att slippa scroll (\r)
        print(f"\r[{i}/{total}] Slår upp {name[:30]:<30}...{' ' * 10}", end="", flush=True)
        res = ext_resolver.resolve(name)
        if res:
            canonical, source = res
            prolog_resolver.add_external_truth(name, canonical, source)
            
    print("\nFrågar Prolog om förslag på sammanslagningar...")
    suggestions = list(prolog_resolver.prolog.query("suggest_merge(A, B, Canonical, Reason)"))
    
    if not suggestions:
        print("Inga nya sammanslagningar kunde föreslås baserat på extern data.")
        return
        
    seen = set()
    output_path = Path("to_be_merged.csv")
    with output_path.open("w", encoding="utf-8") as fp:
        for s in suggestions:
            # Undvik dubbletter eftersom Prolog kan returnera både (A,B) och (B,A)
            pair = frozenset([s["A"], s["B"]])
            if pair in seen:
                continue
            seen.add(pair)
            
            a, b = s["A"], s["B"]
            canonical = s["Canonical"]
            reason = s["Reason"]
            
            print(f"\nFÖRESLÅR MERGE: {a} <-> {b}")
            print(f"  -> {reason}")
            
            # Format: Canonical|A|B (undvik dubbletter i raden)
            members = [canonical]
            for name in [a, b]:
                if name != canonical and name not in members:
                    members.append(name)
            
            fp.write(f"{'|'.join(members)}\n")
            
    print(f"\nSkrev {len(seen)} förslag till {output_path}. Du kan nu köra apply_merge_candidates.py för att uppdatera DB.")

if __name__ == '__main__':
    main()
