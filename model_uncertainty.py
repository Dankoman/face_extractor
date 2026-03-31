#!/usr/bin/env python3
"""Analysera modellens osäkerhet och generera CSV med topp-N mest osäkra personer."""
from __future__ import annotations

import argparse
import csv
import difflib
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

import processed_db


# load_alias_map ersatt av processed_db.get_alias_map()


def load_similar_exclusions(path: Path, alias_map: Dict[str, str]) -> Set[frozenset[str]]:
    """Ladda par som inte ska mergas (trots likhet) från en fil (namn1|namn2)."""
    exclusions: Set[frozenset[str]] = set()
    if not path.exists():
        return exclusions

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                # Upplös alias direkt vid inläsning
                p1 = alias_map.get(parts[0], parts[0])
                p2 = alias_map.get(parts[1], parts[1])
                exclusions.add(frozenset([p1, p2]))
    return exclusions


def load_embeddings(emb_path: Path) -> Tuple[List[np.ndarray], List[str]]:
    if not emb_path.exists():
        raise SystemExit(f"Embeddings-filen hittades inte: {emb_path}")
    with emb_path.open("rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"]


# load_processed_stats ersatt av processed_db.get_stats_by_person()


def compute_person_metrics(
    X: List[np.ndarray],
    y: List[str],
    proc_stats: Dict[str, Dict],
    exclusions: Set[frozenset[str]],
) -> Dict[str, Dict]:
    """Beräkna per-person-mått från embeddings."""

    # Gruppera embeddings per person
    person_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    for emb, label in zip(X, y):
        person_embeddings[label].append(emb)

    persons = sorted(person_embeddings.keys())
    metrics: Dict[str, Dict] = {}

    # Beräkna centroider
    centroids: Dict[str, np.ndarray] = {}
    for person in persons:
        embs = np.vstack(person_embeddings[person])
        centroids[person] = embs.mean(axis=0)

    # Beräkna centroid-matris för inter-klass-avstånd
    centroid_labels = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[p] for p in centroid_labels])
    centroid_dists = cosine_distances(centroid_matrix)
    # Sätt diagonalen till inf så vi inte matchar mot sig själv
    np.fill_diagonal(centroid_dists, np.inf)

    for i, person in enumerate(centroid_labels):
        embs = np.vstack(person_embeddings[person])
        n_samples = len(embs)

        # --- Intra-klass-varians (medel cosine-avstånd till centroid) ---
        if n_samples >= 2:
            dists_to_centroid = cosine_distances(embs, centroids[person].reshape(1, -1)).flatten()
            intra_variance = float(dists_to_centroid.mean())
            intra_std = float(dists_to_centroid.std())

            # Outliers: embeddings > 2 std från centroid
            if intra_std > 0:
                outlier_threshold = intra_variance + 2 * intra_std
                outliers = int((dists_to_centroid > outlier_threshold).sum())
            else:
                outliers = 0
        else:
            intra_variance = 0.0
            intra_std = 0.0
            outliers = 0

        # --- Närmaste annan person (inter-klass) ---
        # Kopiera avståndsraden för att kunna maska exkluderade
        dists = centroid_dists[i].copy()
        for j, other_person in enumerate(centroid_labels):
            if frozenset([person, other_person]) in exclusions:
                dists[j] = np.inf

        nearest_idx = int(dists.argmin())
        nearest_person = centroid_labels[nearest_idx]
        nearest_distance = float(dists[nearest_idx])

        # --- Fail-rate ---
        ps = proc_stats.get(person, {"total": 0, "ok": 0, "fail": 0, "reasons": {}})
        total_processed = ps["total"]
        fail_count = ps["fail"]
        fail_rate = fail_count / total_processed if total_processed > 0 else 0.0
        top_fail_reasons = sorted(ps["reasons"].items(), key=lambda x: -x[1])[:3]
        fail_reasons_str = "; ".join(f"{r}({c})" for r, c in top_fail_reasons) if top_fail_reasons else ""

        metrics[person] = {
            "samples": n_samples,
            "intra_variance": intra_variance,
            "intra_std": intra_std,
            "outliers": outliers,
            "nearest_person": nearest_person,
            "nearest_distance": nearest_distance,
            "fail_rate": fail_rate,
            "fail_count": fail_count,
            "total_processed": total_processed,
            "fail_reasons": fail_reasons_str,
        }

    return metrics


def detect_confusion_pairs(metrics: Dict[str, Dict]) -> Dict[str, str]:
    """Hitta ömsesidiga förväxlingspar (A:s närmaste = B, B:s närmaste = A)."""
    mutual: Dict[str, str] = {}
    for person, m in metrics.items():
        nearest = m["nearest_person"]
        if nearest in metrics and metrics[nearest]["nearest_person"] == person:
            mutual[person] = nearest
    return mutual


def compute_uncertainty_score(m: Dict, is_mutual_confusion: bool) -> float:
    """Beräkna sammanvägt osäkerhetspoäng (0-1, högre = mer osäker)."""
    scores = []

    # Få träningsprover (max bidrag vid ≤ 3 samples)
    sample_score = max(0, 1.0 - m["samples"] / 15.0)
    scores.append(("few_samples", sample_score, 0.25))

    # Hög intra-klass-varians (normaliserad, typiskt 0-0.8 för cosine)
    variance_score = min(1.0, m["intra_variance"] / 0.5)
    scores.append(("high_intra_variance", variance_score, 0.25))

    # Lågt avstånd till närmaste person (hög förväxlingsrisk)
    confusion_score = max(0, 1.0 - m["nearest_distance"] / 0.6)
    if is_mutual_confusion:
        confusion_score = min(1.0, confusion_score * 1.3)  # Extra boost för ömsesidig förväxling
    scores.append(("confusion_risk", confusion_score, 0.30))

    # Hög fail-rate
    fail_score = m["fail_rate"]
    scores.append(("high_fail_rate", fail_score, 0.10))

    # Outliers
    outlier_score = min(1.0, m["outliers"] / max(1, m["samples"] * 0.3))
    scores.append(("outlier_embeddings", outlier_score, 0.10))

    total = sum(score * weight for _, score, weight in scores)
    return total


def determine_primary_issue(m: Dict, is_mutual: bool, partner: Optional[str]) -> str:
    """Bestäm den primära orsaken till osäkerhet i klartext."""
    issues = []

    if m["samples"] <= 3:
        issues.append(f"Mycket få träningsbilder ({m['samples']} st)")
    elif m["samples"] <= 7:
        issues.append(f"Få träningsbilder ({m['samples']} st)")

    if m["intra_variance"] > 0.3:
        issues.append("Hög varians – troligen blandade identiteter i mappen")
    elif m["intra_variance"] > 0.2:
        issues.append("Medelhög varians – möjligen blandade bilder")

    if is_mutual and partner:
        issues.append(f"Ömsesidig förväxling med {partner} (avstånd {m['nearest_distance']:.3f})")
    elif m["nearest_distance"] < 0.3:
        issues.append(f"Mycket lik {m['nearest_person']} (avstånd {m['nearest_distance']:.3f})")
    elif m["nearest_distance"] < 0.4:
        issues.append(f"Ganska lik {m['nearest_person']} (avstånd {m['nearest_distance']:.3f})")

    if m["fail_rate"] > 0.5:
        issues.append(f"Hög andel misslyckade bilder ({m['fail_rate']:.0%})")

    if m["outliers"] > 0:
        issues.append(f"{m['outliers']} outlier-embedding(s)")

    if not issues:
        issues.append("Låg men mätbar osäkerhet")

    return "; ".join(issues)


def name_similarity(a: str, b: str) -> float:
    """Räkna ut namnsimilaritet (0-1) med SequenceMatcher."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def generate_recommendation(row: Dict, m_a: Dict, m_b: Optional[Dict]) -> str:
    """Generera en konkret rekommendation i klartext."""
    person_a = row["person_a"]
    person_b = row.get("person_b", "")
    recs = []

    if person_b:
        # --- Förväxlingspar ---
        sim = name_similarity(person_a, person_b)

        if sim > 0.75:
            # Mycket lika namn
            recs.append(
                f"Namnen är nästan identiska – troligen samma person. "
                f"→ MERGE: Slå ihop '{person_a}' och '{person_b}' i merge.txt."
            )
        elif row["confusion_dist"] < 0.15:
            # Extremt nära embeddings
            recs.append(
                f"Extremt lika embeddings (avstånd {row['confusion_dist']:.3f}). "
                f"Kan vara samma person under olika namn, eller blandade bilder i bådas mappar. "
                f"→ Kontrollera bilderna i båda mapparna. Om samma person: merge. Om olika: rensa felplacerade bilder."
            )
        elif m_a["intra_variance"] > 0.25 or (m_b and m_b["intra_variance"] > 0.25):
            # Hög varians i minst en mapp
            high_var_persons = []
            if m_a["intra_variance"] > 0.25:
                high_var_persons.append(person_a)
            if m_b and m_b["intra_variance"] > 0.25:
                high_var_persons.append(person_b)
            recs.append(
                f"Hög varians i {' och '.join(high_var_persons)} tyder på blandade identiteter. "
                f"→ Granska bilderna i {'dessa mappar' if len(high_var_persons) > 1 else 'mappen'} "
                f"och flytta felplacerade bilder till rätt person."
            )
        else:
            recs.append(
                f"Personerna har liknande ansiktsdrag (avstånd {row['confusion_dist']:.3f}). "
                f"→ Kontrollera att bilderna ligger i rätt mappar. Överväg att lägga till fler distinkta bilder."
            )

        # Tillägg för fail-rate
        for label, m in [(person_a, m_a), (person_b, m_b)]:
            if m and m["fail_rate"] > 0.5:
                recs.append(
                    f"{label} har {m['fail_rate']:.0%} misslyckade bilder "
                    f"– byt ut lågkvalitetsbilder mot bättre."
                )

        # Tillägg för få samples
        for label, m in [(person_a, m_a), (person_b, m_b)]:
            if m and m["samples"] <= 3:
                recs.append(f"{label} har bara {m['samples']} bilder – samla fler träningsbilder.")

    else:
        # --- Enskild person ---
        if m_a["samples"] <= 3:
            recs.append(
                f"Bara {m_a['samples']} träningsbild(er). "
                f"→ Samla fler bilder för att förbättra igenkänningen."
            )

        if m_a["intra_variance"] > 0.3:
            recs.append(
                f"Hög varians tyder på att mappen innehåller bilder av flera personer. "
                f"→ Granska och rensa mappen."
            )

        if m_a["fail_rate"] > 0.5:
            recs.append(
                f"{m_a['fail_rate']:.0%} av bilderna misslyckades. "
                f"→ Byt ut dåliga bilder mot tydligare ansiktsfoton."
            )

        if m_a["nearest_distance"] < 0.3:
            recs.append(
                f"Ligger nära {m_a['nearest_person']} (avstånd {m_a['nearest_distance']:.3f}). "
                f"→ Kontrollera att det inte är samma person."
            )

        if m_a["outliers"] > 2:
            recs.append(
                f"{m_a['outliers']} outlier-embeddings. "
                f"→ Några bilder sticker ut kraftigt – granska och ta bort felaktiga."
            )

        if not recs:
            recs.append("Låg men mätbar osäkerhet – ingen akut åtgärd krävs.")

    return " | ".join(recs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analysera modellens osäkerhet")
    parser.add_argument("--embeddings", default="arcface_work-ppic/embeddings_ppic.pkl",
                        help="Pickle-fil med embeddings")
    parser.add_argument("--db", default="arcface_work-ppic/processed.db",
                        help="SQLite-databas med processade bilder")
    parser.add_argument("--output", default="uncertainty_report.csv",
                        help="Output CSV-fil")
    parser.add_argument("--top", type=int, default=50,
                        help="Antal mest osäkra att visa")
    parser.add_argument("--exclusions", default="similar_exclusions.txt",
                        help="Fil med par som INTE ska mergas (namn1|namn2)")
    args = parser.parse_args()

    print("Laddar databas...")
    conn = processed_db.open_db(Path(args.db))
    print("Laddar alias-mappning...")
    alias_map = processed_db.get_alias_map(conn)
    print(f"  {len(alias_map)} alias-mappningar")

    print("Laddar embeddings...")
    X, y = load_embeddings(Path(args.embeddings))
    # Upplös alias till primärnamn
    y = [alias_map.get(label, label) for label in y]
    print(f"  {len(X)} embeddings, {len(set(y))} unika personer (efter alias-upplösning)")

    print("Laddar processed-statistik...")
    proc_stats = processed_db.get_stats_by_person(conn)
    conn.close()
    # Upplös alias i processed-stats också
    resolved_stats: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "ok": 0, "fail": 0, "reasons": defaultdict(int)})
    for person, ps in proc_stats.items():
        primary = alias_map.get(person, person)
        resolved_stats[primary]["total"] += ps["total"]
        resolved_stats[primary]["ok"] += ps["ok"]
        resolved_stats[primary]["fail"] += ps["fail"]
        for reason, count in ps["reasons"].items():
            resolved_stats[primary]["reasons"][reason] += count
    proc_stats = resolved_stats

    print("Laddar exkluderingar...")
    exclusions = load_similar_exclusions(Path(args.exclusions), alias_map)
    print(f"  {len(exclusions)} exkluderade lika-par")

    print("Beräknar mått per person...")
    metrics = compute_person_metrics(X, y, proc_stats, exclusions)

    print("Detekterar förväxlingspar...")
    mutual_pairs = detect_confusion_pairs(metrics)

    # Beräkna osäkerhetspoäng
    for person, m in metrics.items():
        is_mutual = person in mutual_pairs
        m["score"] = compute_uncertainty_score(m, is_mutual)
        m["is_mutual"] = is_mutual
        m["mutual_partner"] = mutual_pairs.get(person)

    # Hantera förväxlingspar: slå ihop till en rad
    # Vi visar paret en gång, med den högsta poängen
    seen_pairs: Set[str] = set()
    rows = []

    # Sortera alla efter poäng
    sorted_persons = sorted(metrics.keys(), key=lambda p: -metrics[p]["score"])

    for person in sorted_persons:
        if person in seen_pairs:
            continue
        m = metrics[person]
        partner = m.get("mutual_partner")

        if partner and partner not in seen_pairs:
            # Förväxlingspar – slå ihop till en rad
            m2 = metrics[partner]
            combined_score = max(m["score"], m2["score"])
            issue_a = determine_primary_issue(m, True, partner)
            issue_b = determine_primary_issue(m2, True, person)

            row_data = {
                "person_a": person,
                "person_b": partner,
                "samples_a": m["samples"],
                "samples_b": m2["samples"],
                "fail_rate_a": m["fail_rate"],
                "fail_rate_b": m2["fail_rate"],
                "fail_reasons_a": m["fail_reasons"],
                "fail_reasons_b": m2["fail_reasons"],
                "intra_var_a": m["intra_variance"],
                "intra_var_b": m2["intra_variance"],
                "confusion_dist": m["nearest_distance"],
                "outliers_a": m["outliers"],
                "outliers_b": m2["outliers"],
                "issue_a": issue_a,
                "issue_b": issue_b,
                "score": combined_score,
            }
            row_data["recommendation"] = generate_recommendation(row_data, m, m2)
            rows.append(row_data)
            seen_pairs.add(person)
            seen_pairs.add(partner)
        else:
            # Enskild person
            issue = determine_primary_issue(m, False, None)
            row_data = {
                "person_a": person,
                "person_b": "",
                "samples_a": m["samples"],
                "samples_b": "",
                "fail_rate_a": m["fail_rate"],
                "fail_rate_b": "",
                "fail_reasons_a": m["fail_reasons"],
                "fail_reasons_b": "",
                "intra_var_a": m["intra_variance"],
                "intra_var_b": "",
                "confusion_dist": m["nearest_distance"],
                "outliers_a": m["outliers"],
                "outliers_b": "",
                "issue_a": issue,
                "issue_b": "",
                "score": m["score"],
            }
            row_data["recommendation"] = generate_recommendation(row_data, m, None)
            rows.append(row_data)
            seen_pairs.add(person)

    # Sortera efter poäng och ta topp N
    rows.sort(key=lambda r: -r["score"])
    rows = rows[: args.top]

    # Skriv CSV
    output_path = Path(args.output)
    fieldnames = [
        "Rank",
        "Person A", "Person B",
        "Recommendation",
        "Uncertainty Score",
        "Samples A", "Samples B",
        "Fail Rate A", "Fail Rate B",
        "Fail Reasons A", "Fail Reasons B",
        "Intra Variance A", "Intra Variance B",
        "Confusion Distance",
        "Outliers A", "Outliers B",
        "Issue A", "Issue B",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({
                "Rank": rank,
                "Person A": row["person_a"],
                "Person B": row["person_b"],
                "Samples A": row["samples_a"],
                "Samples B": row["samples_b"],
                "Fail Rate A": f"{row['fail_rate_a']:.0%}" if isinstance(row["fail_rate_a"], float) else "",
                "Fail Rate B": f"{row['fail_rate_b']:.0%}" if isinstance(row["fail_rate_b"], float) else "",
                "Fail Reasons A": row["fail_reasons_a"],
                "Fail Reasons B": row["fail_reasons_b"],
                "Intra Variance A": f"{row['intra_var_a']:.4f}" if isinstance(row["intra_var_a"], float) else "",
                "Intra Variance B": f"{row['intra_var_b']:.4f}" if isinstance(row["intra_var_b"], float) else "",
                "Confusion Distance": f"{row['confusion_dist']:.4f}",
                "Outliers A": row["outliers_a"],
                "Outliers B": row["outliers_b"],
                "Issue A": row["issue_a"],
                "Issue B": row["issue_b"],
                "Recommendation": row["recommendation"],
                "Uncertainty Score": f"{row['score']:.3f}",
            })

    print(f"\n✅ Rapport sparad: {output_path}")
    print(f"   {len(rows)} rader (varav {sum(1 for r in rows if r['person_b'])} förväxlingspar)")

    # Visa topp 5 i terminalen
    print(f"\nTopp 5:")
    for i, row in enumerate(rows[:5], 1):
        pair = f"{row['person_a']} ↔ {row['person_b']}" if row["person_b"] else row["person_a"]
        print(f"  {i}. {pair} (score={row['score']:.3f})")
        print(f"     {row['issue_a']}")
        if row["issue_b"]:
            print(f"     {row['issue_b']}")


if __name__ == "__main__":
    main()
