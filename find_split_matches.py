#!/usr/bin/env python3
"""
Analysera splittade mappar (t.ex. Alice_1, Alice_2) från DBSCAN-klustringen
och jämför deras ArcFace-centroider mot alla kända primärmappar för att ge
förslag på vem de mest liknar.

Filtrerar automatiskt bort historiska spök-etiketter från pickle-filen
där mappen redan har raderats eller slagits ihop på disk.
"""

import argparse
import csv
import pickle
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity

console = Console()

# Standardrotmappar där bilder lagras
DEFAULT_ROOTS = [
    Path("/home/marqs/Bilder/pBook"),
    Path("/home/marqs/Bilder/Nya"),
    Path("/home/marqs/Bilder/Innie"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Hitta matchningar för splittade mappar (_1, _2 osv).")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("arcface_work-ppic/embeddings_ppic_merged.pkl"),
        help="Sökväg till embeddings-filen.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("split_matches_report.csv"),
        help="Sökväg till CSV-rapporten som genereras.",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.50,
        help="Minsta cosinus-likhet för att visa i topplistan i terminalen.",
    )
    parser.add_argument(
        "--show-historical",
        action="store_true",
        help="Visa även gamla/historiska splits som redan har åtgärdats på disk.",
    )
    return parser.parse_args()


def exists_on_disk(name: str, roots: list[Path]) -> bool:
    """Kontrollera om mappen faktiskt existerar i någon av bildkatalogerna."""
    for r in roots:
        if (r / name).is_dir():
            return True
    return False


def get_base_name(name: str) -> str:
    """Extrahera ursprungsnamnet från ett splittat namn (t.ex. 'Alice_1' -> 'Alice')."""
    if "_" in name:
        parts = name.split("_")
        base_parts = [p for p in parts if not p.isdigit()]
        if base_parts:
            return "_".join(base_parts)
        return parts[0]
    return name


def main():
    args = parse_args()

    if not args.embeddings.exists():
        console.print(f"[red]Fel: Hittade inte embeddings-filen {args.embeddings}[/red]")
        return

    console.print(f"[bold blue]Laddar embeddings från {args.embeddings}...[/bold blue]")
    with open(args.embeddings, "rb") as f:
        data = pickle.load(f)

    X = np.vstack(data["X"])
    y = np.array(data["y"])
    unique_labels = np.unique(y)

    console.print(f"[cyan]Beräknar centroider för {len(unique_labels)} unika mappar...[/cyan]")
    centroids = {}
    counts = {}
    for lab in unique_labels:
        mask = y == lab
        centroids[lab] = X[mask].mean(axis=0)
        counts[lab] = np.sum(mask)

    labs = list(centroids.keys())
    C = np.vstack([centroids[l] for l in labs])

    # Filtrera fram splittade och primära mappar
    raw_split_indices = [i for i, l in enumerate(labs) if "_" in l]
    primary_indices = [i for i, l in enumerate(labs) if "_" not in l]

    # Verifiera mot disk om inte --show-historical är satt
    if args.show_historical:
        split_indices = raw_split_indices
        console.print(f"[green]Hittade {len(split_indices)} splittade mappar (inkl. historiska).[/green]\n")
    else:
        split_indices = [i for i in raw_split_indices if exists_on_disk(labs[i], DEFAULT_ROOTS)]
        ghost_count = len(raw_split_indices) - len(split_indices)
        console.print(
            f"[green]Hittade {len(split_indices)} aktiva splittade mappar på disk "
            f"(filtrerade bort {ghost_count} historiska/åtgärdade mappar).[/green]\n"
        )

    if not split_indices or not primary_indices:
        console.print("[yellow]Saknar antingen splittade eller primära mappar att jämföra.[/yellow]")
        return

    # Beräkna cosinus-likhet mellan splittade och primära
    C_split = C[split_indices]
    C_prim = C[primary_indices]
    S = cosine_similarity(C_split, C_prim)  # (N_split, N_prim)

    results = []
    for i, s_idx in enumerate(split_indices):
        split_name = labs[s_idx]
        split_count = counts[split_name]
        base_name = get_base_name(split_name)

        sim_row = S[i]
        top_prim_idx = np.argsort(-sim_row)[:5]
        matches = [
            (labs[primary_indices[p_idx]], float(sim_row[p_idx]), counts[labs[primary_indices[p_idx]]])
            for p_idx in top_prim_idx
        ]
        results.append((split_name, split_count, base_name, matches))

    results.sort(key=lambda x: -x[3][0][1])

    # Spara till CSV
    console.print(f"[bold yellow]Sparar rapport till {args.output}...[/bold yellow]")
    with open(args.output, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="|")
        writer.writerow([
            "Splittad Mapp", "Antal Bilder", "Ursprungsmapp",
            "Match 1 Namn", "Match 1 Likhet", "Match 1 Bilder",
            "Match 2 Namn", "Match 2 Likhet", "Match 2 Bilder",
            "Match 3 Namn", "Match 3 Likhet", "Match 3 Bilder",
        ])
        for s_name, s_cnt, b_name, matches in results:
            row = [s_name, s_cnt, b_name]
            for m_name, m_sim, m_cnt in matches[:3]:
                row.extend([m_name, f"{m_sim:.3f}", m_cnt])
            writer.writerow(row)

    # --- Presentera i terminalen ---
    external_matches = []
    internal_matches = []

    for s_name, s_cnt, b_name, matches in results:
        best_m_name, best_m_sim, best_m_cnt = matches[0]
        if best_m_name.lower() == b_name.lower():
            internal_matches.append((s_name, s_cnt, b_name, matches))
        else:
            external_matches.append((s_name, s_cnt, b_name, matches))

    # Tabell 1: Externa matchningar
    ext_filtered = [r for r in external_matches if r[3][0][1] >= args.min_sim]
    if ext_filtered:
        table = Table(title=f"🌟 EXTERNA MATCHNINGAR: Aktiva splittade mappar som liknar ANDRA personer (Topp {min(30, len(ext_filtered))})")
        table.add_column("Splittad mapp", style="cyan", no_wrap=True)
        table.add_column("Bilder", style="blue", justify="right")
        table.add_column("Ursprungsmapp", style="magenta")
        table.add_column("Bästa Matchning", style="bold green")
        table.add_column("Likhet", style="yellow", justify="right")
        table.add_column("Match Bilder", style="blue", justify="right")
        table.add_column("Näst bästa match", style="white")

        for s_name, s_cnt, b_name, matches in ext_filtered[:30]:
            m1_name, m1_sim, m1_cnt = matches[0]
            m2_name, m2_sim, m2_cnt = matches[1] if len(matches) > 1 else ("-", 0.0, 0)
            table.add_row(
                s_name,
                str(s_cnt),
                b_name,
                m1_name,
                f"{m1_sim:.1%}",
                str(m1_cnt),
                f"{m2_name} ({m2_sim:.1%})",
            )
        console.print(table)
    else:
        console.print("[yellow]Inga externa matchningar nådde upp till min-sim tröskeln.[/yellow]")

    console.print("\n")

    # Tabell 2: Interna matchningar
    int_filtered = [r for r in internal_matches if r[3][0][1] >= args.min_sim]
    if int_filtered:
        table = Table(title=f"🔄 INTERNA MATCHNINGAR: Aktiva splittade mappar som liknar sin EGEN ursprungsmapp bäst (Topp {min(20, len(int_filtered))})")
        table.add_column("Splittad mapp", style="cyan", no_wrap=True)
        table.add_column("Bilder", style="blue", justify="right")
        table.add_column("Ursprungsmapp / Bästa Match", style="bold green")
        table.add_column("Likhet", style="yellow", justify="right")
        table.add_column("Näst bästa match (annan person)", style="white")

        for s_name, s_cnt, b_name, matches in int_filtered[:20]:
            m1_name, m1_sim, m1_cnt = matches[0]
            other_m_str = "-"
            for m_name, m_sim, m_cnt in matches[1:]:
                if m_name.lower() != b_name.lower():
                    other_m_str = f"{m_name} ({m_sim:.1%})"
                    break

            table.add_row(s_name, str(s_cnt), m1_name, f"{m1_sim:.1%}", other_m_str)
        console.print(table)

    console.print(f"\n[bold green]✅ Analys slutförd! Rapport med alla {len(results)} aktiva mappar sparas i: {args.output}[/bold green]")
    if not args.show_historical:
        console.print("[italic]Tips: Kör med --show-historical om du vill se gamla/åtgärdade mappar från cachen.[/italic]")


if __name__ == "__main__":
    main()
