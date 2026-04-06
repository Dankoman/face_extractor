"""Ta bort embeddings för listade namn (med alias-stöd)."""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


import processed_db

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# load_alias_map ersatt av processed_db.get_alias_map()


def expand_remove_set(remove_names: Iterable[str], alias_to_primary: Dict[str, str],
                      primary_to_aliases: Dict[str, Set[str]]) -> tuple[Set[str], Dict[str, int]]:
    primary_counts = {alias_to_primary.get(name, name): 0 for name in remove_names}
    expanded: Set[str] = set()
    for primary in primary_counts:
        expanded.update(primary_to_aliases.get(primary, {primary}))
    return expanded, primary_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ta bort embeddings för namn i remove.txt")
    parser.add_argument("--embeddings", default="arcface_work-ppic/embeddings_ppic.pkl", help="Pickle-fil med embeddings")
    parser.add_argument("--remove", default="remove.txt", help="Textfil med namn att ta bort")
    parser.add_argument("--db", default="arcface_work-ppic/processed.db", help="SQLite-databas")
    parser.add_argument("--no-alias", action="store_true", help="Utöka inte alias – ta bara bort exakt angivna namn")
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    remove_path = Path(args.remove)
    db_path = Path(args.db)

    if not embeddings_path.exists():
        raise SystemExit(f"Hittar inte embeddingsfilen: {embeddings_path}")
    if not remove_path.exists():
        raise SystemExit(f"Hittar inte remove-filen: {remove_path}")

    remove_names = read_lines(remove_path)
    conn = processed_db.open_db(db_path)
    
    if args.no_alias:
        alias_to_primary: Dict[str, str] = {}
        primary_to_aliases: Dict[str, Set[str]] = {}
        expanded_remove = set(remove_names)
        counters = {name: 0 for name in remove_names}
    else:
        alias_to_primary = processed_db.get_alias_map(conn)
        primary_to_aliases = defaultdict(set)
        for alias, primary in alias_to_primary.items():
            primary_to_aliases[primary].add(alias)
            
        expanded_remove, counters = expand_remove_set(remove_names, alias_to_primary, primary_to_aliases)
    conn.close()

    with embeddings_path.open("rb") as f:
        data = pickle.load(f)

    X: Sequence = data["X"]
    y: Sequence[str] = data["y"]

    keep_idx: List[int] = []
    removed = 0
    for idx, label in enumerate(y):
        primary = alias_to_primary.get(label, label)
        if label in expanded_remove or primary in counters:
            counters[primary] = counters.get(primary, 0) + 1
            removed += 1
            continue
        keep_idx.append(idx)

    X_new = [X[i] for i in keep_idx]
    y_new = [y[i] for i in keep_idx]

    with embeddings_path.open("wb") as f:
        pickle.dump({"X": X_new, "y": y_new}, f)

    missing = [name for name, count in counters.items() if count == 0]
    if HAS_RICH:
        console = Console()
        console.print(f"✅ Tog bort [bold red]{removed}[/bold red] poster från '{embeddings_path}' baserat på '{remove_path}'.")
        if missing:
            console.print("[yellow]Ej hittade i embeddings:[/yellow]")
            for name in missing:
                console.print(f"  - [dim]{name}[/dim]")
    else:
        print(f"Tog bort {removed} poster från '{embeddings_path}' baserat på '{remove_path}'.")
        if missing:
            print("Ej hittade i embeddings:")
            for name in missing:
                print(f"  - {name}")


if __name__ == "__main__":
    main()
