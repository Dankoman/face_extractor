#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import processed_db
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Rensa embeddings som saknar mapp på disk (orphans).")
    parser.add_argument("--embeddings", required=True, type=Path, help="Sökväg till embeddings (.pkl)")
    parser.add_argument("--db", required=True, type=Path, help="Sökväg till processed.db")
    parser.add_argument("--data-root", required=True, type=Path, help="Sökväg till pBook (modeller)")
    parser.add_argument("--dry-run", action="store_true", help="Kör utan att spara ändringar")
    args = parser.parse_args()

    if not args.embeddings.exists():
        console.print(f"[red]Embeddings-fil hittades inte: {args.embeddings}[/red]")
        return
    if not args.db.exists():
        console.print(f"[red]DB-fil hittades inte: {args.db}[/red]")
        return
    if not args.data_root.exists():
        console.print(f"[red]Data-root hittades inte: {args.data_root}[/red]")
        return

    # Ladda DB
    conn = processed_db.open_db(args.db)
    alias_map = processed_db.get_resolved_alias_map(conn)
    conn.close()

    # Ladda embeddings
    with open(args.embeddings, "rb") as f:
        data = pickle.load(f)
    
    X = data["X"]
    y = data["y"]
    
    original_count = len(X)
    original_people = len(set(y))
    
    # Upplös namn
    resolved_y = [alias_map.get(label, label) for label in y]
    
    # Identifiera vilka som finns på disk
    valid_names = set()
    for name in set(resolved_y):
        path = args.data_root / name
        if path.exists() and path.is_dir():
            valid_names.add(name)
            
    # Filtrera
    new_X = []
    new_y = []
    
    for emb, orig_label, res_label in zip(X, y, resolved_y):
        if res_label in valid_names:
            new_X.append(emb)
            new_y.append(orig_label) # Behåller originaletiketten så merge.py fungerar som tänkt
            
    new_count = len(new_X)
    new_people = len(set(new_y))
    removed_count = original_count - new_count
    
    console.print(f"Började med: {original_count} embeddings ({original_people} unika personer)")
    console.print(f"Tar bort:    {removed_count} embeddings som saknar mapp i {args.data_root.name}")
    console.print(f"Behåller:    {new_count} embeddings ({new_people} unika personer)")
    
    if args.dry_run:
        console.print("[yellow]Dry-run: inga ändringar sparades.[/yellow]")
    elif removed_count > 0:
        with open(args.embeddings, "wb") as f:
            pickle.dump({"X": new_X, "y": new_y}, f)
        console.print(f"[green]✅ Sparade uppdaterad {args.embeddings.name}[/green]")
    else:
        console.print("[green]✅ Inget att rensa![/green]")

if __name__ == "__main__":
    main()
