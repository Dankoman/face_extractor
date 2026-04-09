#!/usr/bin/env python3
import os
import shutil
import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich.progress import Progress

import processed_db
from identity_resolver import IdentityResolver
from external_resolver import ExternalIdentityResolver

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Rätta artistnamn på filsystemet och i databasen.")
    parser.add_argument("--data-root", required=True, type=Path, help="Rot-mapp för artister.")
    parser.add_argument("--db", required=True, type=Path, help="Sökväg till processed.db.")
    parser.add_argument("--apply", action="store_true", help="Utför faktiska ändringar (standard är Dry Run).")
    parser.add_argument("--merge-txt", default="merge.txt", type=Path)
    parser.add_argument("--exclusions-txt", default="similar_exclusions.txt", type=Path)
    return parser.parse_args()

def backup_db(db_path: Path):
    backup_path = db_path.with_suffix(db_path.suffix + ".bak")
    console.print(f"[yellow]Skapar backup av databasen: [bold]{backup_path}[/bold]...[/yellow]")
    shutil.copy2(db_path, backup_path)

def merge_directories(src: Path, dst: Path, apply: bool):
    """Flyttar filer från src till dst. Hanterar namnkonflikter."""
    files = list(src.iterdir())
    moves = []
    for f in files:
        if not f.is_file():
            continue
        target = dst / f.name
        if target.exists():
            # Rename conflict: Name.jpg -> Name-merge.jpg
            stem, suffix = f.stem, f.suffix
            counter = 1
            while target.exists():
                target = dst / f"{stem}-merge{counter}{suffix}"
                counter += 1
        
        moves.append((f, target))
    
    if apply:
        if not dst.exists():
            dst.mkdir(parents=True, exist_ok=True)
        for s, d in moves:
            s.rename(d)
        # Ta bort src om den är tom
        try:
            src.rmdir()
        except OSError:
            console.print(f"[red]Kunde inte ta bort mappen {src} (kanske inte tom).[/red]")
    return len(moves)

def main():
    args = parse_args()
    data_root = args.data_root
    db_path = args.db
    
    if not data_root.is_dir():
        console.print(f"[red]Fel: {data_root} är inte en mapp.[/red]")
        return
    
    if not db_path.exists():
        console.print(f"[red]Fel: Databasen {db_path} hittades inte.[/red]")
        return

    # 1. Initiera Resolvers
    resolver = IdentityResolver(args.merge_txt, args.exclusions_txt)
    external = ExternalIdentityResolver()
    
    # 2. Skanna mappar och hämta externa sanningar
    console.print("[bold blue]Skannar mappar och hämtar data från StashDB/TPDB...[/bold blue]")
    performer_dirs = [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Beräknar rättningar...", total=len(performer_dirs))
        for d in performer_dirs:
            original_name = d.name
            progress.update(task, description=f"[cyan]Kollar: [bold]{original_name}[/bold]...")
            
            # Kolla externa källor (använder cache i första hand)
            res = external.resolve(original_name)
            if res:
                canonical, source = res
                resolver.add_external_truth(original_name, canonical, source)
            
            progress.update(task, advance=1)

    # 3. Beräkna faktiska ändringar baserat på all samlad logik
    actions = [] # list of (old_name, new_name, action_type)
    for d in performer_dirs:
        original_name = d.name
        canonical_name = resolver.resolve(original_name)
        
        if original_name != canonical_name:
            target_path = data_root / canonical_name
            if target_path.exists():
                actions.append((original_name, canonical_name, "MERGE"))
            else:
                actions.append((original_name, canonical_name, "MOVE"))

    if not actions:
        console.print("[green]✅ Alla mappar har redan korrekta namn![/green]")
        return

    # 4. Presentera förslag
    table = Table(title="Föreslagna Identitets-rättningar")
    table.add_column("Originalnamn", style="red")
    table.add_column("Nytt Namn (Canonical)", style="green")
    table.add_column("Typ", style="yellow")
    
    for old, new, action in actions:
        table.add_row(old, new, action)
    
    console.print(table)
    
    if not args.apply:
        console.print("\n[bold yellow]Detta är en DRY RUN. Inga ändringar gjordes.[/bold yellow]")
        console.print("Kör med [bold]--apply[/bold] för att utföra ändringarna.")
        return

    # 5. Utför ändringar
    if not Confirm.ask(f"Är du säker på att du vill utföra {len(actions)} ändringar?"):
        return

    backup_db(db_path)
    conn = processed_db.open_db(db_path)
    
    try:
        with console.status("[bold green]Utför rättningar..."):
            for old_name, new_name, action in actions:
                src = data_root / old_name
                dst = data_root / new_name
                
                # A. Filesystem
                if action == "MOVE":
                    src.rename(dst)
                else: # MERGE
                    merge_directories(src, dst, apply=True)
                
                # B. Database (Paths)
                updated_count = processed_db.update_directory_path(conn, str(src), str(dst))
                
                # C. Database (Aliases)
                # Spara mappningen gammalt -> nytt så att vi kan slå upp alias i framtiden
                processed_db.add_aliases_batch(conn, {old_name: new_name})
                
                console.print(f"  [green]Fixat:[/green] {old_name} -> {new_name} ({updated_count} poster uppdaterade)")
                
        console.print("\n[bold green]✅ Klart! Allt är rättat.[/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]Ett fel uppstod: {e}[/bold red]")
        console.print("[yellow]Kontrollera backupen av databasen om något gick fel.[/yellow]")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
