#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

import processed_db
from external_resolver import ExternalIdentityResolver

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Analysera och reparera alias-tabellen i processed.db mha externa källor.")
    parser.add_argument("--db", required=True, type=Path, help="Sökväg till processed.db.")
    parser.add_argument("--apply", action="store_true", help="Utför faktiska ändringar och ta en backup av DB.")
    return parser.parse_args()

def backup_db(db_path: Path):
    backup_path = db_path.with_suffix(db_path.suffix + ".alias_fix.bak")
    console.print(f"[yellow]Skapar backup av databasen: [bold]{backup_path}[/bold]...[/yellow]")
    shutil.copy2(db_path, backup_path)

def main():
    args = parse_args()
    db_path = args.db
    
    if not db_path.exists():
        console.print(f"[red]Fel: Databasen {db_path} hittades inte.[/red]")
        return

    console.print("[bold blue]Initierar uppkoppling mot externa API:er (StashDB, TPDB, PMVStash, FansDB)...[/bold blue]")
    resolver = ExternalIdentityResolver()
    
    conn = processed_db.open_db(db_path)
    # Ladda rå alias-map utan cycle breaking för att inspektera databasens nuvarande exakta state
    raw_aliases = processed_db.get_alias_map(conn)
    
    if not raw_aliases:
        console.print("[green]Inga alias hittades i databasen![/green]")
        conn.close()
        return

    console.print(f"[cyan]Hittade {len(raw_aliases)} alias-kopplingar att verifiera.[/cyan]")
    
    actions = []  # (alias, old_primary, new_primary, reason)
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Analyserar...", total=len(raw_aliases))
        
        for alias, old_primary in raw_aliases.items():
            progress.update(task, description=f"[cyan]Kollar: [bold]{alias}[/bold]...")
            
            # Först testar vi att lösa upp 'alias'
            res_alias = resolver.resolve(alias)
            
            # Sen testar vi 'old_primary' ifall namnet bytts helt
            res_primary = resolver.resolve(old_primary)
            
            # Bestäm ny canonical
            new_primary = old_primary
            reason = ""
            
            if res_alias:
                canonical, source = res_alias
                if canonical != old_primary:
                    new_primary = canonical
                    reason = f"Externt korrigerad via {source}"
            elif res_primary:
                canonical, source = res_primary
                if canonical != old_primary:
                    new_primary = canonical
                    reason = f"Primary korrigerad via {source}"
            
            # Detektera om alias och primary är omvända och bildar en cykel (ex A->B, B->A)
            if new_primary == alias:
                # Detta alias är faktiskt the True Name!
                # Vi bör radera denna regel.
                actions.append((alias, old_primary, "<RADERA (Är True Name)>", "Självreferens / Cykel bruten"))
            elif new_primary != old_primary:
                actions.append((alias, old_primary, new_primary, reason))
                
            progress.update(task, advance=1)
            
    if not actions:
        console.print("[green]✅ Alla alias i databasen pekar redan på korrekt kanoniskt namn![/green]")
        conn.close()
        return
        
    # Presentera förslag
    table = Table(title=f"Föreslagna Rättningar ({len(actions)} st)")
    table.add_column("Alias", style="cyan")
    table.add_column("Gammal Primary", style="red")
    table.add_column("Ny Primary", style="green")
    table.add_column("Anledning", style="yellow")
    
    for alias, old, new, reason in actions:
        table.add_row(alias, old, new, reason)
        
    console.print(table)
    
    if not args.apply:
        console.print("\n[bold yellow]Detta är en DRY RUN. Inga ändringar gjordes i databasen.[/bold yellow]")
        console.print("Kör med [bold]--apply[/bold] för att utföra ändringarna.")
        conn.close()
        return
        
    # Utför
    backup_db(db_path)
    
    try:
        with console.status("[bold green]Uppdaterar databasen..."):
            updates = {}
            deletes = []
            for alias, old, new, reason in actions:
                if new.startswith("<RADERA"):
                    deletes.append(alias)
                else:
                    updates[alias] = new
                    
            if updates:
                processed_db.add_aliases_batch(conn, updates)
            
            if deletes:
                # delete the ones that point to themselves
                for alias in deletes:
                    conn.execute("DELETE FROM aliases WHERE alias = ?", (alias,))
                conn.commit()
                
        console.print(f"[bold green]✅ Uppdaterade {len(updates)} poster och raderade {len(deletes)} felaktiga poster i databasen.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Misslyckades med att uppdatera DB: {e}[/bold red]")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
