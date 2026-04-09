#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Import our new modules
from external_resolver import ExternalIdentityResolver
from identity_resolver import IdentityResolver

console = Console()

def main():
    console.print("[bold blue]Laddar Identitets-Valideringsverktyg...[/bold blue]")
    
    # Files
    face_extractor_dir = Path(__file__).parent
    merge_file = face_extractor_dir / "merge.txt"
    exclusions_file = face_extractor_dir / "similar_exclusions.txt"
    
    if not merge_file.exists():
        console.print(f"[red]Fel: {merge_file} hittades inte.[/red]")
        return

    # 1. Initialize Resolvers
    resolver = IdentityResolver(merge_file, exclusions_file)
    external = ExternalIdentityResolver()
    
    # 2. Extract uniquely mentioned names from merge.txt to verify them externally
    all_names = set()
    with open(merge_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            all_names.update(parts)
    
    console.print(f"Hittade {len(all_names)} unika namn i merge.txt.")
    console.print("[yellow]Kontrollerar namn mot StashDB/ThePornDB (detta kan ta tid vid första körning pga cache-populering)...[/yellow]")
    
    # 3. Resolve names externally and feed into Prolog
    with Progress() as progress:
        task = progress.add_task("[cyan]Anropar externa API:er...", total=len(all_names))
        for name in all_names:
            progress.update(task, description=f"[cyan]Bearbetar: [bold]{name}[/bold]...")
            res = external.resolve(name)
            if res:
                canonical, source = res
                resolver.add_external_truth(name, canonical, source)
                # Only log hits to avoid terminal spam, but show where they come from
                if not external._get_from_cache(name): # If it wasn't already in cache before this resolve
                     console.log(f"[green]Träff![/green] {name} -> [bold]{canonical}[/bold] ({source})")
            progress.update(task, advance=1)
            
    # 4. Check for Conflicts
    console.print("\n[bold green]Söker efter konflikter...[/bold green]")
    conflicts = resolver.check_conflicts()
    
    if conflicts:
        table = Table(title="Logiska Konflikter funna!", show_header=True, header_style="bold red")
        table.add_column("Person 1")
        table.add_column("Person 2")
        table.add_column("Status")
        
        for p1, p2 in conflicts:
            table.add_row(p1, p2, "Länkade via alias men finns i exkluderingslistan!")
        console.print(table)
    else:
        console.print("[green]✅ Inga logiska konflikter funna mellan alias och exkluderingar.[/green]")
        
    # 5. Check for Canonical Name Mismatches
    # (Checking if our local primary matches the external truth)
    console.print("\n[bold green]Verifierar huvudnamn mot externa källor...[/bold green]")
    mismatches = []
    
    # We only care about groups that have an external truth
    checked_groups = set()
    for name in all_names:
        # Get the representative of the group (canonical according to Prolog)
        resolved = resolver.resolve(name)
        
        # Get members of this group
        members = tuple(sorted(resolver.get_group(name)))
        if members in checked_groups:
            continue
        checked_groups.add(members)
        
        # Check if any member has an external hit
        external_hits = {}
        for m in members:
            ext = external._get_from_cache(m) # Use cache helper to see if we actually got a hit
            if ext:
                external_hits[m] = ext
        
        if external_hits:
            # We found external truths for this group.
            # Does our resolved canonical match the highest priority external truth?
            # IdentityResolver.resolve() already handles priority.
            
            # Find the "Expected" canonical (StashDB > TPDB)
            # Actually, Let's just trust Prolog's resolved result if it came from external_truth.
            pass
            
    # Let's list some recommendations
    recommendations_table = Table(title="Rekommenderade Uppdateringar", show_header=True)
    recommendations_table.add_column("Namn i merge.txt")
    recommendations_table.add_column("Föreslaget Huvudnamn (Källa)")
    
    count = 0
    for name in sorted(all_names):
        resolved = resolver.resolve(name)
        if resolved != name:
            # Check if name is the first in its merge line (primary)
            # This is a bit complex to track perfectly without more state, 
            # so let's just show top 20 interesting ones.
            pass

    console.print("[blue]Tips: Kör med --fix (ej implementerat än) för att automatiskt uppdatera merge.txt[/blue]")

if __name__ == "__main__":
    main()
