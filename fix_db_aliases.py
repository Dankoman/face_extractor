#!/usr/bin/env python3
"""
Analysera och reparera alias-tabellen i processed.db mot externa källor
(StashDB, ThePornDB, PMVStash, FansDB).

Fyra typer av åtgärder:
  1. UPPDATERA – alias pekar på rätt person men namnet har ändrats externt
  2. RADERA (självreferens) – alias pekar på sig själv / cykel
  3. BRYT ISÄR – de externa källorna bekräftar att alias och primary är OLIKA personer
  4. BRYT OVERIFIERBARA – namn som inte hittas externt och är tillräckligt olika
     (aktiveras med --break-unverified-different)
"""

import argparse
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

import processed_db
from external_resolver import ExternalIdentityResolver

console = Console()

# Åtgärdstyper
ACTION_UPDATE = "UPDATE"
ACTION_DELETE_SELF = "DELETE_SELF"
ACTION_DELETE_DIFFERENT = "DELETE_DIFFERENT"
ACTION_DELETE_UNVERIFIED = "DELETE_UNVERIFIED"

# Tröskel för ordlikhet – under detta anses namnen vara "helt olika"
SIMILARITY_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analysera och reparera alias-tabellen i processed.db mha externa källor."
    )
    parser.add_argument("--db", required=True, type=Path, help="Sökväg till processed.db.")
    parser.add_argument("--apply", action="store_true", help="Utför faktiska ändringar och ta en backup av DB.")
    parser.add_argument(
        "--break-unverified-different", action="store_true",
        help="Bryt även isär overifierbara alias där namnen är helt olika "
             "(under 50%% ordlikhet). Alias med liknande stavning lämnas ifred.",
    )
    return parser.parse_args()


def backup_db(db_path: Path):
    backup_path = db_path.with_suffix(db_path.suffix + ".alias_fix.bak")
    console.print(f"[yellow]Skapar backup av databasen: [bold]{backup_path}[/bold]...[/yellow]")
    shutil.copy2(db_path, backup_path)


def names_match(a: str, b: str) -> bool:
    """Case-insensitive name comparison."""
    return a.strip().lower() == b.strip().lower()


def name_similarity(a: str, b: str) -> float:
    """Jaccard-likhet baserat på ord. 0.0 = inga gemensamma ord, 1.0 = identiska."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


def main():
    args = parse_args()
    db_path = args.db

    if not db_path.exists():
        console.print(f"[red]Fel: Databasen {db_path} hittades inte.[/red]")
        return

    console.print("[bold blue]Initierar uppkoppling mot externa API:er (StashDB, TPDB, PMVStash, FansDB)...[/bold blue]")
    resolver = ExternalIdentityResolver()

    conn = processed_db.open_db(db_path)
    raw_aliases = processed_db.get_alias_map(conn)

    if not raw_aliases:
        console.print("[green]Inga alias hittades i databasen![/green]")
        conn.close()
        return

    console.print(f"[cyan]Hittade {len(raw_aliases)} alias-kopplingar att verifiera.[/cyan]")

    # (alias, old_primary, new_value_or_None, reason, action_type)
    actions = []
    stats = {"confirmed": 0, "updated": 0, "broken": 0, "self_ref": 0, "unverifiable": 0, "unverified_broken": 0}

    with Progress() as progress:
        task = progress.add_task("[cyan]Analyserar...", total=len(raw_aliases))

        for alias, old_primary in raw_aliases.items():
            progress.update(task, description=f"[cyan]Kollar: [bold]{alias}[/bold]...")

            # Resolve both sides against external sources
            res_alias = resolver.resolve(alias)
            res_primary = resolver.resolve(old_primary)

            canonical_alias = res_alias[0] if res_alias else None
            source_alias = res_alias[1] if res_alias else None
            canonical_primary = res_primary[0] if res_primary else None
            source_primary = res_primary[1] if res_primary else None

            if canonical_alias and canonical_primary:
                # === Båda namnen hittades externt ===
                if names_match(canonical_alias, canonical_primary):
                    # Samma person – bekräftad!
                    stats["confirmed"] += 1
                    if not names_match(canonical_alias, old_primary):
                        actions.append((
                            alias, old_primary, canonical_alias,
                            f"Bekräftad & namn uppdaterat via {source_alias}",
                            ACTION_UPDATE,
                        ))
                else:
                    # OLIKA PERSONER – bryt isär!
                    stats["broken"] += 1
                    actions.append((
                        alias, old_primary, None,
                        f"'{alias}' → '{canonical_alias}' ({source_alias}), "
                        f"'{old_primary}' → '{canonical_primary}' ({source_primary})",
                        ACTION_DELETE_DIFFERENT,
                    ))

            elif canonical_alias:
                # === Bara alias hittades externt ===
                if names_match(canonical_alias, old_primary):
                    stats["confirmed"] += 1
                elif names_match(canonical_alias, alias):
                    # Alias ÄR det kanoniska namnet (självreferens)
                    stats["self_ref"] += 1
                    actions.append((
                        alias, old_primary, None,
                        f"'{alias}' är det kanoniska namnet ({source_alias}), "
                        f"'{old_primary}' okänt externt",
                        ACTION_DELETE_SELF,
                    ))
                else:
                    # Alias pekar på en tredje person, primary okänd
                    stats["updated"] += 1
                    actions.append((
                        alias, old_primary, canonical_alias,
                        f"Externt korrigerad via {source_alias}",
                        ACTION_UPDATE,
                    ))

            elif canonical_primary:
                # === Bara primary hittades externt ===
                if not names_match(canonical_primary, old_primary):
                    stats["updated"] += 1
                    actions.append((
                        alias, old_primary, canonical_primary,
                        f"Primary korrigerad via {source_primary}",
                        ACTION_UPDATE,
                    ))
                else:
                    stats["unverifiable"] += 1
            else:
                # === Inget namn hittades externt ===
                if args.break_unverified_different:
                    sim = name_similarity(alias, old_primary)
                    if sim < SIMILARITY_THRESHOLD:
                        stats["unverified_broken"] += 1
                        actions.append((
                            alias, old_primary, None,
                            f"Overifierbar & olika namn (likhet {sim:.0%})",
                            ACTION_DELETE_UNVERIFIED,
                        ))
                    else:
                        stats["unverifiable"] += 1
                else:
                    stats["unverifiable"] += 1

            progress.update(task, advance=1)

    # Sammanfattning
    console.print(f"\n[bold]Resultat:[/bold]")
    console.print(f"  ✅ Bekräftade:      {stats['confirmed']}")
    console.print(f"  ✏️  Att uppdatera:   {stats['updated']}")
    console.print(f"  💔 Olika personer:   {stats['broken']}")
    console.print(f"  🔄 Självreferenser:  {stats['self_ref']}")
    if stats["unverified_broken"]:
        console.print(f"  🟠 Overifierbara brutna: {stats['unverified_broken']}")
    console.print(f"  ❓ Overifierbara:    {stats['unverifiable']}")

    if not actions:
        console.print("\n[green]✅ Alla verifierbara alias pekar redan korrekt![/green]")
        conn.close()
        return

    # --- Tabeller per åtgärdstyp ---

    broken = [a for a in actions if a[4] == ACTION_DELETE_DIFFERENT]
    if broken:
        table = Table(title=f"🔴 Alias att BRYTA ISÄR – Olika personer ({len(broken)} st)")
        table.add_column("Alias", style="cyan")
        table.add_column("Gammal Primary", style="red")
        table.add_column("Bevis", style="yellow")
        for alias, old, _, reason, _ in broken:
            table.add_row(alias, old, reason)
        console.print(table)

    self_refs = [a for a in actions if a[4] == ACTION_DELETE_SELF]
    if self_refs:
        table = Table(title=f"🔄 Självreferenser att RADERA ({len(self_refs)} st)")
        table.add_column("Alias", style="cyan")
        table.add_column("Gammal Primary", style="red")
        table.add_column("Anledning", style="yellow")
        for alias, old, _, reason, _ in self_refs:
            table.add_row(alias, old, reason)
        console.print(table)

    updates_list = [a for a in actions if a[4] == ACTION_UPDATE]
    if updates_list:
        table = Table(title=f"✏️  Alias att UPPDATERA ({len(updates_list)} st)")
        table.add_column("Alias", style="cyan")
        table.add_column("Gammal Primary", style="red")
        table.add_column("Ny Primary", style="green")
        table.add_column("Anledning", style="yellow")
        for alias, old, new, reason, _ in updates_list:
            table.add_row(alias, old, new or "?", reason)
        console.print(table)

    unverified_broken = [a for a in actions if a[4] == ACTION_DELETE_UNVERIFIED]
    if unverified_broken:
        table = Table(title=f"🟠 Overifierbara med OLIKA namn att BRYTA ISÄR ({len(unverified_broken)} st)")
        table.add_column("Alias", style="cyan")
        table.add_column("Gammal Primary", style="red")
        table.add_column("Anledning", style="yellow")
        for alias, old, _, reason, _ in unverified_broken:
            table.add_row(alias, old, reason)
        console.print(table)

    if not args.apply:
        console.print("\n[bold yellow]Detta är en DRY RUN. Inga ändringar gjordes i databasen.[/bold yellow]")
        console.print("Kör med [bold]--apply[/bold] för att utföra ändringarna.")
        conn.close()
        return

    # --- Utför ändringar ---
    backup_db(db_path)

    try:
        with console.status("[bold green]Uppdaterar databasen..."):
            updates = {}
            deletes = []
            for alias, old, new, reason, action_type in actions:
                if action_type in (ACTION_DELETE_DIFFERENT, ACTION_DELETE_SELF, ACTION_DELETE_UNVERIFIED):
                    deletes.append(alias)
                elif action_type == ACTION_UPDATE and new:
                    updates[alias] = new

            if updates:
                processed_db.add_aliases_batch(conn, updates)

            if deletes:
                for alias in deletes:
                    conn.execute("DELETE FROM aliases WHERE alias = ?", (alias,))
                conn.commit()

        console.print(
            f"[bold green]✅ Uppdaterade {len(updates)} poster och "
            f"raderade {len(deletes)} felaktiga poster i databasen.[/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Misslyckades med att uppdatera DB: {e}[/bold red]")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
