import argparse
from pathlib import Path
from collections import defaultdict

def load_alias_map(merge_path: Path):
    alias_to_primary = {}
    if not merge_path.exists():
        return alias_to_primary
    
    for line in merge_path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.strip().split("|") if p.strip()]
        if len(parts) >= 2:
            primary = parts[0]
            for alias in parts[1:]:
                alias_to_primary[alias] = primary
    return alias_to_primary

def main():
    root = Path("/home/marqs/Bilder/pBook")
    remove_file = Path("remove.txt")
    merge_file = Path("merge.txt")

    if not remove_file.exists():
        print("remove.txt saknas.")
        return

    removals = [line.strip() for line in remove_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    alias_map = load_alias_map(merge_file)

    print(f"Checking {len(removals)} names from remove.txt against {root}...\n")

    found_conflict = False
    for name in removals:
        primary = alias_map.get(name, name)
        
        # Check if primary folder exists
        primary_folder = root / primary
        
        if primary_folder.exists() and primary_folder.is_dir():
            found_conflict = True
            msg = f"❌ {name}"
            if primary != name:
                msg += f" (alias för '{primary}')"
            msg += " kommer ligga kvar"
            print(f"{msg} eftersom mappen '{primary}' finns.")
    
    if not found_conflict:
        print("✅ Inga konflikter hittades. Alla listade namn verkar sakna mappar.")
    else:
        print("\nFörklaring: Om mappen finns kvar kommer modellen träna på den igen, även om du lagt namnet i remove.txt.")
        print("Lösning: Ta bort mappen för 'Primary' också, eller ta bort aliaset i merge.txt om det är felaktigt.")

if __name__ == "__main__":
    main()
