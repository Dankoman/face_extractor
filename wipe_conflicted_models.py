import sqlite3
import os
import shutil
from pathlib import Path

# Config
BASE_DIR = Path("/home/marqs/Programmering/Python/3.11")
FACE_EXTRACTOR_DIR = BASE_DIR / "face_extractor"
DOPPELGANGER_DIR = BASE_DIR / "doppelganger"

PROCESSED_DB = FACE_EXTRACTOR_DIR / "arcface_work-ppic" / "processed.db"
SCRAPER_DB = DOPPELGANGER_DIR / "ppic_scraper_state.db"
MERGE_TXT = FACE_EXTRACTOR_DIR / "merge.txt"

def get_groups():
    groups = []
    with open(MERGE_TXT, "r", encoding="utf-8") as f:
        for line in f:
            names = [n.strip() for n in line.split("|") if n.strip()]
            if names:
                groups.append(names)
    return groups

def wipe_model(name, conn_proc, conn_scrap):
    print(f"🔥 Wiping {name}...")
    
    # 1. Wipe from Processed DB (facial analysis)
    # We find paths in the DB and delete folders
    cur = conn_proc.execute("SELECT DISTINCT path FROM processed")
    paths_to_delete = set()
    rows = cur.fetchall()
    for (path,) in rows:
        p = Path(path)
        if p.parent.name == name:
            paths_to_delete.add(p.parent)
    
    # Actually delete folders
    for folder in paths_to_delete:
        if folder.exists():
            print(f"  🗑️ Deleting directory: {folder}")
            shutil.rmtree(folder)
            
    # Remove from processed table
    # We need to find all paths that belong to this name
    conn_proc.execute("DELETE FROM processed WHERE path LIKE ?", (f"%/{name}/%",))
    
    # 2. Wipe from Scraper DB
    conn_scrap.execute("DELETE FROM models WHERE name = ?", (name,))
    conn_scrap.execute("DELETE FROM galleries WHERE model_name = ?", (name,))
    conn_scrap.execute("DELETE FROM images WHERE model_name = ?", (name,))
    
    # Also check for alternate names in scraper DB
    # (Since synonyms might have been used before merging)
    # We'll do this for all names in the group in the outer loop

def main():
    groups = get_groups()
    
    conn_proc = sqlite3.connect(str(PROCESSED_DB))
    conn_scrap = sqlite3.connect(str(SCRAPER_DB))
    
    # Detektera alla namn i DB för att veta vilka som finns
    print("🔍 Analyserar databaser efter existerande modeller...")
    
    # Från Processed DB (mappnamn)
    cur = conn_proc.execute("SELECT DISTINCT path FROM processed")
    existing_folders = set()
    for (path,) in cur:
        existing_folders.add(Path(path).parent.name)
        
    # Från Scraper DB
    cur = conn_scrap.execute("SELECT name FROM models")
    existing_in_scraper = {row[0] for row in cur}

    for group in groups:
        primary = group[0]
        all_names = set(group)
        
        # Vi torkar OM:
        # 1. Gruppen innehåller "Piper Fawn" (eftersom användaren bad om det explicit)
        # 2. Mer än ett namn i gruppen har data i någon DB (indikerar dubbel identitet)
        
        names_with_data = all_names.intersection(existing_folders.union(existing_in_scraper))
        
        needs_wipe = False
        if "Piper Fawn" in all_names:
            needs_wipe = True
        elif len(names_with_data) > 1:
            print(f"🚩 Konflikt detekterad för grupp {primary}: {names_with_data}")
            needs_wipe = True
            
        if needs_wipe:
            print(f"🧹 Totalsanering av grupp: {primary} ({len(all_names)} namn)")
            for name in all_names:
                wipe_model(name, conn_proc, conn_scrap)
            # Se till att även primärnamnet är borta om det inte fanns i all_names (borde finnas)
            wipe_model(primary, conn_proc, conn_scrap)

    conn_proc.commit()
    conn_scrap.commit()
    conn_proc.close()
    conn_scrap.close()
    print("✨ Alla konflikter har rensats.")

if __name__ == "__main__":
    main()
