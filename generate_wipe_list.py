import sqlite3
from pathlib import Path

# Config
BASE_DIR = Path("/home/marqs/Programmering/Python/3.11")
FACE_EXTRACTOR_DIR = BASE_DIR / "face_extractor"
PROCESSED_DB = FACE_EXTRACTOR_DIR / "arcface_work-ppic" / "processed.db"
MERGE_TXT = FACE_EXTRACTOR_DIR / "merge.txt"
OUTPUT_FILE = FACE_EXTRACTOR_DIR / "wiped_targets.txt"

def get_primary_names():
    primaries = []
    with open(MERGE_TXT, "r", encoding="utf-8") as f:
        for line in f:
            names = [n.strip() for n in line.split("|") if n.strip()]
            if names:
                primaries.append(names[0])
    return sorted(list(set(primaries)))

def main():
    primaries = get_primary_names()
    conn = sqlite3.connect(str(PROCESSED_DB))
    
    # Get all names that currently have data in processed.db
    cur = conn.execute("SELECT DISTINCT path FROM processed")
    existing_parents = set()
    for (path,) in cur:
        existing_parents.add(Path(path).parent.name)
    conn.close()
    
    # We want to find primaries that are MISSING from the DB 
    # (since we wiped them)
    wiped_targets = []
    for p in primaries:
        if p not in existing_parents:
            wiped_targets.append(p)
            
    print(f"🔍 Hittade {len(wiped_targets)} modeller som saknar data i DB (potentiellt raderade).")
    
    # Special: Se till att Piper Fawn är med om hon saknas
    if "Piper Fawn" not in wiped_targets and "Piper Fawn" in primaries:
        # Om hon faktiskt saknas i DB men inte hamnade i listan, kolla varför.
        pass

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name in wiped_targets:
            f.write(f"{name}\n")
            
    print(f"✅ Skrev {len(wiped_targets)} namn till {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
