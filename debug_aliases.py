import processed_db
from pathlib import Path
import json

def test():
    db_path = "arcface_work-ppic/processed.db"
    conn = processed_db.open_db(db_path)
    amap = processed_db.get_alias_map(conn)
    
    root = Path("/home/marqs/Bilder/pBook")
    existing_aliases = sorted([alias for alias in amap if (root / alias).is_dir()], key=str.casefold)

    problems = []
    
    for alias in existing_aliases:
        main = amap[alias]
        alias_dir = root / alias
        main_dir = root / main
        
        if alias_dir.resolve() != main_dir.resolve():
            num_files = len([p for p in alias_dir.iterdir() if p.is_file()])
            if num_files > 0:
                problems.append({
                    "alias": alias,
                    "main": main,
                    "files": num_files,
                    "alias_dir": str(alias_dir),
                    "main_dir": str(main_dir)
                })

    with open("debug_aliases.json", "w") as f:
        json.dump(problems, f, indent=2)

if __name__ == "__main__":
    test()
