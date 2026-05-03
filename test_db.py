import sqlite3
import processed_db

conn = processed_db.open_db("arcface_work-ppic/processed.db")
amap = processed_db.get_alias_map(conn)
# find mapping that could be causing 691 files to move every time
import os
from pathlib import Path

root = Path("/home/marqs/Bilder/pBook")
existing_aliases = sorted([alias for alias in amap if (root / alias).is_dir()], key=str.casefold)

for alias in existing_aliases:
    main = amap[alias]
    alias_dir = root / alias
    main_dir = root / main
    if alias_dir.resolve() != main_dir.resolve():
        num_files = len([p for p in alias_dir.iterdir() if p.is_file()])
        if num_files > 0:
            print(f"Alias '{alias}' -> Main '{main}'. Files: {num_files}")
