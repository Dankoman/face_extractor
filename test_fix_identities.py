import os
import shutil
import sqlite3
from pathlib import Path
import subprocess

def test_fix():
    print("--- Testing fix_identities_fs.py ---")
    
    # 1. Setup Dummy Data
    root = Path("test_data_root")
    root.mkdir(exist_ok=True)
    (root / "OlderDir").mkdir(exist_ok=True)
    with open(root / "OlderDir" / "image1.jpg", "w") as f:
        f.write("test")
        
    # 2. Setup Dummy DB
    db_path = Path("test_processed.db")
    if db_path.exists(): db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE processed (path TEXT PRIMARY KEY, ok INTEGER, reason TEXT)")
    conn.execute("CREATE TABLE aliases (alias TEXT PRIMARY KEY, primary_name TEXT)")
    conn.execute("INSERT INTO processed VALUES (?, ?, ?)", (str(root / "OlderDir" / "image1.jpg"), 1, "ok"))
    conn.commit()
    conn.close()
    
    # 3. Setup Identity Logic (via merge.txt)
    with open("test_merge.txt", "w") as f:
        f.write("NewCanonical|OlderDir\n")
        
    # 4. Run Fixer (Apply)
    cmd = [
        "python", "fix_identities_fs.py",
        "--data-root", str(root),
        "--db", str(db_path),
        "--merge-txt", "test_merge.txt",
        "--apply"
    ]
    # Simulate 'yes' to the prompt
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input="y\n")
    
    print(stdout)
    if stderr: print(f"Errors: {stderr}")
    
    # 5. Verify Results
    print("\n--- Verifying Results ---")
    new_dir = root / "NewCanonical"
    old_dir = root / "OlderDir"
    
    assert new_dir.exists()
    assert not old_dir.exists()
    assert (new_dir / "image1.jpg").exists()
    
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT path FROM processed")
    paths = [row[0] for row in cur]
    print(f"Paths in DB: {paths}")
    assert str(new_dir / "image1.jpg") in paths
    
    cur = conn.execute("SELECT alias, primary_name FROM aliases")
    aliases = {row[0]: row[1] for row in cur}
    print(f"Aliases in DB: {aliases}")
    assert aliases["OlderDir"] == "NewCanonical"
    
    conn.close()
    print("\n✅ Fixer test passed!")
    
    # Cleanup
    shutil.rmtree(root)
    db_path.unlink()
    Path("test_merge.txt").unlink()
    if Path("test_processed.db.bak").exists(): Path("test_processed.db.bak").unlink()

if __name__ == "__main__":
    test_fix()
