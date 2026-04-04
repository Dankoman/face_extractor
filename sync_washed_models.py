#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Set

from PIL import Image

# Standardinställningar
INNE_DIR = Path("/home/marqs/Bilder/Innie")
PBOOK_DIR = Path("/home/marqs/Bilder/pBook")
BACKUP_DIR = Path("/home/marqs/Bilder/pBook_backups")
MERGE_FILE = Path("/home/marqs/Programmering/Python/3.11/face_extractor/merge.txt")
UNCERTAINTY_SCRIPT = Path("/home/marqs/Programmering/Python/3.11/face_extractor/model_uncertainty.py")
DB_PATH = Path("/home/marqs/Programmering/Python/3.11/face_extractor/arcface_work-ppic/processed.db")
EMB_PATH = Path("/home/marqs/Programmering/Python/3.11/face_extractor/arcface_work-ppic/embeddings_ppic.pkl")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG"}


def load_merge_map(path: Path) -> Dict[str, str]:
    """Läs merge.txt och skapa Alias -> Huvudnamn mappning."""
    merge_map = {}
    if not path.exists():
        print(f"⚠️ Varning: {path} hittades inte.")
        return merge_map

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            primary = parts[0]
            for alias in parts:
                merge_map[alias] = primary
    return merge_map


def run_analysis() -> Set[str]:
    """Kör model_uncertainty.py och returnera flaggade namn."""
    report_file = "sync_pipeline_report.csv"
    cmd = [
        "python3", str(UNCERTAINTY_SCRIPT),
        "--db", str(DB_PATH),
        "--embeddings", str(EMB_PATH),
        "--output", report_file,
        "--top", "1000"
    ]
    
    print(f"🔍 Kör analys: {' '.join(cmd)}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysen misslyckades kapitalt: {e}")
        print("⚠️ Avbryter synk för att inte missa flaggade mappar pga miljöfel.")
        exit(1)

    flagged = set()
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if row.get("Person A"): flagged.add(row["Person A"])
                if row.get("Person B"): flagged.add(row["Person B"])
    else:
        print(f"❌ Fel: {report_file} skapades inte av analysscriptet.")
        exit(1)
    
    print(f"✅ Analys klar. {len(flagged)} flaggade personer hittades.")
    return flagged


def create_btrfs_snapshot(source: Path, target_base: Path, name: str, dry_run: bool):
    """Skapa en backup-kopia med Btrfs reflinks (ögonblicklig, CoW)."""
    if not source.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = target_base / timestamp / name
    
    if not dry_run:
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"📸 Skapar snapshot: {source} -> {snapshot_dir}")
        # Använd cp --reflink=always för btrfs-magi
        subprocess.run(["cp", "--reflink=always", "-r", str(source), str(snapshot_dir)], check=True)
    else:
        print(f"DRY-RUN: Skulle skapat snapshot av {source} i {snapshot_dir}")


def sync_person(source_folder: Path, primary_name: str, dry_run: bool):
    """Hantera flytt, rensning och konvertering för en person."""
    target_dir = PBOOK_DIR / primary_name
    
    # 1. Analysera befintlig målmapp
    existing_images = []
    small_images = []
    if target_dir.exists():
        existing_images = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        small_images = [f for f in existing_images if f.stat().st_size <= 100 * 1024] # 100 KB

    # 2. Beslut om rensning och Snapshot
    if existing_images:
        create_btrfs_snapshot(target_dir, BACKUP_DIR, primary_name, dry_run)
        
        should_wipe_all = len(existing_images) >= 20
        
        if should_wipe_all:
            if not dry_run:
                print(f"🧹 Rensar ALLA ({len(existing_images)}) bilder i {target_dir} (Stor mapp)...")
                for img in existing_images:
                    img.unlink()
            else:
                print(f"DRY-RUN: Skulle ha rensat ALLA {len(existing_images)} bilder i {target_dir}")
        else:
            if small_images:
                if not dry_run:
                    print(f"🧹 Rensar {len(small_images)} småbilder (<=100KB) i {target_dir} (Behåller {len(existing_images) - len(small_images)} stora)...")
                    for img in small_images:
                        img.unlink()
                else:
                    print(f"DRY-RUN: Skulle ha rensat {len(small_images)} småbilder i {target_dir}")
            else:
                print(f"ℹ️ Behåller alla {len(existing_images)} bilder i {target_dir} (Liten mapp, inga småfiler).")
    else:
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"📂 Skapar ny mapp: {target_dir}")
        else:
            print(f"DRY-RUN: Skulle skapat mapp {target_dir}")

    # 3. Konvertera och flytta de nya bilderna
    files = sorted([f for f in source_folder.iterdir() if f.is_file()])
    for f in files:
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
            
        target_file = target_dir / f"{f.stem}.jpg"
        
        if not dry_run:
            try:
                if f.suffix.lower() == ".jpg":
                    shutil.move(str(f), str(target_file))
                else:
                    # Konvertera till JPG
                    with Image.open(f) as img:
                        rgb_img = img.convert("RGB")
                        rgb_img.save(target_file, "JPEG", quality=95)
                    f.unlink() # Ta bort original
                # print(f"  ➡️ {f.name} -> {target_file.name}")
            except Exception as e:
                print(f"  ❌ Fel vid hantering av {f.name}: {e}")
        else:
            action = "Flytta" if f.suffix.lower() == ".jpg" else "Konvertera"
            print(f"DRY-RUN: {action} {f.name} -> {target_file.name}")

    # 4. Ta bort tom källmapp
    if not dry_run:
        try:
            # Kolla om den är tom (förutom eventuella dolda filer)
            if not any(source_folder.iterdir()):
                source_folder.rmdir()
                print(f"✅ Klar med {primary_name}, källmapp borttagen.")
            else:
                print(f"⚠️ Källmapp {source_folder} inte tom, sparar den.")
        except Exception as e:
            print(f"⚠️ Kunde inte ta bort {source_folder}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Synka tvättade modeller från Innie till pBook")
    parser.add_argument("--confirm", action="store_true", help="Genomför faktiska ändringar (utan denna körs dry-run)")
    parser.add_argument("--skip-analysis", action="store_true", help="Använd befintlig rapport istället för att köra ny analys")
    args = parser.parse_args()

    dry_run = not args.confirm
    if dry_run:
        print("🚀 KÖR I DRY-RUN LÄGE (inga filer ändras)")
        print("Använd --confirm för att faktiskt flytta filer.\n")

    # Ladda mappningar
    merge_map = load_merge_map(MERGE_FILE)
    
    # Kör analys
    flagged_names = set()
    if not args.skip_analysis:
        flagged_names = run_analysis()
    else:
        print("⏭️ Hoppar över analys, läser befintlig rapport...")
        # (Logik för att läsa befintlig rapport om den finns)

    # Skanna Innie
    if not INNE_DIR.exists():
        print(f"❌ Källmapp {INNE_DIR} finns inte!")
        return

    innie_folders = sorted([d for d in INNE_DIR.iterdir() if d.is_dir()])
    
    for folder in innie_folders:
        name = folder.name
        primary = merge_map.get(name, name)
        
        is_flagged = (name in flagged_names) or (primary in flagged_names)
        is_new = not (PBOOK_DIR / primary).exists()
        
        if is_flagged or is_new:
            reason = "FLAGGAD" if is_flagged else "NY"
            print(f"\n📦 Bearbetar {name} -> {primary} ({reason})")
            sync_person(folder, primary, dry_run)
        else:
            # print(f"skipping {name} (ej flaggad och finns redan)")
            pass

    if dry_run:
        print("\n✨ Dry-run klar. Ingen skada skedd.")
    else:
        print("\n✨ Synk klar!")


if __name__ == "__main__":
    main()
