#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple

from PIL import Image

# Standardinställningar
# Standardinställningar
DEFAULT_SOURCE_DIR = Path("/home/marqs/Bilder/Innie")
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


def parse_report(report_file: str) -> Tuple[Set[str], Set[str]]:
    """Läs rapporten och returnera (flagged_names, wipe_candidates)."""
    flagged = set()
    wipe_candidates = set()
    
    def should_wipe(issue_text: str) -> bool:
        text = issue_text.lower()
        return any(k in text for k in ["varians", "outlier", "misslyckade"])
        
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                p_a = row.get("Person A", "")
                p_b = row.get("Person B", "")
                i_a = row.get("Issue A", "")
                i_b = row.get("Issue B", "")
                if p_a:
                    flagged.add(p_a)
                    if should_wipe(i_a): wipe_candidates.add(p_a)
                if p_b:
                    flagged.add(p_b)
                    if should_wipe(i_b): wipe_candidates.add(p_b)
        print(f"✅ Analys-rapport inläst. {len(flagged)} flaggade (varav {len(wipe_candidates)} rensas helt).")
    else:
        print(f"⚠️ Rapportfil {report_file} saknas.")
    return flagged, wipe_candidates


def run_analysis(report_file: str) -> None:
    """Kör model_uncertainty.py och generera rapport."""
    cmd = [
        "python3", str(UNCERTAINTY_SCRIPT),
        "--db", str(DB_PATH),
        "--embeddings", str(EMB_PATH),
        "--output", report_file,
        "--top", "1000",
        "--exclusions", str(UNCERTAINTY_SCRIPT.parent / "similar_exclusions.txt"),
        "--ignore", str(UNCERTAINTY_SCRIPT.parent / "uncertainty_exceptions.txt")
    ]
    
    print(f"🔍 Kör analys: {' '.join(cmd)}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysen misslyckades kapitalt: {e}")
        print("⚠️ Avbryter synk för att inte missa flaggade mappar pga miljöfel.")
        exit(1)


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


def get_image_count(directory: Path) -> int:
    """Räkna antal bilder i en mapp."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


def sync_person(source_folder: Path, primary_name: str, dry_run: bool, force_wipe: bool = False):
    """Hantera flytt, rensning och konvertering för en person."""
    target_dir = PBOOK_DIR / primary_name
    
    # 1. Analysera befintliga bilder
    existing_images = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS] if target_dir.exists() else []
    small_images = [f for f in existing_images if f.stat().st_size <= 100 * 1024] # 100 KB
    new_images = [f for f in source_folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

    # 2. Beslut om rensning och Snapshot
    if existing_images:
        create_btrfs_snapshot(target_dir, BACKUP_DIR, primary_name, dry_run)
        
        # FULL WIPE sker om force_wipe (blandade identiteter) är satt
        # ELLER endast om BÅDA mapparna är stora (>= 20 bilder).
        should_wipe_all = force_wipe or ((len(existing_images) >= 20) and (len(new_images) >= 20))
        
        if should_wipe_all:
            if not dry_run:
                print(f"🧹 Rensar ALLA ({len(existing_images)}) bilder i {target_dir} (Fullständig tvätt)...")
                for img in existing_images:
                    img.unlink()
            else:
                print(f"DRY-RUN: Skulle ha rensat ALLA {len(existing_images)} bilder i {target_dir}")
        else:
            # MERGE-läge: Vi vill att mappen ska växa.
            # Vi rensar ändå bort småfiler (skräp) för att höja kvaliteten.
            if small_images:
                if not dry_run:
                    print(f"🧹 Kvalitetsrens: Tar bort {len(small_images)} småbilder (<=100KB) i {target_dir}.")
                    for img in small_images:
                        img.unlink()
                else:
                    print(f"DRY-RUN: Skulle ha rensat {len(small_images)} småbilder i {target_dir}")
            
            print(f"ℹ️ Mergar in {len(new_images)} bilder från källmappen till {len(existing_images) - len(small_images)} befintliga stora bilder i pBook.")
    else:
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"📂 Skapar ny mapp: {target_dir}")
        else:
            print(f"DRY-RUN: Skulle skapat mapp {target_dir}")

    # 3. Konvertera och flytta de nya bilderna
    for f in sorted(new_images):
        target_file = target_dir / f"{f.stem}.jpg"
        
        if not dry_run:
            try:
                if f.suffix.lower() == ".jpg":
                    shutil.move(str(f), str(target_file))
                else:
                    # Konvertera till JPG (Higher Quality)
                    with Image.open(f) as img:
                        rgb_img = img.convert("RGB")
                        rgb_img.save(target_file, "JPEG", quality=95)
                    f.unlink()
            except Exception as e:
                print(f"  ❌ Fel vid hantering av {f.name}: {e}")
        else:
            action = "Flytta" if f.suffix.lower() == ".jpg" else "Konvertera"
            print(f"DRY-RUN: {action} {f.name} -> {target_file.name}")

    # 4. Ta bort tom källmapp
    if not dry_run:
        try:
            if not any(source_folder.iterdir()):
                source_folder.rmdir()
                print(f"✅ Klar med {primary_name}, källmapp borttagen.")
        except Exception as e:
            print(f"⚠️ Kunde inte ta bort {source_folder}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Synka tvättade modeller från Innie till pBook")
    parser.add_argument("source_dir", type=str, nargs="?", help="Källmapp (valfri positional)")
    parser.add_argument("--confirm", action="store_true", help="Genomför faktiska ändringar (utan denna körs dry-run)")
    parser.add_argument("--skip-analysis", action="store_true", help="Använd befintlig rapport istället för att köra ny analys")
    parser.add_argument("--source", type=str, help=f"Källmapp (standard: {DEFAULT_SOURCE_DIR})")
    args = parser.parse_args()

    # Prioritera positional argument om det finns, annars --source, annars default
    source_raw = args.source_dir or args.source or str(DEFAULT_SOURCE_DIR)
    source_dir = Path(source_raw)

    dry_run = not args.confirm
    if dry_run:
        print("🚀 KÖR I DRY-RUN LÄGE (inga filer ändras)")
        print("Använd --confirm för att faktiskt flytta filer.\n")

    merge_map = load_merge_map(MERGE_FILE)
    
    report_file = "sync_pipeline_report.csv"
    if not args.skip_analysis:
        run_analysis(report_file)
    else:
        print("⏭️ Hoppar över ny analys.")

    flagged_names, wipe_candidates = parse_report(report_file)

    if not source_dir.exists():
        print(f"❌ Källmapp {source_dir} finns inte!")
        return

    source_folders = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    
    for folder in source_folders:
        name = folder.name
        primary = merge_map.get(name, name)
        dest_path = PBOOK_DIR / primary
        
        # Filter-logik Version 3
        is_flagged = (name in flagged_names) or (primary in flagged_names)
        is_wipe = (name in wipe_candidates) or (primary in wipe_candidates)
        is_new = not dest_path.exists()
        
        # Kolla om mappen är liten (färre än 10 bilder)
        pbook_count = get_image_count(dest_path)
        is_small = (not is_new) and (pbook_count < 10)
        
        if is_flagged or is_new or is_small:
            reason = []
            if is_wipe: reason.append("FULL WIPE")
            elif is_flagged: reason.append("FLAGGAD")
            if is_new: reason.append("NY")
            if is_small: reason.append(f"LITEN ({pbook_count} bilder)")
            
            print(f"\n📦 Bearbetar {name} -> {primary} ({', '.join(reason)})")
            sync_person(folder, primary, dry_run, force_wipe=is_wipe)

    if dry_run:
        print("\n✨ Dry-run klar. Ingen skada skedd.")
    else:
        print("\n✨ Synk klar!")

    if dry_run:
        print("\n✨ Dry-run klar. Ingen skada skedd.")
    else:
        print("\n✨ Synk klar!")


if __name__ == "__main__":
    main()
