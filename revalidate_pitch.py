#!/usr/bin/env python3
import sys
import argparse
import pickle
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
from tqdm.auto import tqdm

import processed_db
from face_arc_pipeline import (
    init_app, 
    load_image_rgb, 
    rgb_to_bgr, 
    upsample_if_needed, 
    estimate_pose_from_kps,
    normalize_angle
)

STATE_FILE = Path("reval_pitch_state.json")

def get_people_to_check(conn, state_checked: set) -> dict[str, list[str]]:
    c = conn.cursor()
    # Hämta alla sökvägar som redan varit markerade som "ok"
    c.execute("SELECT path FROM processed WHERE ok = 1")
    rows = c.fetchall()
    
    person_paths = {}
    for (path,) in rows:
        person = Path(path).parent.name
        if person in state_checked:
            continue
        if person not in person_paths:
            person_paths[person] = []
        person_paths[person].append(path)
        
    return person_paths

def main():
    parser = argparse.ArgumentParser(description="Re-validerar gamla bilder baserat på pitch.")
    parser.add_argument("--batch-size", type=int, default=100, help="Antal personer att verifiera åt gången")
    parser.add_argument("--max-pitch", type=float, default=35.0, help="Max pitch-vinkel")
    args = parser.parse_args()

    db_path = Path("arcface_work-ppic/processed.db")
    emb_path = Path("arcface_work-ppic/embeddings_ppic.pkl")
    
    if not db_path.exists() or not emb_path.exists():
        print("❌ Databasen eller embeddings-filen saknas!")
        return

    conn = processed_db.open_db(db_path)

    state_checked = set()
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            state_checked = set(json.load(f))

    person_paths = get_people_to_check(conn, state_checked)
    
    if not person_paths:
        print("✅ Alla personer är redan kontrollerade!")
        conn.close()
        sys.exit(2)

    # Välj ut vår batch för körningen
    people_to_process = list(person_paths.keys())[:args.batch_size]
    print(f"🔍 Valde ut {len(people_to_process)} o-verifierade personer (av {len(person_paths)} återstående) för granskning (max {args.max_pitch} grader)...")

    # Starta InsightFace
    app = init_app(["CPUExecutionProvider"])
    
    failed_people = set()
    passed_people = set()

    pbar = tqdm(people_to_process, unit="person")
    for person in pbar:
        paths = person_paths[person]
        person_failed = False
        
        for path in paths:
            pbar.set_description(f"👤 {person[:20]:<20} | 🖼️ {Path(path).name[:25]:<25}")
            
            img_rgb = load_image_rgb(path)
            if img_rgb is None:
                continue
            
            # Använd samma logik som din vanliga pipeline för att få korrekta dimensioner
            img_for_det = upsample_if_needed(img_rgb)
            bgr = rgb_to_bgr(img_for_det)
            faces = app.get(bgr)
            
            if len(faces) == 1:
                face = faces[0]
                pitch = None
                
                pose = getattr(face, "pose", None)
                if pose is not None and len(pose) >= 2:
                    pitch = normalize_angle(float(pose[0]))
                else:
                    kps = getattr(face, "kps", None)
                    if kps is not None:
                        h_det, w_det = bgr.shape[:2]
                        pitch, yaw = estimate_pose_from_kps(kps, w_det, h_det)
                
                if pitch is not None and abs(pitch) > args.max_pitch:
                    tqdm.write(f"❌ {person} raderas: Bilden '{Path(path).name}' hade pitch {pitch:.1f}°")
                    person_failed = True
                    break # Om EN bild fallerar måste vi ändå bygga om personen. Inte lönt att kolla resten.
        
        if person_failed:
            failed_people.add(person)
        else:
            passed_people.add(person)

    # Uppdatera vilka vi behandlat
    state_checked.update(passed_people)
    state_checked.update(failed_people)

    if failed_people:
        print(f"\n❌ Hittade {len(failed_people)} personer med för hög pitch! Kastar ut dem...")

        # 1. Rensa från databasen så att face_arc_pipeline tvingas bygga om dem (och då sorterar bort dåliga bilder)
        removed_db_count = processed_db.remove_by_persons(conn, failed_people)
        print(f"🗑️ Raderade {removed_db_count} rader från cachen.")

        # 2. Ta bort deras data från embeddings-filen
        with open(emb_path, "rb") as f:
            data = pickle.load(f)
        X = data["X"]
        y = data["y"]
        
        alias_to_primary = processed_db.get_alias_map(conn)
        
        keep_idx = []
        for i, label in enumerate(y):
            primary = alias_to_primary.get(label, label)
            if label not in failed_people and primary not in failed_people:
                keep_idx.append(i)
                
        X_new = [X[i] for i in keep_idx]
        y_new = [y[i] for i in keep_idx]
        
        tmp = emb_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"X": X_new, "y": y_new}, f)
        tmp.rename(emb_path)

        print(f"🗑️ Raderade {len(y) - len(y_new)} berörda bilder från själva modellen!")
    else:
        print("\n✅ Inga dåliga bilder hittades i den här batchen.")

    with open(STATE_FILE, "w") as f:
        json.dump(list(state_checked), f)
        
    print(f"💾 Förlopp sparat ({len(state_checked)} personer klara totalt).\nKör samma kommando igen för att ta nästa batch!")
    conn.close()

if __name__ == "__main__":
    main()
