import sys
import numpy as np
from pathlib import Path
from face_arc_pipeline import init_app, load_image_rgb, rgb_to_bgr, upsample_if_needed, estimate_pose_from_kps

def main():
    root = Path("/home/marqs/Bilder/pBook")
    if not root.exists():
        print("Kan inte hitta pBook!")
        return

    app = init_app(["CPUExecutionProvider"])
    
    images_checked = 0
    pitch_values = []
    yaw_values = []
    
    for person_dir in root.iterdir():
        if not person_dir.is_dir(): continue
        for img_path in person_dir.glob("*.jpg"):
            img_rgb = load_image_rgb(str(img_path))
            if img_rgb is None: continue
            
            bgr = rgb_to_bgr(upsample_if_needed(img_rgb))
            faces = app.get(bgr)
            
            if len(faces) == 1:
                face = faces[0]
                pitch = None
                yaw = None
                
                pose = getattr(face, "pose", None)
                if pose is not None and len(pose) >= 2:
                    pitch = float(pose[0])
                    yaw = float(pose[1])
                    mode = "pose"
                else:
                    kps = getattr(face, "kps", None)
                    if kps is not None:
                        h_det, w_det = bgr.shape[:2]
                        pitch, yaw = estimate_pose_from_kps(kps, w_det, h_det)
                        mode = "kps"
                
                if pitch is not None and yaw is not None:
                    pitch_values.append(pitch)
                    yaw_values.append(yaw)
                    images_checked += 1
                    
                    if images_checked % 10 == 0:
                        print(f"[{mode}] Pitch: {pitch:6.1f} | Yaw: {yaw:6.1f} | Abs Pitch: {abs(pitch):6.1f}")
                        
            if images_checked >= 100:
                break
        if images_checked >= 100:
            break

    if not pitch_values:
        print("Kunde inte extrahera vinklar från någon bild!")
        return

    print("\n--- STATISTIK (100 bilder) ---")
    print(f"Genomsnitt pitch:     {np.mean(pitch_values):.1f}")
    print(f"Genomsnitt abs pitch: {np.mean(np.abs(pitch_values)):.1f}")
    print(f"Max abs pitch:        {np.max(np.abs(pitch_values)):.1f}")
    print(f"Min abs pitch:        {np.min(np.abs(pitch_values)):.1f}")
    print(f"---")
    print(f"Genomsnitt abs yaw:   {np.mean(np.abs(yaw_values)):.1f}")
    print(f"Max abs yaw:          {np.max(np.abs(yaw_values)):.1f}")

if __name__ == "__main__":
    main()
