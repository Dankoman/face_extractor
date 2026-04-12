import sys
import cv2
import numpy as np
from face_arc_pipeline import init_app, estimate_pose_from_kps

def main():
    app = init_app(["CPUExecutionProvider"])
    
    # Ladda en godtycklig bild med cv2 istället
    path = "/home/marqs/Bilder/pBook/Aaliyah Envy/Aaliyah Envy_001.jpg"
    bgr = cv2.imread(path)
    if bgr is None:
        # Hitta nån bild
        import glob
        files = glob.glob("/home/marqs/Bilder/pBook/*/*.jpg")
        if not files:
            print("Hittade inga")
            return
        bgr = cv2.imread(files[0])
        path = files[0]

    faces = app.get(bgr)
    if len(faces) >= 1:
        face = faces[0]
        
        # Test pose via face.pose
        pose = getattr(face, "pose", None)
        if pose is not None and len(pose) >= 2:
            print("Via face.pose:")
            print(f"  Pitch: {pose[0]}, Yaw: {pose[1]}")
            
        kps = getattr(face, "kps", None)
        if kps is not None:
            h, w = bgr.shape[:2]
            pitch, yaw = estimate_pose_from_kps(kps, w, h)
            print("Via landmarks:")
            print(f"  Pitch: {pitch}, Yaw: {yaw}")

if __name__ == "__main__":
    main()
