#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArcFace-pipeline med progressbar & checkpoints.

- Läser data_root/person/*.jpg|png|...
- Detekterar ansikten med InsightFace (SCRFD), beräknar embeddings (ArcFace)
- Upsamplar små bilder innan detektion (valbart)
- Fallback: om ingen detektion -> direkt embedding på 112x112 (valbart)
- Tillämpar alias från merge.txt för labels
- Sparar embeddings + etiketter i embeddings.pkl
- Tränar KNN (cosine) och sparar modell i face_knn_arcface.pkl
- Checkpoints: processed.jsonl + embeddings.pkl
"""
import sys
import pickle
import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

import processed_db

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from insightface.app import FaceAnalysis

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    from rich.style import Style
    from rich.color import Color
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

import subprocess
import shutil

# Dynamisk sökväg till script-katalogen för modeller
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# ------------------ Default ------------------
DEFAULT_DATA_ROOT   = "/home/marqs/Bilder/pBook"
DEFAULT_WORKDIR     = "./arcface_work-ppic"
EMB_PKL             = "embeddings_ppic.pkl"
PROC_DB             = "processed.db"
MODEL_PKL           = "face_knn_arcface_ppic.pkl"

MIN_WIDTH           = 40
MIN_HEIGHT          = 40
UPSAMPLE_TARGET_MIN = 180   # minsta sida efter upsampling-försök

K_DEFAULT           = 3
MIN_PER_CLASS_DEF   = 2
MAX_ABS_YAW_DEFAULT = 35.0  # max absolut yaw-vinkel (grader) innan vi skippar bilden
MIN_DET_SCORE_DEFAULT = 0.25  # lägsta detektor-score vi accepterar
MIN_FOCUS_DEFAULT   = 150.0   # lägsta Laplacian-variance för att inte klassas som suddig
MODEL_3D_5POINTS = np.array(
    [
        (-30.0,  0.0, -30.0),  # left eye
        ( 30.0,  0.0, -30.0),  # right eye
        (  0.0,  0.0,   0.0),  # nose
        (-20.0, -35.0, -30.0), # left mouth
        ( 20.0, -35.0, -30.0), # right mouth
    ],
    dtype=np.float32,
)
# ---------------------------------------------

# load_aliases / append_processed ersatta av processed_db-modulen

# ---------------- Gender-CNN ------------------
GENDER_PROTO = str(MODELS_DIR / "deploy_gender.prototxt")
GENDER_MODEL = str(MODELS_DIR / "gender_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
FEMALE_THRESH = 0.0
# ---------------------------------------------

# -------------- Hjälpfunktioner --------------

def init_app(providers: List[str]) -> FaceAnalysis:
    app = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
        allowed_modules=['detection','recognition','genderage']
    )
    app.prepare(ctx_id=0)
    return app


# load_processed / append_processed ersatta av processed_db-modulen


def load_embeddings(emb_path: Path) -> Tuple[List[np.ndarray], List[str]]:
    if not emb_path.exists():
        return [], []
    with emb_path.open("rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"]


def save_embeddings(emb_path: Path, X, y):
    emb_path = Path(emb_path)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = emb_path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    tmp.rename(emb_path)


def iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for person_dir in root.iterdir():
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        for p in person_dir.iterdir():
            if p.suffix.lower() in exts:
                yield str(p), label


def load_image_rgb(path: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    return np.array(img)


def rgb_to_bgr(arr: np.ndarray) -> np.ndarray:
    return arr[:, :, ::-1]


def upsample_if_needed(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    m = min(h, w)
    if m >= UPSAMPLE_TARGET_MIN:
        return img_rgb
    scale = UPSAMPLE_TARGET_MIN / m
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return np.array(Image.fromarray(img_rgb).resize((new_w, new_h), Image.BICUBIC))


def get_embedding_direct(rec_model, img_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Direkt embedding utan detektor (112x112 resize)."""
    if img_rgb is None:
        return None
    img_112 = Image.fromarray(img_rgb).resize((112, 112), Image.BICUBIC)
    bgr = rgb_to_bgr(np.array(img_112)).astype(np.uint8)
    emb = rec_model.get(bgr)
    if emb is None or emb.size == 0:
        return None
    return emb.astype(np.float32)


def estimate_yaw_from_kps(kps: Optional[np.ndarray], width: int, height: int) -> Optional[float]:
    if kps is None or len(kps) < 5:
        return None
    focal = float(max(width, height))
    if focal <= 0:
        return None
    image_points = np.array(kps, dtype=np.float32)
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    camera_matrix = np.array(
        [
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    try:
        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_3D_5POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None
    if not success:
        return None
    rot_mat, _ = cv2.Rodrigues(rotation_vec)
    proj = np.hstack((rot_mat, translation_vec))
    try:
        *_unused, euler_angles = cv2.decomposeProjectionMatrix(proj)
    except cv2.error:
        return None
    # euler_angles är (3,1); välj elementet explicit för att undvika NumPy-scalar warning
    if euler_angles is not None and euler_angles.size >= 2:
        yaw = float(euler_angles[1, 0])
    else:
        yaw = None
    return yaw

def render_face_preview(face_crop_bgr, width=60, height=30):
    """Renderar en ANSI-preview av ansiktet med högre upplösning."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return "" if not HAS_RICH else Text("")
    
    # Försök chafa först om det finns
    if shutil.which("chafa"):
        try:
            # Skriv till temporär fil och kör chafa
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                cv2.imwrite(tmp.name, face_crop_bgr)
                res = subprocess.check_output(
                    ["chafa", "--size", f"{width}x{height}", "--symbols", "all", "--colors", "full", tmp.name],
                    text=True
                )
                return Text.from_ansi(res) if HAS_RICH else res
        except Exception:
            pass

    # Fallback: Enkel ANSI-block-renderare (gråskala/färg om möjligt)
    try:
        h, w = face_crop_bgr.shape[:2]
        small = cv2.resize(face_crop_bgr, (width, height), interpolation=cv2.INTER_AREA)
        
        if HAS_RICH:
            res = Text()
            for y in range(height):
                for x in range(width):
                    b, g, r = small[y, x]
                    # I terminaler ser ett space ut som ca 1:2 aspect ratio.
                    # Vi använder ett space per pixel för fallback.
                    res.append(" ", style=Style(bgcolor=Color.from_rgb(r, g, b)))
                if y < height - 1:
                    res.append("\n")
            return res
        else:
            output = []
            for y in range(height):
                line = ""
                for x in range(width):
                    b, g, r = small[y, x]
                    # ANSI truecolor escape
                    line += f"\033[48;2;{r};{g};{b}m "
                output.append(line + "\033[0m")
            return "\n".join(output)
    except Exception:
        return "[Preview Error]" if not HAS_RICH else Text("[Preview Error]", style="bold red")

# ----------- Embedding-funktion -----------

def compute_embedding(app: FaceAnalysis,
                      rec_model,
                      img_path: str,
                      allow_fallback: bool,
                      allow_upsample: bool,
                      max_abs_yaw: Optional[float],
                      min_det_score: Optional[float],
                      min_focus: Optional[float],
                      verbose: bool = False) -> tuple[Optional[np.ndarray], int, bool, str]:
    """Returnerar (embedding, antal detekterade ansikten, fallback användes, reason)."""
    def log(msg: str) -> None:
        if verbose:
            print(f"[VERBOSE] {img_path}: {msg}")

    img_rgb = load_image_rgb(img_path)
    if img_rgb is None:
        log("failed to load image")
        return None, 0, False, "load_failed"

    h, w = img_rgb.shape[:2]
    log(f"image size {w}x{h}")

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        log("image below minimum size")
        if allow_fallback:
            log("trying direct fallback because image is too small")
            emb = get_embedding_direct(rec_model, img_rgb)
            if emb is None:
                log("fallback produced no embedding")
                return None, 0, True, "fallback_failed"
            log("fallback embedding succeeded (small image)")
            return emb, 0, True, "ok"
        return None, 0, False, "too_small"

    img_for_det = img_rgb
    if allow_upsample:
        try:
            upsampled = upsample_if_needed(img_rgb)
            if upsampled is not img_rgb:
                log("upsampled image before detection")
            img_for_det = upsampled
        except Exception as exc:
            log(f"upsample failed: {exc}")
            return None, 0, False, "upsample_failed"

    bgr = rgb_to_bgr(img_for_det)
    faces = app.get(bgr)
    n_faces = len(faces)
    log(f"detector found {n_faces} face(s)")

    if n_faces == 0 and allow_fallback:
        log("no faces detected; trying direct fallback embedding")
        emb = get_embedding_direct(rec_model, img_rgb)
        if emb is None:
            log("fallback produced no embedding")
            return None, 0, True, "fallback_failed"
        log("fallback embedding succeeded (no detection)")
        return emb, 0, True, "ok"

    if n_faces == 0:
        log("no faces detected")
        return None, 0, False, "no_faces"

    if n_faces != 1:
        log(f"rejecting because {n_faces} faces were detected")
        return None, n_faces, False, "multiple_faces"

    face = faces[0]
    h_det, w_det = img_for_det.shape[:2]
    det_score = float(getattr(face, "det_score", 1.0))
    log(f"detector score={det_score:.3f}")
    if min_det_score is not None and det_score < min_det_score:
        log(f"rejecting due to det_score {det_score:.3f} < {min_det_score:.3f}")
        return None, 1, False, "low_det_score"

    if max_abs_yaw is not None:
        pose = getattr(face, "pose", None)
        yaw = None
        if pose is not None and len(pose) >= 2:
            yaw = float(pose[1])
            log(f"pose yaw={yaw:.1f}\u00b0")
        if yaw is None:
            kps = getattr(face, "kps", None)
            if kps is not None:
                yaw = estimate_yaw_from_kps(kps, w_det, h_det)
                if yaw is not None:
                    log(f"estimated yaw from landmarks={yaw:.1f}\u00b0")
                else:
                    log("failed to estimate yaw from landmarks")
            else:
                log("no landmarks available for yaw estimation")
        if yaw is None:
            log("pose information missing; cannot apply yaw filter")
        elif abs(yaw) > max_abs_yaw:
            log(f"rejecting due to |yaw|={abs(yaw):.1f}\u00b0 > {max_abs_yaw}\u00b0")
            return None, 1, False, "high_yaw"
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w_det), min(y2, h_det)
    if x2 <= x1 or y2 <= y1:
        log("invalid bounding box after clipping")
        return None, 1, False, "invalid_bbox"

    face_crop = bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        log("face crop is empty")
        return None, 1, False, "empty_crop"

    if min_focus is not None and min_focus > 0:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        log(f"focus variance={focus:.1f}")
        if focus < min_focus:
            log(f"rejecting due to focus variance {focus:.1f} < {min_focus}")
            return None, 1, False, "low_focus"

    try:
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
    except Exception as exc:
        log(f"failed to build gender blob: {exc}")
        return None, 1, False, "gender_blob_failed"

    gender_net.setInput(blob)
    preds = gender_net.forward()[0]
    female_prob = float(preds[1])
    log(f"gender female probability={female_prob:.3f}")
    if female_prob < FEMALE_THRESH:
        log(f"rejecting due to female probability < {FEMALE_THRESH}")
        return None, 1, False, "gender_filter"

    emb = face.embedding
    if emb is None or emb.size == 0:
        log("embedding empty despite single face")
        return None, 1, False, "empty_embedding"

    log("embedding computed successfully")
    return emb.astype(np.float32), 1, False, "ok"

# ----------------- Pipeline -----------------

def encode(args) -> None:
    data_root = Path(args.data_root)
    workdir = Path(args.workdir); workdir.mkdir(parents=True, exist_ok=True)
    emb_path = workdir/EMB_PKL
    db_path = workdir/PROC_DB
    X,y = load_embeddings(emb_path)
    conn = processed_db.open_db(db_path)
    alias_map = processed_db.get_alias_map(conn)
    processed = processed_db.load_processed_set(conn)
    all_imgs = [(path, alias_map.get(label, label)) for path, label in iter_images(data_root)]
    todo=[(p,lbl) for p,lbl in all_imgs if p not in processed]
    if not todo:
        print("✅ Inget att göra – allt är redan processat.")
        conn.close()
        return
    print(f"🔍 Att processa: {len(todo)} bilder (skippar {len(processed)})")
    app=init_app(["CPUExecutionProvider"]); rec_model=app.models["recognition"]
    
    if args.ui and HAS_RICH:
        run_encode_rich(todo, app, rec_model, conn, X, y, emb_path, args)
    else:
        run_encode_tqdm(todo, app, rec_model, conn, X, y, emb_path, args)


def run_encode_tqdm(todo, app, rec_model, conn, X, y, emb_path, args):
    try:
        pbar = tqdm(todo, unit="img")
        for path, label in pbar:
            pbar.set_description(f"👤 {label[:20]:<20}")
            ok = False
            reason = "unknown"
            try:
                emb, n, fb, reason = compute_embedding(
                    app, rec_model, path,
                    args.allow_fallback, args.allow_upsample,
                    args.max_yaw,
                    args.min_det_score,
                    args.min_focus,
                    verbose=args.verbose
                )
                ok = emb is not None
                if args.verbose:
                    status = "OK" if ok else "FAIL"
                    print(f"[VERBOSE] {path}: final status {status} (faces={n}, fallback={fb}, reason={reason})")
                if ok:
                    X.append(emb)
                    y.append(label)
            except Exception as e:
                reason="exception"
                print(f"❌ {path}: {e}",file=sys.stderr)
            processed_db.add_processed(conn, path, ok, reason)
            if ok and len(X)%args.flush_every==0:
                save_embeddings(emb_path,X,y)
    except KeyboardInterrupt:
        print("\n⏹️ Avbrutet. Sparar state...")
    finally:
        save_embeddings(emb_path,X,y)


def run_encode_rich(todo, app, rec_model, conn, X, y, emb_path, args):
    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[bold cyan]{task.speed:>4.1f} img/s"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        expand=True
    )
    
    task_id = progress.add_task("Processar bilder...", total=len(todo))
    last_results = []
    MAX_LOG = 8
    
    layout = Layout()
    layout.split_row(
        Layout(name="main", ratio=1),
        Layout(name="visual", ratio=1, visible=args.visual)
    )
    
    def get_log_table():
        table = Table(title="Senaste händelser", expand=True, box=None)
        table.add_column("Person", style="magenta", no_wrap=True)
        table.add_column("Fil", style="cyan", no_wrap=True, overflow="ellipsis", max_width=50)
        table.add_column("Status", style="bold", width=10)
        table.add_column("Anledning", style="dim")
        for res in last_results[-MAX_LOG:]:
            clr = "green" if res['ok'] else "red"
            st = "✅ OK" if res['ok'] else "❌ FAIL"
            table.add_row(res['label'], Path(res['path']).name, Text(st, style=clr), res['reason'])
        return table

    preview_content = ""

    with Live(layout, console=console, refresh_per_second=4, screen=False) as live:
        try:
            for path, label in todo:
                ok = False
                reason = "unknown"
                face_crop_for_visual = None
                
                # Om vi vill ha visual, måste vi få tag i bildrutan. 
                # Vi ändrar compute_embedding lite i framtiden eller gör det här.
                # För nu: kör som vanligt.
                
                try:
                    # En fuling för att få tag i bildrutan för visual mode
                    if args.visual:
                        img_rgb = load_image_rgb(path)
                        if img_rgb is not None:
                            # Enkel detektion för preview
                            bgr = rgb_to_bgr(img_rgb)
                            faces = app.get(bgr)
                            if faces:
                                f = faces[0]
                                x1, y1, x2, y2 = f.bbox.astype(int)
                                h_img, w_img = bgr.shape[:2]
                                x1, y1, x2, y2 = max(x1,0), max(y1,0), min(x2,w_img), min(y2,h_img)
                                face_crop_for_visual = bgr[y1:y2, x1:x2]
                                preview_content = render_face_preview(face_crop_for_visual)

                    emb, n, fb, reason = compute_embedding(
                        app, rec_model, path,
                        args.allow_fallback, args.allow_upsample,
                        args.max_yaw,
                        args.min_det_score,
                        args.min_focus,
                        verbose=False # Tvinga False i UI-mode
                    )
                    ok = emb is not None
                    if ok:
                        X.append(emb)
                        y.append(label)
                except Exception as e:
                    reason = f"error: {str(e)[:30]}"
                
                processed_db.add_processed(conn, path, ok, reason)
                last_results.append({'path': path, 'label': label, 'ok': ok, 'reason': reason})
                
                # Uppdatera UI
                progress.update(task_id, advance=1, description=f"Handlägger: {label} / [bold]{Path(path).name}[/bold]")
                
                main_panel = Panel.fit(
                    Layout(progress),
                    title="ArcFace Pipeline",
                    subtitle=f"Totalt: {len(X)} embeddings"
                )
                
                # Vi bygger om layouten varje steg
                log_table = get_log_table()
                
                # Skapa en layout-struktur för 'main'
                main_layout = Layout()
                main_layout.split_column(
                    Layout(Align.center(Panel(progress, title="Framsteg")), size=5),
                    Layout(Panel(log_table, title="Logg"))
                )
                
                layout["main"].update(main_layout)
                if args.visual:
                    layout["visual"].update(Panel(Align.center(preview_content), title="Face Preview"))
                
                if ok and len(X) % args.flush_every == 0:
                    save_embeddings(emb_path, X, y)
                    
        except KeyboardInterrupt:
            console.print("\n[bold yellow]⏹️ Avbrutet av användare. Sparar...[/bold yellow]")
        finally:
            save_embeddings(emb_path, X, y)
        conn.close()
        print(f"💾 Sparade embeddings ({len(X)}) till {emb_path}")


def train(args) -> None:
    """
    Tränar KNN på embeddings.
    args.embeddings kan vara absolut eller relativ stig till pkl-fil.
    """
    # Läs in embeddings-fil direkt från argumentet
    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        print(f"❌ Embeddings-filen hittades inte: {emb_path}"); sys.exit(1)

    # Ladda embeddings
    X_list, y_list = load_embeddings(emb_path)
    if not X_list:
        print("❌ Inga embeddings – kör encode först."); sys.exit(1)

    # Stacka och koda etiketter
    X = np.vstack(X_list)
    y = np.array(y_list)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Bestäm k baserat på minsta klassstorlek
    counts = {cls: np.sum(y == cls) for cls in np.unique(y)}
    k = 1 if min(counts.values()) < args.min_per_class else args.k

    # Träna KNN
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine")
    clf.fit(X, y_enc)

    # Spara modell
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump({"model": clf, "label_encoder": le}, f)

    print(f"✅ Modell sparad: {model_out} (k={k}, klasser={len(np.unique(y))})")


def main():
    global EMB_PKL
    ap=argparse.ArgumentParser(description="ArcFace-pipeline (encode/train, fallback & upsampling)")
    ap.add_argument("--data-root",default=DEFAULT_DATA_ROOT)
    ap.add_argument("--workdir",default=DEFAULT_WORKDIR)
    ap.add_argument("--embeddings",default=EMB_PKL)
    ap.add_argument("--model-out",default=MODEL_PKL)
    ap.add_argument("--mode",choices=["encode","train","both"],default="both")
    ap.add_argument("--flush-every",type=int,default=200)
    ap.add_argument("-k",type=int,default=K_DEFAULT)
    ap.add_argument("--min-per-class",type=int,default=MIN_PER_CLASS_DEF)
    ap.add_argument("--allow-fallback",action="store_true")
    ap.add_argument("--allow-upsample",action="store_true")
    ap.add_argument("--ui", action="store_true", help="Visa grafiskt TUI istället för tqdm")
    ap.add_argument("--visual", action="store_true", help="Visa bild-preview i terminalen (kräver --ui)")
    ap.add_argument(
        "--max-yaw",
        type=float,
        default=MAX_ABS_YAW_DEFAULT,
        help="Max absolut yaw-vinkel (grader) som accepteras; sätt <=0 för att stänga av filtret",
    )
    ap.add_argument(
        "--min-det-score",
        type=float,
        default=MIN_DET_SCORE_DEFAULT,
        help="Minsta SCRFD-detektionsscore som accepteras; sätt <=0 för att stänga av",
    )
    ap.add_argument(
        "--min-focus",
        type=float,
        default=MIN_FOCUS_DEFAULT,
        help="Minsta Laplacian-variance på face crop (suddighetsfilter); sätt <=0 för att stänga av",
    )
    ap.add_argument("--verbose", action="store_true")
    args=ap.parse_args();EMB_PKL=args.embeddings
    if args.max_yaw is not None and args.max_yaw <= 0:
        args.max_yaw = None
    if args.min_det_score is not None and args.min_det_score <= 0:
        args.min_det_score = None
    if args.min_focus is not None and args.min_focus <= 0:
        args.min_focus = None
    if args.mode in ("encode","both"): encode(args)
    if args.mode in ("train","both"): train(args)

if __name__=="__main__": main()
