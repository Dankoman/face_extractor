#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Extractor – API + web UI + bildproxy (samlad i EN fil)
-----------------------------------------------------------
- /recognize      : topp‑K ansiktsigenkänning. Tar råbytes (clipboard/drag & drop) eller multipart 'image'.
- /web            : enkelt test-UI (klistra in, dra‑och‑släpp, välj fil). Ritar bbox och listar topp‑3.
- /resolve_image  : hämtar profilbilder via backend (lokal Stash → StashDB) **utan cache**, timeout och retry.
                    Standard: 302‑redirect till bildens URL (passar direkt som <img src>).
                    Lägg ?format=json för JSON, eller ?format=bytes för att returnera själva bildbytes (CSP‑vänligt).
- /api/health, /api/config, index → /web

Miljövariabler:
  STASH_URL            (ex: http://192.168.0.50:9999)
  STASH_API_KEY        (lokal Stash API‑nyckel, om din Stash kräver auth)
  STASHDB_ENDPOINT     (default: https://stashdb.org/graphql)
  STASHDB_API_KEY      (din StashDB‑nyckel – skickas i HTTP‑headern ApiKey)

Kräver: Flask, flask_cors, requests, numpy, Pillow, OpenCV, insightface (FaceAnalysis), din test_model.
"""
from __future__ import annotations

import os
import re
import time
import json
import pickle
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, Any, Iterable

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template_string, redirect, Response
from flask_cors import CORS

# === Importer från din kodbas ===
from test_model import FaceAnalysis, detect_once, cosine_similarity

# -------------------------------------------------
# Flask-app
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# Modell- och runtimeinit
# -------------------------------------------------
MODEL_PATH = "arcface_work-ppic/face_knn_arcface_ppic.pkl"
app_insight = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # justera providers om du har GPU
app_insight.prepare(ctx_id=0)

_model_lock = Lock()
_model_mtime: float = 0.0
clf = None
le = None


def _maybe_reload_model():
    """Ladda om modellen om pkl-filen har ändrats på disk."""
    global clf, le, _model_mtime
    try:
        current_mtime = os.path.getmtime(MODEL_PATH)
    except OSError:
        return
    if current_mtime == _model_mtime:
        return
    with _model_lock:
        # Dubbelkolla efter att vi tagit låset
        if current_mtime == _model_mtime:
            return
        print(f"[model] Loading model from {MODEL_PATH} (mtime {current_mtime})")
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        clf = bundle["model"]
        le = bundle["label_encoder"]
        _model_mtime = current_mtime


# Ladda modellen direkt vid uppstart
_maybe_reload_model()

THRESHOLD = 0.2  # legacy

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
GENDER_PROTO_PATH = MODELS_DIR / "deploy_gender.prototxt"
GENDER_MODEL_PATH = MODELS_DIR / "gender_net.caffemodel"
FEMALE_THRESHOLD = float(os.environ.get("FACE_EXTRACTOR_FEMALE_THRESHOLD", "0.7"))
_gender_net: Any | None = None
_gender_filter_available: bool | None = None
_gender_lock = Lock()


# -------------------------------------------------
# Proxy‑konfiguration (env)
# -------------------------------------------------
STASH_URL = os.environ.get("STASH_URL", "http://127.0.0.1:9999").rstrip("/")
STASH_GRAPHQL = f"{STASH_URL}/graphql"
STASH_API_KEY = os.environ.get("STASH_API_KEY", "")

STASHDB_ENDPOINT_DEFAULT = os.environ.get("STASHDB_ENDPOINT", "https://stashdb.org/graphql")
STASHDB_API_KEY = os.environ.get("STASHDB_API_KEY", "")

DEBUG_IMAGES = True

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "Content-Type": "application/json"})


def _post_graphql(url: str, query: str, variables: dict, headers: dict | None = None,
                  timeout: float = 7.0, retries: int = 1) -> Optional[dict]:
    """Liten wrapper som returnerar JSON eller None; loggar GraphQL errors."""
    hdrs = dict(SESSION.headers)
    if headers:
        hdrs.update(headers)
    payload = {"query": query, "variables": variables}
    for attempt in range(retries + 1):
        try:
            r = SESSION.post(url, json=payload, headers=hdrs, timeout=timeout)
            # serverfel: prova om
            if r.status_code >= 500 and attempt < retries:
                time.sleep(0.4 * (attempt + 1))
                continue
            data = r.json()
            # GraphQL-level errors (HTTP 200 men "errors" i kroppen)
            if isinstance(data, dict) and data.get("errors"):
                if DEBUG_IMAGES:
                    print(f"[gql][{url}] errors: {str(data['errors'])[:200]}")
                return None
            return data
        except Exception as e:
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
                continue
            if DEBUG_IMAGES:
                print(f"[gql][{url}] exception: {e}")
            return None


def _mask(token: str | None) -> str:
    if not token:
        return "<none>"
    return token[:6] + "..." + token[-4:]

# -------------------------------------------------
# Igenkänning (hjälpare)
# -------------------------------------------------

def _faces_to_boxes(faces):
    boxes = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        boxes.append({"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)})
    return boxes


def _ensure_gender_net() -> Any | None:
    global _gender_net, _gender_filter_available
    if _gender_net is not None:
        return _gender_net
    if _gender_filter_available is False:
        return None
    if not GENDER_PROTO_PATH.exists() or not GENDER_MODEL_PATH.exists():
        print(f"[gender] Missing model files: {GENDER_PROTO_PATH} / {GENDER_MODEL_PATH}")
        _gender_filter_available = False
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(str(GENDER_PROTO_PATH), str(GENDER_MODEL_PATH))
    except Exception as exc:
        print(f"[gender] Failed to load gender net: {exc}")
        _gender_filter_available = False
        return None
    _gender_net = net
    _gender_filter_available = True
    return _gender_net


def _female_probability(face, img_bgr: np.ndarray) -> tuple[Optional[float], bool]:
    net = _ensure_gender_net()
    if net is None:
        return None, False
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None, True
    face_crop = img_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, True
    try:
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
    except Exception:
        return None, True
    with _gender_lock:
        try:
            net.setInput(blob)
            preds = net.forward()[0]
        except Exception:
            return None, True
    try:
        female_prob = float(preds[1]) if len(preds) > 1 else float(preds[0])
    except Exception:
        return None, True
    return female_prob, True


def _predict_top_k(emb: np.ndarray, k: int) -> list[dict]:
    n_neighbors = max(1, int(k))
    neigh_idx = clf.kneighbors([emb], n_neighbors=n_neighbors, return_distance=False)[0]
    candidates, seen = [], set()
    for idx in neigh_idx:
        proto = clf._fit_X[idx]
        sim = float(cosine_similarity([emb], [proto])[0, 0])
        if hasattr(clf, "_y"):
            cls_idx = clf._y[idx]
            name = str(le.inverse_transform([cls_idx])[0])
        else:
            name = str(le.inverse_transform([idx])[0])
        if name not in seen:
            candidates.append({"name": name, "score": sim})
            seen.add(name)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

# -------------------------------------------------
# GraphQL helpers + introspektion
# -------------------------------------------------

def _gql_post_raw(ep: str, headers: dict, query: str, variables: dict | None = None, timeout: float = 10):
    r = SESSION.post(ep, json={"query": query, "variables": variables or {}}, headers=headers, timeout=timeout)
    ok = True
    try:
        r.raise_for_status()
    except Exception:
        ok = False
    return ok, r


def _gql_post_json(ep: str, headers: dict, query: str, variables: dict | None = None, timeout: float = 10):
    ok, r = _gql_post_raw(ep, headers, query, variables, timeout)
    if not ok:
        if DEBUG_IMAGES:
            print(f"[stashdb] HTTP {r.status_code}: {r.text[:400]}")
        r.raise_for_status()
    return r.json()

_schema_caps_cache: Optional[dict] = None


def _introspect_stashdb_schema(endpoint: str, api_key: str) -> dict:
    headers = {"ApiKey": api_key}
    caps = {
        "has_queryPerformers": False,
        "qp_accepts_input": False,
        "input_fields": {},  # name->type
        "performer_fields": [],
    }

    # 1) Query‑fält
    q_fields = """
    query __Q { __schema { queryType { fields { name args { name type { kind name ofType { kind name } } } } } } }
    """
    data = _gql_post_json(endpoint, headers, q_fields)
    qtype = (((data or {}).get('data') or {}).get('__schema') or {}).get('queryType') or {}
    fields = qtype.get('fields') or []
    for f in fields:
        if f.get('name') == 'queryPerformers':
            caps['has_queryPerformers'] = True
            for a in (f.get('args') or []):
                if a.get('name') == 'input':
                    caps['qp_accepts_input'] = True

    # 2) Input‑typ
    if caps['qp_accepts_input']:
        q_input = """
        query __T { __type(name:"PerformerQueryInput"){ inputFields { name type { kind name ofType { kind name ofType { kind name } } } } } }
        """
        d2 = _gql_post_json(endpoint, headers, q_input)
        it = (((d2 or {}).get('data') or {}).get('__type') or {})
        for fld in (it.get('inputFields') or []):
            # spara name -> enkel typrepr
            t = fld.get('type') or {}
            def tname(tn):
                if not tn: return None
                if tn.get('name'): return tn['name']
                return tname(tn.get('ofType'))
            caps['input_fields'][fld.get('name')] = tname(t)

    # 3) Hämta performer-fält för att kunna forma svaren
    q_perf = """
    query __P { __type(name:"Performer"){ fields { name } } }
    """
    try:
        d3 = _gql_post_json(endpoint, headers, q_perf)
        perf_type = (((d3 or {}).get('data') or {}).get('__type') or {})
        fields = perf_type.get('fields') or []
        caps['performer_fields'] = [f.get('name') for f in fields if f.get('name')]
    except Exception as exc:
        if DEBUG_IMAGES:
            print(f"[stashdb] performer field introspection failed: {exc}")


    if DEBUG_IMAGES:
        print("[stashdb] schema caps:", json.dumps(caps))
    return caps


def _extract_first_image(perf: Dict[str, Any]) -> Optional[str]:
    for im in (perf.get('images') or []):
        u = im.get('url')
        if u and isinstance(u, str) and u.startswith('http'):
            return u
    ip = perf.get('image_path')
    if ip and isinstance(ip, str) and ip.startswith('http'):
        return ip
    return None


def _select_exact(perfs: list[dict], name: str) -> Optional[dict]:
    """Case‑insensitiv exaktmatch mot name eller aliases.
    Tar även höjd för att inparametern kan vara citerad ("Namn").
    """
    lname = name.strip().strip('"').lower()
    for p in perfs:
        if (p.get('name') or '').strip().lower() == lname:
            return p
        aliases = [(a or '').strip().lower() for a in (p.get('aliases') or [])]
        if lname in aliases:
            return p
    return None


# -------------------------------------------------
# Bildproxy lookups – **strikt exaktmatch**, ingen cache
# -------------------------------------------------

def _lookup_local_stash_image(name: str, alias_sink: Optional[list[str]] = None) -> Optional[str]:
    """Sök i lokal Stash med strikt exaktmatch (case‑insensitivt) på name/alias.
    Ingen fallback till CONTAINS. Fyller optional alias_sink med matchande namn/alias."""
    local_headers = {"ApiKey": STASH_API_KEY} if STASH_API_KEY else None

    gql_equals = """
    query FindPerformerImageEq($name:String!){
      findPerformers(
        performer_filter:{ OR:{
          name:{value:$name,modifier:EQUALS},
          aliases:{value:$name,modifier:EQUALS}
        }}
        filter:{ per_page: 25 }
      ){
        performers{ id name aliases image_path images{ url } }
      }
    }
    """
    data = _post_graphql(STASH_GRAPHQL, gql_equals, {"name": name}, headers=local_headers, timeout=5.0, retries=1)
    perfs = (data or {}).get("data", {}).get("findPerformers", {}).get("performers") or []

    # Strikt urval
    perf = _select_exact(perfs, name)
    if not perf:
        if DEBUG_IMAGES:
            print(f"[resolve_image][local] strict miss: {name}")
        return None

    if alias_sink is not None:
        seen = { (a or '').strip().lower() for a in alias_sink if isinstance(a, str) }
        def _push(value: str | None):
            if not value:
                return
            val = value.strip()
            if not val:
                return
            key = val.lower()
            if key in seen:
                return
            seen.add(key)
            alias_sink.append(val)
        _push(perf.get("name"))
        for alias in perf.get("aliases") or []:
            _push(alias)

    images = perf.get("images") or []
    url = (images[0].get("url") if images else None) or perf.get("image_path")
    url = url if (url or '').startswith('http') else f"{STASH_URL}{url}" if url else None
    if DEBUG_IMAGES:
        print(f"[resolve_image][local] strict hit: {name} -> {url}")
    return url



def _stashdb_search_exact(
    name: str,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_terms: Optional[Iterable[str]] = None,
    requested_fields: Optional[Iterable[str]] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Strict lookup against StashDB that returns the performer dict plus the matched term."""
    global _schema_caps_cache
    api_key = api_key or STASHDB_API_KEY
    if not api_key:
        if DEBUG_IMAGES:
            print(f"[stashdb] no API key set for {name}")
        return None, None

    ep = (endpoint or STASHDB_ENDPOINT_DEFAULT).strip()
    headers = {"ApiKey": api_key}

    if _schema_caps_cache is None:
        try:
            _schema_caps_cache = _introspect_stashdb_schema(ep, api_key)
        except Exception as e:
            if DEBUG_IMAGES:
                print(f"[stashdb] introspection failed: {e}")
            _schema_caps_cache = {
                "has_queryPerformers": True,
                "qp_accepts_input": True,
                "input_fields": {"name": "String", "alias": "String", "names": "String", "per_page": "Int", "page": "Int"},
                "performer_fields": ["id", "name", "aliases", "images", "urls"],
            }

    caps = _schema_caps_cache or {}
    if not (caps.get("has_queryPerformers") and caps.get("qp_accepts_input")):
        return None, None

    inp_fields: dict = caps.get("input_fields", {})
    performer_fields: list[str] = list(caps.get("performer_fields") or [])

    default_wanted = [
        "disambiguation",
        "gender",
        "birthdate",
        "death_date",
        "deathdate",
        "ethnicity",
        "country",
        "eye_color",
        "hair_color",
        "height",
        "height_cm",
        "weight",
        "measurements",
        "instagram",
        "twitter",
        "tiktok",
        "urls",
        "images",
        "stash_ids",
    ]
    if requested_fields:
        for fld in requested_fields:
            if fld not in default_wanted:
                default_wanted.append(fld)

    nested_templates = {
        "urls": "urls { url }",
        "images": "images { url }",
        "stash_ids": "stash_ids { site stash_id endpoint }",
    }

    selection_parts: list[str] = []
    for fld in ("id", "name", "aliases"):
        if fld in performer_fields or fld in ("id", "name"):
            if fld not in selection_parts:
                selection_parts.append(fld)
    for fld in default_wanted:
        if fld in nested_templates:
            if fld in performer_fields and nested_templates[fld] not in selection_parts:
                selection_parts.append(nested_templates[fld])
        elif fld in performer_fields and fld not in selection_parts:
            selection_parts.append(fld)

    if not selection_parts:
        selection_parts = ["id", "name", "aliases"]

    selection = " ".join(selection_parts)

    def _post(q: str, vars: dict) -> Optional[dict]:
        ok, r = _gql_post_raw(ep, headers, q, vars, timeout=10)
        if not ok:
            if DEBUG_IMAGES:
                print(f"[stashdb] HTTP {r.status_code}: {r.text[:300]}")
            return None
        try:
            data = r.json()
            if isinstance(data, dict) and data.get("errors"):
                if DEBUG_IMAGES:
                    print(f"[stashdb] GQL errors: {str(data['errors'])[:200]}")
                return None
            return data
        except Exception:
            if DEBUG_IMAGES:
                print(f"[stashdb] non-json body: {r.text[:120]}")
            return None

    q_tpl = f"""
    query($inp:PerformerQueryInput!){{
      queryPerformers(input:$inp){{ count performers{{ {selection} }} }}
    }}
    """

    def _query_term(term: str, allowed_fields: Iterable[str]) -> tuple[Optional[dict], Optional[str]]:
        if not term:
            return None, None
        term = term.strip()
        if not term:
            return None, None
        patterns = [f'"{term}"', term]
        for pattern in patterns:
            candidates: list[dict] = []
            for field in allowed_fields:
                if field not in inp_fields:
                    continue
                entry = {field: pattern}
                if "per_page" in inp_fields:
                    entry["per_page"] = 40
                candidates.append(entry)
            if not candidates:
                continue
            for inp in candidates:
                data = _post(q_tpl, {"inp": inp})
                perfs = (((data or {}).get("data") or {}).get("queryPerformers") or {}).get("performers") or []
                if not perfs:
                    continue
                match = _select_exact(perfs, term)
                if match:
                    return match, term
        return None, None

    performer, matched = _query_term(name, ("name", "names"))
    if performer:
        return performer, matched

    alias_terms: list[str] = []
    seen_alias: set[str] = set()

    def _push_alias(value: str | None):
        if not value:
            return
        val = value.strip()
        if not val:
            return
        key = val.lower()
        if key in seen_alias:
            return
        seen_alias.add(key)
        alias_terms.append(val)

    _push_alias(name)
    if extra_terms:
        for alias in extra_terms:
            _push_alias(alias)

    for alias_term in alias_terms:
        if alias_term.lower() == name.strip().lower():
            continue
        performer, matched = _query_term(alias_term, ("alias", "names"))
        if performer:
            return performer, matched

    return None, None


def _lookup_stashdb_image(
    name: str,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_terms: Optional[Iterable[str]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Reuse strict lookup to extract the first image URL."""
    alias_terms: list[str] = []
    seen_alias: set[str] = set()

    def _collect(term: str | None):
        if not term:
            return
        val = term.strip()
        if not val:
            return
        key = val.lower()
        if key in seen_alias:
            return
        seen_alias.add(key)
        alias_terms.append(val)

    _collect(name)
    if extra_terms:
        for alias in extra_terms:
            _collect(alias)

    performer, matched = _stashdb_search_exact(
        name,
        endpoint=endpoint,
        api_key=api_key,
        extra_terms=[t for t in alias_terms if t.lower() != name.strip().lower()],
        requested_fields=("images", "image_path"),
    )
    if performer:
        url = _extract_first_image(performer)
        if url:
            return url, matched

    if DEBUG_IMAGES:
        alias_info = f"; alias tried: {', '.join(alias_terms)}" if alias_terms else ""
        print(f"[resolve_image][stashdb] strict miss: {name}{alias_info}")
    return None, matched


@app.get("/stashdb/performer")
def stashdb_performer_details():
    name = (request.args.get("name") or "").strip()
    if not name:
        return jsonify({"error": "missing name"}), 400

    stashdb_ep = (request.args.get("stashdb_endpoint") or STASHDB_ENDPOINT_DEFAULT).strip()
    if request.args.get("stashdb_api_key") and DEBUG_IMAGES:
        print("[stashdb] Ignoring client-supplied stashdb_api_key")

    stashdb_key = STASHDB_API_KEY.strip()
    if not stashdb_key:
        return jsonify({"error": "stashdb_api_key not configured"}), 503

    alias_terms = [a.strip() for a in request.args.getlist("alias") if isinstance(a, str) and a.strip()]

    performer, matched = _stashdb_search_exact(
        name,
        endpoint=stashdb_ep,
        api_key=stashdb_key,
        extra_terms=alias_terms,
        requested_fields=(
            "disambiguation",
            "aliases",
            "gender",
            "birthdate",
            "death_date",
            "deathdate",
            "ethnicity",
            "country",
            "eye_color",
            "hair_color",
            "height",
            "height_cm",
            "weight",
            "measurements",
            "urls",
            "images",
            "stash_ids",
            "instagram",
            "twitter",
            "tiktok",
        ),
    )
    if not performer:
        return jsonify({"performer": None, "matched": None}), 404

    def _clean_str(value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        if value is None:
            return None
        return str(value)

    def _clean_list(values):
        cleaned: list[str] = []
        for val in values or []:
            if isinstance(val, str):
                norm = val.strip()
                if norm:
                    cleaned.append(norm)
        return cleaned

    def _extract_urls(values):
        urls: list[str] = []
        for entry in values or []:
            if isinstance(entry, str):
                norm = entry.strip()
                if norm:
                    urls.append(norm)
            elif isinstance(entry, dict):
                norm = _clean_str(entry.get("url"))
                if norm:
                    urls.append(norm)
        uniq: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            uniq.append(url)
        return uniq

    stash_ids_raw = performer.get("stash_ids") or []
    stash_ids: list[dict[str, Optional[str]]] = []
    for entry in stash_ids_raw:
        if not isinstance(entry, dict):
            continue
        sid = _clean_str(entry.get("stash_id") or entry.get("id"))
        if not sid:
            continue
        stash_ids.append({
            "stash_id": sid,
            "endpoint": _clean_str(entry.get("endpoint") or entry.get("url") or stashdb_ep),
            "site": _clean_str(entry.get("site")),
        })
    if not stash_ids and performer.get("id"):
        stash_ids.append({
            "stash_id": _clean_str(performer.get("id")),
            "endpoint": stashdb_ep,
            "site": None,
        })

    payload = {
        "id": _clean_str(performer.get("id")),
        "name": _clean_str(performer.get("name")) or name,
        "disambiguation": _clean_str(performer.get("disambiguation")),
        "aliases": _clean_list(performer.get("aliases") or []),
        "gender": _clean_str(performer.get("gender")),
        "birthdate": _clean_str(performer.get("birthdate")),
        "death_date": _clean_str(performer.get("death_date") or performer.get("deathdate")),
        "ethnicity": _clean_str(performer.get("ethnicity")),
        "country": _clean_str(performer.get("country")),
        "eye_color": _clean_str(performer.get("eye_color")),
        "hair_color": _clean_str(performer.get("hair_color")),
        "height": performer.get("height_cm") or performer.get("height"),
        "weight": performer.get("weight"),
        "measurements": _clean_str(performer.get("measurements")),
        "urls": _extract_urls(performer.get("urls")),
        "stash_ids": stash_ids,
        "social": {
            "instagram": _clean_str(performer.get("instagram")),
            "twitter": _clean_str(performer.get("twitter")),
            "tiktok": _clean_str(performer.get("tiktok")),
        },
    }

    image_url = _extract_first_image(performer)

    return jsonify({
        "performer": payload,
        "matched": matched,
        "image_url": image_url,
        "source_endpoint": stashdb_ep,
    })


# -------------------------------------------------
# API – topp‑K och raw‑bytesstöd
# -------------------------------------------------
def _detect_all_angles_unique(app, img_rgb):
    H, W = img_rgb.shape[:2]
    all_faces = []
    
    # 0 deg
    f0 = detect_once(app, img_rgb)
    for face in f0:
        all_faces.append({"f": face, "bbox": face.bbox.astype(float)})
        
    # 90 CW: x_orig = y_rot, y_orig = H - x_rot
    img_90 = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
    f90 = detect_once(app, img_90)
    for face in f90:
        bx = face.bbox.astype(float)
        nx1, ny1, nx2, ny2 = bx[1], H - bx[2], bx[3], H - bx[0]
        # x_orig = y_rot, y_orig = H - x_rot
        all_faces.append({"f": face, "bbox": [nx1, ny1, nx2, ny2]})
        
    # 180: x_orig = W - x_rot, y_orig = H - y_rot
    img_180 = cv2.rotate(img_rgb, cv2.ROTATE_180)
    f180 = detect_once(app, img_180)
    for face in f180:
        bx = face.bbox.astype(float)
        nx1, ny1, nx2, ny2 = W - bx[2], H - bx[3], W - bx[0], H - bx[1]
        all_faces.append({"f": face, "bbox": [nx1, ny1, nx2, ny2]})
        
    # 270 CW (90 CCW): x_orig = W - y_rot, y_orig = x_rot
    img_270 = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    f270 = detect_once(app, img_270)
    for face in f270:
        bx = face.bbox.astype(float)
        nx1, ny1, nx2, ny2 = W - bx[3], bx[0], W - bx[1], bx[2]
        all_faces.append({"f": face, "bbox": [nx1, ny1, nx2, ny2]})

    def compute_iou(b1, b2):
        x1_inter = max(b1[0], b2[0])
        y1_inter = max(b1[1], b2[1])
        x2_inter = min(b1[2], b2[2])
        y2_inter = min(b1[3], b2[3])
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter_area / float(max(b1_area + b2_area - inter_area, 1e-6))

    unique_faces = []
    for item in all_faces:
        b1 = item["bbox"]
        is_dup = False
        for uf in unique_faces:
            if compute_iou(b1, uf["bbox"]) > 0.4:
                is_dup = True
                if item["f"].det_score > uf["f"].det_score:
                    uf["f"] = item["f"]
                    uf["bbox"] = b1
                break
        if not is_dup:
            unique_faces.append(item)
            
    res = []
    for uf in unique_faces:
        bx = uf["bbox"]
        # Säkra koordinaterna (min/max)
        rx1, ry1, rx2, ry2 = min(bx[0], bx[2]), min(bx[1], bx[3]), max(bx[0], bx[2]), max(bx[1], bx[3])
        uf["f"].bbox = np.array([rx1, ry1, rx2, ry2], dtype=np.float32)
        res.append(uf["f"])
        
    return res

@app.post("/recognize")
def recognize():
    try:
        _maybe_reload_model()
        top_k = int(request.args.get("top_k", 3))
        file = request.files.get("image") if "image" in request.files else None
        data = file.read() if file is not None else request.get_data()
        if not data:
            return jsonify({"error": "Ingen bild mottagen"}), 400

        img_arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Kunde inte läsa bilden"}), 400
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        all_angles = request.args.get("all_angles", "0") == "1"
        if all_angles:
            faces = _detect_all_angles_unique(app_insight, img_rgb)
        else:
            faces = detect_once(app_insight, img_rgb)
            
        if not faces:
            return jsonify([])

        raw_faces = request.args.get("raw_faces", "0") == "1"

        boxes = _faces_to_boxes(faces)
        result = []
        for face, box in zip(faces, boxes):
            female_prob, filter_active = _female_probability(face, img_bgr)
            if filter_active and (female_prob is None or female_prob < FEMALE_THRESHOLD) and not raw_faces:
                continue
            emb = getattr(face, "embedding", None)
            entry = {"box": box, "candidates": []}
            if female_prob is not None:
                entry["female_probability"] = female_prob
            if emb is None or getattr(emb, "size", 0) == 0:
                result.append(entry)
                continue
            candidates = _predict_top_k(emb, k=top_k)
            entry["candidates"] = candidates
            result.append(entry)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Serverfel: {e}"}), 500


# -------------------------------------------------
# Bildproxy – lokal Stash → StashDB **utan cache**
# -------------------------------------------------
@app.get("/resolve_image")
def resolve_image():
    name = (request.args.get("name") or "").strip()
    if not name:
        return jsonify({"url": None, "error": "missing name"}), 400

    source = (request.args.get("source") or "both").lower()  # local|stashdb|both
    fmt = (request.args.get("format") or "redirect").lower() # redirect|json|bytes

    stashdb_ep = (request.args.get("stashdb_endpoint") or STASHDB_ENDPOINT_DEFAULT).strip()
    #stashdb_key = (request.args.get("stashdb_api_key") or STASHDB_API_KEY).strip()
    stashdb_key = STASHDB_API_KEY.strip()
    if request.args.get("stashdb_api_key"):
        if DEBUG_IMAGES: print("[resolve_image] Ignoring client-supplied stashdb_api_key")

    # **Ingen cache** – alltid färsk uppslagning
    url: Optional[str] = None
    tried: list[str] = []
    stashdb_matched_term: Optional[str] = None

    alias_candidates: list[str] = []
    seen_alias: set[str] = set()

    def _collect_alias(value: str | None):
        if not value:
            return
        val = value.strip()
        if not val:
            return
        key = val.strip('"').lower()
        if key in seen_alias:
            return
        seen_alias.add(key)
        alias_candidates.append(val)

    def _collect_variants(raw: str):
        if not raw:
            return
        # dela upp på typiska alias-separatorer: "/", "|", "aka", kommatecken/semi-kolon
        parts = re.split(r'\s*(?:/|\||\baka\b|\bAKA\b|,|;)\s*', raw)
        for part in parts:
            cleaned = part.strip()
            if cleaned and cleaned.lower() != raw.strip().lower():
                _collect_alias(cleaned)

    _collect_variants(name)

    if source in ("local", "both"):
        tried.append("local")
        local_aliases: list[str] = []
        url = _lookup_local_stash_image(name, alias_sink=local_aliases)
        for alias in local_aliases:
            _collect_alias(alias)

    if not url and source in ("stashdb", "both"):
        tried.append("stashdb")
        stashdb_url, stashdb_matched_term = _lookup_stashdb_image(
            name,
            endpoint=stashdb_ep,
            api_key=stashdb_key,
            extra_terms=alias_candidates
        )
        url = stashdb_url

    alias_note = ''
    if stashdb_matched_term and stashdb_matched_term.strip().lower() != name.strip().lower():
        alias_note = f" via '{stashdb_matched_term}'"

    if DEBUG_IMAGES:
        print(f"[resolve_image] {name} -> {url} (strict{alias_note}; tried: {','.join(tried)})")

    if fmt == "json":
        return jsonify({"url": url})

    if fmt == "bytes":
        if not url:
            return Response(status=204)
        try:
            r = SESSION.get(url, timeout=10)
            r.raise_for_status()
        except Exception as e:
            return jsonify({"error": f"fetch failed: {e}"}), 502
        ctype = r.headers.get("Content-Type", "image/jpeg")
        return Response(r.content, status=200, headers={
            "Content-Type": ctype,
            # client‑side cache är OK, men ingen server‑side mellanlagring
            "Cache-Control": "public, max-age=86400"
        })

    # default: redirect
    if url:
        return redirect(url, code=302)
    return Response(status=204)


# -------------------------------------------------
# Enkel webbsida – topp 3, clipboard & drag/drop (demo)
# -------------------------------------------------
@app.get("/web")
def web_ui():
    html = """
<!doctype html>
<html lang=\"sv\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Face Extractor – Webbtest</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;margin:0;background:#0e0f12;color:#eaecef}
    header{padding:16px 20px;border-bottom:1px solid #23252b}
    main{display:grid;grid-template-columns:420px 1fr;gap:16px;padding:16px}
    .pane{background:#14161b;border:1px solid #23252b;border-radius:12px}
    .pane h2{margin:0;padding:12px 14px;border-bottom:1px solid #23252b;font-size:15px;color:#b6bbc7}
    .pane .content{padding:12px 14px}
    #drop{display:flex;align-items:center;justify-content:center;height:280px;border:2px dashed #3a3f4b;border-radius:12px;background:#0e1014;color:#9aa0ac;text-align:center}
    #drop.drag{background:#111726;border-color:#6b83ff;color:#cfd6ff}
    canvas{display:block;max-width:100%;background:#000;border-radius:10px;border:1px solid #23252b}
    .hint{color:#9aa0ac;font-size:12px;margin-top:8px}
    .actions{display:flex;gap:10px}
    button{background:#2a61ff;border:0;color:#fff;padding:8px 12px;border-radius:10px;cursor:pointer}
    button:disabled{opacity:.6;cursor:not-allowed}
    input[type=file]{display:none}
    label.btn{display:inline-block;background:#1d2027;border:1px solid #2b2f39;color:#cbd1dd;padding:8px 12px;border-radius:10px;cursor:pointer}
    .meta{color:#9aa0ac;font-size:12px}

    /* Canvas-wrapper med overlay */
    #canvas-wrapper{position:relative;display:inline-block;max-width:100%}

    /* Bounding-box overlay */
    .frp-overlay{position:absolute;inset:0;pointer-events:none;z-index:10;font-family:ui-sans-serif,system-ui}
    .frp-face-box{position:absolute;border:2px solid rgba(0,200,255,0.9);border-radius:6px;box-shadow:0 0 0 1px rgba(0,0,0,0.35),0 4px 14px rgba(0,0,0,0.4);pointer-events:auto;cursor:pointer;transition:border-color .15s}
    .frp-face-box:hover{border-color:rgba(0,200,255,1);z-index:100}

    /* Namn-dropdown bredvid boxen */
    .frp-suggestions{position:absolute;left:0;top:100%;margin-top:6px;min-width:220px;max-width:none;width:max-content;background:rgba(18,18,18,0.92);color:#f2f2f2;border:1px solid rgba(255,255,255,0.12);border-radius:10px;overflow:visible;backdrop-filter:blur(6px);pointer-events:auto;display:none}
    .frp-suggestions.visible{display:block}
    .frp-face-box.active{border-color:rgba(0,200,255,1);z-index:100}
    .frp-suggestion{display:flex;align-items:center;gap:10px;padding:8px 10px;line-height:1.25;border-bottom:1px solid rgba(255,255,255,0.06);width:max-content;overflow:visible;cursor:default}
    .frp-suggestion:last-child{border-bottom:none}
    .frp-suggestion span{font-size:14px;font-weight:600;color:#f7f7f7;text-shadow:0 1px 1px rgba(0,0,0,0.4)}
    .frp-suggestion .conf{font-size:12px;font-weight:400;color:#9aa0ac;margin-left:4px}
    .frp-suggestion.is-low{opacity:0.6}

    /* Flip dropdown ovanför boxen om den hamnar utanför */
    .frp-face-box[data-flip=\"top\"] .frp-suggestions{top:auto;bottom:100%;margin-top:0;margin-bottom:6px}

    /* Hover-preview tooltip */
    .frp-preview{position:fixed;left:0;top:0;border-radius:10px;border:1px solid rgba(255,255,255,0.12);background:#0f1115;box-shadow:0 8px 18px rgba(0,0,0,0.35);pointer-events:none;z-index:2147483647;overflow:visible;max-width:150px;max-height:90vh}
    .frp-preview img{display:block;width:100%;height:auto;object-fit:contain;max-width:150px;max-height:90vh;border-radius:10px}
  </style>
</head>
<body>
  <header>
    <strong>Face Extractor – Webbtest</strong>
    <span class=\"meta\">(klistra in bild med Ctrl+V, eller dra‑och‑släpp / välj fil)</span>
  </header>
  <main>
    <section class=\"pane\">
      <h2>Bildinmatning</h2>
      <div class=\"content\">
        <div id=\"drop\">Släpp bild här eller <label class=\"btn\" for=\"file\">välj fil</label><input type=\"file\" id=\"file\" accept=\"image/*\"></div>
        <div class=\"hint\">Tips: markera en bild i clipboard och tryck <b>Ctrl+V</b> här i fönstret.</div>
        <div class=\"actions\" style=\"margin-top:10px\">
          <button id=\"run\" disabled>Kör igenkänning (topp 3)</button>
        </div>
      </div>
    </section>
    <section class=\"pane\">
      <h2>Resultat</h2>
      <div class=\"content\">
        <div id=\"canvas-wrapper\">
          <canvas id=\"canvas\"></canvas>
          <div id=\"overlay\" class=\"frp-overlay\"></div>
        </div>
      </div>
    </section>
  </main>
  <script>
  const drop = document.getElementById('drop');
  const fileInput = document.getElementById('file');
  const runBtn = document.getElementById('run');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const overlay = document.getElementById('overlay');
  let imageBlob = null;
  let img = new Image();

  function setImageFromBlob(blob){
    imageBlob = blob;
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      canvas.width = img.width; canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      overlay.innerHTML = '';
      runBtn.disabled = false;
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }

  ['dragenter','dragover'].forEach(evt=>drop.addEventListener(evt, e=>{e.preventDefault(); drop.classList.add('drag');}))
  ;['dragleave','drop'].forEach(evt=>drop.addEventListener(evt, e=>{e.preventDefault(); drop.classList.remove('drag');}))
  drop.addEventListener('drop', e=>{
    const f = e.dataTransfer.files?.[0];
    if (f) setImageFromBlob(f);
  });

  fileInput.addEventListener('change', e=>{
    const f = fileInput.files?.[0];
    if (f) setImageFromBlob(f);
  });

  window.addEventListener('paste', e=>{
    const items = e.clipboardData?.items || [];
    for (const it of items){
      if (it.type.startsWith('image/')){
        const blob = it.getAsFile();
        if (blob){ setImageFromBlob(blob); break; }
      }
    }
  });

  /* ---- Hover-preview (som i stashpluginet) ---- */

  function makePreviewTooltip(){
    const tip = document.createElement('div');
    tip.className = 'frp-preview';
    const im = document.createElement('img');
    im.alt = 'preview';
    tip.appendChild(im);
    return { tip, img: im };
  }

  function attachHoverPreview(rowEl, name){
    let tipRef = null;
    let enterTimer = null;
    let ctrl = null;

    function placeTip(el, tip){
      const r = el.getBoundingClientRect();
      const tr = tip.getBoundingClientRect();
      const pad = 12;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      let x = r.right + pad;
      if (x + tr.width > vw - pad) x = Math.max(pad, r.left - tr.width - pad);
      let y = r.top + (r.height - tr.height) / 2;
      y = Math.max(pad, Math.min(vh - tr.height - pad, y));
      if (x < pad) x = pad;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    }

    function removeTip(){
      if (tipRef){ tipRef.remove(); tipRef = null; }
    }

    rowEl.addEventListener('mouseenter', () => {
      if (enterTimer) clearTimeout(enterTimer);
      enterTimer = setTimeout(async () => {
        if (tipRef) return;
        const ac = (typeof AbortController !== 'undefined') ? new AbortController() : null;
        ctrl = ac;
        let url;
        try {
          const resp = await fetch('/resolve_image?name=' + encodeURIComponent(name) + '&format=bytes', {
            signal: ac ? ac.signal : undefined
          });
          if (!resp.ok || resp.status === 204) return;
          const blob = await resp.blob();
          url = URL.createObjectURL(blob);
        } catch(err){
          if (err?.name !== 'AbortError') console.error('Preview-fetch misslyckades:', err);
          return;
        }
        if (ctrl !== ac) return;
        ctrl = null;
        if (!url || tipRef) return;

        const { tip, img: tImg } = makePreviewTooltip();
        tipRef = tip;
        tImg.onload = () => {
          if (tipRef !== tip) return;
          if (!tip.parentNode) document.body.appendChild(tip);
          placeTip(rowEl, tip);
        };
        tImg.onerror = () => { tip.remove(); if (tipRef === tip) tipRef = null; };
        if (!tip.parentNode) document.body.appendChild(tip);
        tImg.src = url;
      }, 150);
    });

    rowEl.addEventListener('mousemove', () => {
      if (tipRef) placeTip(rowEl, tipRef);
    });

    rowEl.addEventListener('mouseleave', () => {
      if (enterTimer){ clearTimeout(enterTimer); enterTimer = null; }
      if (ctrl){ try { ctrl.abort(); } catch(_){} ctrl = null; }
      removeTip();
    });
  }

  /* ---- Igenkänning ---- */

  async function recognize(){
    runBtn.disabled = true;
    overlay.innerHTML = '';
    const resp = await fetch('/recognize?top_k=3', { method: 'POST', body: imageBlob });
    if(!resp.ok){ overlay.textContent = 'Fel: ' + resp.status; runBtn.disabled = false; return; }
    const data = await resp.json();

    // Rita om bilden (rensa eventuella gamla linjer)
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 3;

    // Beräkna skalfaktor – canvasen kan vara skalad via CSS max-width
    const displayW = canvas.clientWidth || canvas.width;
    const displayH = canvas.clientHeight || canvas.height;
    const sx = displayW / canvas.width;
    const sy = displayH / canvas.height;

    data.forEach((d, i) => {
      const {x,y,w,h} = d.box;
      const best = (d.candidates?.[0]?.score) || 0;

      // Rita rektangeln på canvasen
      let color = 'rgb(70,220,120)';
      if (best < 0.3) color = 'rgb(220,60,60)'; else if (best < 0.7) color = 'rgb(230,200,70)';
      ctx.strokeStyle = color; ctx.strokeRect(x,y,w,h);

      // Skapa HTML-overlay boundingbox
      const box = document.createElement('div');
      box.className = 'frp-face-box';
      box.style.left   = (x * sx) + 'px';
      box.style.top    = (y * sy) + 'px';
      box.style.width  = (w * sx) + 'px';
      box.style.height = (h * sy) + 'px';

      // Flippa dropdown uppåt om boxen är nära botten
      if ((y + h) * sy > displayH * 0.7) box.dataset.flip = 'top';

      // Skapa suggestion-lista
      const sug = document.createElement('div');
      sug.className = 'frp-suggestions';

      const cands = (d.candidates || []).slice(0, 3);
      if (cands.length === 0){
        const row = document.createElement('div');
        row.className = 'frp-suggestion';
        row.innerHTML = '<span style=\"color:#9aa0ac\">(inga kandidater)</span>';
        sug.appendChild(row);
      } else {
        cands.forEach(c => {
          const row = document.createElement('div');
          row.className = 'frp-suggestion';
          if (c.score < 0.3) row.classList.add('is-low');
          row.innerHTML = '<span>' + c.name + '</span><span class=\"conf\">' + Math.round(c.score*100) + '%</span>';
          sug.appendChild(row);
          attachHoverPreview(row, c.name);
        });
      }

      box.appendChild(sug);

      // Visa/dölj namnlistan med fördröjning
      let hideTimer = null;
      function showSug(){ clearTimeout(hideTimer); sug.classList.add('visible'); box.classList.add('active'); }
      function schedulHide(){ hideTimer = setTimeout(() => { sug.classList.remove('visible'); box.classList.remove('active'); }, 400); }
      box.addEventListener('mouseenter', showSug);
      box.addEventListener('mouseleave', schedulHide);
      sug.addEventListener('mouseenter', showSug);
      sug.addEventListener('mouseleave', schedulHide);

      overlay.appendChild(box);
    });

    runBtn.disabled = false;
  }

  runBtn.addEventListener('click', recognize);
  </script>
</body>
</html>
    """
    return render_template_string(html)


# -------------------------------------------------
# Hälsa/konfig och index
# -------------------------------------------------
@app.get("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "service": "face_extractor",
        "version": "1.8.0-strict-no-cache",
        "model_loaded": clf is not None,
        "classes": len(le.classes_) if hasattr(le, 'classes_') else None,
    })


@app.get("/api/config")
def api_get_config():
    return jsonify({
        "threshold": THRESHOLD,
        "model_path": MODEL_PATH,
        "classes": le.classes_.tolist() if hasattr(le, 'classes_') else []
    })


@app.post("/api/config")
def api_set_config():
    global THRESHOLD
    data = request.get_json(silent=True) or {}
    if 'threshold' in data:
        try:
            new_threshold = float(data['threshold'])
        except Exception:
            return jsonify({"error": "Threshold måste vara flyttal"}), 400
        if 0.0 <= new_threshold <= 1.0:
            THRESHOLD = new_threshold
            return jsonify({"message": "Threshold uppdaterad", "threshold": THRESHOLD})
        return jsonify({"error": "Threshold måste vara mellan 0.0 och 1.0"}), 400
    return jsonify({"error": "Ingen giltig konfiguration skickad"}), 400


@app.get("/")
def index_redirect():
    return ("<meta http-equiv='refresh' content='0;url=/web'>"), 302


if __name__ == "__main__":
    print("🚀 Startar Face Extractor API…")
    print("   /web                    – web UI (clipboard & drag/drop)")
    print("   POST /recognize?top_k=N – topp‑K API")
    print("   GET  /resolve_image     – bildproxy (Stash/StashDB, strikt, ingen cache)")
    print("   GET  /api/health        – hälsa")
    print("   GET/POST /api/config    – konfigurera")
    print("🌐 STASH_URL:", STASH_URL)
    print("🌐 STASH_GRAPHQL:", STASH_GRAPHQL)
    print("🔑 STASH_API_KEY set:", bool(STASH_API_KEY))
    print("🌐 STASHDB_ENDPOINT_DEFAULT:", STASHDB_ENDPOINT_DEFAULT)
    print("🔑 STASHDB_API_KEY:", _mask(STASHDB_API_KEY))

    debug_flag = os.environ.get("FACE_EXTRACTOR_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=5000, debug=debug_flag, use_reloader=False)
