#!/usr/bin/env python3
"""Build a valid, runnable Layer2D v1 (.l2d) avatar from scratch."""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def paint_starter_textures() -> dict[str, tuple[bytes, int, int, float, float]]:
    """Return {layer_id: (png_bytes, w, h, center_x, center_y)} on a 1200x2000 canvas."""
    out = {}

    # Back hair (H01)
    im = Image.new("RGBA", (540, 680), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((30, 30, 510, 650), radius=180, fill=(38, 42, 56, 255), outline=(20, 22, 30, 255), width=6)
    out["H01"] = (png_bytes(im), im.width, im.height, 600.0, 540.0)

    # Torso (B01)
    im = Image.new("RGBA", (420, 640), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((40, 40, 380, 620), radius=90, fill=(28, 30, 38, 255), outline=(190, 30, 60, 255), width=6)
    out["B01"] = (png_bytes(im), im.width, im.height, 600.0, 1080.0)

    # Face base (F01 — clipTarget for eyes/blush/shadows)
    im = Image.new("RGBA", (360, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((20, 15, 340, 395), fill=(252, 232, 222, 255), outline=(80, 60, 65, 255), width=5)
    out["F01"] = (png_bytes(im), im.width, im.height, 600.0, 490.0)

    # Eyes (E01R / E01L)
    for side, cx in (("R", 525.0), ("L", 675.0)):
        im = Image.new("RGBA", (96, 86), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.ellipse((8, 12, 88, 76), fill=(255, 255, 255, 255), outline=(35, 30, 40, 255), width=5)
        d.ellipse((24, 18, 72, 68), fill=(190, 30, 60, 255))
        d.ellipse((38, 32, 58, 52), fill=(24, 18, 28, 255))
        d.ellipse((30, 24, 42, 36), fill=(255, 255, 255, 255))
        out[f"E01{side}"] = (png_bytes(im), im.width, im.height, cx, 490.0)

    # Neutral mouth (M01)
    im = Image.new("RGBA", (90, 46), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.arc((12, 4, 78, 36), start=15, end=165, fill=(135, 58, 68, 255), width=5)
    out["M01"] = (png_bytes(im), im.width, im.height, 600.0, 600.0)

    # Front bangs (H03C)
    im = Image.new("RGBA", (420, 250), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(30, 190), (80, 20), (340, 20), (390, 190), (310, 150), (260, 215), (210, 145), (155, 215), (105, 150)],
              fill=(54, 60, 78, 255), outline=(20, 22, 30, 255))
    out["H03C"] = (png_bytes(im), im.width, im.height, 600.0, 365.0)

    return out


def build_starter_l2d(out_path: Path) -> None:
    tex_defs = paint_starter_textures()
    draw_order = [
        ("H01",  True,  0, "back",  False, None),
        ("B01",  False, 1, "face",  False, None),
        ("F01",  True,  0, "face",  True,  None),
        ("E01R", True,  0, "face",  False, "F01"),
        ("E01L", True,  0, "face",  False, "F01"),
        ("M01",  True,  0, "face",  False, "F01"),
        ("H03C", True,  0, "bangs", False, None),
    ]

    layers = []
    pitch_groups = {}
    for lid, is_head, kind, pg, is_target, clip in draw_order:
        _, w, h, _, _ = tex_defs[lid]
        pitch_groups[lid] = pg
        layers.append({
            "id": lid, "w": w, "h": h, "kind": kind, "head": is_head,
            "jaw": False, "noFade": is_head, "hole": False,
            "clip": clip, "clipTarget": is_target, "pad": 320.0,
            "pitchGroup": pg,
            **({"chains": [0], "chainSigma": 180.0} if lid == "H01" else {}),
        })

    # 5 continuous yaw keyforms from -0.22 rad (-32.5 deg) to +0.22 rad (+32.5 deg)
    angles = [-0.22, -0.11, 0.0, 0.11, 0.22]
    samples = []
    for ang in angles:
        s_layers = []
        for lid, is_head, _, pg, _, _ in draw_order:
            _, _, _, cx, cy = tex_defs[lid]
            depth = -0.35 if pg == "back" else (0.14 if pg == "bangs" else 0.0)
            tx = cx + (math.sin(ang) * 260.0 * depth if is_head else 0.0)
            s_layers.append({
                "A": [1.0, 0.0, 0.0, 1.0],
                "t": [round(tx, 2), round(cy, 2)],
                "op": 1.0,
                **({"turn": round(ang * (0.45 if pg == "back" else 0.75), 4)} if is_head else {}),
            })
        samples.append({"angle": ang, "jaw": 0.0, "chin": 0.0, "chinAt": [600.0, 680.0], "layers": s_layers})

    pack = {
        "title": "Starter — Layer2D Avatar",
        "canvas": [1200, 2000],
        "bg": [0.05, 0.05, 0.07],
        "views": {"bust": [180, 80, 1020, 1180], "full": [0, 0, 1200, 2000]},
        "texExt": "png",
        "samples": samples,
        "layers": layers,
        "head": {
            "centre": [600.0, 490.0], "radius": 260.0,
            "eu": [1.0, 0.0], "ev": [0.0, 1.0],
            "chin": 185.0, "neck": 260.0, "jawNear": 110.0,
            "pivot": [600.0, 740.0], "turn": {"R": -0.22, "L": 0.22},
        },
        "motion": {
            "sway": [3.2, 0.006, 0.011], "stepBounce": 4.0, "breath": [3.6, 0.0045],
            "bodyFollow": [0.14, 0.28], "bounces": [[7.0, 7.5], 12.0, 0.12], "dur": 13.0,
            "earSoft": 5.0, "sew": 160.0,
            "limit": {"bend": 0.28, "lift": 0.06, "earring": 0.6, "arm": 0.08},
            "lockSigma": 50.0, "lockLower": [1.5, 0.8, 3.0],
            "springs": {"skirt": [1.1, 0.25, 0.4, 4.0, 0.4], "armR": [0.9, 0.35, 0.7, 8.0, 0.0], "armL": [0.9, 0.35, 0.7, 8.0, 0.0]},
            "timeline": {"a": [[0, 0], [1.8, -0.8], [4.2, 0.8], [6.0, 0]], "roll": [[0, 0], [1.8, 3], [4.2, -3], [6.0, 0]]},
            "roll0": 0.0, "waist": [600.0, 1020.0], "footY": 1920.0, "legTop": 1250.0,
            "shoulder": {"armR": [445.0, 690.0], "armL": [755.0, 690.0]},
            "bodyC": 600.0, "bodyR": 420.0, "twistY": [640.0, 1040.0], "skirtY": [1010.0, 1380.0],
            "seam": {"armR": [-0.3, 0.95, -45.0, -25.0, 50.0, 100.0], "armL": [0.3, 0.95, 45.0, -25.0, -50.0, 100.0]},
            "armReach": 780.0, "hair": [], "field": [], "earrings": {}, "locks": [],
            "elbow": {"armR": [395.0, 930.0], "armL": [805.0, 930.0]},
            "hand": {"armR": [375.0, 1190.0], "armL": [825.0, 1190.0]},
            "shin": [1550.0, 1850.0], "hips": {"R": [495.0, 1250.0], "L": [705.0, 1250.0]},
            "puppetFace": [600.0, 490.0, 220.0, 240.0],
        },
        "skirt": {"y0": 0, "xl": [0.0], "xr": [1.0], "ml": [0.0], "mr": [0.0]},
        "expr": {
            "pieces": {},
            "faces": {
                "Neutral": {"hide": [], "add": [], "move": {}},
                "Happy":   {"hide": [], "add": [], "move": {}, "eyes": [0.25, 0.25]},
                "Wink":    {"hide": [], "add": [], "move": {}, "eyes": [1.0, 0.08]},
            },
            "holds": {}, "raiseSleeves": {}, "eyes": {}, "blink": [2.8, 0.14],
            "faceTimeline": [], "holdTimeline": [], "vowels": {},
        },
        "hairTint": None, "eyeTint": None,
        "chains": [{
            "name": "back_hair", "parent": "head", "k": [0.55, 0.18], "damp": 0.16,
            "joints": [600.0, 340.0, 600.0, 520.0, 600.0, 700.0, 600.0, 860.0],
            "depth": 1.0, "pin": False,
        }],
        "chainWind": [0.0, 0.0], "sheer": None, "outfits": {},
        "hairShadowId": None, "highlightIds": [], "pitchGroups": pitch_groups,
    }

    manifest = {
        "format": "layer2d/v1", "id": "starter", "name": "Starter",
        "title": pack["title"], "version": "1.0.0", "author": "Agent", "license": "MIT",
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entry": "pack.json", "texExt": "png", "canvas": pack["canvas"],
        "stats": {
            "layers": len(layers), "chains": len(pack["chains"]), "samples": len(samples),
            "expressions": len(pack["expr"]["faces"]), "holds": 0, "outfits": 0, "exprPieces": 0, "vowels": [],
        },
        "capabilities": {
            "headYawSamples": len(samples), "headPitch2D": True, "springChains": True,
            "tailRig": False, "obsAlpha": True, "micLipSync": False, "webcamTracking": True,
        },
    }

    members: dict[str, bytes] = {
        "layer2d.json": (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        "pack.json": json.dumps(pack, separators=(",", ":")).encode("utf-8"),
    }
    for lid, (raw_png, *_rest) in tex_defs.items():
        members[f"tex/{lid}.png"] = raw_png

    sums = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(members.items())}
    members["SHA256SUMS.json"] = (json.dumps(sums, indent=2, sort_keys=True) + "\n").encode("utf-8")

    with zipfile.ZipFile(out_path, "w") as zf:
        for name in sorted(members):
            comp = zipfile.ZIP_DEFLATED if name.endswith(".json") else zipfile.ZIP_STORED
            zf.writestr(name, members[name], compress_type=comp)
    print(f"Built {out_path} ({out_path.stat().st_size} bytes, {len(members)} members)")


if __name__ == "__main__":
    import sys
    build_starter_l2d(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("starter.l2d"))