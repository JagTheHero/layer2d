"""Layer2D Rapid Avatar Scaffolder (`rigkit.model_scaffold`).

Creates a brand-new, immediately runnable Layer2D avatar project complete with:
- `avatar_spec.json` (`layer2d-spec/v1`)
- `rig.json` (data-driven head, springs, outfits, highlightIds, hairShadowId)
- `placed/` starter clean layered RGBA anatomy (back hair, body, face, hair shadow,
  eyes, brows, mouth, front hair) so the model boots in the Layer2D WebGL2 Player
  out-of-the-box before bespoke art is dropped in
- `viewer/pack.json` + `viewer/index.html` + `viewer/tex/*.webp`
- `<slug>.l2d` validated open Layer2D container archive

Proves the unified base system works cleanly for ANY avatar with zero viewer edits.

Usage:
    python -m rigkit.model_scaffold demo_fox --name "Demo Fox" --out /tmp/demo_fox
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from rigkit.expr_library import AvatarFaceAnchors, build_default_timelines, build_standard_faces
from rigkit.format import pack_l2d, validate_l2d
from rigkit.meshpack import build_pack
from rigkit.schema import AvatarSpec, HeadAnatomy, SPEC_VERSION, load_avatar_spec

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIEWER = ROOT / "viewer/index.html"


def _save_layer(placed: Path, lid: str, box: tuple[int, int, int, int], draw_fn) -> dict[str, Any]:
    x0, y0, x1, y1 = box
    im = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(im), x1 - x0, y1 - y0)
    im.save(placed / f"{lid}.png")
    return {"id": lid, "box": [x0, y0, x1, y1]}


def scaffold_avatar(
    slug: str,
    *,
    name: str,
    out_dir: Path,
    spec_path: Path | None = None,
    accent_rgb: tuple[int, int, int] = (190, 30, 60),
    hair_rgb: tuple[int, int, int] = (212, 214, 224),
    eye_rgb: tuple[int, int, int] = (205, 52, 72),
    viewer_template: Path = DEFAULT_VIEWER,
) -> dict[str, Any]:
    """Scaffold a complete, immediately runnable Layer2D avatar + `.l2d` bundle.

    When `spec_path` is provided, reads and validates an authored `avatar_spec.json`
    (`layer2d-spec/v1`) and derives slug, name, title, canvas, and head anchors from it.
    """
    out_dir = Path(out_dir)
    placed = out_dir / "placed"
    placed.mkdir(parents=True, exist_ok=True)
    if spec_path is not None:
        loaded = load_avatar_spec(spec_path)
        slug = str(loaded["slug"])
        name = str(loaded["name"])
        W, H = int(loaded["canvas"][0]), int(loaded["canvas"][1])
        cx, cy = int(round(loaded["head"]["centre"][0])), int(round(loaded["head"]["centre"][1]))
        spec_dict = loaded
        (out_dir / "avatar_spec.json").write_text(json.dumps(spec_dict, indent=2) + "\n")
    else:
        W, H = 1200, 2000
        cx, cy = 600, 440
        spec = AvatarSpec(
            spec_version=SPEC_VERSION,
            slug=slug,
            name=name,
            title=f"{name} — Layer2D Avatar",
            version="1.0.0",
            author="Layer2D",
            canvas=[W, H],
            head=HeadAnatomy(
                centre=[float(cx), float(cy)],
                radius=210.0,
                chin=144.0,
                neck=202.0,
                neck_pivot=[float(cx), 610.0],
                eye_r=[535.0, 445.0],
                eye_l=[665.0, 445.0],
                mouth=[600.0, 522.0],
                forehead_y=365.0,
            ),
            outfits=["accessory"],
            expression_pack="standard-core",
        )
        spec.save(out_dir / "avatar_spec.json")
        spec_dict = spec.to_dict()

    # Minimal clean layered starter art (can be replaced 1-for-1 with final art sheets)
    skin = (252, 232, 222, 255)
    skin_edge = (182, 146, 138, 255)
    hair_dark = (max(0, hair_rgb[0] - 48), max(0, hair_rgb[1] - 48), max(0, hair_rgb[2] - 42), 255)
    hair_fill = (*hair_rgb, 255)
    acc_fill = (*accent_rgb, 255)
    eye_fill = (*eye_rgb, 255)

    layers_meta: list[dict[str, Any]] = []

    def add(lid: str, box: tuple[int, int, int, int], kind: str, draw_fn, **extra: Any) -> None:
        rec = _save_layer(placed, lid, box, draw_fn)
        rec["kind"] = kind
        rec.update(extra)
        layers_meta.append(rec)

    add("H01", (360, 210, 840, 860), "hair",
        lambda d, w, h: d.rounded_rectangle((12, 12, w - 12, h - 12), radius=140, fill=hair_dark),
        depth=-0.55, group="hair_back", noFade=True)
    add("B01", (410, 570, 790, 1420), "torso",
        lambda d, w, h: (
            d.rectangle((w // 2 - 42, 0, w // 2 + 42, 115), fill=skin),
            d.rounded_rectangle((36, 95, w - 36, h - 20), radius=90, fill=(36, 38, 48, 255), outline=acc_fill, width=8),
        ))
    add("B07R", (330, 665, 460, 1240), "arm",
        lambda d, w, h: d.rounded_rectangle((12, 12, w - 12, h - 12), radius=48, fill=(44, 46, 58, 255)),
        side="R")
    add("B07L", (740, 665, 870, 1240), "arm",
        lambda d, w, h: d.rounded_rectangle((12, 12, w - 12, h - 12), radius=48, fill=(44, 46, 58, 255)),
        side="L")
    add("F01", (450, 270, 750, 600), "head",
        lambda d, w, h: d.ellipse((14, 14, w - 14, h - 14), fill=skin, outline=skin_edge, width=4),
        depth=1.0)
    add("F02", (465, 290, 735, 425), "head",
        lambda d, w, h: d.ellipse((8, 8, w - 8, h - 8), fill=(110, 96, 130, 52)),
        depth=1.02, clip="F01")
    add("E01R", (495, 415, 575, 475), "eye",
        lambda d, w, h: d.ellipse((4, 6, w - 4, h - 6), fill=(255, 255, 255, 255)),
        side="R", depth=1.05, clip="F01")
    add("E02R", (513, 421, 557, 469), "iris",
        lambda d, w, h: d.ellipse((2, 2, w - 2, h - 2), fill=eye_fill),
        side="R", depth=1.06, clip="E01R")
    add("E04R", (521, 428, 535, 442), "lash",
        lambda d, w, h: d.ellipse((1, 1, w - 1, h - 1), fill=(255, 255, 255, 255)),
        side="R", depth=1.07, clip="E01R")
    add("E05R", (490, 408, 580, 442), "lash",
        lambda d, w, h: d.arc((4, 6, w - 4, h + 16), 195, 345, fill=(32, 26, 36, 255), width=7),
        side="R", depth=1.08)
    add("E01L", (625, 415, 705, 475), "eye",
        lambda d, w, h: d.ellipse((4, 6, w - 4, h - 6), fill=(255, 255, 255, 255)),
        side="L", depth=1.05, clip="F01")
    add("E02L", (643, 421, 687, 469), "iris",
        lambda d, w, h: d.ellipse((2, 2, w - 2, h - 2), fill=eye_fill),
        side="L", depth=1.06, clip="E01L")
    add("E04L", (651, 428, 665, 442), "lash",
        lambda d, w, h: d.ellipse((1, 1, w - 1, h - 1), fill=(255, 255, 255, 255)),
        side="L", depth=1.07, clip="E01L")
    add("E05L", (620, 408, 710, 442), "lash",
        lambda d, w, h: d.arc((4, 6, w - 4, h + 16), 195, 345, fill=(32, 26, 36, 255), width=7),
        side="L", depth=1.08)
    add("BR01R", (495, 386, 575, 406), "brow",
        lambda d, w, h: d.line((6, 12, w - 6, 8), fill=(72, 68, 82, 255), width=5),
        side="R", depth=1.08)
    add("BR01L", (625, 386, 705, 406), "brow",
        lambda d, w, h: d.line((6, 8, w - 6, 12), fill=(72, 68, 82, 255), width=5),
        side="L", depth=1.08)
    add("M01", (568, 506, 632, 540), "head",
        lambda d, w, h: d.arc((8, 4, w - 8, h - 4), 15, 165, fill=(145, 76, 82, 255), width=4),
        depth=1.07)
    add("H03C", (430, 225, 770, 415), "hair",
        lambda d, w, h: d.polygon(
            [(20, 170), (45, 35), (w // 2, 8), (w - 45, 35), (w - 20, 170),
             (w - 90, 130), (w // 2 + 25, 178), (w // 2, 118), (w // 2 - 25, 178), (90, 130)],
            fill=hair_fill, outline=hair_dark,
        ),
        depth=1.18, group="bangs", noFade=True)
    add("A01", (685, 295, 755, 365), "accessory_head",
        lambda d, w, h: d.ellipse((8, 8, w - 8, h - 8), fill=acc_fill, outline=(255, 255, 255, 220), width=4),
        depth=1.22)

    (placed / "layers.json").write_text(json.dumps({"canvas": [W, H], "layers": layers_meta}, indent=2) + "\n")

    rig = {
        "name": name,
        "title": f"{name} — Layer2D Avatar",
        "bg": [0.09, 0.09, 0.12],
        "centre": [float(cx), float(cy)],
        "radius": 210.0,
        "tilt": 0.0,
        "chin": 144.0,
        "neck": 202.0,
        "neck_pivot": [float(cx), 610.0],
        "waist": [float(cx), 1020.0],
        "shoulder": {"R": [445.0, 690.0], "L": [755.0, 690.0]},
        "turn": {"R": -0.22, "L": 0.22},
        "eyes": {"R": "E01R", "L": "E01L"},
        "upper_lash": ["E05R", "E05L"],
        "eye_pin": True,
        "hairShadowId": "F02",
        "highlightIds": ["E04R", "E04L"],
        "pitchGroups": {
            "H01": "back", "F01": "face", "F02": "face",
            "E01R": "face", "E02R": "face", "E04R": "face", "E05R": "face",
            "E01L": "face", "E02L": "face", "E04L": "face", "E05L": "face",
            "BR01R": "face", "BR01L": "face", "M01": "face",
            "H03C": "bangs", "A01": "bangs",
        },
        "views": {"bust": [240, 120, 960, 1120], "full": [140, 80, 1060, 1940]},
        "sigma": {"hair_back": 180.0, "bangs": 120.0},
        "chain_depth": {"hair_back": -0.45, "bangs": 0.9},
        "chains": {
            "hair_back_C": ["head", "hair_back", [0.92, 0.24], 0.18, [[600, 260], [600, 520], [600, 820]]],
            "bangs_C": ["head", "bangs", [0.96, 0.45], 0.18, [[600, 240], [600, 320], [600, 405]]],
        },
        "body": {
            "footY": 1920.0,
            "legTop": 1250.0,
            "bodyR": 420.0,
            "twistY": [640.0, 1040.0],
            "skirtY": [1010.0, 1380.0],
            "armReach": 780.0,
            "seam": {
                "armR": [-0.3, 0.95, -45.0, -25.0, 50.0, 100.0],
                "armL": [0.3, 0.95, 45.0, -25.0, -50.0, 100.0],
            },
            "elbow": {"armR": [395.0, 930.0], "armL": [805.0, 930.0]},
            "hand": {"armR": [375.0, 1190.0], "armL": [825.0, 1190.0]},
            "hips": {"R": [495.0, 1250.0], "L": [705.0, 1250.0]},
            "shin": [1550.0, 1850.0],
            "puppetFace": [600.0, 450.0, 220.0, 240.0],
        },
        "outfits": {
            "accessory": {"label": "Pin", "default": True, "hide": ["A01"], "off": [25, -25], "secs": 0.35},
        },
    }
    (out_dir / "rig.json").write_text(json.dumps(rig, indent=2) + "\n")

    pack_info = build_pack(out_dir, viewer_template=viewer_template if viewer_template.exists() else None)

    # Wire minimal expressions using rigkit.expr_library so every scaffolded model has working faces
    pack_path = out_dir / "viewer/pack.json"
    pack = json.loads(pack_path.read_text())
    faces = {
        "Neutral": {"hide": [], "add": [], "move": {}},
        "Happy": {"hide": [], "add": [], "move": {}, "eyes": 0.15},
        "Wink": {"hide": [], "add": [], "move": {}, "eyes": [1.0, 0.05]},
        "Surprise": {"hide": [], "add": [], "move": {"BR01R": [0, -14], "BR01L": [0, -14]}, "eyes": 1.25},
        "Sleepy": {"hide": [], "add": [], "move": {"BR01R": [0, 8], "BR01L": [0, 8]}, "eyes": 0.42},
    }
    pack["expr"]["faces"] = faces
    pack["expr"]["faceTimeline"] = [[0.0, None], [2.5, "Happy"], [4.8, "Wink"], [7.2, None]]
    pack["expr"]["holdTimeline"] = [[0.0, None]]
    pack_path.write_text(json.dumps(pack, separators=(",", ":")))

    # Build a crisp preview.png from the layered composition
    preview = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for L in layers_meta:
        part = Image.open(placed / f"{L['id']}.png").convert("RGBA")
        x0, y0, _, _ = L["box"]
        preview.alpha_composite(part, (x0, y0))
    preview_path = out_dir / "preview.png"
    preview.crop((180, 100, 1020, 1220)).resize((600, 800), Image.LANCZOS).save(preview_path)

    l2d_path = out_dir / f"{slug}.l2d"
    l2d_meta = {**spec_dict, "license": spec_dict.get("license") or "MIT"}
    l2d_info = pack_l2d(out_dir / "viewer", l2d_path, meta=l2d_meta, preview_path=preview_path)
    val = validate_l2d(l2d_path)
    if not val["ok"]:
        raise RuntimeError(f"Scaffolded .l2d failed validation: {val['errors']}")

    return {
        "slug": slug,
        "out_dir": str(out_dir),
        "pack": pack_info,
        "l2d": l2d_info,
        "validation": val,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold a new Layer2D avatar ready to run in the WebGL2 Player")
    ap.add_argument("slug", nargs="?", default=None, help="Short lowercase model slug (optional if --spec is given)")
    ap.add_argument("--name", default=None, help="Display name (optional if --spec is given)")
    ap.add_argument("--spec", type=Path, default=None, help="Path to an authored avatar_spec.json (layer2d-spec/v1)")
    ap.add_argument("--out", type=Path, required=True, help="Output model directory")
    a = ap.parse_args()
    if a.spec is None and (not a.slug or not a.name):
        ap.error("Provide either --spec <avatar_spec.json> or both <slug> and --name")
    res = scaffold_avatar(a.slug or "avatar", name=a.name or "Avatar", out_dir=a.out, spec_path=a.spec)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
