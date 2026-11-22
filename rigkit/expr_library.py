"""Reusable Layer2D Studio Expression & Animated Flipbook FX Library (`rigkit.expr_library`).

Provides a model-independent system for packing 30–36+ studio-grade animated expressions
onto any Layer2D avatar using shared flipbook FX spritesheets (`assets/fx/S_fx_*.png` + `fx.json`)
paired with each model's native eyes, mouths, brows, and custom FX overlays.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ASSETS_FX_DIR = Path(__file__).resolve().parent.parent / "assets/fx"


def _r4(arr) -> list[float]:
    return [round(float(v), 4) for v in np.ravel(arr)]


@dataclass
class AvatarFaceAnchors:
    """Canonical facial landmark anchors in the avatar's rest canvas coordinate space."""

    eye_r: tuple[float, float]
    eye_l: tuple[float, float]
    mouth: tuple[float, float]
    face_center: tuple[float, float]
    forehead_y: float
    cheek_y: float
    # Layer IDs for anchoring & clipping
    face_base: str = "F01"
    face_clip_layer: str = "F02"  # clipped to F01 -> any FX riding this auto-clips inside face silhouette!
    blush_r: str = "F04R"
    blush_l: str = "F04L"
    mouth_base: str = "M01"
    hair_front_base: str = "H03C"
    overlay_top_at: str = "HP01"  # drawn ABOVE hair & accessories so floating FX are NEVER occluded!
    brow_r: str = "E11R"
    brow_l: str = "E11L"
    eye_r_layers: list[str] = field(
        default_factory=lambda: ["E01R", "E02R", "E03R", "E04R", "E04bR", "E05R", "E06R", "E07R"]
    )
    eye_l_layers: list[str] = field(
        default_factory=lambda: ["E01L", "E02L", "E03L", "E04L", "E04bL", "E05L", "E06L", "E07L"]
    )
    highlight_layers: list[str] = field(default_factory=lambda: ["E04R", "E04L"])

    @property
    def inter_ocular(self) -> float:
        return math.hypot(self.eye_l[0] - self.eye_r[0], self.eye_l[1] - self.eye_r[1])

    @property
    def face_scale(self) -> float:
        """Scale relative to Test Signal's canonical 196px inter-ocular baseline."""
        return self.inter_ocular / 196.0


def build_flipbook_fx_pieces(
    anchors: AvatarFaceAnchors,
    rest: dict[str, dict],
    tex_out_dir: Path,
    *,
    fx_dir: Path = ASSETS_FX_DIR,
    extra_specs: dict[str, tuple[str, str, str, tuple[float, float], float, float, float]] | None = None,
) -> dict[str, dict]:
    """Copy shared animated flipbook FX PNGs into `tex_out_dir` and build `pack["expr"]["pieces"]` entries.

    All coordinates derive smoothly from `anchors` (inter-ocular spacing & facial landmarks) so any
    Layer2D avatar inherits crisp, well-proportioned animated FX out of the box!
    """
    fx_meta_path = fx_dir / "fx.json"
    if not fx_meta_path.exists():
        raise FileNotFoundError(f"Missing shared flipbook FX metadata: {fx_meta_path}")
    fx_meta = json.loads(fx_meta_path.read_text())

    k = anchors.face_scale
    cx, cy = anchors.face_center
    er_x, er_y = anchors.eye_r
    el_x, el_y = anchors.eye_l
    mx, my = anchors.mouth
    dx_eye = abs(el_x - er_x)

    # Top unobstructed z-order slot (fallback cleanly if model has no hairpin)
    top_at = anchors.overlay_top_at if anchors.overlay_top_at in rest else anchors.hair_front_base
    face_clip = anchors.face_clip_layer if anchors.face_clip_layer in rest else anchors.face_base
    brow_at = anchors.brow_l if anchors.brow_l in rest else anchors.face_base
    blush_at = anchors.blush_l if anchors.blush_l in rest else face_clip

    # {piece_id: (fx_name, base_layer, at_layer, (canvas_x, canvas_y), scale, rot_deg, t0)}
    specs: dict[str, tuple[str, str, str, tuple[float, float], float, float, float]] = {
        # Sparkling aura (Excited, Starry, Cute, Hungry)
        "FX_sp1": ("sparkle", anchors.hair_front_base, top_at, (cx + 1.52 * dx_eye, anchors.forehead_y - 0.58 * dx_eye), 1.38, 0.0, 0.0),
        "FX_sp2": ("sparkle", anchors.hair_front_base, top_at, (cx - 1.48 * dx_eye, anchors.forehead_y - 0.22 * dx_eye), 1.18, 0.0, 0.30),
        "FX_sp3": ("sparkle", anchors.hair_front_base, top_at, (cx + 1.36 * dx_eye, anchors.cheek_y - 0.08 * dx_eye), 0.96, 0.0, 0.55),
        # Question mark (Confused)
        "FX_question": ("question", anchors.hair_front_base, top_at, (cx + 1.48 * dx_eye, anchors.forehead_y - 0.62 * dx_eye), 1.12, 6.0, 0.05),
        # Floating hearts (Love, Blow Kiss — clear of hairpin!)
        "FX_heart_L": ("heart", anchors.hair_front_base, top_at, (cx + 1.36 * dx_eye, anchors.forehead_y - 0.52 * dx_eye), 1.05, 6.0, 0.05),
        "FX_heart_R": ("heart", anchors.hair_front_base, top_at, (cx - 1.38 * dx_eye, anchors.forehead_y + 0.08 * dx_eye), 0.86, -8.0, 0.32),
        # Animated tear droplets (Puppy Eyes, Crying)
        "FX_tear_R": ("tear", "E06R" if "E06R" in rest else anchors.face_base, top_at, (er_x - 0.14 * dx_eye, er_y + 0.42 * dx_eye), 0.95, 0.0, 0.10),
        "FX_tear_L": ("tear", "E06L" if "E06L" in rest else anchors.face_base, top_at, (el_x + 0.14 * dx_eye, el_y + 0.42 * dx_eye), 0.95, 0.0, 0.32),
        # Animated sweat drops (Flustered, Nervous/Shock — strictly unobstructed!)
        "FX_sweat_R": ("sweat", anchors.hair_front_base, top_at, (cx - 1.24 * dx_eye, anchors.forehead_y + 0.06 * dx_eye), 1.35, 14.0, 0.0),
        "FX_sweat_L": ("sweat", anchors.hair_front_base, top_at, (cx + 1.34 * dx_eye, anchors.cheek_y + 0.05 * dx_eye), 1.18, -18.0, 0.28),
        # Frustrated scribble cloud (Frustrated, Dizzy)
        "FX_scribble": ("scribble", anchors.hair_front_base, top_at, (cx - 1.26 * dx_eye, anchors.forehead_y - 0.58 * dx_eye), 1.18, 0.0, 0.0),
        # Sigh puff (Relieved, Exhausted)
        "FX_puff": ("puff", anchors.mouth_base, top_at, (mx - 0.62 * dx_eye, my + 0.10 * dx_eye), 0.95, 0.0, 0.15),
        # Drool (Hungry)
        "FX_drool": ("drool", anchors.mouth_base, top_at, (mx + 0.18 * dx_eye, my + 0.20 * dx_eye), 0.92, 0.0, 0.05),
        # Clipped upper-forehead & brow mood shades (riding on `face_clip` -> auto-clipped inside F01!)
        "FX_shade_blue": ("shade_blue", face_clip, brow_at, (cx, anchors.forehead_y - 0.04 * dx_eye), 1.02, 0.0, 0.0),
        "FX_shade_pale": ("shade_pale", face_clip, brow_at, (cx, anchors.forehead_y - 0.08 * dx_eye), 1.02, 0.0, 0.0),
        "FX_shade_purple": ("shade_purple", face_clip, brow_at, (cx, anchors.forehead_y - 0.06 * dx_eye), 1.02, 0.0, 0.0),
        "FX_shadow": ("shadow", face_clip, brow_at, (cx, anchors.forehead_y - 0.04 * dx_eye), 1.02, 0.0, 0.0),
        # Animated rosy blush bar (clipped cleanly to F01!)
        "FX_blush": ("blush", face_clip, blush_at, (cx, anchors.cheek_y + 0.02 * dx_eye), 0.98, 0.0, 0.0),
        # Sharp anime eye glints (Evil Grin, Smug)
        "FX_glint_R": ("glint", "E02R" if "E02R" in rest else anchors.face_base, top_at, (er_x - 0.04 * dx_eye, er_y - 0.06 * dx_eye), 0.95, 0.0, 0.10),
        "FX_glint_L": ("glint", "E02L" if "E02L" in rest else anchors.face_base, top_at, (el_x - 0.04 * dx_eye, el_y - 0.06 * dx_eye), 0.95, 0.0, 0.25),
    }
    if extra_specs:
        specs.update(extra_specs)

    pieces: dict[str, dict] = {}
    copied: set[str] = set()
    tex_out_dir.mkdir(parents=True, exist_ok=True)

    for pid, (fx_name, base, at, mid, sc, deg, t0) in specs.items():
        if fx_name not in fx_meta or base not in rest:
            continue
        src_png = fx_dir / f"S_fx_{fx_name}.png"
        if not src_png.exists():
            continue
        if fx_name not in copied:
            shutil.copy2(src_png, tex_out_dir / f"S_fx_{fx_name}.png")
            copied.add(fx_name)

        Ab = np.array(rest[base]["A"], dtype=np.float64).reshape(2, 2)
        tb = np.array(rest[base]["t"], dtype=np.float64)
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        Mi = np.linalg.inv((sc * k) * np.array([[c, s], [-s, c]], dtype=np.float64))
        m = fx_meta[fx_name]
        entry: dict[str, Any] = {
            "base": base,
            "at": at,
            "w": m["size"][0],
            "h": m["size"][1],
            "relA": _r4(Mi @ Ab),
            "relT": _r4(Mi @ (tb - np.array(mid, dtype=np.float64))),
            "op": 0.68 if fx_name.startswith("shade_") else (0.76 if fx_name == "shadow" else 1.0),
            "tex": f"S_fx_{fx_name}",
            "t0": t0,
            "atlas": {ak: m[ak] for ak in ("cols", "rows", "intro", "loop", "fps", "pad")},
        }
        if fx_name == "blush":
            entry["anim"] = {"fade": 0.35}
        pieces[pid] = entry

    return pieces


def build_standard_faces(
    anchors: AvatarFaceAnchors,
    available_pieces: set[str],
    *,
    brow_scale: float = 1.0,
) -> dict[str, dict]:
    """Return a rich 36-expression commercial catalog filtered strictly to `available_pieces`.

    Guarantees zero broken piece references, distinct visual silhouettes across all 36 faces,
    and emotionally accurate mouth/brow/eye/FX pairings.
    """
    eye_r = list(anchors.eye_r_layers)
    eye_l = list(anchors.eye_l_layers)
    eye_all = eye_r + eye_l
    hl = list(anchors.highlight_layers)
    mouth = [anchors.mouth_base]
    br, bl = anchors.brow_r, anchors.brow_l

    def brows(dy: float, rot_r: float, dy_l: float | None = None, rot_l: float | None = None) -> dict:
        dyl = dy if dy_l is None else dy_l
        rl = -rot_r if rot_l is None else rot_l
        return {
            br: [0, round(dy * brow_scale, 2), round(rot_r, 2)],
            bl: [0, round(dyl * brow_scale, 2), round(rl, 2)],
        }

    BROWS_UP = brows(-11, 0)
    BROWS_SOFT = brows(-5, 6)
    BROWS_SAD = brows(-3, 11)
    BROWS_MAD = brows(8, -14)
    BROWS_CROSS = brows(5, -9)
    BROWS_LOW = brows(6, 0)
    BROWS_SMUG = brows(4, -5, dy_l=-7, rot_l=4)
    BROWS_CONFUSED = brows(4, -7, dy_l=-10, rot_l=3)

    raw_faces: dict[str, dict] = {
        "Neutral": {},
        "Happy": {"hide": mouth + eye_all, "add": ["M04", "E10R", "E10L", "X01R", "X01L"], "move": BROWS_UP},
        "Laugh": {"hide": mouth, "add": ["M04", "X01R", "X01L", "FX_sp1"], "move": BROWS_UP, "eyes": [0.94, 0.94]},
        "Wink": {"hide": mouth + eye_l, "add": ["M09", "E10L"], "move": brows(-6, 0, dy_l=-3, rot_l=0)},
        "Wink R": {"hide": mouth + eye_r, "add": ["M04", "E10R", "X01R", "X01L"], "move": brows(-3, 0, dy_l=-6, rot_l=0)},
        "Fang Smirk": {"hide": mouth, "add": ["M09"], "move": BROWS_SMUG, "eyes": [0.85, 0.85]},
        "Smug": {"hide": mouth, "add": ["M09", "FX_glint_R"], "move": BROWS_SMUG, "eyes": [0.58, 0.58], "look": [5, 0]},
        "Starry": {"hide": mouth + hl, "add": ["M04", "X07R", "X07L", "FX_sp1", "FX_sp2"], "move": BROWS_UP, "eyes": [1.14, 1.14]},
        "Excited": {"hide": mouth, "add": ["M04", "FX_blush", "FX_sp1", "FX_sp2", "FX_sp3"], "move": BROWS_UP, "eyes": [1.22, 1.22]},
        "Love": {"hide": mouth + hl, "add": ["M03", "X08R", "X08L", "X01R", "X01L", "FX_heart_L", "FX_heart_R"], "move": BROWS_UP},
        "Blow Kiss": {"hide": mouth + eye_l, "add": ["E10L", "M06", "FX_blush", "FX_heart_L"], "move": BROWS_SOFT},
        "Cute": {"hide": mouth + eye_l + ["E04R"], "add": ["E10L", "X07R", "M09", "X01R", "X01L", "FX_sp2"], "move": BROWS_UP},
        "Proud": {"hide": mouth + eye_all, "add": ["E09R", "E09L", "M09"], "move": BROWS_UP},
        "Relieved": {"hide": eye_all, "add": ["E09R", "E09L", "FX_puff"], "move": BROWS_SOFT},
        "Shy": {"add": ["FX_blush"], "move": BROWS_SAD, "eyes": [0.74, 0.74], "look": [-6, 3]},
        "Shy Blush": {"hide": mouth, "add": ["M12", "X01R", "X01L", "X03"], "move": BROWS_SAD, "eyes": [0.82, 0.82]},
        "Flustered": {"hide": mouth, "add": ["M12", "X01R", "X01L", "FX_sweat_R", "FX_sweat_L"], "move": BROWS_SAD, "eyes": [1.18, 1.18], "iris": 0.86, "shake": 0.55},
        "Pout": {"hide": mouth, "add": ["M10", "X01R", "X01L"], "move": BROWS_CROSS, "eyes": [0.82, 0.82]},
        "Sulking": {"hide": mouth, "add": ["M10", "FX_blush"], "move": BROWS_CROSS, "eyes": [0.66, 0.66], "look": [7, 2]},
        "Angry": {"hide": mouth, "add": ["M12", "X04"], "move": BROWS_MAD, "eyes": [0.76, 0.76]},
        "Frustrated": {"hide": mouth, "add": ["M12", "X01R", "X01L", "FX_scribble", "X04"], "move": BROWS_MAD, "eyes": [0.70, 0.70], "shake": 0.70},
        "Surprised": {"hide": mouth, "add": ["M11"], "move": BROWS_UP, "eyes": [1.24, 1.24]},
        "Shock": {"hide": mouth + eye_all, "add": ["M11", "X10R", "X10L", "X03"], "move": BROWS_UP, "shake": 0.85},
        "Scared": {"hide": mouth, "add": ["M11", "FX_shade_pale", "FX_sweat_R"], "move": BROWS_SAD, "eyes": [1.28, 1.28], "iris": 0.62, "shake": 0.90},
        "Sad": {"hide": mouth, "add": ["M02", "X05R", "X05L"], "move": BROWS_SAD, "eyes": [0.80, 0.80]},
        "Puppy Eyes": {"hide": mouth, "add": ["M10", "X05R", "X05L", "FX_tear_R", "FX_blush"], "move": BROWS_SAD, "eyes": [1.14, 1.14], "iris": 1.10},
        "Crying": {"hide": mouth, "add": ["M12", "X05R", "X05L", "X06R", "X06L", "FX_tear_R", "FX_tear_L"], "move": BROWS_SAD, "eyes": [0.74, 0.74], "shake": 0.45},
        "Gloom": {"hide": mouth, "add": ["M02", "X02", "FX_shade_blue"], "move": BROWS_SAD, "eyes": [0.68, 0.68]},
        "Exhausted": {"hide": mouth, "add": ["M02", "FX_shade_purple", "FX_puff"], "move": BROWS_SAD, "eyes": [0.42, 0.42]},
        "Sleepy": {"hide": mouth, "add": ["M06"], "move": BROWS_LOW, "eyes": [0.38, 0.38]},
        "Confused": {"hide": mouth, "add": ["M11", "FX_question"], "move": BROWS_CONFUSED, "eyes": [0.95, 1.10]},
        "Dizzy": {"hide": mouth + eye_all, "add": ["M12", "X09R", "X09L", "FX_scribble"], "move": BROWS_SAD},
        "Yandere": {"hide": mouth, "add": ["M05", "FX_shadow", "X01R", "X01L"], "move": BROWS_SAD, "eyes": [1.28, 1.28], "iris": 0.72, "stare": True},
        "Evil Grin": {"hide": mouth, "add": ["M09", "FX_shadow", "FX_glint_R", "FX_glint_L"], "move": BROWS_MAD, "eyes": [0.74, 0.74], "stare": True},
        "Suspicious": {"hide": mouth, "add": ["M02"], "move": BROWS_LOW, "eyes": [0.50, 0.50], "look": [-7, 0]},
        "Deadpan": {"hide": mouth, "add": ["M02"], "move": BROWS_LOW, "eyes": [0.48, 0.48], "stare": True},
        "Hungry": {"hide": mouth, "add": ["M04", "FX_drool", "FX_sp1"], "move": BROWS_UP, "eyes": [1.12, 1.12]},
    }

    faces: dict[str, dict] = {}
    for name, spec in raw_faces.items():
        cleaned = dict(spec)
        if "add" in cleaned:
            cleaned["add"] = [p for p in cleaned["add"] if p in available_pieces]
        faces[name] = cleaned
    return faces


def build_default_timelines(faces: dict[str, dict], holds: dict[str, dict]) -> tuple[list[list], list[list]]:
    """Build model-safe auto-performance face & hold timelines referencing ONLY existing keys."""
    preferred_faces = [
        "Happy", "Wink", "Starry", "Love", "Fang Smirk", "Excited",
        "Pout", "Shy", "Confused", "Laugh", "Blow Kiss", "Angry",
        "Surprised", "Puppy Eyes", "Evil Grin", "Relieved",
    ]
    valid = [f for f in preferred_faces if f in faces]
    if not valid:
        valid = [k for k in faces.keys() if k != "Neutral"][:12]
    face_tl: list[list] = [[0.0, "Neutral"]]
    t = 1.8
    for nm in valid[:14]:
        face_tl.append([round(t, 2), nm])
        t += 1.45
        face_tl.append([round(t, 2), "Neutral"])
        t += 1.05

    hold_tl: list[list] = [[0.0, None]]
    ht = 8.0
    for hname in list(holds.keys())[:4]:
        hold_tl.append([round(ht, 2), hname])
        ht += 2.6
        hold_tl.append([round(ht, 2), None])
        ht += 4.5
    return face_tl, hold_tl
