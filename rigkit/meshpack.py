"""Build an interactive WebGL2 rig pack (viewer/pack.json + viewer/tex/*.webp) from placed/ layers.

Features:
- Signed 3D head depth parallax (negative depth for back hair -> shifts opposite the face on head turns!)
- Zero-pop turn keyforms from work/turn_keys.json (continuous single-mesh D-grid warps, zero opacity crossfades on hair/face)
- Full outfit/accessory toggles (including default-OFF toggles like glasses & wolf mask)
- Spring chains for kemonomimi ears, fluffy tail, hair locks, ribbons, tassels & skirt chains
- Expressions, blinks (squash + drawn closed eye swap), gaze tracking, and hand poses
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SAMPLES = [-1.0, -0.5, 0.0, 0.5, 1.0]
HEAD_KINDS = {"head", "eye", "iris", "lash", "brow", "hair", "ear", "wolf_ear", "accessory_head", "glasses"}
EYE_MODE = {"eye": 1, "iris": 3}
EYE_COLS = 24

PERFORMER = {
    "sway": [3.2, 0.009, 0.012],
    "stepBounce": 4.0,
    "breath": [3.6, 0.004],
    "bodyFollow": [0.12, 0.25],
    "bodyFollowLR": [0.025, 0.025],
    "bounces": [[7.0, 7.5, 8.0], 14.0, 0.12],
    "dur": 13.0,
    "earSoft": 5.0,
    "sew": 160.0,
    "limit": {"bend": 0.25, "lift": 0.06, "earring": 0.6, "arm": 0.08},
    "lockSigma": 50.0,
    "lockLower": [1.5, 0.8, 3.0],
    "springs": {
        "skirt": [1.1, 0.25, 0.4, 4.0, 0.4],
        "armR": [0.9, 0.35, 0.7, 8.0, 0.0],
        "armL": [0.9, 0.35, 0.7, 8.0, 0.0],
    },
    "timeline": {
        "a": [
            [0, 0], [0.9, 0], [1.7, -1], [3.0, -1], [4.3, 1], [5.6, 1],
            [6.5, 0], [9.0, 0], [9.45, -0.45], [9.95, 0.45], [10.4, 0], [13, 0],
        ],
        "roll": [
            [0, 0], [0.9, 0], [1.7, 3], [3.0, 3], [4.3, -3], [5.6, -3],
            [6.5, 0], [10.6, 0], [11.2, 9], [12.0, 9], [12.6, 0], [13, 0],
        ],
    },
}


def r4(v) -> list[float]:
    return [round(float(x), 4) for x in np.asarray(v, dtype=np.float64).ravel()]


def kind_of(L: dict) -> int:
    k, side = L.get("kind", "torso"), L.get("side")
    if k == "hair_body":
        return 1
    if k in HEAD_KINDS or (k == "ribbon" and not side):
        return 0
    if k in ("arm", "sleeve", "hand", "ribbon") and side:
        return 2 if side == "R" else 3
    if k in ("skirt", "overskirt", "hips", "tail"):
        return 5
    return 6 if k in ("leg", "sock", "boot", "foot") else 1


def eye_profile(placed_dir: Path, L: dict) -> dict:
    a = np.asarray(Image.open(placed_dir / f"{L['id']}.png").getchannel("A")) > 96
    x0, y0, x1, y1 = L["box"]
    xs = np.nonzero(a.any(0))[0]
    lo, hi = int(xs.min()), int(xs.max())
    cols = []
    for k in range(EYE_COLS):
        c = int(round(lo + (hi - lo) * k / (EYE_COLS - 1)))
        ys = np.nonzero(a[:, max(lo, c - 1) : c + 2].any(1))[0]
        top, bot = (y0 + int(ys.min()), y0 + int(ys.max())) if len(ys) else (y0 + a.shape[0] / 2.0,) * 2
        cols.append([float(top), float(bot), float(bot - top)])
    return {"x0": float(x0 + lo), "dx": float(hi - lo) / (EYE_COLS - 1), "cols": cols}


def build_pack(model_dir: Path, viewer_template: Path | None = None) -> dict:
    rig = json.loads((model_dir / "rig.json").read_text())
    placed = model_dir / "placed"
    out = model_dir / "viewer"
    tex_dir = out / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)
    for f in tex_dir.iterdir():
        if f.is_file():
            f.unlink()

    if viewer_template and viewer_template.exists():
        shutil.copy2(viewer_template, out / "index.html")
        for sib in ("arm3d.js", "mouth_keyshape.js", "mesh3d.js"):
            sib_p = viewer_template.parent / sib
            if sib_p.exists():
                shutil.copy2(sib_p, out / sib)

    meta = json.loads((placed / "layers.json").read_text())
    # Only include active/toggleable rig layers (skip PSD-only hidden swap parts if tagged `psd_only: true`)
    active_meta = [L for L in meta["layers"] if not L.get("psd_only", False)]
    by_id = {L["id"]: L for L in active_meta}

    tilt = math.radians(float(rig.get("tilt", 0.0)))
    eu, ev = [math.cos(tilt), -math.sin(tilt)], [math.sin(tilt), math.cos(tilt)]
    chain_names = list(rig.get("chains", {}).keys())

    # Support multi-sample turn_keys.json (e.g., 17 smooth samples across [-1.0..+1.0]) or default 5 samples
    tk_path = model_dir / "work/turn_keys.json"
    TK = json.loads(tk_path.read_text()) if tk_path.exists() else None
    sample_as = [float(x) for x in TK["samples_a"]] if (TK and "samples_a" in TK) else list(SAMPLES)

    layers = []
    samples = [{"angle": 0.0, "jaw": 0.0, "chin": 0.0, "layers": []} for _ in sample_as]

    for L in active_meta:
        lid = L["id"]
        hires_p = placed / f"{lid}_hires.png"
        im = Image.open(hires_p if hires_p.exists() else placed / f"{lid}.png").convert("RGBA")
        im.save(tex_dir / f"{lid}.webp", quality=93, method=5, alpha_quality=100)
        x0, y0, x1, y1 = L["box"]
        kind = kind_of(L)
        group = L.get("group")
        d = {
            "id": lid,
            "w": x1 - x0,
            "h": y1 - y0,
            "kind": kind,
            "head": kind == 0,
            "jaw": False,
            "noFade": bool(L.get("noFade", False)),
            "hole": False,
            "clip": L.get("clip"),
            "clipTarget": any(o.get("clip") == lid or lid in (o.get("behind") or []) for o in active_meta),
            "pad": float(L.get("pad", 320.0 if kind == 0 else 80.0)),
        }
        if L.get("behind"):
            d["behind"] = L["behind"]
        if L.get("sheer"):
            sheer_p = placed / f"{lid}_sheer.png"
            if sheer_p.exists():
                Image.open(sheer_p).save(tex_dir / f"{lid}_sheer.png", optimize=True)
                d["sheer"] = True

        chains = [k for k, n in enumerate(chain_names) if rig["chains"][n][1] == group] if group else []
        if chains:
            d["chains"] = chains
            d["chainSigma"] = rig.get("sigma", {}).get(group, 140.0)
            d["pad"] = max(d["pad"], 420.0)

        mode = 2 if lid in rig.get("upper_lash", []) else EYE_MODE.get(L.get("kind"))
        if mode and L.get("side"):
            d["eye"] = {"side": L["side"], "mode": mode, "A": [1, 0, 0, 1], "t": [(x0 + x1) / 2.0, (y0 + y1) / 2.0]}
            if mode == 1:
                im.save(tex_dir / f"{lid}_open.png", optimize=True)

        layers.append(d)
        for s, a in zip(samples, sample_as):
            entry = {"A": [1, 0, 0, 1], "t": [(x0 + x1) / 2.0, (y0 + y1) / 2.0], "op": 1.0}
            if kind == 0:
                # Signed 3D depth parallax: negative depth on back hair shifts opposite the face!
                turn_ang = rig["turn"]["R" if a < 0 else "L"]
                entry["turn"] = round(abs(a) * turn_ang * float(L.get("depth", 1.0)), 5)
            s["layers"].append(entry)

    for s, a in zip(samples, sample_as):
        s["angle"] = round(abs(a) * rig["turn"]["R" if a < 0 else "L"], 5)
        s["chinAt"] = [0.0, 0.0]

    # Apply fitted continuous turn keyforms (work/turn_keys.json)
    if TK:
        for j, L in enumerate(active_meta):
            if L["id"] in TK.get("hide", []):
                for s in samples:
                    s["layers"][j]["op"] = 0.0
        if "per_sample" in TK:
            # Multi-sample exact keyforms: per_sample[sample_idx][layer_id] -> {A, t, op, D?}
            for si, smap in enumerate(TK["per_sample"]):
                for j, L in enumerate(active_meta):
                    k = smap.get(L["id"])
                    if k:
                        samples[si]["layers"][j].update(k)
        else:
            for side, (half_idx, full_idx) in (("R", (1, 0)), ("L", (3, 4))):
                for j, L in enumerate(active_meta):
                    k = TK.get(side, {}).get(L["id"])
                    if not k:
                        continue
                    r_l, h_l, f_l = samples[2]["layers"][j], samples[half_idx]["layers"][j], samples[full_idx]["layers"][j]
                    f_l.update(A=k["A"], t=k["t"], op=k.get("op", 1.0))
                    h_l.update(
                        A=r4((np.array(r_l["A"]) + np.array(k["A"])) / 2.0),
                        t=r4((np.array(r_l["t"]) + np.array(k["t"])) / 2.0),
                        op=round((r_l["op"] + k.get("op", 1.0)) / 2.0, 4),
                    )
                    if "D" in k:
                        f_l["D"] = k["D"]
                        h_l["D"] = r4(np.array(k["D"]) / 2.0)
                    if "half" in k:
                        h_l.update(k["half"])
                    if "rest" in k:
                        r_l.update(k["rest"])
        for j in range(len(active_meta)):
            n_d = max((len(s["layers"][j].get("D", [])) for s in samples), default=0)
            if n_d:
                for s in samples:
                    s["layers"][j].setdefault("D", [0.0] * n_d)

    eyes = {}
    for side, lid in rig["eyes"].items():
        E = eye_profile(placed, by_id[lid])
        pupil = (by_id.get(f"E03{side}") or by_id.get(f"E02{side}"))["box"]
        E["c"] = [(pupil[0] + pupil[2]) / 2.0, (pupil[1] + pupil[3]) / 2.0]
        if rig.get("eye_pin"):
            E["pin"] = True
        eyes[side] = E

    chains = []
    for n in chain_names:
        parent, group, (k0, k1), damp, J = rig["chains"][n]
        chains.append(
            {
                "name": n,
                "parent": parent,
                "k": [k0, k1],
                "damp": damp,
                "joints": r4(J),
                "depth": float(rig.get("chain_depth", {}).get(group, 1.0)),
                "pin": n in rig.get("pin_tip", []),
                **({"hang": rig["chain_hang"][n]} if n in rig.get("chain_hang", {}) else {}),
            }
        )

    B = rig.get("body", {})
    shoulder = {"arm" + s: list(v) for s, v in rig["shoulder"].items()}
    motion = {
        **PERFORMER,
        "roll0": 0.0,
        "waist": rig["waist"],
        "footY": B["footY"],
        "legTop": B["legTop"],
        "shoulder": shoulder,
        "bodyC": rig["waist"][0],
        "bodyR": B["bodyR"],
        "twistY": B["twistY"],
        "skirtY": B["skirtY"],
        "seam": B["seam"],
        "armReach": B["armReach"],
        "hair": [],
        "field": [],
        "earrings": {},
        "locks": [],
    }
    for k in (
        "sway", "breath", "bodyFollow", "limit", "springs", "timeline",
        "elbow", "hand", "wrist", "turnBody", "bodyFollowLR",
        "shin", "elbowRound", "elbowLine", "sleeves",
        "hips", "puppetFace", "elbowHalf", "turnChain",
    ):
        if k in B:
            motion[k] = B[k]

    pack = {
        "title": rig.get("title", rig.get("name", "Layer2D Model")),
        "canvas": meta["canvas"],
        "bg": rig.get("bg", [0.94, 0.94, 0.96]),
        "views": rig["views"],
        "texExt": "webp",
        "samples": samples,
        "layers": layers,
        "head": {
            "centre": rig["centre"],
            "radius": rig["radius"],
            "eu": r4(eu),
            "ev": r4(ev),
            "chin": rig["chin"],
            "neck": rig["neck"],
            "jawNear": 150.0,
            "pivot": rig["neck_pivot"],
            "turn": rig["turn"],
        },
        "motion": motion,
        "skirt": {"y0": 0, "xl": [0.0], "xr": [1.0], "ml": [0.0], "mr": [0.0]},
        "expr": {
            "pieces": {},
            "faces": {},
            "holds": {},
            "raiseSleeves": {},
            "eyes": eyes,
            "blink": {"hide": [], "add": [], "drop": [], "squash": True},
        },
        "hairTint": {"base": [0.85, 0.0, 0.0], "chroma": [0.04, 0.08], "colours": {}},
        "eyeTint": {"base": [0.55, 0.16, 0.08], "colours": {}},
        "chains": chains,
        "chainWind": float(rig.get("chainWind", 1.2)),
        "sheer": {"op": 0.5, "boreal": 0.0, **rig.get("sheer", {})},
    }
    if "outfits" in rig:
        pack["outfits"] = rig["outfits"]
    if "tail" in rig:
        pack["tail"] = rig["tail"]
    if "hairShadowId" in rig:
        pack["hairShadowId"] = rig["hairShadowId"]
    if "highlightIds" in rig:
        pack["highlightIds"] = rig["highlightIds"]
    if "pitchGroups" in rig:
        pack["pitchGroups"] = rig["pitchGroups"]
    if "shadows" in rig:
        pack["shadows"] = rig["shadows"]

    (out / "pack.json").write_text(json.dumps(pack, separators=(",", ":")))
    tex_mb = sum(f.stat().st_size for f in tex_dir.iterdir() if f.is_file()) / 1e6
    pack_kb = (out / "pack.json").stat().st_size // 1024
    print(f"Packed {len(layers)} layers, {len(chains)} chains -> {out}/pack.json ({pack_kb} KB, tex {tex_mb:.1f} MB)")
    return {"layers": len(layers), "chains": len(chains), "pack_kb": pack_kb, "tex_mb": round(tex_mb, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--viewer", type=Path, default=Path(__file__).resolve().parent.parent / "viewer/index.html")
    a = ap.parse_args()
    build_pack(a.model_dir, viewer_template=a.viewer)


if __name__ == "__main__":
    main()
