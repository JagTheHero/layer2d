"""Layer2D Automated Model Polish & Attachment Auditor (`rigkit.polish`).

Quantitatively audits how polished a 2D avatar model is — checking limb alignment,
joint centerline continuity, width matching, fitted clothing & accessories (e.g.
choker-to-neck width & zero air overhang, belt-to-skirt, socks-to-shins & shoes,
shoulders/elbows/wrists), unblended cut-edge seams, floating islands / gaps,
bilateral L/R symmetry, and kinematic rig anchor consistency across ALL outfit
toggle states.

Works on EITHER:
  - A workspace model directory (`placed/layers.json` + `placed/*.png` + `rig.json`)
  - Any standalone `.l2d` archive (`pack.json` + `tex/*.webp|png`)

Usage:
    python -m rigkit.polish models/nori --out models/nori/review/polish
    python -m rigkit.polish models/test-signal/test-signal.l2d --out /tmp/ts_polish
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


ALPHA_THRESH = 28


@dataclass
class LayerCanvas:
    lid: str
    kind: str
    side: str | None
    clip: str | None
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) in canvas coords
    rgba: np.ndarray  # (h, w, 4) uint8 matching box
    mask: np.ndarray  # (h, w) bool (alpha > ALPHA_THRESH)
    visible_default: bool = True
    head: bool = False


@dataclass
class ModelBundle:
    name: str
    source: str
    canvas: tuple[int, int]  # (W, H)
    layers: list[LayerCanvas]
    by_id: dict[str, LayerCanvas]
    head: dict[str, Any]
    motion: dict[str, Any]
    tail: dict[str, Any] | None
    outfits: dict[str, dict[str, Any]]
    ref_paths: dict[str, Path] = field(default_factory=dict)
    all_layer_ids: set[str] = field(default_factory=set)
    sample_count: int = 1


@dataclass
class Finding:
    category: str
    check_id: str
    severity: str  # "pass" | "warn" | "fail"
    outfit_state: str
    title: str
    summary: str
    metrics: dict[str, Any]
    fix_hint: str = ""
    roi: tuple[int, int, int, int] | None = None  # canvas bbox for diagnostic crop
    guides: dict[str, Any] = field(default_factory=dict)


def _infer_kind_and_side(lid: str, raw_kind: Any = None, raw_side: Any = None) -> tuple[str, str | None]:
    side = raw_side if raw_side in ("R", "L") else None
    if side is None:
        if lid.endswith("R") or lid.endswith("R_in") or "_R" in lid:
            side = "R"
        elif lid.endswith("L") or lid.endswith("L_in") or "_L" in lid:
            side = "L"
        elif lid == "F05":
            side = "R"
        elif lid == "F06":
            side = "L"
    if isinstance(raw_kind, int):
        # pack.json numeric kinds: 0=head, 1=torso, 2=armR, 3=armL, 5=skirt/tail, 6=leg
        if raw_kind == 2:
            side = side or "R"
            return ("hand" if lid.startswith("G") else ("sleeve" if lid.startswith("J") else "arm")), side
        if raw_kind == 3:
            side = side or "L"
            return ("hand" if lid.startswith("G") else ("sleeve" if lid.startswith("J") else "arm")), side
        if raw_kind == 6:
            return "leg", side
        if raw_kind == 5:
            return ("tail" if lid.startswith("T") else "skirt"), side
        if raw_kind == 1:
            if lid.startswith("A"):
                return "accessory", side
            if lid.startswith("J"):
                return "outerwear", side
            return "torso", side
    if isinstance(raw_kind, str) and raw_kind:
        return raw_kind, side
    if lid.startswith("H") and not lid.startswith("HC"):
        return "hair", side
    if lid.startswith("HC"):
        return "accessory_head", side
    if lid.startswith("CE"):
        return "wolf_ear", side
    if lid in ("F05", "F06"):
        return "ear", side
    if lid.startswith("F") or lid.startswith("M"):
        return "head", side
    if lid.startswith("E"):
        return "eye", side
    if lid.startswith("T"):
        return "tail", side
    if lid.startswith("J"):
        return "sleeve" if side and lid[:3] in ("J04", "J05", "J06") else "outerwear", side
    if lid.startswith("G"):
        return "hand", side
    if lid.startswith("S"):
        return "leg", side
    if lid.startswith("K"):
        return "skirt", side
    if lid.startswith("A"):
        return "accessory", side
    if lid.startswith("C"):
        return "torso", side
    if lid.startswith("B"):
        if lid[:3] in ("B03", "B04", "B06", "B07", "B08") and side:
            return "arm", side
        if lid[:3] in ("B05", "B06", "B09", "B10", "B12") and side:
            return "leg", side
        return "torso", side
    return "other", side


def _transform_pack_layer(
    im: Image.Image,
    lw: float,
    lh: float,
    A: list[float],
    t: list[float],
    D: list[float] | None,
    W: int,
    H: int,
    ox_shift: int = 0,
    oy_shift: int = 0,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Place a pack.json texture onto canonical canvas space via p = A @ (l - [lw/2, lh/2]) + t
    (supporting 2x HiDPI textures where im.size != (lw, lh) and inverse UV warp grid D).
    """
    tw, th = im.size
    sx_tex = lw / max(1.0, float(tw))
    sy_tex = lh / max(1.0, float(th))
    # Effective matrix mapping pixel offsets (px - tw/2, py - th/2) -> canvas coords
    a00, a01, a10, a11 = float(A[0]) * sx_tex, float(A[1]) * sy_tex, float(A[2]) * sx_tex, float(A[3]) * sy_tex
    tx, ty = float(t[0]) + ox_shift, float(t[1]) + oy_shift

    # Compute exact output bounding box from transformed corners
    corners = np.array(
        [[-tw / 2, -th / 2], [tw / 2, -th / 2], [-tw / 2, th / 2], [tw / 2, th / 2]],
        dtype=np.float64,
    )
    M = np.array([[a00, a01], [a10, a11]], dtype=np.float64)
    pts = corners @ M.T + np.array([tx, ty])
    margin = int(math.ceil(max(lw, lh) * 0.14)) if D else 2
    x0 = max(0, int(math.floor(pts[:, 0].min())) - margin)
    y0 = max(0, int(math.floor(pts[:, 1].min())) - margin)
    x1 = min(W, int(math.ceil(pts[:, 0].max())) + margin)
    y1 = min(H, int(math.ceil(pts[:, 1].max())) + margin)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 1, 1), np.zeros((1, 1, 4), dtype=np.uint8)

    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-8:
        return (0, 0, 1, 1), np.zeros((1, 1, 4), dtype=np.uint8)

    if D and len(D) >= 18:
        # Exact inverse UV lookup with D grid (identical to viewer/index.html shader!)
        gd = int(round(math.sqrt(len(D) / 2)))
        if gd * gd * 2 == len(D):
            d_arr = np.asarray(D, dtype=np.float64).reshape(gd, gd, 2)
            # Map canvas grid (y0..y1, x0..x1) -> normalized UV [0..1] via inverse of A_orig (in lw, lh space)
            oa00, oa01, oa10, oa11 = [float(v) for v in A]
            odet = oa00 * oa11 - oa01 * oa10
            ia00, ia01, ia10, ia11 = oa11 / odet, -oa01 / odet, -oa10 / odet, oa00 / odet
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx = xx - tx
            dy = yy - ty
            u = (ia00 * dx + ia01 * dy) / max(1.0, lw) + 0.5
            v = (ia10 * dx + ia11 * dy) / max(1.0, lh) + 0.5
            # Sample D at clamped (v*(gd-1), u*(gd-1))
            gv = np.clip(v, 0.0, 1.0) * (gd - 1)
            gu = np.clip(u, 0.0, 1.0) * (gd - 1)
            du = ndimage.map_coordinates(d_arr[..., 0], [gv, gu], order=1, mode="nearest")
            dv = ndimage.map_coordinates(d_arr[..., 1], [gv, gu], order=1, mode="nearest")
            src_x = (u + du) * tw - 0.5
            src_y = (v + dv) * th - 0.5
            src = np.asarray(im, dtype=np.float32)
            out = np.stack(
                [ndimage.map_coordinates(src[..., c], [src_y, src_x], order=1, mode="constant", cval=0.0) for c in range(4)],
                axis=-1,
            )
            arr = np.clip(out, 0, 255).round().astype(np.uint8)
        else:
            D = None
    if not D:
        if abs(a00 - 1.0) < 1e-4 and abs(a01) < 1e-4 and abs(a10) < 1e-4 and abs(a11 - 1.0) < 1e-4:
            px0 = int(round(tx - tw / 2.0))
            py0 = int(round(ty - th / 2.0))
            src = np.asarray(im, dtype=np.uint8)
            cx0, cy0 = max(0, px0), max(0, py0)
            cx1, cy1 = min(W, px0 + tw), min(H, py0 + th)
            if cx1 <= cx0 or cy1 <= cy0:
                return (0, 0, 1, 1), np.zeros((1, 1, 4), dtype=np.uint8)
            arr = src[cy0 - py0 : cy1 - py0, cx0 - px0 : cx1 - px0].copy()
            x0, y0, x1, y1 = cx0, cy0, cx1, cy1
        else:
            ia00, ia01 = a11 / det, -a01 / det
            ia10, ia11 = -a10 / det, a00 / det
            ox, oy = x0 - tx, y0 - ty
            coeffs = (
                ia00,
                ia01,
                ia00 * ox + ia01 * oy + tw / 2.0,
                ia10,
                ia11,
                ia10 * ox + ia11 * oy + th / 2.0,
            )
            out_im = im.transform((x1 - x0, y1 - y0), Image.AFFINE, coeffs, Image.BILINEAR)
            arr = np.asarray(out_im, dtype=np.uint8)

    nz_y, nz_x = np.nonzero(arr[..., 3] > 4)
    if len(nz_y) == 0:
        return (x0, y0, x0 + 1, y0 + 1), np.zeros((1, 1, 4), dtype=np.uint8)
    ty0, ty1 = int(nz_y.min()), int(nz_y.max()) + 1
    tx0, tx1 = int(nz_x.min()), int(nz_x.max()) + 1
    trimmed = arr[ty0:ty1, tx0:tx1].copy()
    return (x0 + tx0, y0 + ty0, x0 + tx1, y0 + ty1), trimmed


def load_model(source: Path) -> ModelBundle:
    """Load layers & rig metadata from either a workspace model_dir or a standalone .l2d archive."""
    source = Path(source).resolve()

    if source.is_file() and source.suffix.lower() == ".l2d":
        with zipfile.ZipFile(source, "r") as zf:
            pack = json.loads(zf.read("pack.json").decode("utf-8"))
            cw, ch = int(pack["canvas"][0]), int(pack["canvas"][1])
            full_v = (pack.get("views") or {}).get("full")
            min_x = min(0, int(math.floor(full_v[0])) - 20) if full_v else 0
            min_y = min(0, int(math.floor(full_v[1])) - 20) if full_v else 0
            max_x = max(cw, int(math.ceil(full_v[2])) + 20) if full_v else cw
            max_y = max(ch, int(math.ceil(full_v[3])) + 20) if full_v else ch
            ox_shift, oy_shift = -min_x, -min_y
            W, H = max_x - min_x, max_y - min_y

            tex_ext = pack.get("texExt", "webp")
            samples = pack.get("samples", [])
            s0 = min(samples, key=lambda s: abs(float(s.get("angle", 0.0)))) if samples else None
            names_in_zip = set(zf.namelist())
            layers: list[LayerCanvas] = []
            for idx, L in enumerate(pack.get("layers", [])):
                lid = L["id"]
                rel = f"tex/{lid}.{tex_ext}"
                if rel not in names_in_zip:
                    alt = f"tex/{lid}.png"
                    rel = alt if alt in names_in_zip else f"tex/{lid}.webp"
                if rel not in names_in_zip:
                    continue
                im = Image.open(io.BytesIO(zf.read(rel))).convert("RGBA")
                sl = s0["layers"][idx] if (s0 and idx < len(s0["layers"])) else {}
                A = sl.get("A", [1.0, 0.0, 0.0, 1.0])
                t = sl.get("t", [cw / 2.0, ch / 2.0])
                D = sl.get("D")
                op = float(sl.get("op", 1.0))
                lw = float(L.get("w") or im.width)
                lh = float(L.get("h") or im.height)
                box, rgba = _transform_pack_layer(im, lw, lh, A, t, D, W, H, ox_shift, oy_shift)
                kind, side = _infer_kind_and_side(lid, L.get("kind"), (L.get("eye") or {}).get("side"))
                layers.append(
                    LayerCanvas(
                        lid=lid,
                        kind=kind,
                        side=side,
                        clip=L.get("clip"),
                        box=box,
                        rgba=rgba,
                        mask=rgba[..., 3] > ALPHA_THRESH,
                        visible_default=(op > 0.01),
                        head=bool(L.get("head", False)),
                    )
                )
        head = json.loads(json.dumps(pack.get("head") or {}))
        if "centre" in head:
            head["centre"] = [head["centre"][0] + ox_shift, head["centre"][1] + oy_shift]
        motion = json.loads(json.dumps(pack.get("motion") or {}))
        if "elbow" in motion and isinstance(motion["elbow"], dict):
            for k, v in motion["elbow"].items():
                if isinstance(v, list) and len(v) == 2:
                    motion["elbow"][k] = [v[0] + ox_shift, v[1] + oy_shift]
        tail = json.loads(json.dumps(pack.get("tail"))) if pack.get("tail") else None
        if tail and "pivot" in tail:
            tail["pivot"] = [tail["pivot"][0] + ox_shift, tail["pivot"][1] + oy_shift]
        outfits = pack.get("outfits") or {}
        return ModelBundle(
            name=pack.get("title") or source.stem,
            source=str(source),
            canvas=(W, H),
            layers=layers,
            by_id={L.lid: L for L in layers},
            head=head,
            motion=motion,
            tail=tail,
            outfits=outfits,
            all_layer_ids={L["id"] for L in pack.get("layers", [])},
            sample_count=len(pack.get("samples") or []),
        )

    # Workspace directory mode: prefer placed/layers.json + rig.json
    placed_json = source / "placed/layers.json"
    if placed_json.exists():
        meta = json.loads(placed_json.read_text(encoding="utf-8"))
        W, H = int(meta["canvas"][0]), int(meta["canvas"][1])
        rig = json.loads((source / "rig.json").read_text(encoding="utf-8")) if (source / "rig.json").exists() else {}
        pack = json.loads((source / "viewer/pack.json").read_text(encoding="utf-8")) if (source / "viewer/pack.json").exists() else {}
        layers = []
        for L in meta["layers"]:
            if L.get("psd_only", False) or not L.get("visible", True):
                continue
            lid = L["id"]
            p = source / "placed" / f"{lid}.png"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGBA")
            arr = np.asarray(im, dtype=np.uint8)
            x0, y0, x1, y1 = [int(v) for v in L["box"]]
            # Clip cleanly to canvas
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(W, x0 + arr.shape[1]), min(H, y0 + arr.shape[0])
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            sub = arr[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0].copy()
            kind, side = _infer_kind_and_side(lid, L.get("kind"), L.get("side"))
            is_head = kind in ("head", "eye", "iris", "lash", "brow", "hair", "ear", "wolf_ear", "accessory_head", "glasses")
            layers.append(
                LayerCanvas(
                    lid=lid,
                    kind=kind,
                    side=side,
                    clip=L.get("clip"),
                    box=(cx0, cy0, cx1, cy1),
                    rgba=sub,
                    mask=sub[..., 3] > ALPHA_THRESH,
                    visible_default=True,
                    head=is_head,
                )
            )
        # Prefer fresh rig.json when auditing workspace placed/ (fallback to viewer/pack.json if no rig.json)
        if rig:
            head = {
                "centre": rig.get("centre", [W / 2.0, H * 0.14]),
                "radius": rig.get("radius", W * 0.16),
                "chin": rig.get("chin", 250.0),
                "neck": rig.get("neck", 350.0),
                "neck_pivot": rig.get("neck_pivot", [W / 2.0, H * 0.21]),
            }
            motion = {
                "waist": rig.get("waist"),
                "shoulder": rig.get("shoulder"),
                **(rig.get("body") or {}),
            }
            tail = rig.get("tail")
            outfits = rig.get("outfits") or {}
        else:
            head = pack.get("head") or {"centre": [W / 2.0, H * 0.14], "radius": W * 0.16}
            motion = pack.get("motion") or {}
            tail = pack.get("tail")
            outfits = pack.get("outfits") or {}
        refs: dict[str, Path] = {}
        if (source / "refs/canon_full.png").exists():
            refs["default"] = source / "refs/canon_full.png"
        if (source / "refs/canon_no_cardigan.png").exists():
            refs["without_cardigan"] = source / "refs/canon_no_cardigan.png"
        return ModelBundle(
            name=rig.get("name") or source.name,
            source=str(source),
            canvas=(W, H),
            layers=layers,
            by_id={L.lid: L for L in layers},
            head=head,
            motion=motion,
            tail=tail,
            outfits=outfits,
            ref_paths=refs,
            all_layer_ids={L["id"] for L in meta["layers"] if not L.get("psd_only", False)},
            sample_count=len(pack.get("samples") or []) if pack else 1,
        )

    # Fallback: check if a .l2d exists inside model_dir
    l2ds = sorted(source.glob("*.l2d"))
    if l2ds:
        return load_model(l2ds[0])
    raise FileNotFoundError(f"Could not find placed/layers.json or .l2d archive at {source}")


def _composite_state(model: ModelBundle, hidden: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Composite visible layers into RGB (on white) + Alpha (0..255) canvas arrays."""
    W, H = model.canvas
    rgb = np.full((H, W, 3), 255.0, dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)
    clip_masks: dict[str, np.ndarray] = {}
    clip_targets = {L.clip for L in model.layers if L.clip}

    for L in model.layers:
        if not L.visible_default or L.lid in hidden:
            continue
        x0, y0, x1, y1 = L.box
        a = L.rgba[..., 3].astype(np.float32) / 255.0
        c = L.rgba[..., :3].astype(np.float32)
        if L.clip and L.clip in clip_masks:
            cm = clip_masks[L.clip][y0:y1, x0:x1]
            a = a * cm
        if L.lid in clip_targets:
            full_m = np.zeros((H, W), dtype=np.float32)
            full_m[y0:y1, x0:x1] = a
            clip_masks[L.lid] = full_m
        inv = 1.0 - a
        rgb[y0:y1, x0:x1] = c * a[..., None] + rgb[y0:y1, x0:x1] * inv[..., None]
        alpha[y0:y1, x0:x1] = a + alpha[y0:y1, x0:x1] * inv

    return np.clip(rgb, 0, 255).round().astype(np.uint8), np.clip(alpha * 255.0, 0, 255).round().astype(np.uint8)


def _layer_points(L: LayerCanvas, stride: int = 2) -> np.ndarray:
    """Return Nx2 float64 array of opaque pixel coordinates (x, y) in canvas space."""
    ys, xs = np.nonzero(L.mask[::stride, ::stride])
    if len(ys) == 0:
        ys, xs = np.nonzero(L.mask)
        if len(ys) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([xs + L.box[0], ys + L.box[1]], axis=1).astype(np.float64)
    return np.stack([xs * stride + L.box[0], ys * stride + L.box[1]], axis=1).astype(np.float64)


def _row_spans(L: LayerCanvas) -> dict[int, tuple[int, int]]:
    """Map canvas y -> (x_min, x_max) for every row where layer has opaque pixels."""
    x0, y0, _, _ = L.box
    spans: dict[int, tuple[int, int]] = {}
    h = L.mask.shape[0]
    any_row = L.mask.any(axis=1)
    for ry in np.nonzero(any_row)[0]:
        xs = np.nonzero(L.mask[ry])[0]
        spans[y0 + int(ry)] = (x0 + int(xs[0]), x0 + int(xs[-1]))
    return spans


def _layer_overlap_px(A: LayerCanvas, B: LayerCanvas) -> int:
    ix0 = max(A.box[0], B.box[0])
    iy0 = max(A.box[1], B.box[1])
    ix1 = min(A.box[2], B.box[2])
    iy1 = min(A.box[3], B.box[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    ma = A.mask[iy0 - A.box[1] : iy1 - A.box[1], ix0 - A.box[0] : ix1 - A.box[0]]
    mb = B.mask[iy0 - B.box[1] : iy1 - B.box[1], ix0 - B.box[0] : ix1 - B.box[0]]
    return int((ma & mb).sum())


def _union_roi(boxes: list[tuple[int, int, int, int]], pad: int, W: int, H: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(W, max(b[2] for b in boxes) + pad)
    y1 = min(H, max(b[3] for b in boxes) + pad)
    return (int(x0), int(y0), int(x1), int(y1))


def _joint_axis_metrics(
    P: LayerCanvas,
    C: LayerCanvas,
    rgb_comp: np.ndarray,
    *,
    force_vertical: bool = False,
) -> dict[str, Any]:
    """Measure oriented parent->child joint alignment along the parent's natural axis."""
    pts_p = _layer_points(P, stride=2)
    pts_c = _layer_points(C, stride=2)
    if len(pts_p) < 10 or len(pts_c) < 10:
        return {"valid": False}

    # Estimate parent primary axis via PCA pointing toward child centroid
    mean_p = pts_p.mean(axis=0)
    mean_c = pts_c.mean(axis=0)
    to_c = mean_c - mean_p
    if force_vertical:
        axis_p = np.array([0.0, 1.0 if to_c[1] >= 0 else -1.0], dtype=np.float64)
    else:
        cov_p = np.cov((pts_p - mean_p).T)
        evals, evecs = np.linalg.eigh(cov_p)
        axis_p = evecs[:, int(np.argmax(evals))]
        if np.dot(axis_p, to_c) < 0:
            axis_p = -axis_p

    # Child axis for angle-jump check
    cov_c = np.cov((pts_c - mean_c).T)
    evals_c, evecs_c = np.linalg.eigh(cov_c)
    axis_c = evecs_c[:, int(np.argmax(evals_c))]
    if np.dot(axis_c, axis_p) < 0:
        axis_c = -axis_c
    cos_ang = float(np.clip(np.dot(axis_p, axis_c), -1.0, 1.0))
    angle_jump_deg = math.degrees(math.acos(cos_ang))

    perp = np.array([-axis_p[1], axis_p[0]], dtype=np.float64)

    # Project both point clouds onto (u, v) centered at mean_p
    dp = pts_p - mean_p
    up = dp @ axis_p
    vp = dp @ perp
    dc = pts_c - mean_p
    uc = dc @ axis_p
    vc = dc @ perp

    u_p_min, u_p_max = float(np.percentile(up, 1)), float(np.percentile(up, 99))
    u_c_min, u_c_max = float(np.percentile(uc, 1)), float(np.percentile(uc, 99))
    len_p = max(1.0, u_p_max - u_p_min)
    len_c = max(1.0, u_c_max - u_c_min)

    gap_px = u_c_min - u_p_max  # >0 is literal air gap; <0 is overlap
    overlap_u = max(0.0, -gap_px)

    # Sample cross-section right near the parent end (last 18%..5% before tip to avoid rounded cap/fade)
    # and right near the child entry (first 5%..20% after entry)
    p_band = (up >= u_p_max - max(28.0, len_p * 0.22)) & (up <= u_p_max - max(8.0, len_p * 0.05))
    if p_band.sum() < 12:
        p_band = up >= u_p_max - max(24.0, len_p * 0.25)
    c_band = (uc >= u_c_min + max(8.0, len_c * 0.05)) & (uc <= u_c_min + max(28.0, len_c * 0.22))
    if c_band.sum() < 12:
        c_band = uc <= u_c_min + max(24.0, len_c * 0.25)

    vp_lo, vp_hi = float(np.percentile(vp[p_band], 3)), float(np.percentile(vp[p_band], 97))
    vc_lo, vc_hi = float(np.percentile(vc[c_band], 3)), float(np.percentile(vc[c_band], 97))
    w_p = max(1.0, vp_hi - vp_lo)
    w_c = max(1.0, vc_hi - vc_lo)
    v_p_mid = 0.5 * (vp_lo + vp_hi)
    v_c_mid = 0.5 * (vc_lo + vc_hi)

    # Also measure directly AT the overlap midpoint u_mid if they overlap significantly
    if overlap_u >= 18.0:
        u_mid = 0.5 * (u_c_min + u_p_max)
        half_w = max(10.0, min(22.0, overlap_u * 0.35))
        pb_ov = np.abs(up - u_mid) <= half_w
        cb_ov = np.abs(uc - u_mid) <= half_w
        if pb_ov.sum() >= 12 and cb_ov.sum() >= 12:
            vp_lo_o, vp_hi_o = float(np.percentile(vp[pb_ov], 3)), float(np.percentile(vp[pb_ov], 97))
            vc_lo_o, vc_hi_o = float(np.percentile(vc[cb_ov], 3)), float(np.percentile(vc[cb_ov], 97))
            w_p_ov = max(1.0, vp_hi_o - vp_lo_o)
            w_c_ov = max(1.0, vc_hi_o - vc_lo_o)
            # Use overlap slice when parent isn't heavily faded there
            if w_p_ov >= 0.65 * w_p and w_c_ov >= 0.65 * w_c:
                v_p_mid = 0.5 * (vp_lo_o + vp_hi_o)
                v_c_mid = 0.5 * (vc_lo_o + vc_hi_o)
                w_p, w_c = w_p_ov, w_c_ov

    offset_px = abs(v_c_mid - v_p_mid)
    offset_ratio = offset_px / max(w_p, 1.0)
    width_ratio = w_c / max(w_p, 1.0)

    # Seam-line ink detector: check luminance drop right across the parent end or child start line inside overlap
    seam_contrast = 0.0
    if overlap_u >= 6.0:
        H_c, W_c = rgb_comp.shape[:2]
        lum = rgb_comp.mean(axis=-1) / 255.0
        # Test both seam boundaries (u = u_p_max and u = u_c_min) across the interior 70% of the limb width
        best_seam = 0.0
        v_lo_i = max(vp_lo, vc_lo) + 0.15 * min(w_p, w_c)
        v_hi_i = min(vp_hi, vc_hi) - 0.15 * min(w_p, w_c)
        if v_hi_i > v_lo_i + 8.0:
            vs = np.linspace(v_lo_i, v_hi_i, 24)
            for u_seam in (u_p_max - 2.0, u_c_min + 2.0):
                drops = []
                for vv in vs:
                    # Sample luminance at seam vs +-10px along axis
                    p0 = mean_p + axis_p * u_seam + perp * vv
                    p_bef = mean_p + axis_p * (u_seam - 10.0) + perp * vv
                    p_aft = mean_p + axis_p * (u_seam + 10.0) + perp * vv
                    xs_i = [int(round(p[0])) for p in (p0, p_bef, p_aft)]
                    ys_i = [int(round(p[1])) for p in (p0, p_bef, p_aft)]
                    if all(0 <= x < W_c and 0 <= y < H_c for x, y in zip(xs_i, ys_i)):
                        l0 = lum[ys_i[0], xs_i[0]]
                        l_surr = 0.5 * (lum[ys_i[1], xs_i[1]] + lum[ys_i[2], xs_i[2]])
                        drops.append(max(0.0, l_surr - l0))
                if len(drops) >= 10:
                    # Fraction of cross-section with noticeable dark cut-line (>0.14 lum drop)
                    frac_dark = float(np.mean(np.array(drops) > 0.14))
                    med_drop = float(np.median(drops))
                    best_seam = max(best_seam, frac_dark * min(1.0, med_drop / 0.20))
        seam_contrast = best_seam

    # Compute joint anchor point in canvas space for diagnostic overlay
    u_joint = 0.5 * (u_p_max + u_c_min)
    p_end_xy = mean_p + axis_p * u_p_max + perp * v_p_mid
    c_start_xy = mean_p + axis_p * u_c_min + perp * v_c_mid
    joint_xy = mean_p + axis_p * u_joint + perp * (0.5 * (v_p_mid + v_c_mid))

    return {
        "valid": True,
        "gap_px": round(float(gap_px), 1),
        "overlap_px": round(float(overlap_u), 1),
        "offset_px": round(float(offset_px), 1),
        "offset_ratio": round(float(offset_ratio), 3),
        "parent_w": round(float(w_p), 1),
        "child_w": round(float(w_c), 1),
        "width_ratio": round(float(width_ratio), 3),
        "angle_jump_deg": round(float(angle_jump_deg), 1),
        "seam_line_score": round(float(seam_contrast), 3),
        "p_end_xy": [round(float(p_end_xy[0]), 1), round(float(p_end_xy[1]), 1)],
        "c_start_xy": [round(float(c_start_xy[0]), 1), round(float(c_start_xy[1]), 1)],
        "joint_xy": [round(float(joint_xy[0]), 1), round(float(joint_xy[1]), 1)],
    }


def _audit_connectivity(
    model: ModelBundle, state_name: str, hidden: set[str], alpha: np.ndarray
) -> list[Finding]:
    findings: list[Finding] = []
    W, H = model.canvas
    S = H / 4800.0
    solid = alpha > ALPHA_THRESH
    struct = np.ones((3, 3), dtype=np.int32)
    lbl, n_comp = ndimage.label(solid, structure=struct)
    if n_comp == 0:
        return [
            Finding(
                category="connectivity",
                check_id=f"{state_name}:empty",
                severity="fail",
                outfit_state=state_name,
                title="Empty Composite",
                summary="No opaque pixels rendered in composite.",
                metrics={},
            )
        ]

    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    main_id = int(np.argmax(counts))
    main_mask = lbl == main_id

    # 1. Floating / disconnected components (ignore tiny hair-tip dust < (6*S)^2 or within 10*S px of main)
    min_island_px = max(36, int(round((8.0 * S) ** 2)))
    dist_to_main = ndimage.distance_transform_edt(~main_mask)
    islands: list[dict[str, Any]] = []
    for cid in range(1, n_comp + 1):
        if cid == main_id or counts[cid] < min_island_px:
            continue
        cm = lbl == cid
        min_d = float(dist_to_main[cm].min())
        if min_d <= max(4.0, 10.0 * S):
            continue
        ys, xs = np.nonzero(cm)
        bx = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        # Find which visible layer owns this island
        owners = []
        for L in model.layers:
            if not L.visible_default or L.lid in hidden:
                continue
            ix0, iy0 = max(bx[0], L.box[0]), max(bx[1], L.box[1])
            ix1, iy1 = min(bx[2], L.box[2]), min(bx[3], L.box[3])
            if ix1 > ix0 and iy1 > iy0:
                ov = cm[iy0:iy1, ix0:ix1] & L.mask[iy0 - L.box[1] : iy1 - L.box[1], ix0 - L.box[0] : ix1 - L.box[0]]
                if ov.any():
                    owners.append(L.lid)
        islands.append({"area_px": int(counts[cid]), "gap_px": round(min_d, 1), "box": list(bx), "layers": owners})

    if islands:
        worst = max(islands, key=lambda d: d["area_px"])
        findings.append(
            Finding(
                category="connectivity",
                check_id=f"{state_name}:floating_parts",
                severity="fail",
                outfit_state=state_name,
                title=f"Detached / Floating Piece(s) [{state_name}]",
                summary=f"Found {len(islands)} detached silhouette island(s) separated from the main body (largest: {worst['layers']}, gap {worst['gap_px']}px, area {worst['area_px']}px²).",
                metrics={"island_count": len(islands), "islands": islands[:5]},
                fix_hint=f"Reconnect detached layer(s) {worst['layers']} so they overlap their parent layer by at least {int(round(12 * S))}px with zero air gap.",
                roi=_union_roi([tuple(worst["box"])], int(round(180 * S)), W, H),
            )
        )
    else:
        findings.append(
            Finding(
                category="connectivity",
                check_id=f"{state_name}:silhouette_connected",
                severity="pass",
                outfit_state=state_name,
                title=f"Silhouette Connected [{state_name}]",
                summary="Zero detached or floating body/clothing islands.",
                metrics={"main_area_px": int(counts[main_id]), "island_count": 0},
            )
        )

    # 2. Enclosed interior air-gap holes inside the silhouette (e.g. gap between sock & shin or under shoe strap)
    inv_lbl, n_holes = ndimage.label(~solid)
    border_labels = set(np.unique(inv_lbl[0, :])) | set(np.unique(inv_lbl[-1, :])) | set(np.unique(inv_lbl[:, 0])) | set(np.unique(inv_lbl[:, -1]))
    h_counts = np.bincount(inv_lbl.ravel())
    min_hole_px = max(20, int(round((6.0 * S) ** 2)))
    max_hole_px = int(round((140.0 * S) ** 2))
    suspicious_holes = []
    for hid in range(1, n_holes + 1):
        if hid in border_labels:
            continue
        area = int(h_counts[hid])
        if min_hole_px <= area <= max_hole_px:
            hm = inv_lbl == hid
            ys, xs = np.nonzero(hm)
            bx = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            # Exclude intentional jewelry loops (A*, K06, C03), arm-to-torso pockets, AND hair-to-shoulder air pockets!
            dil = ndimage.binary_dilation(hm, iterations=2)
            touching_body = []
            touches_hair = False
            for L in model.layers:
                if not L.visible_default or L.lid in hidden:
                    continue
                ix0, iy0 = max(bx[0] - 2, L.box[0]), max(bx[1] - 2, L.box[1])
                ix1, iy1 = min(bx[2] + 2, L.box[2]), min(bx[3] + 2, L.box[3])
                if ix1 > ix0 and iy1 > iy0:
                    ov = dil[iy0:iy1, ix0:ix1] & L.mask[iy0 - L.box[1] : iy1 - L.box[1], ix0 - L.box[0] : ix1 - L.box[0]]
                    if ov.any():
                        if L.kind == "hair":
                            touches_hair = True
                        elif not L.head:
                            touching_body.append(L.lid)
            body_touch = [lid for lid in touching_body if not lid.startswith("A") and lid != "K06" and lid != "C03"]
            touch_kinds = {model.by_id[lid].kind for lid in body_touch if lid in model.by_id}
            is_arm_torso_pocket = bool(touch_kinds & {"arm", "sleeve", "hand"}) and bool(touch_kinds & {"torso", "skirt", "outerwear"})
            if len(set(body_touch)) >= 2 and not is_arm_torso_pocket and not touches_hair:
                suspicious_holes.append({"area_px": area, "box": list(bx), "touching_layers": sorted(set(body_touch))})

    if suspicious_holes:
        worst_h = max(suspicious_holes, key=lambda d: d["area_px"])
        findings.append(
            Finding(
                category="connectivity",
                check_id=f"{state_name}:interior_gaps",
                severity="warn" if worst_h["area_px"] < int(round((28.0 * S) ** 2)) else "fail",
                outfit_state=state_name,
                title=f"Enclosed Inter-Layer Air Gap [{state_name}]",
                summary=f"Found {len(suspicious_holes)} enclosed air gap(s) between body/clothing layers (largest: {worst_h['area_px']}px² touching {worst_h['touching_layers']}).",
                metrics={"hole_count": len(suspicious_holes), "holes": suspicious_holes[:5]},
                fix_hint=f"Close gap between {worst_h['touching_layers']} by overlapping the pieces cleanly.",
                roi=_union_roi([tuple(worst_h["box"])], int(round(160 * S)), W, H),
            )
        )

    return findings


def _discover_joint_pairs(model: ModelBundle, hidden: set[str]) -> list[tuple[str, str, str, str]]:
    """Discover (parent_id, child_id, rel_type, label) joint pairs active in this outfit state."""
    vis = {L.lid for L in model.layers if L.visible_default and L.lid not in hidden}
    pairs: list[tuple[str, str, str, str]] = []

    def add_if(p: str, c: str, rel: str, lbl: str) -> None:
        if p in vis and c in vis:
            pairs.append((p, c, rel, lbl))

    # 1. Bare arms (Nori convention B03/B04/G01 or Test-Signal convention B06/B07/B08)
    for s in ("R", "L"):
        if f"B03{s}" in vis and f"B04{s}" in vis:
            # Only audit bare arm when not covered by cardigan sleeve J04/J05
            if f"J04{s}" not in vis:
                add_if("B02", f"B03{s}", "shoulder_attach", f"Shoulder {s} (B02->B03{s})")
                add_if(f"B03{s}", f"B04{s}", "skin_joint", f"Bare Elbow {s} (B03{s}->B04{s})")
                if f"G01{s}" in vis:
                    add_if(f"B04{s}", f"G01{s}", "wrist_joint", f"Bare Wrist {s} (B04{s}->G01{s})")
        elif f"B06{s}" in vis and f"B07{s}" in vis:
            # Test Signal naming: B06=upper arm, B07=forearm, B08=hand; C04=short sleeve covering top of B06
            if f"C04{s}" in vis:
                add_if(f"C04{s}", f"B06{s}", "cuff_to_arm", f"Sleeve {s} (C04{s}->B06{s})")
            add_if(f"B06{s}", f"B07{s}", "skin_joint", f"Elbow {s} (B06{s}->B07{s})")
            if f"B08{s}" in vis:
                add_if(f"B07{s}", f"B08{s}", "wrist_joint", f"Wrist {s} (B07{s}->B08{s})")

        # 2. Cardigan / jacket sleeves (J03->J04->J05->G01)
        if f"J04{s}" in vis and f"J05{s}" in vis:
            if f"J03{s}" in vis:
                add_if(f"J03{s}", f"J04{s}", "shoulder_attach", f"Jacket/Sweater Shoulder {s} (J03{s}->J04{s})")
            add_if(f"J04{s}", f"J05{s}", "sleeve_joint", f"Cardigan Elbow {s} (J04{s}->J05{s})")
            if f"G01{s}" in vis:
                add_if(f"J05{s}", f"G01{s}", "cuff_to_hand", f"Cardigan Cuff->Hand {s} (J05{s}->G01{s})")

        # 3. Legs, socks, shoes (Nori B05/B06/S01/S02 or Test Signal B09/B10/S01/S02)
        if f"B05{s}" in vis and f"B06{s}" in vis:
            add_if(f"B05{s}", f"B06{s}", "skin_joint", f"Knee {s} (B05{s}->B06{s})")
            if f"S01{s}" in vis:
                add_if(f"B06{s}", f"S01{s}", "leg_to_sock", f"Shin->Sock {s} (B06{s}->S01{s})")
                if f"S02{s}" in vis:
                    add_if(f"S01{s}", f"S02{s}", "sock_to_shoe", f"Sock->Shoe {s} (S01{s}->S02{s})")
            elif f"S02{s}" in vis:
                add_if(f"B06{s}", f"S02{s}", "sock_to_shoe", f"Shin->Shoe {s} (B06{s}->S02{s})")
        elif f"B09{s}" in vis and f"B10{s}" in vis:
            add_if(f"B09{s}", f"B10{s}", "skin_joint", f"Knee {s} (B09{s}->B10{s})")
            if f"S01{s}" in vis:
                add_if(f"B10{s}", f"S01{s}", "leg_to_sock", f"Shin->Sock {s} (B10{s}->S01{s})")
                if f"S02{s}" in vis:
                    add_if(f"S01{s}", f"S02{s}", "sock_to_shoe", f"Sock->Shoe {s} (S01{s}->S02{s})")

    return pairs


def _audit_partial_masking(
    model: ModelBundle, state_name: str, hidden: set[str]
) -> list[Finding]:
    """Detect cylindrical garments/bands (socks, sleeve cuffs, collars/bands) drawn ON TOP of an
    entering limb/neck that still contain an unmasked hollow back-rim opening over the skin!
    Proper 2D rigging requires partial masking: splitting the back inner rim behind the limb
    and keeping only the front lip curving in front of the limb.
    """
    findings: list[Finding] = []
    W, H = model.canvas
    S = H / 4800.0
    vis_order = [L.lid for L in model.layers if L.visible_default and L.lid not in hidden]
    vis_idx = {lid: i for i, lid in enumerate(vis_order)}

    # Candidate (entering_limb_id, covering_garment_id, entry_edge, label)
    candidates: list[tuple[str, str, str, str]] = []
    for s in ("R", "L"):
        for shin in (f"B06{s}", f"B10{s}"):
            if shin in vis_idx and f"S01{s}" in vis_idx and vis_idx[f"S01{s}"] > vis_idx[shin]:
                candidates.append((shin, f"S01{s}", "top", f"Sock {s} Partial Mask ({shin}->{f'S01{s}'})"))
        if f"G01{s}" in vis_idx and f"J05{s}" in vis_idx and vis_idx[f"J05{s}"] > vis_idx[f"G01{s}"]:
            candidates.append((f"G01{s}", f"J05{s}", "bot", f"Sleeve Cuff {s} Partial Mask (G01{s}->J05{s})"))
        if f"B06{s}" in vis_idx and f"C04{s}" in vis_idx and vis_idx[f"C04{s}"] > vis_idx[f"B06{s}"]:
            candidates.append((f"B06{s}", f"C04{s}", "bot", f"Short Sleeve {s} Partial Mask (B06{s}->C04{s})"))
    if "B01" in vis_idx:
        neck = model.by_id["B01"]
        for lid in vis_order:
            L = model.by_id[lid]
            if lid == "B01" or L.head or vis_idx[lid] <= vis_idx["B01"]:
                continue
            if (lid.startswith("A") or L.kind == "accessory") and _layer_overlap_px(L, neck) > 0.20 * max(1, int(L.mask.sum())):
                candidates.append(("B01", lid, "top", f"Neck Band Partial Mask (B01->{lid})"))

    for p_id, c_id, edge, label in candidates:
        P, C = model.by_id[p_id], model.by_id[c_id]
        ix0, ix1 = max(P.box[0], C.box[0]), min(P.box[2], C.box[2])
        ow = ix1 - ix0
        if ow < 16:
            continue
        cx0, cy0, cx1, cy1 = C.box
        cw, ch = cx1 - cx0, cy1 - cy0
        x_lo = ix0 + int(ow * 0.20)
        x_hi = ix1 - int(ow * 0.20)
        if x_hi <= x_lo + 6:
            continue

        rgb_f = C.rgba[..., :3].astype(np.float32) / 255.0
        lum = rgb_f.mean(axis=-1)
        # Pixel saturation (max-min)/max so saturated clothing stripes (e.g. rainbow sleeves) aren't confused with dark ink rims!
        cmax = rgb_f.max(axis=-1)
        cmin = rgb_f.min(axis=-1)
        sat = np.where(cmax > 0.05, (cmax - cmin) / np.maximum(cmax, 1e-5), 0.0)
        a = C.rgba[..., 3] > 40
        search_h = max(18, int(ch * 0.16))
        step = max(1, (x_hi - x_lo) // 32)

        ys_edge: list[tuple[int, int]] = []
        double_rim_hits = 0
        for x in range(x_lo, x_hi, step):
            rx = x - cx0
            col_ys = np.nonzero(a[:, rx])[0]
            if len(col_ys) < 16:
                continue
            if edge == "top":
                ye = int(col_ys[0])
                ys_edge.append((x, ye))
                col_lum = lum[ye : min(ch, ye + search_h), rx]
                col_sat = sat[ye : min(ch, ye + search_h), rx]
            else:
                ye = int(col_ys[-1])
                ys_edge.append((x, ye))
                col_lum = lum[max(0, ye - search_h) : ye + 1, rx][::-1]
                col_sat = sat[max(0, ye - search_h) : ye + 1, rx][::-1]
            if len(col_lum) > 16:
                rest = col_lum[10:]
                sec_idx = 10 + int(np.argmin(rest))
                sec_val = float(col_lum[sec_idx])
                sec_sat = float(col_sat[sec_idx])
                peak_mid = float(np.max(col_lum[4 : sec_idx + 1]))
                after_sl = col_lum[min(len(col_lum), sec_idx + 4) : min(len(col_lum), sec_idx + 14)]
                after_bright = float(np.max(after_sl)) if len(after_sl) else 0.0
                # Genuine front-lip ink line: dark (lum < 0.56), non-neon-stripe (sat < 0.55), bounded by hollow cavity & garment body!
                if sec_val < 0.56 and sec_sat < 0.55 and (peak_mid - sec_val > 0.14) and (after_bright - sec_val > 0.12):
                    double_rim_hits += 1

        if len(ys_edge) < 5:
            continue
        frac_double = double_rim_hits / len(ys_edge)
        y_ends = 0.5 * (ys_edge[0][1] + ys_edge[-1][1])
        y_mid = ys_edge[len(ys_edge) // 2][1]
        # Positive arch = outer edge curves AWAY from garment body (classic hollow back-wall ellipse!)
        arch_px = (y_ends - y_mid) if edge == "top" else (y_mid - y_ends)

        needs_mask = (frac_double >= 0.35 and arch_px >= 4.0 * S) or (frac_double >= 0.55 and arch_px >= 3.2 * S)
        sev = "fail" if needs_mask else ("warn" if (frac_double >= 0.30 and arch_px >= 2.8 * S) else "pass")

        # Crop tightly around the opening rim
        ey_canvas = cy0 + int(np.median([y for _, y in ys_edge]))
        pad_r = int(round(max(140 * S, ow * 0.85)))
        roi = (max(0, int(0.5 * (ix0 + ix1) - pad_r)), max(0, ey_canvas - pad_r), min(W, int(0.5 * (ix0 + ix1) + pad_r)), min(H, ey_canvas + pad_r))

        findings.append(
            Finding(
                category="partial_masking",
                check_id=f"{state_name}:mask_{c_id}",
                severity=sev,
                outfit_state=state_name,
                title=f"{label} [{state_name}]",
                summary=(
                    f"Clean front-lip occlusion (double_rim={frac_double*100:.0f}%, arch={arch_px:+.1f}px)."
                    if sev == "pass"
                    else f"Unmasked hollow back-rim opening drawn ON TOP of {p_id} (double_rim={frac_double*100:.0f}%, back-rim arch={arch_px:+.1f}px) — needs partial masking!"
                ),
                metrics={
                    "double_rim_frac": round(frac_double, 3),
                    "opening_arch_px": round(float(arch_px), 1),
                    "entry_edge": edge,
                },
                fix_hint=(
                    f"Split {c_id} along its front lip curve: move the hollow back inner rim into {c_id}_back BEHIND {p_id} and keep ONLY the front lip on {c_id} in front of {p_id}."
                    if sev != "pass"
                    else ""
                ),
                roi=roi,
            )
        )

    return findings


def _audit_limb_chains(
    model: ModelBundle, state_name: str, hidden: set[str], rgb_comp: np.ndarray
) -> list[Finding]:
    findings: list[Finding] = []
    W, H = model.canvas
    S = H / 4800.0
    pairs = _discover_joint_pairs(model, hidden)

    for p_id, c_id, rel, label in pairs:
        P, C = model.by_id[p_id], model.by_id[c_id]
        roi = _union_roi([P.box, C.box], int(round(60 * S)), W, H)

        if rel == "shoulder_attach":
            # Audit bare shoulder: (1) overlap, (2) dark cut-seam, (3) top shoulder silhouette V-notch/gap,
            # AND (4) interior alpha translucency gap (white background bleeding through shoulder/armpit)!
            ov_px = _layer_overlap_px(P, C)
            c_area = max(1, int(C.mask.sum()))
            ov_ratio = ov_px / c_area
            ix0, iy0 = max(P.box[0], C.box[0]), max(P.box[1], C.box[1])
            ix1, iy1 = min(P.box[2], C.box[2]), min(P.box[3], C.box[3])
            # Composite ONLY P (torso/jacket front) + C (upper arm/sleeve) without hair to inspect TRUE shoulder seam,
            # top contour V-notch & interior alpha solidity!
            sh_x0 = max(0, min(P.box[0], C.box[0]))
            sh_y0 = max(0, min(P.box[1], C.box[1]))
            sh_x1 = min(W, max(P.box[2], C.box[2]))
            sh_y1 = min(H, min(P.box[3], C.box[1] + int((C.box[3] - C.box[1]) * 0.65)))
            sh_h = max(1, sh_y1 - sh_y0)
            sh_w = max(1, sh_x1 - sh_x0)
            comb_a = np.zeros((sh_h, sh_w), dtype=np.float32)
            comb_rgb = np.full((sh_h, sh_w, 3), 255.0, dtype=np.float32)
            c_idx = next(i for i, L in enumerate(model.layers) if L.lid == c_id)
            p_idx = next(i for i, L in enumerate(model.layers) if L.lid == p_id)
            draw_pair = (P, C) if c_idx > p_idx else (C, P)
            for L_sub in draw_pair:
                bx0, by0, bx1, by1 = max(sh_x0, L_sub.box[0]), max(sh_y0, L_sub.box[1]), min(sh_x1, L_sub.box[2]), min(sh_y1, L_sub.box[3])
                if bx1 > bx0 and by1 > by0:
                    sub_rgba = L_sub.rgba[by0 - L_sub.box[1] : by1 - L_sub.box[1], bx0 - L_sub.box[0] : bx1 - L_sub.box[0]]
                    la = sub_rgba[..., 3].astype(np.float32) / 255.0
                    lc = sub_rgba[..., :3].astype(np.float32)
                    sl_y = slice(by0 - sh_y0, by1 - sh_y0)
                    sl_x = slice(bx0 - sh_x0, bx1 - sh_x0)
                    cur_a = comb_a[sl_y, sl_x]
                    comb_rgb[sl_y, sl_x] = lc * la[..., None] + comb_rgb[sl_y, sl_x] * (1.0 - la[..., None])
                    comb_a[sl_y, sl_x] = la + cur_a * (1.0 - la)
            comb_u8 = np.clip(comb_a * 255.0, 0, 255).round().astype(np.uint8)

            seam_score = 0.0
            if ix1 > ix0 and iy1 > iy0:
                iy1_sh = min(sh_y1, iy0 + max(12, int((iy1 - iy0) * (0.62 if p_id.startswith("J03") else 1.0))))
                if iy1_sh > iy0:
                    mp = P.mask[iy0 - P.box[1] : iy1_sh - P.box[1], ix0 - P.box[0] : ix1 - P.box[0]]
                    mc = C.mask[iy0 - C.box[1] : iy1_sh - C.box[1], ix0 - C.box[0] : ix1 - C.box[0]]
                    ov = mp & mc
                    er_in = max(4, int(round(8 * S)))
                    lum = comb_rgb[iy0 - sh_y0 : iy1_sh - sh_y0, ix0 - sh_x0 : ix1 - sh_x0].mean(axis=-1) / 255.0
                    if ov.sum() > 20 and c_idx > p_idx:
                        c_border = mc & ~ndimage.binary_erosion(mc, iterations=3) & ndimage.binary_erosion(mp, iterations=er_in)
                        if c_border.sum() > 15:
                            inner_skin = ov & ~c_border
                            if inner_skin.sum() > 20:
                                seam_score = float(max(0.0, np.median(lum[inner_skin]) - np.percentile(lum[c_border], 25)))
                    elif ov.sum() > 20 and p_idx > c_idx:
                        p_border = mp & ~ndimage.binary_erosion(mp, iterations=3) & ndimage.binary_erosion(mc, iterations=er_in)
                        # For jacket/cardigan front panels (J03R/L), inspect ONLY the outer armhole edge
                        # (exclude the chest-side placket/collar trim which intentionally has dark ink!).
                        if p_id.startswith("J03"):
                            bw = p_border.shape[1]
                            if (C.side or ("R" if c_id.endswith("R") else "L")) == "R":
                                p_border[:, int(bw * 0.55) :] = False
                            else:
                                p_border[:, : int(bw * 0.45)] = False
                        if p_border.sum() > 15:
                            inner_skin = ov & ~p_border
                            if inner_skin.sum() > 20:
                                seam_score = float(max(0.0, np.median(lum[inner_skin]) - np.percentile(lum[p_border], 25)))

            # Trace top silhouette ridge across the shoulder joint x-band (around the junction between torso & upper arm)
            side_c = C.side or ("R" if (C.box[0] + C.box[2]) < (P.box[0] + P.box[2]) else "L")
            j_cx = int(round(0.5 * (ix0 + ix1))) if ix1 > ix0 else int(round(0.5 * (C.box[0] + C.box[2])))
            scan_x0 = max(sh_x0 + 4, j_cx - int(round(120 * S)))
            scan_x1 = min(sh_x1 - 4, j_cx + int(round(120 * S)))
            solid_sh = comb_u8 > 40
            ridge_pts: list[tuple[int, int]] = []
            min_int_alpha = 255
            for x in range(scan_x0, scan_x1, max(1, int(round(4 * S)))):
                rx = x - sh_x0
                ys = np.nonzero(solid_sh[:, rx])[0]
                if len(ys) > 25:
                    yt = int(ys[0])
                    ridge_pts.append((x, sh_y0 + yt))
                    # Check alpha solidity inside the shoulder 10..40px below the top ridge (excluding outer 12px border)
                    if abs(x - j_cx) <= int(round(65 * S)):
                        sub_a = comb_u8[yt + max(6, int(round(8 * S))) : min(sh_h, yt + int(round(55 * S))), rx]
                        if len(sub_a):
                            min_int_alpha = min(min_int_alpha, int(sub_a.min()))

            # Detect V-notch dip along the top shoulder ridge: compare each point y against upper convex envelope!
            notch_depth_px = 0.0
            if len(ridge_pts) >= 7:
                rxs = np.array([p[0] for p in ridge_pts], dtype=np.float64)
                rys = np.array([p[1] for p in ridge_pts], dtype=np.float64)
                # Walk inward->outward (from torso toward arm); a smooth convex shoulder has y <= chord between left & right neighbors!
                for i in range(2, len(rxs) - 2):
                    # Best (highest / smallest y) point on left and right within 80*S px
                    left_i = [j for j in range(0, i) if rxs[i] - rxs[j] <= 95 * S]
                    right_i = [j for j in range(i + 1, len(rxs)) if rxs[j] - rxs[i] <= 95 * S]
                    if left_i and right_i:
                        jL = min(left_i, key=lambda j: rys[j])
                        jR = min(right_i, key=lambda j: rys[j])
                        t_ch = (rxs[i] - rxs[jL]) / max(1.0, rxs[jR] - rxs[jL])
                        y_chord = rys[jL] * (1.0 - t_ch) + rys[jR] * t_ch
                        dip = rys[i] - y_chord  # positive = downward V-notch indentation into shoulder!
                        if dip > notch_depth_px:
                            notch_depth_px = float(dip)

            # For outer jacket/sweater shoulder joins (J03->J04), also catch football-pad humps where
            # the upper sleeve C rises above the jacket/cardigan front panel P's collar/shoulder crest!
            sleeve_hump_px = 0.0
            if p_id.startswith("J03"):
                p_ys = np.nonzero(P.mask.any(axis=1))[0]
                c_ys = np.nonzero(C.mask.any(axis=1))[0]
                if len(p_ys) and len(c_ys):
                    sleeve_hump_px = float((P.box[1] + int(p_ys[0])) - (C.box[1] + int(c_ys[0])))

            issues = []
            sev = "pass"
            if ov_px < int(round(120 * S * S)):
                sev = "fail"
                issues.append(f"barely overlaps torso ({ov_px}px²)")
            if sleeve_hump_px > 4.0 * S:
                sev = "fail" if sleeve_hump_px > 10.0 * S else ("warn" if sev == "pass" else sev)
                issues.append(f"sleeve shoulder cap pokes {sleeve_hump_px:.1f}px higher than {p_id} collar/shoulder slope")
            if seam_score > 0.22:
                sev = "fail" if seam_score > 0.32 else ("warn" if sev == "pass" else sev)
                issues.append(f"visible dark shoulder cut-seam (contrast={seam_score:.2f})")
            if notch_depth_px > 9.0 * S:
                sev = "fail" if notch_depth_px > 14.0 * S else ("warn" if sev == "pass" else sev)
                issues.append(f"top shoulder V-notch gap ({notch_depth_px:.1f}px dip)")
            if min_int_alpha < 225:
                sev = "fail" if min_int_alpha < 180 else ("warn" if sev == "pass" else sev)
                issues.append(f"translucent shoulder gap/hole (min_alpha={min_int_alpha}/255)")

            sh_roi = (
                max(0, j_cx - int(round(190 * S))),
                max(0, min(P.box[1], C.box[1]) - int(round(40 * S))),
                min(W, j_cx + int(round(190 * S))),
                min(H, min(P.box[1], C.box[1]) + int(round(360 * S))),
            )
            findings.append(
                Finding(
                    category="limb_chains",
                    check_id=f"{state_name}:{p_id}_{c_id}",
                    severity=sev,
                    outfit_state=state_name,
                    title=f"{label} [{state_name}]",
                    summary=(
                        f"Clean solid shoulder contour (notch={notch_depth_px:.1f}px, min_alpha={min_int_alpha})."
                        if not issues
                        else f"Shoulder gap/seam issue: {'; '.join(issues)}."
                    ),
                    metrics={
                        "overlap_px": ov_px,
                        "overlap_ratio": round(ov_ratio, 3),
                        "seam_contrast": round(seam_score, 3),
                        "notch_depth_px": round(notch_depth_px, 1),
                        "min_interior_alpha": int(min_int_alpha),
                    },
                    fix_hint=(
                        f"Seat {c_id} flush into {p_id}'s shoulder contour with 100% opaque top ridge & armpit (fade ONLY inside {p_id}'s opaque interior) so notch=0px and alpha=255."
                        if issues
                        else ""
                    ),
                    roi=sh_roi,
                )
            )
            continue

        if rel == "leg_to_sock":
            # Compare shin RIGHT AT the sock's top rim (NOT at the shin's lower ankle tip inside the sock!)
            p_sp = _row_spans(P)
            c_sp = _row_spans(C)
            if not p_sp or not c_sp:
                continue
            sock_top_y = min(c_sp)
            sock_h = max(1, max(c_sp) - sock_top_y)
            # Sample sock rim (top 2%..10% of sock height)
            rim_ys = [y for y in range(sock_top_y + max(2, int(sock_h * 0.02)), sock_top_y + max(10, int(sock_h * 0.10))) if y in c_sp]
            if not rim_ys:
                rim_ys = sorted(c_sp)[:10]
            sock_w = float(np.median([c_sp[y][1] - c_sp[y][0] for y in rim_ys]))
            sock_cx = float(np.median([0.5 * (c_sp[y][0] + c_sp[y][1]) for y in rim_ys]))

            # Check shin coverage immediately above sock_top_y (sock_top_y - 40*S .. sock_top_y + 15*S)
            # AND verify ZERO visible white/transparent gap right above the sock rim!
            above_band = range(max(0, sock_top_y - int(round(35 * S))), sock_top_y)
            # Check composite alpha right above sock rim across the central 65% of sock rim width
            x_chk0 = max(0, int(round(sock_cx - sock_w * 0.34)))
            x_chk1 = min(W, int(round(sock_cx + sock_w * 0.34)))
            y_chk0 = max(0, sock_top_y - max(6, int(round(18 * S))))
            y_chk1 = sock_top_y + 2
            # Detect white/empty gap right above sock rim from rgb_comp (pure white >=252) OR missing shin pixels
            shin_above_ys = [y for y in range(sock_top_y - int(round(65 * S)), sock_top_y + int(round(12 * S))) if y in p_sp]
            shin_at_rim_ys = [y for y in range(sock_top_y - int(round(14 * S)), sock_top_y + int(round(18 * S))) if y in p_sp]
            gap_above_px = 0
            if not shin_at_rim_ys:
                gap_above_px = sock_top_y - max(p_sp) if p_sp else 999
            else:
                # Check if shin alpha faded out right above sock rim (leaving visible gap in composite)
                if y_chk1 > y_chk0 and x_chk1 > x_chk0:
                    strip = rgb_comp[y_chk0:y_chk1, x_chk0:x_chk1]
                    white_frac = float(((strip[..., 0] > 248) & (strip[..., 1] > 248) & (strip[..., 2] > 248)).mean())
                    if white_frac > 0.12:
                        gap_above_px = max(gap_above_px, int(round(white_frac * (y_chk1 - y_chk0) * 1.5)))

            ref_ys = shin_at_rim_ys or shin_above_ys
            if ref_ys:
                shin_w = float(np.median([p_sp[y][1] - p_sp[y][0] for y in ref_ys]))
                shin_cx = float(np.median([0.5 * (p_sp[y][0] + p_sp[y][1]) for y in ref_ys]))
            else:
                shin_w = float(P.box[2] - P.box[0])
                shin_cx = 0.5 * (P.box[0] + P.box[2])

            # Check shin overlap depth inside sock AND whether shin pokes sideways outside sock below rim
            overlap_y = max(p_sp) - sock_top_y
            poke_px = 0
            for y in range(sock_top_y + int(round(12 * S)), min(max(p_sp), max(c_sp)) + 1):
                if y in p_sp and y in c_sp:
                    poke_px = max(poke_px, max(0, c_sp[y][0] - p_sp[y][0]), max(0, p_sp[y][1] - c_sp[y][1]))

            w_ratio = sock_w / max(shin_w, 1.0)
            cx_err = abs(sock_cx - shin_cx)
            cx_ratio = cx_err / max(shin_w, 1.0)

            issues = []
            sev = "pass"
            if gap_above_px > 2:
                sev = "fail"
                issues.append(f"visible gap/fade above sock cuff (~{gap_above_px}px)")
            if overlap_y < 14 * S:
                sev = "fail"
                issues.append(f"shin barely enters sock (overlap={overlap_y}px)")
            if poke_px > 8 * S:
                sev = "fail"
                issues.append(f"shin pokes {poke_px}px sideways outside sock below cuff")
            if w_ratio < 0.94:
                sev = "fail"
                issues.append(f"sock rim narrower than shin ({sock_w:.0f}px vs {shin_w:.0f}px)")
            elif w_ratio > 1.24:
                sev = "fail" if w_ratio > 1.32 else ("warn" if sev == "pass" else sev)
                issues.append(f"sock rim too wide for shin ({sock_w:.0f}px vs {shin_w:.0f}px, ratio {w_ratio:.2f}x)")
            if cx_ratio > 0.12:
                sev = "fail" if cx_ratio > 0.20 else ("warn" if sev == "pass" else sev)
                issues.append(f"sock off-center vs shin by {cx_err:.1f}px ({cx_ratio*100:.0f}%)")

            pad_y = int(round(220 * S))
            joint_roi = (
                max(0, int(min(P.box[0], C.box[0]) - 80 * S)),
                max(0, int(sock_top_y - pad_y)),
                min(W, int(max(P.box[2], C.box[2]) + 80 * S)),
                min(H, int(sock_top_y + pad_y)),
            )
            findings.append(
                Finding(
                    category="limb_chains",
                    check_id=f"{state_name}:{p_id}_{c_id}",
                    severity=sev,
                    outfit_state=state_name,
                    title=f"{label} [{state_name}]",
                    summary=(
                        f"Shin enters sock snugly (sock={sock_w:.0f}px vs shin={shin_w:.0f}px, ratio={w_ratio:.2f}x, offset={cx_err:.1f}px)."
                        if not issues
                        else f"Sock cuff issue: {'; '.join(issues)}."
                    ),
                    metrics={
                        "shin_w_at_rim": round(shin_w, 1),
                        "sock_rim_w": round(sock_w, 1),
                        "width_ratio": round(w_ratio, 3),
                        "center_offset_px": round(cx_err, 1),
                        "gap_above_px": int(gap_above_px),
                        "overlap_y_px": int(overlap_y),
                        "side_poke_px": int(poke_px),
                    },
                    fix_hint=(
                        f"Extend {p_id} solidly into {c_id} (no fade above y={sock_top_y}), match {c_id} top width to ~{shin_w*1.06:.0f}px centered at x={shin_cx:.0f}."
                        if issues
                        else ""
                    ),
                    roi=joint_roi,
                    guides={"p_end_xy": [round(shin_cx, 1), float(sock_top_y - 10)], "c_start_xy": [round(sock_cx, 1), float(sock_top_y + 10)]},
                )
            )
            continue

        if rel == "sock_to_shoe":
            # Vertical penetration + horizontal centering of sock/shin entering shoe opening
            s_spans = _row_spans(P)
            sh_spans = _row_spans(C)
            if not s_spans or not sh_spans:
                continue
            sock_bot_y = max(s_spans.keys())
            shoe_top_y = min(sh_spans.keys())
            shoe_bot_y = max(sh_spans.keys())
            shoe_h = max(1, shoe_bot_y - shoe_top_y)
            penetration_px = sock_bot_y - shoe_top_y
            penetration_ratio = penetration_px / shoe_h

            # Measure horizontal alignment near shoe top opening (top 22% of shoe)
            open_ys = [y for y in range(shoe_top_y, shoe_top_y + max(10, int(shoe_h * 0.22))) if y in sh_spans]
            sock_ys = [y for y in range(max(min(s_spans), shoe_top_y - int(round(90 * S))), min(sock_bot_y + 1, shoe_top_y + int(round(60 * S)))) if y in s_spans]
            sh_cx = float(np.mean([0.5 * (sh_spans[y][0] + sh_spans[y][1]) for y in open_ys])) if open_ys else 0.5 * (C.box[0] + C.box[2])
            sh_w = float(np.median([sh_spans[y][1] - sh_spans[y][0] for y in open_ys])) if open_ys else float(C.box[2] - C.box[0])
            s_cx = float(np.mean([0.5 * (s_spans[y][0] + s_spans[y][1]) for y in sock_ys[-30:]])) if sock_ys else 0.5 * (P.box[0] + P.box[2])
            s_w = float(np.median([s_spans[y][1] - s_spans[y][0] for y in sock_ys[-30:]])) if sock_ys else float(P.box[2] - P.box[0])
            cx_err = abs(s_cx - sh_cx)
            cx_ratio = cx_err / max(sh_w, 1.0)

            # Check FULL wall-to-wall open arch cutout inside shoe (e.g. Mary Jane U-arch under buckle strap)!
            # Any enclosed transparent hole inside C's mask (between left/right walls & strap) MUST be 100% filled by P,
            # AND P's own dark outer outline should NOT cut down vertically through the interior of that open arch!
            c_filled = ndimage.binary_fill_holes(C.mask)
            arch_hole = c_filled & ~C.mask
            arch_area = int(arch_hole.sum())
            shoe_has_open_arch = arch_area > max(120, int(round((24.0 * S) ** 2)))
            unfilled_window = 0
            arch_inner_outline_frac = 0.0
            if shoe_has_open_arch:
                p_in_c = np.zeros_like(C.mask)
                iy0, iy1 = max(C.box[1], P.box[1]), min(C.box[3], P.box[3])
                ix0, ix1 = max(C.box[0], P.box[0]), min(C.box[2], P.box[2])
                if iy1 > iy0 and ix1 > ix0:
                    p_in_c[iy0 - C.box[1] : iy1 - C.box[1], ix0 - C.box[0] : ix1 - C.box[0]] = P.mask[
                        iy0 - P.box[1] : iy1 - P.box[1], ix0 - P.box[0] : ix1 - P.box[0]
                    ]
                unfilled_window = int((arch_hole & ~p_in_c).sum())
                # Also check if P's own dark side outline cuts INSIDE the open arch (meaning sock stops short of shoe inner wall!)
                arch_core = ndimage.binary_erosion(arch_hole, iterations=max(2, int(round(6 * S))))
                if arch_core.sum() > 40:
                    p_lum_in_c = np.full(C.mask.shape, 1.0, dtype=np.float32)
                    if iy1 > iy0 and ix1 > ix0:
                        sub_rgba = P.rgba[iy0 - P.box[1] : iy1 - P.box[1], ix0 - P.box[0] : ix1 - P.box[0]]
                        p_lum_in_c[iy0 - C.box[1] : iy1 - C.box[1], ix0 - C.box[0] : ix1 - C.box[0]] = sub_rgba[..., :3].mean(axis=-1) / 255.0
                    # Check per-row inside arch_core: is there a sharp dark line (lum < 0.72) inside a bright sock arch?
                    a_ys = np.nonzero(arch_core.any(axis=1))[0]
                    med_arch_lum = float(np.median(p_lum_in_c[arch_core & p_in_c])) if (arch_core & p_in_c).any() else 0.0
                    if med_arch_lum > 0.80 and len(a_ys) >= 6:
                        dark_rows = sum(1 for ry in a_ys if float(p_lum_in_c[ry, arch_core[ry]].min()) < med_arch_lum - 0.16)
                        arch_inner_outline_frac = dark_rows / len(a_ys)

            issues = []
            sev = "pass"
            if penetration_px < 4 * S:
                sev = "fail"
                issues.append(f"sock does not enter shoe (gap/overlap={penetration_px}px)")
            elif shoe_has_open_arch and penetration_ratio < 0.38:
                sev = "fail"
                issues.append(f"open-arch shoe needs deeper sock tuck ({penetration_ratio*100:.0f}%, needs >=40%)")
            if unfilled_window > max(40, int(round((10.0 * S) ** 2))):
                sev = "fail"
                issues.append(f"unfilled gap inside shoe arch opening ({unfilled_window}px² / {arch_area}px²)")
            if arch_inner_outline_frac > 0.30:
                sev = "fail"
                issues.append(f"sock side outline cuts inside shoe arch ({arch_inner_outline_frac*100:.0f}% of arch rows) — sock doesn't fill shoe wall-to-wall")
            if cx_ratio > 0.16:
                sev = "fail" if cx_ratio > 0.24 else ("warn" if sev == "pass" else sev)
                issues.append(f"ankle off-center vs shoe by {cx_err:.1f}px ({cx_ratio*100:.0f}%)")

            joint_roi = _union_roi([C.box, (P.box[0], max(P.box[1], shoe_top_y - int(round(200 * S))), P.box[2], P.box[3])], int(round(80 * S)), W, H)
            findings.append(
                Finding(
                    category="limb_chains",
                    check_id=f"{state_name}:{p_id}_{c_id}",
                    severity=sev,
                    outfit_state=state_name,
                    title=f"{label} [{state_name}]",
                    summary=("Sock tucks cleanly into shoe." if not issues else f"Footwear fit issue: {'; '.join(issues)}."),
                    metrics={
                        "penetration_px": int(penetration_px),
                        "penetration_ratio": round(penetration_ratio, 3),
                        "unfilled_window_px": unfilled_window,
                        "center_offset_px": round(cx_err, 1),
                        "sock_w": round(s_w, 1),
                        "shoe_open_w": round(sh_w, 1),
                    },
                    fix_hint=(
                        f"Lower {p_id} into {c_id} (or extend {p_id} downward) so the sock fills the shoe arch behind the strap and tucks >=40% into the shoe, centered at x={sh_cx:.0f}."
                        if issues
                        else ""
                    ),
                    roi=joint_roi,
                )
            )
            continue

        # Standard articulated joint (elbow, wrist, sleeve elbow, cuff->hand, knee, shin->sock)
        m = _joint_axis_metrics(P, C, rgb_comp, force_vertical=(rel in ("leg_to_sock", "skin_joint") and "Knee" in label))
        if not m.get("valid"):
            continue

        issues = []
        sev = "pass"
        gap = m["gap_px"]
        off_r = m["offset_ratio"]
        wr = m["width_ratio"]
        ang = m["angle_jump_deg"]
        seam = m["seam_line_score"]

        # 1. Gap / insufficient overlap
        min_ov = 10.0 * S
        if gap > 1.5 * S:
            sev = "fail"
            issues.append(f"air gap of {gap:.1f}px")
        elif m["overlap_px"] < min_ov:
            sev = "fail" if m["overlap_px"] < 2.0 * S else "warn"
            issues.append(f"tiny overlap ({m['overlap_px']:.1f}px < {min_ov:.0f}px)")

        # 2. Centerline offset
        if rel in ("skin_joint", "sleeve_joint"):
            off_warn, off_fail = 0.08, 0.14
        elif rel == "leg_to_sock":
            off_warn, off_fail = 0.08, 0.15
        else:  # wrist_joint, cuff_to_hand, cuff_to_arm
            off_warn, off_fail = 0.12, 0.20

        if off_r > off_fail:
            sev = "fail"
            issues.append(f"centerline misaligned by {m['offset_px']:.1f}px ({off_r*100:.0f}% of width)")
        elif off_r > off_warn:
            sev = "warn" if sev == "pass" else sev
            issues.append(f"slight centerline shift {m['offset_px']:.1f}px ({off_r*100:.0f}%)")

        # 3. Width continuity
        if rel == "skin_joint":
            if wr < 0.72 or wr > 1.32:
                sev = "fail"
                issues.append(f"width mismatch {m['child_w']:.0f}px vs {m['parent_w']:.0f}px (ratio {wr:.2f}x)")
            elif wr < 0.80 or wr > 1.22:
                sev = "warn" if sev == "pass" else sev
                issues.append(f"mild width step (ratio {wr:.2f}x)")
        elif rel == "sleeve_joint":
            if wr < 0.80 or wr > 1.35:
                sev = "fail"
                issues.append(f"sleeve width mismatch {m['child_w']:.0f}px vs {m['parent_w']:.0f}px (ratio {wr:.2f}x)")
            elif wr < 0.88 or wr > 1.24:
                sev = "warn" if sev == "pass" else sev
                issues.append(f"sleeve width step (ratio {wr:.2f}x)")
        elif rel == "leg_to_sock":
            if wr < 0.94:
                sev = "fail"
                issues.append(f"sock narrower than shin ({m['child_w']:.0f}px vs {m['parent_w']:.0f}px) — skin spills out")
            elif wr > 1.32:
                sev = "fail"
                issues.append(f"sock rim too wide for shin ({m['child_w']:.0f}px vs {m['parent_w']:.0f}px, ratio {wr:.2f}x)")
            elif wr > 1.20:
                sev = "warn" if sev == "pass" else sev
                issues.append(f"sock rim slightly wider than shin (ratio {wr:.2f}x)")
        elif rel == "wrist_joint":
            if wr > 1.45 or wr < 0.65:
                sev = "fail"
                issues.append(f"wrist width mismatch {m['child_w']:.0f}px vs {m['parent_w']:.0f}px (ratio {wr:.2f}x)")
        elif rel == "cuff_to_hand":
            if wr > 1.05:
                sev = "fail"
                issues.append(f"hand wrist wider than sleeve cuff ({m['child_w']:.0f}px vs {m['parent_w']:.0f}px)")

        # 4. Angle jump check on straight A-pose joints
        if rel in ("skin_joint", "sleeve_joint") and ang > 18.0:
            sev = "fail" if ang > 28.0 else ("warn" if sev == "pass" else sev)
            issues.append(f"axis kink of {ang:.1f}° between {p_id} and {c_id}")

        # 5. Unblended horizontal cut-line across continuous skin or knit sleeve
        if rel in ("skin_joint", "sleeve_joint") and seam > 0.38:
            sev = "fail" if seam > 0.55 else ("warn" if sev == "pass" else sev)
            issues.append(f"visible hard seam line across joint (score={seam:.2f})")

        jx, jy = m["joint_xy"]
        j_pad = int(round(max(220 * S, max(m["parent_w"], m["child_w"]) * 1.45)))
        joint_roi = (max(0, int(jx - j_pad)), max(0, int(jy - j_pad)), min(W, int(jx + j_pad)), min(H, int(jy + j_pad)))

        findings.append(
            Finding(
                category="limb_chains",
                check_id=f"{state_name}:{p_id}_{c_id}",
                severity=sev,
                outfit_state=state_name,
                title=f"{label} [{state_name}]",
                summary=(
                    f"Aligned (overlap={m['overlap_px']:.0f}px, offset={m['offset_px']:.1f}px, width={wr:.2f}x)."
                    if not issues
                    else f"Misaligned: {'; '.join(issues)}."
                ),
                metrics=m,
                fix_hint=(
                    f"Align {c_id} entry to {p_id} end (shift {m['offset_px']:.0f}px toward {m['p_end_xy']}, match width ratio ~1.0, overlap >= {int(round(20*S))}px, and feather/erase cut-edge outline)."
                    if issues
                    else ""
                ),
                roi=joint_roi,
                guides={"p_end_xy": m["p_end_xy"], "c_start_xy": m["c_start_xy"], "joint_xy": m["joint_xy"]},
            )
        )

    return findings


def _audit_garment_fit(
    model: ModelBundle, state_name: str, hidden: set[str], rgb_comp: np.ndarray
) -> list[Finding]:
    """Audit fitted clothing & accessories against their underlying host anatomy."""
    findings: list[Finding] = []
    W, H = model.canvas
    S = H / 4800.0
    vis = {L.lid for L in model.layers if L.visible_default and L.lid not in hidden}

    # 1. CHOKER / NECK ACCESSORY vs NECK (B01)
    if "B01" in vis:
        neck = model.by_id["B01"]
        n_spans = _row_spans(neck)
        neck_cx_approx = 0.5 * (neck.box[0] + neck.box[2])
        neck_w_approx = max(1.0, float(neck.box[2] - neck.box[0]))
        # Find centered neck accessory layers (A03, A01, etc.) — exclude head earrings (L.head or far off-center!)
        neck_accs = []
        for L in model.layers:
            if L.lid not in vis or L.lid == "B01" or L.head:
                continue
            if not (L.lid.startswith("A") or L.kind == "accessory"):
                continue
            if L.box[1] >= neck.box[3] or L.box[3] <= neck.box[1]:
                continue
            acc_cx = 0.5 * (L.box[0] + L.box[2])
            if abs(acc_cx - neck_cx_approx) > 0.65 * neck_w_approx:
                continue
            ov_px = _layer_overlap_px(L, neck)
            if ov_px < 0.20 * max(1, int(L.mask.sum())):
                continue
            neck_accs.append(L)
        for acc in neck_accs:
            a_spans = _row_spans(acc)
            if not a_spans or not n_spans:
                continue
            # Identify strap rows (top 40% of accessory height OR rows with width >= 72% of max accessory width)
            all_aw = [x1 - x0 for x0, x1 in a_spans.values()]
            max_aw = max(all_aw)
            ay_min, ay_max = min(a_spans), max(a_spans)
            strap_ys = [
                y
                for y, (ax0, ax1) in a_spans.items()
                if y in n_spans and (y <= ay_min + (ay_max - ay_min) * 0.42) and (ax1 - ax0) >= 0.70 * max_aw
            ]
            if not strap_ys:
                strap_ys = [y for y in a_spans if y in n_spans]
            if not strap_ys:
                continue

            over_L = [max(0, n_spans[y][0] - a_spans[y][0]) for y in strap_ys]
            over_R = [max(0, a_spans[y][1] - n_spans[y][1]) for y in strap_ys]
            nw_list = [n_spans[y][1] - n_spans[y][0] for y in strap_ys]
            aw_list = [a_spans[y][1] - a_spans[y][0] for y in strap_ys]
            cx_errs = [abs(0.5 * (a_spans[y][0] + a_spans[y][1]) - 0.5 * (n_spans[y][0] + n_spans[y][1])) for y in strap_ys]

            med_nw = float(np.median(nw_list))
            med_aw = float(np.median(aw_list))
            med_ov_L = float(np.median(over_L))
            med_ov_R = float(np.median(over_R))
            max_ov = max(med_ov_L, med_ov_R)
            w_ratio = med_aw / max(med_nw, 1.0)
            cx_err = float(np.median(cx_errs))

            issues = []
            sev = "pass"
            if w_ratio > 1.20 or max_ov > max(8.0 * S, 0.11 * med_nw):
                sev = "fail"
                issues.append(
                    f"too wide for neck ({med_aw:.0f}px vs neck {med_nw:.0f}px, ratio {w_ratio:.2f}x; overhangs {med_ov_L:.0f}px L / {med_ov_R:.0f}px R into air)"
                )
            elif w_ratio > 1.13 or max_ov > max(5.0 * S, 0.075 * med_nw):
                sev = "warn"
                issues.append(f"slightly wide vs neck ({med_aw:.0f}px vs {med_nw:.0f}px, overhang L={med_ov_L:.0f}px R={med_ov_R:.0f}px)")
            elif w_ratio < 0.92:
                sev = "fail"
                issues.append(f"narrower than neck ({med_aw:.0f}px vs {med_nw:.0f}px)")
            if cx_err > max(5.0 * S, 0.06 * med_nw):
                sev = "fail" if cx_err > 0.10 * med_nw else ("warn" if sev == "pass" else sev)
                issues.append(f"off-center vs neck by {cx_err:.1f}px")

            target_w = round(med_nw * 1.05)
            findings.append(
                Finding(
                    category="garment_fit",
                    check_id=f"{state_name}:choker_{acc.lid}",
                    severity=sev,
                    outfit_state=state_name,
                    title=f"Choker/Neck Accessory Fit ({acc.lid} on B01) [{state_name}]",
                    summary=(
                        f"Snug fit on neck ({med_aw:.0f}px vs {med_nw:.0f}px neck, ratio {w_ratio:.2f}x)."
                        if not issues
                        else f"Choker fit issue: {'; '.join(issues)}."
                    ),
                    metrics={
                        "neck_w_px": round(med_nw, 1),
                        "accessory_w_px": round(med_aw, 1),
                        "width_ratio": round(w_ratio, 3),
                        "overhang_L_px": round(med_ov_L, 1),
                        "overhang_R_px": round(med_ov_R, 1),
                        "center_err_px": round(cx_err, 1),
                        "target_w_px": target_w,
                    },
                    fix_hint=(
                        f"Resize {acc.lid} width from ~{med_aw:.0f}px to ~{target_w}px (centered on neck B01) so the strap hugs the neck with zero overhang."
                        if issues
                        else ""
                    ),
                    roi=_union_roi([acc.box, (neck.box[0], acc.box[1] - int(round(80 * S)), neck.box[2], acc.box[3] + int(round(80 * S)))], int(round(90 * S)), W, H),
                )
            )

        # 2. NECK (B01) TO CHIN (F01) & NECK-TO-TORSO (B02) BLEND
        if "F01" in vis:
            face = model.by_id["F01"]
            f_spans = _row_spans(face)
            if f_spans and n_spans:
                face_bot = max(f_spans)
                face_top = min(f_spans)
                neck_top = min(n_spans)
                ov_chin = face_bot - neck_top
                # Compare neck top center X against LOWER JAW/CHIN center X (bottom 22% of face, robust to hair/ear asymmetry!)
                chin_ys = [y for y in f_spans if y >= face_bot - max(12, int((face_bot - face_top) * 0.22))]
                f_cx = float(np.median([0.5 * (f_spans[y][0] + f_spans[y][1]) for y in chin_ys])) if chin_ys else 0.5 * (face.box[0] + face.box[2])
                n_top_ys = sorted(n_spans)[: min(30, len(n_spans))]
                n_cx = float(np.median([0.5 * (n_spans[y][0] + n_spans[y][1]) for y in n_top_ys]))
                cx_err = abs(n_cx - f_cx)
                # Check neck-to-head contour smoothness: B01's upper silhouette just below chin should taper smoothly
                # WITHOUT abrupt horizontal stair-step flares/notches (e.g. an uncleaned wider jaw stub on top of B01)!
                neck_step_px = 0.0
                top_stub_ratio = 1.0
                if ov_chin >= 10 * S:
                    # Compare true mid-neck cylinder width (face_bot + 15..55*S) vs upper neck/jaw-stub width (face_bot - 35..+8*S)
                    mid_ys = [y for y in range(face_bot + int(round(12 * S)), min(max(n_spans), face_bot + int(round(55 * S))) + 1) if y in n_spans]
                    top_ys = [y for y in range(max(neck_top, face_bot - int(round(38 * S))), min(max(n_spans), face_bot + int(round(8 * S))) + 1) if y in n_spans]
                    if mid_ys and top_ys:
                        mid_nw = float(np.median([n_spans[y][1] - n_spans[y][0] for y in mid_ys]))
                        max_top_nw = float(np.percentile([n_spans[y][1] - n_spans[y][0] for y in top_ys], 92))
                        top_stub_ratio = max_top_nw / max(mid_nw, 1.0)
                        if top_stub_ratio > 1.12:
                            neck_step_px = max(neck_step_px, 0.5 * (max_top_nw - mid_nw))
                    # Also check sudden per-side outward stair-steps on B01's OWN upper neck silhouette (NOT F01's curved chin!)
                    y_chk_lo = max(neck_top + int(round(4 * S)), face_bot - int(round(32 * S)))
                    y_chk_hi = min(max(n_spans), face_bot + int(round(28 * S)))
                    u_ys = [y for y in range(y_chk_lo, y_chk_hi + 1, max(1, int(round(4 * S)))) if y in n_spans]
                    prev_l, prev_r = None, None
                    for y in u_ys:
                        lx, rx = n_spans[y]
                        if prev_l is not None and y <= face_bot + int(round(8 * S)):
                            # Abrupt inward drop walking downward from top stub into slender neck!
                            neck_step_px = max(neck_step_px, float(max(0, lx - prev_l)), float(max(0, prev_r - rx)))
                        prev_l, prev_r = lx, rx

                issues = []
                sev = "pass"
                if ov_chin < 10 * S:
                    sev = "fail"
                    issues.append(f"neck barely reaches chin (overlap={ov_chin:.0f}px)")
                if cx_err > 18 * S:
                    sev = "warn" if cx_err <= 32 * S else "fail"
                    issues.append(f"neck center X shifted {cx_err:.1f}px from chin center")
                if neck_step_px > 6.5 * S or top_stub_ratio > 1.14:
                    sev = "fail" if (neck_step_px > 9.5 * S or top_stub_ratio > 1.18) else ("warn" if sev == "pass" else sev)
                    issues.append(f"non-smooth jaw/neck transition (step={neck_step_px:.1f}px, upper-neck flare={top_stub_ratio:.2f}x)")
                findings.append(
                    Finding(
                        category="garment_fit",
                        check_id=f"{state_name}:neck_chin",
                        severity=sev,
                        outfit_state=state_name,
                        title=f"Neck->Head Alignment (B01->F01) [{state_name}]",
                        summary=("Neck cleanly centered & smooth behind chin." if not issues else "; ".join(issues)),
                        metrics={
                            "chin_overlap_px": int(ov_chin),
                            "center_offset_px": round(cx_err, 1),
                            "jaw_neck_step_px": round(neck_step_px, 1),
                            "upper_neck_flare_ratio": round(top_stub_ratio, 3),
                        },
                        fix_hint=(
                            "Trim any flared jaw stub from top of B01 so the neck cylinder rises smoothly inside F01's jawline with zero step/gap."
                            if issues
                            else ""
                        ),
                        roi=_union_roi([neck.box, (face.box[0], face_bot - int(round(120 * S)), face.box[2], face_bot + int(round(60 * S)))], int(round(60 * S)), W, H),
                    )
                )

        if "B02" in vis:
            torso = model.by_id["B02"]
            # Check if bottom edge of B01 creates a visible seam across upper chest of B02
            ix0, iy0 = max(neck.box[0], torso.box[0]), max(neck.box[1], torso.box[1])
            ix1, iy1 = min(neck.box[2], torso.box[2]), min(neck.box[3], torso.box[3])
            if ix1 > ix0 and iy1 > iy0:
                ny_bot = neck.box[3] - 4
                if torso.box[1] + 15 < ny_bot < torso.box[3] - 15:
                    # Check edge gradient along B01's bottom boundary inside torso
                    lum = rgb_comp.mean(axis=-1) / 255.0
                    # Find pixels on bottom contour of neck.mask
                    nb_ys = [max(0, neck.mask.shape[0] - 5), neck.mask.shape[0] - 1]
                    cols = np.nonzero(neck.mask[nb_ys[0] : nb_ys[1] + 1].any(axis=0))[0]
                    diffs = []
                    for rx in cols[::4]:
                        cx = neck.box[0] + int(rx)
                        col_ys = np.nonzero(neck.mask[:, rx])[0]
                        if len(col_ys) < 6:
                            continue
                        cy = neck.box[1] + int(col_ys[-1])
                        if torso.box[0] + 20 < cx < torso.box[2] - 20 and torso.box[1] + 20 < cy < torso.box[3] - 20:
                            if torso.mask[cy - torso.box[1], cx - torso.box[0]]:
                                # Compare pixel right at edge vs 8px above/below
                                l_e = lum[cy, cx]
                                l_a = lum[max(0, cy - 8), cx]
                                l_b = lum[min(H - 1, cy + 8), cx]
                                diffs.append(max(abs(l_e - 0.5 * (l_a + l_b)), abs(l_a - l_b) * 0.5))
                    if len(diffs) >= 10:
                        seam_d = float(np.percentile(diffs, 80))
                        sev = "pass" if seam_d < 0.08 else ("warn" if seam_d < 0.14 else "fail")
                        findings.append(
                            Finding(
                                category="garment_fit",
                                check_id=f"{state_name}:neck_torso_seam",
                                severity=sev,
                                outfit_state=state_name,
                                title=f"Neck->Chest Blend Seam (B01 on B02) [{state_name}]",
                                summary=(
                                    f"Smooth collarbone blend (edge delta={seam_d:.3f})."
                                    if sev == "pass"
                                    else f"Visible horizontal seam where neck B01 ends on chest B02 (edge delta={seam_d:.3f})."
                                ),
                                metrics={"edge_lum_delta": round(seam_d, 4)},
                                fix_hint=(
                                    "Add soft fade={'bottom': 55} to B01 and remove any dark bottom boundary stroke so neck skin blends invisibly into chest B02."
                                    if sev != "pass"
                                    else ""
                                ),
                                roi=_union_roi([neck.box], int(round(80 * S)), W, H),
                            )
                        )

    # 3. WAIST BELT (K01) vs SKIRT (K02) FIT (only when K01 is actually a narrow waistband/belt < 38% of K02 height & <= 1.25x width!)
    if "K01" in vis and "K02" in vis:
        belt, skirt = model.by_id["K01"], model.by_id["K02"]
        bh = max(1, belt.box[3] - belt.box[1])
        kh = max(1, skirt.box[3] - skirt.box[1])
        bw_box = max(1, belt.box[2] - belt.box[0])
        kw_box = max(1, skirt.box[2] - skirt.box[0])
        if bh > 0.38 * kh or bw_box > 1.25 * kw_box:
            belt, skirt = None, None  # e.g. K01=skirt front, K02=waistband/back
    else:
        belt, skirt = None, None
    if belt is not None and skirt is not None:
        b_sp = _row_spans(belt)
        k_sp = _row_spans(skirt)
        if b_sp and k_sp:
            b_y0, b_y1 = min(b_sp), max(b_sp)
            k_y0 = min(k_sp)
            # Measure belt max width vs skirt waistband width (top 10% of skirt)
            k_top_ys = [y for y in range(k_y0, k_y0 + max(8, int((max(k_sp) - k_y0) * 0.12))) if y in k_sp]
            b_mid_ys = [y for y in range(b_y0 + int((b_y1 - b_y0) * 0.25), b_y0 + int((b_y1 - b_y0) * 0.75)) if y in b_sp]
            bw = float(np.median([b_sp[y][1] - b_sp[y][0] for y in b_mid_ys])) if b_mid_ys else float(belt.box[2] - belt.box[0])
            kw = float(np.median([k_sp[y][1] - k_sp[y][0] for y in k_top_ys])) if k_top_ys else float(skirt.box[2] - skirt.box[0])
            bcx = float(np.median([0.5 * (b_sp[y][0] + b_sp[y][1]) for y in b_mid_ys])) if b_mid_ys else 0.5 * (belt.box[0] + belt.box[2])
            kcx = float(np.median([0.5 * (k_sp[y][0] + k_sp[y][1]) for y in k_top_ys])) if k_top_ys else 0.5 * (skirt.box[0] + skirt.box[2])
            vert_ov = b_y1 - k_y0  # how many px belt overlaps skirt top
            belt_h = max(1, b_y1 - b_y0)
            ov_frac = vert_ov / belt_h
            w_ratio = bw / max(kw, 1.0)
            cx_err = abs(bcx - kcx)

            # Check if belt left/right ends overhang into thin air past skirt top waistband (use robust median top span!)
            air_overhang_px = 0
            k_top_x0 = float(np.median([k_sp[y][0] for y in k_top_ys])) if k_top_ys else float(skirt.box[0])
            k_top_x1 = float(np.median([k_sp[y][1] for y in k_top_ys])) if k_top_ys else float(skirt.box[2])
            torso = model.by_id.get("B02") if "B02" in vis else None
            t_sp = _row_spans(torso) if torso else {}
            for y in b_mid_ys:
                bx0, bx1 = b_sp[y]
                host_x0, host_x1 = k_top_x0, k_top_x1
                if y in t_sp:
                    host_x0, host_x1 = min(host_x0, float(t_sp[y][0])), max(host_x1, float(t_sp[y][1]))
                air_overhang_px = max(air_overhang_px, int(round(max(0.0, host_x0 - bx0))), int(round(max(0.0, bx1 - host_x1))))

            issues = []
            sev = "pass"
            if ov_frac < 0.45:
                sev = "fail" if ov_frac < 0.25 else "warn"
                issues.append(f"belt sits too high above skirt waistband (overlap only {ov_frac*100:.0f}% of belt height)")
            if w_ratio > 1.12 or w_ratio < 0.85:
                sev = "fail" if (w_ratio > 1.20 or w_ratio < 0.78) else ("warn" if sev == "pass" else sev)
                issues.append(f"belt width {bw:.0f}px vs skirt waistband {kw:.0f}px (ratio {w_ratio:.2f}x)")
            if air_overhang_px > 18 * S:
                sev = "fail" if air_overhang_px > 32 * S else ("warn" if sev == "pass" else sev)
                issues.append(f"belt corners overhang body/skirt by {air_overhang_px:.0f}px into air")
            if cx_err > 12 * S:
                sev = "warn" if sev == "pass" else sev
                issues.append(f"belt off-center by {cx_err:.1f}px")

            findings.append(
                Finding(
                    category="garment_fit",
                    check_id=f"{state_name}:belt_skirt",
                    severity=sev,
                    outfit_state=state_name,
                    title=f"Belt->Skirt Waistband Fit (K01 on K02) [{state_name}]",
                    summary=("Belt hugs skirt waistband flush." if not issues else f"Waistband fit issue: {'; '.join(issues)}."),
                    metrics={
                        "belt_w_px": round(bw, 1),
                        "skirt_top_w_px": round(kw, 1),
                        "width_ratio": round(w_ratio, 3),
                        "overlap_frac": round(ov_frac, 3),
                        "air_overhang_px": int(air_overhang_px),
                        "center_err_px": round(cx_err, 1),
                    },
                    fix_hint=(
                        f"Seat K01 flush over K02's top waistband (overlap >=65% of belt height) and match width to ~{kw:.0f}px."
                        if issues
                        else ""
                    ),
                    roi=_union_roi([belt.box, (skirt.box[0], skirt.box[1], skirt.box[2], skirt.box[1] + int(round(220 * S)))], int(round(70 * S)), W, H),
                )
            )

    # 4. CAMISOLE / TOP (C01) vs TORSO (B02) FIT
    if "C01" in vis and "B02" in vis:
        top, torso = model.by_id["C01"], model.by_id["B02"]
        c_sp = _row_spans(top)
        t_sp = _row_spans(torso)
        if c_sp and t_sp:
            cy0, cy1 = min(c_sp), max(c_sp)
            # Check lower 60% of camisole (body bodice, excluding neck halter straps)
            bod_ys = [y for y in range(cy0 + int((cy1 - cy0) * 0.40), cy1 - int((cy1 - cy0) * 0.05)) if y in c_sp and y in t_sp]
            if bod_ys:
                c_cx = float(np.median([0.5 * (c_sp[y][0] + c_sp[y][1]) for y in bod_ys]))
                t_cx = float(np.median([0.5 * (t_sp[y][0] + t_sp[y][1]) for y in bod_ys]))
                cx_err = abs(c_cx - t_cx)
                # Skin poking out past sides of camisole bodice (excluding upper bust/armpits)
                mid_ys = [y for y in range(cy0 + int((cy1 - cy0) * 0.55), cy1 - int((cy1 - cy0) * 0.08)) if y in c_sp and y in t_sp]
                skin_poke = [
                    max(0, c_sp[y][0] - t_sp[y][0], t_sp[y][1] - c_sp[y][1])
                    for y in mid_ys
                ]
                max_poke = float(np.percentile(skin_poke, 90)) if skin_poke else 0.0
                # Check top camisole/halter straps attachment (top 14% of C01): must rest cleanly on B01 neck OR B02 shoulders
                # (zero floating overhang out into empty air above/outside the shoulder & neck silhouette!)
                neck = model.by_id.get("B01") if "B01" in vis else None
                n_sp = _row_spans(neck) if neck else {}
                body_sp: dict[int, tuple[int, int]] = {}
                for y in set(t_sp.keys()) | set(n_sp.keys()):
                    xl = min(t_sp[y][0] if y in t_sp else 10**9, n_sp[y][0] if y in n_sp else 10**9)
                    xr = max(t_sp[y][1] if y in t_sp else -10**9, n_sp[y][1] if y in n_sp else -10**9)
                    body_sp[y] = (xl, xr)
                strap_gap = 0.0
                if body_sp:
                    strap_ys = [y for y in range(cy0, cy0 + max(8, int((cy1 - cy0) * 0.14))) if y in c_sp]
                    if strap_ys:
                        overhangs = []
                        for y in strap_ys:
                            if y in body_sp:
                                overhangs.append(max(0, body_sp[y][0] - c_sp[y][0], c_sp[y][1] - body_sp[y][1]))
                            else:
                                # Strap sticks above the entire body/neck top silhouette!
                                overhangs.append(max(1.0, float(min(body_sp) - y)))
                        strap_gap = float(np.median(overhangs)) if overhangs else 0.0
                issues = []
                sev = "pass"
                if cx_err > 12 * S:
                    sev = "warn" if cx_err <= 22 * S else "fail"
                    issues.append(f"top off-center vs torso by {cx_err:.1f}px")
                if max_poke > 14 * S:
                    sev = "fail" if max_poke > 26 * S else ("warn" if sev == "pass" else sev)
                    issues.append(f"torso skin pokes {max_poke:.0f}px past camisole side seams")
                if strap_gap > 16 * S:
                    sev = "warn" if strap_gap <= 32 * S else "fail"
                    issues.append(f"halter straps stick out {strap_gap:.0f}px past neck sides at top")
                findings.append(
                    Finding(
                        category="garment_fit",
                        check_id=f"{state_name}:top_torso",
                        severity=sev,
                        outfit_state=state_name,
                        title=f"Top/Camisole Fit (C01 on B02) [{state_name}]",
                        summary=("Camisole aligns cleanly over torso." if not issues else f"Top fit issue: {'; '.join(issues)}."),
                        metrics={
                            "center_err_px": round(cx_err, 1),
                            "side_skin_poke_px": round(max_poke, 1),
                            "strap_overhang_px": round(strap_gap, 1),
                        },
                        fix_hint=(
                            "Match C01 bodice width to B02 waist/ribcage and tuck top halter straps cleanly behind B01 neck/choker."
                            if issues
                            else ""
                        ),
                        roi=_union_roi([top.box], int(round(80 * S)), W, H),
                    )
                )

    # 5. CARDIGAN HEM CLEARANCE & SHOULDER ALIGNMENT (J03R/L vs K01 & J04R/L)
    if "J03R" in vis and "J03L" in vis:
        jr, jl = model.by_id["J03R"], model.by_id["J03L"]
        issues = []
        sev = "pass"
        hem_diff = abs(jr.box[3] - jl.box[3])
        if hem_diff > 10 * S:
            sev = "warn"
            issues.append(f"L/R cardigan hem mismatch by {hem_diff}px")
        belt_overlap_px = 0
        if "K01" in vis:
            belt = model.by_id["K01"]
            belt_overlap_px = max(jr.box[3], jl.box[3]) - belt.box[1]
            # Cardigan crop hem overlapping deep into the white belt looks cramped unless intentional
            if belt_overlap_px > 48 * S:
                sev = "warn" if sev == "pass" else sev
                issues.append(f"cardigan hem overlaps {belt_overlap_px:.0f}px down over belt K01")
        # Check shoulder seam gap between J03R/L and J04R/L
        sh_ov_min = 999999
        for s in ("R", "L"):
            if f"J04{s}" in vis:
                ov = _layer_overlap_px(model.by_id[f"J03{s}"], model.by_id[f"J04{s}"])
                sh_ov_min = min(sh_ov_min, ov)
        if sh_ov_min < int(round(300 * S * S)):
            sev = "fail"
            issues.append(f"cardigan sleeve J04 barely connects to front panel J03 (overlap={sh_ov_min}px²)")
        findings.append(
            Finding(
                category="garment_fit",
                check_id=f"{state_name}:cardigan_fit",
                severity=sev,
                outfit_state=state_name,
                title=f"Cardigan Panels & Sleeve Join (J03/J04) [{state_name}]",
                summary=("Cardigan panels & sleeve shoulders sit cleanly." if not issues else "; ".join(issues)),
                metrics={
                    "hem_diff_px": int(hem_diff),
                    "belt_overlap_px": int(belt_overlap_px),
                    "min_sleeve_panel_overlap_px": int(sh_ov_min if sh_ov_min < 999999 else 0),
                },
                fix_hint=(
                    "Raise J03R/L slightly so cropped cardigan hem sits cleanly above belt K01 and overlaps J04R/L shoulders solidly."
                    if issues
                    else ""
                ),
                roi=_union_roi([jr.box, jl.box], int(round(80 * S)), W, H),
            )
        )

    # 6. APPENDAGES: HUMAN EARS, KEMONOMIMI EARS, CAT TAIL
    if "F01" in vis:
        face = model.by_id["F01"]
        for eid in ("F05", "F06"):
            if eid in vis:
                ear = model.by_id[eid]
                ov = _layer_overlap_px(face, ear)
                if ov < int(round(12 * S * S)):
                    findings.append(
                        Finding(
                            category="garment_fit",
                            check_id=f"{state_name}:ear_{eid}",
                            severity="fail",
                            outfit_state=state_name,
                            title=f"Human Ear Detached ({eid}->F01) [{state_name}]",
                            summary=f"Ear {eid} does not overlap face silhouette F01 (overlap={ov}px²).",
                            metrics={"overlap_px": ov},
                            fix_hint=f"Slide {eid} inward toward face F01 edge until overlap >= {int(round(30*S*S))}px².",
                            roi=_union_roi([ear.box], int(round(120 * S)), W, H),
                        )
                    )

    if "T01" in vis:
        tail_l = model.by_id["T01"]
        # Tail must overlap skirt/body behind it cleanly
        hosts = [model.by_id[h] for h in ("K04", "K02", "B02", "B05L", "B05R") if h in vis]
        ov_tail = sum(_layer_overlap_px(tail_l, h) for h in hosts)
        sev = "pass" if ov_tail >= int(round(350 * S * S)) else "fail"
        findings.append(
            Finding(
                category="garment_fit",
                check_id=f"{state_name}:tail_attach",
                severity=sev,
                outfit_state=state_name,
                title=f"Tail Root Attachment (T01) [{state_name}]",
                summary=(
                    f"Tail root tucked solidly behind skirt/pelvis (overlap={ov_tail}px²)."
                    if sev == "pass"
                    else f"Tail T01 appears detached or barely touching pelvis/skirt (overlap={ov_tail}px²)."
                ),
                metrics={"pelvis_overlap_px": ov_tail},
                fix_hint=("Shift T01 root inward under skirt K04/K02 so it stays anchored during tail wag." if sev != "pass" else ""),
                roi=_union_roi([tail_l.box], int(round(120 * S)), W, H),
            )
        )

    return findings


def _audit_face_and_eyes(model: ModelBundle, state_name: str, hidden: set[str]) -> list[Finding]:
    """Audit head-face anatomy & gaze quality:
    1. `face_neck_stem`: Detects any hanging neck stem baked onto the bottom of face F01 below the chin outline
       (which detaches sideways from body neck B01 whenever the head turns!).
    2. `double_ears` / `face_baked_ears`: Detects human ears baked onto F01's sides — especially FAIL when separate
       F05/F06 ear layers are ALSO visible (giving the character two pairs of human ears!).
    3. `eye_gaze_focus`: Measures iris/pupil vergence & gaze bias inside sclera E01R/L to catch wall-eyed / unfocused
       outward-diverging eyes or severe cross-eyed/off-center neutral gaze!
    """
    findings: list[Finding] = []
    W, H = model.canvas
    S = H / 4800.0
    vis = {L.lid for L in model.layers if L.visible_default and L.lid not in hidden}

    # 1 & 2. Face F01 stem & baked-on human ears audit via interior skin flood-fill
    if "F01" in vis:
        F = model.by_id["F01"]
        fh, fw = F.rgba.shape[:2]
        alpha = F.rgba[..., 3] > 28
        lum = (0.299 * F.rgba[..., 0] + 0.587 * F.rgba[..., 1] + 0.114 * F.rgba[..., 2]) / 255.0
        skin_int = alpha & (lum > 0.68)
        ys_s, xs_s = np.nonzero(skin_int)
        if len(ys_s) > 200:
            y0, y1 = int(ys_s.min()), int(ys_s.max())
            cx = int(round(float(np.median(xs_s))))
            cy = y0 + int((y1 - y0) * 0.45)
            lbl, _ = ndimage.label(skin_int)
            seed_id = int(lbl[cy, cx])
            if seed_id > 0:
                face_core = ndimage.binary_fill_holes(lbl == seed_id)
                keep_oval = ndimage.binary_dilation(face_core, iterations=max(6, int(round(7 * S))))
                # Count actual protruding skin/body pixels outside the dark outline (ignore outermost 1px of thick dark ink!)
                outside = alpha & (~keep_oval) & (lum > 0.56)

                fc_ys, fc_xs = np.nonzero(face_core)
                fc_y0, fc_y1 = int(fc_ys.min()), int(fc_ys.max())
                fc_x0, fc_x1 = int(fc_xs.min()), int(fc_xs.max())
                fc_h = max(1, fc_y1 - fc_y0)
                fc_w = max(1, fc_x1 - fc_x0)
                yy, xx = np.mgrid[0:fh, 0:fw]

                # A. Hanging neck stem below chin outline (OR wide flat-cut neck stub on lineless art)
                stem_zone = outside & (yy >= fc_y1 - int(fc_h * 0.16)) & (np.abs(xx - cx) <= fc_w * 0.34)
                stem_px = int(stem_zone.sum())
                # Fallback check for lineless F01: bottom 3px row width vs max face width
                a_ys = np.nonzero(alpha.any(axis=1))[0]
                bot_w_ratio = 0.0
                if len(a_ys) > 20:
                    y_bot = int(a_ys[-1])
                    row_b = np.nonzero(alpha[max(0, y_bot - max(2, int(round(4 * S))))])[0]
                    if len(row_b):
                        bot_w_ratio = float(row_b[-1] - row_b[0] + 1) / max(1.0, float(fc_w))

                stem_thresh = max(80, int(round(360 * S * S)))
                has_stem = stem_px >= stem_thresh or bot_w_ratio >= 0.24
                sev_stem = "fail" if has_stem else "pass"
                chin_roi = (
                    max(0, F.box[0] + int(fw * 0.12)),
                    max(0, F.box[1] + int(fh * 0.68)),
                    min(W, F.box[2] - int(fw * 0.12)),
                    min(H, F.box[3] + int(round(120 * S))),
                )
                findings.append(
                    Finding(
                        category="garment_fit",
                        check_id=f"{state_name}:face_neck_stem",
                        severity=sev_stem,
                        outfit_state=state_name,
                        title=f"Face Layer Neck-Stem Check (F01) [{state_name}]",
                        summary=(
                            "Clean chin contour on F01 (zero hanging neck stem)."
                            if sev_stem == "pass"
                            else f"Neck stem baked onto face F01 below chin ({stem_px}px² outside chin outline, bottom_w={bot_w_ratio*100:.0f}%) — detaches from body neck B01 on head turns!"
                        ),
                        metrics={"stem_px": stem_px, "bottom_width_ratio": round(bot_w_ratio, 3)},
                        fix_hint=(
                            "Erase ALL neck-stem pixels below F01's chin jawline outline (F01 must end cleanly at the chin!). Extend body neck B01 upward behind F01 instead."
                            if sev_stem != "pass"
                            else ""
                        ),
                        roi=chin_roi,
                    )
                )

                # B. Baked-on human ears on F01 sides (double ears if F05/F06 are also visible!)
                ear_zone = (
                    outside
                    & (yy >= fc_y0 + int(fc_h * 0.42))
                    & (yy <= fc_y0 + int(fc_h * 0.88))
                    & (np.abs(xx - cx) > fc_w * 0.34)
                )
                ear_px = int(ear_zone.sum())
                has_sep_ears = bool({"F05", "F06"} & vis)
                ear_thresh = max(140, int(round(600 * S * S)))
                if ear_px >= ear_thresh:
                    sev_ear = "fail" if has_sep_ears else "warn"
                    msg = (
                        f"Double human ears! F01 has baked-on side ears ({ear_px}px² outside cheek outline) while separate ear layers F05/F06 are ALSO visible."
                        if has_sep_ears
                        else f"Human ears are baked directly onto F01 ({ear_px}px² outside cheek outline) — separate them onto F05/F06 for clean 3D head-turn parallax."
                    )
                    hint = (
                        "Strip the baked-on side ears cleanly off F01 right along its cheek outline so ONLY the separate F05/F06 ear layers remain!"
                        if has_sep_ears
                        else "Move ears off F01 onto separate F05/F06 layers behind F01."
                    )
                else:
                    sev_ear = "pass"
                    msg = "Clean cheek contour on F01 (zero baked-on duplicate human ears)."
                    hint = ""
                ear_roi = (
                    max(0, F.box[0] - int(round(90 * S))),
                    max(0, F.box[1] + int(fh * 0.42)),
                    min(W, F.box[2] + int(round(90 * S))),
                    min(H, F.box[1] + int(fh * 0.90)),
                )
                findings.append(
                    Finding(
                        category="garment_fit",
                        check_id=f"{state_name}:face_double_ears",
                        severity=sev_ear,
                        outfit_state=state_name,
                        title=f"Face Contour / Duplicate Human Ears (F01) [{state_name}]",
                        summary=msg,
                        metrics={"baked_ear_px": ear_px, "has_separate_F05_F06": has_sep_ears},
                        fix_hint=hint,
                        roi=ear_roi,
                    )
                )

    # 3. Eye Gaze Focus / Vergence Audit (E01R/L sclera vs E02R/L iris & E03R/L pupil)
    if all(k in vis for k in ("E01R", "E02R", "E01L", "E02L")):
        def _mcx_w(lid: str) -> tuple[float, float]:
            L = model.by_id[lid]
            _, xs = np.nonzero(L.mask)
            if not len(xs):
                return 0.5 * (L.box[0] + L.box[2]), float(L.box[2] - L.box[0])
            return L.box[0] + float(np.median(xs)), float(xs.max() - xs.min() + 1)

        sr, swr = _mcx_w("E01R")
        sl, swl = _mcx_w("E01L")
        ir, _ = _mcx_w("E02R")
        il, _ = _mcx_w("E02L")
        pr = _mcx_w("E03R")[0] if "E03R" in vis else ir
        pl = _mcx_w("E03L")[0] if "E03L" in vis else il
        sw = max(1.0, 0.5 * (swr + swl))

        # Positive outward_div = wall-eyed / exotropic (R looks left, L looks right -> unfocused!)
        div_i = (sr - ir) + (il - sl)
        div_p = (sr - pr) + (pl - sl)
        div_ratio = max(div_i, div_p) / sw
        # Negative = inward convergence (slight -0.04..-0.32 is natural focused anime forward eye contact)
        conv_ratio = min(div_i, div_p) / sw
        # Parallel side gaze bias (positive = both looking right, negative = both looking left)
        bias_px = 0.5 * ((pr - sr) + (pl - sl))
        bias_ratio = abs(bias_px) / sw

        issues = []
        sev = "pass"
        if div_ratio > 0.06:
            sev = "fail" if div_ratio > 0.09 else "warn"
            issues.append(
                f"wall-eyed / unfocused outward gaze (irises diverge outward by {div_i:+.1f}px / {div_ratio*100:+.1f}% of sclera width)"
            )
        elif conv_ratio < -0.40:
            sev = "fail"
            issues.append(f"excessively cross-eyed inward convergence ({conv_ratio*100:.1f}% of sclera width)")
        if bias_ratio > 0.14:
            sev = "fail" if bias_ratio > 0.22 else ("warn" if sev == "pass" else sev)
            issues.append(f"neutral gaze drifts sideways by {bias_px:+.1f}px ({bias_ratio*100:.1f}% of sclera width)")

        shift_in = max(4, int(round((max(div_i, div_p) * 0.5) + 0.04 * sw)))
        eye_roi = _union_roi([model.by_id["E01R"].box, model.by_id["E01L"].box], int(round(90 * S)), W, H)
        findings.append(
            Finding(
                category="garment_fit",
                check_id=f"{state_name}:eye_gaze_focus",
                severity=sev,
                outfit_state=state_name,
                title=f"Eye Gaze Focus & Vergence (E02/E03 in E01) [{state_name}]",
                summary=(
                    f"Focused straight-ahead eye contact (vergence={div_p:+.1f}px / {div_p/sw*100:+.1f}%, bias={bias_px:+.1f}px)."
                    if not issues
                    else f"Eye gaze issue: {'; '.join(issues)}."
                ),
                metrics={
                    "iris_outward_div_px": round(div_i, 1),
                    "pupil_outward_div_px": round(div_p, 1),
                    "outward_div_ratio": round(div_ratio, 3),
                    "parallel_bias_px": round(bias_px, 1),
                    "sclera_w_px": round(sw, 1),
                },
                fix_hint=(
                    f"Shift irises/pupils (E02R/E03R right by +{shift_in}px, E02L/E03L left by -{shift_in}px) slightly inward of sclera centers so both eyes focus straight ahead at the viewer, and align E04 highlights to a single key-light direction."
                    if issues
                    else ""
                ),
                roi=eye_roi,
            )
        )

    # 4. Lower Lash Attachment & Span Check (E06R/L vs E01R/L sclera bottom rim!)
    for s in ("R", "L"):
        if f"E01{s}" in vis and f"E06{s}" in vis:
            sc, ll = model.by_id[f"E01{s}"], model.by_id[f"E06{s}"]
            s_ys, s_xs = np.nonzero(sc.mask)
            l_ys, l_xs = np.nonzero(ll.mask)
            if len(s_ys) and len(l_ys):
                sc_bot = sc.box[1] + int(s_ys.max())
                ll_top = ll.box[1] + int(l_ys.min())
                sc_w = float(s_xs.max() - s_xs.min() + 1)
                ll_w = float(l_xs.max() - l_xs.min() + 1)
                v_gap = ll_top - sc_bot  # > 0 means lower lash floats below sclera bottom!
                span_ratio = ll_w / max(1.0, sc_w)
                ov_px = _layer_overlap_px(sc, ll)
                l_issues = []
                l_sev = "pass"
                if v_gap > 3.0 * S or ov_px == 0:
                    l_sev = "fail"
                    l_issues.append(f"lower lash E06{s} floats {max(0, v_gap):.1f}px below sclera bottom E01{s} (overlap={ov_px}px²)")
                if span_ratio < 0.48:
                    l_sev = "warn" if l_sev == "pass" else l_sev
                    l_issues.append(f"lower lash E06{s} spans only {span_ratio*100:.0f}% of sclera width (too tiny/detached)")
                elif span_ratio > 1.06:
                    l_sev = "warn" if l_sev == "pass" else l_sev
                    l_issues.append(f"lower lash E06{s} overhangs sclera width ({span_ratio*100:.0f}% — shrink & sculpt around E01's lower curve)")
                findings.append(
                    Finding(
                        category="garment_fit",
                        check_id=f"{state_name}:lower_lash_{s}",
                        severity=l_sev,
                        outfit_state=state_name,
                        title=f"Lower Lash Contact (E06{s}->E01{s}) [{state_name}]",
                        summary=(
                            f"Lower lash hugs sclera rim cleanly (span={span_ratio*100:.0f}%, overlap={ov_px}px²)."
                            if not l_issues
                            else "; ".join(l_issues)
                        ),
                        metrics={"vertical_gap_px": int(v_gap), "span_ratio": round(span_ratio, 3), "overlap_px": ov_px},
                        fix_hint=(
                            f"Sculpt E06{s} flush onto E01{s}'s exact curved lower rim (span 50%–95% of E01{s} width) so it wraps the eye shape cleanly without floating or jutting straight out past the corners."
                            if l_issues
                            else ""
                        ),
                        roi=_union_roi([sc.box, ll.box], int(round(60 * S)), W, H),
                    )
                )

    return findings


def _audit_symmetry(model: ModelBundle, hidden: set[str]) -> list[Finding]:
    """Audit L/R bilateral symmetry of paired anatomical & clothing layers."""
    W, H = model.canvas
    S = H / 4800.0
    cx_ref = float((model.head.get("centre") or [W / 2.0, 0.0])[0])
    vis = {L.lid: L for L in model.layers if L.visible_default and L.lid not in hidden}

    pairs: list[tuple[str, str]] = []
    seen = set()
    for lid in vis:
        if lid.endswith("R") and (lid[:-1] + "L") in vis:
            pairs.append((lid, lid[:-1] + "L"))
            seen.add(lid)
        elif lid == "F05" and "F06" in vis:
            pairs.append(("F05", "F06"))

    # Separate head symmetry axis from body symmetry axis (allows natural anime contrapposto weight shift!)
    cx_head = float((model.head.get("centre") or [W / 2.0, 0.0])[0])
    body_mids = []
    for rid, lid in pairs:
        if not vis[rid].head:
            R, L = vis[rid], vis[lid]
            body_mids.append(0.25 * (R.box[0] + R.box[2] + L.box[0] + L.box[2]))
    cx_body = float(np.median(body_mids)) if body_mids else cx_head

    asym_items = []
    for rid, lid in pairs:
        R, L = vis[rid], vis[lid]
        r_cx, r_cy = 0.5 * (R.box[0] + R.box[2]), 0.5 * (R.box[1] + R.box[3])
        l_cx, l_cy = 0.5 * (L.box[0] + L.box[2]), 0.5 * (L.box[1] + L.box[3])
        r_w, r_h = R.box[2] - R.box[0], R.box[3] - R.box[1]
        l_w, l_h = L.box[2] - L.box[0], L.box[3] - L.box[1]

        cx_axis = cx_head if R.head else cx_body
        mir_err = abs(0.5 * (r_cx + l_cx) - cx_axis)
        y_err = abs(r_cy - l_cy)
        w_diff = abs(r_w - l_w) / max(r_w, l_w, 1)
        h_diff = abs(r_h - l_h) / max(r_h, l_h, 1)

        # Hair strands, highlights & ear-perspective details have organic asymmetry; keep strict on limbs, sleeves, socks, shoes, jacket panels, main eyes
        strict = not (rid.startswith("H") or rid.startswith("E04") or rid.startswith("E07") or rid.endswith("_in") or rid == "F05")
        if strict and (mir_err > 28 * S or y_err > 24 * S or w_diff > 0.18 or h_diff > 0.18):
            asym_items.append(
                {
                    "pair": f"{rid}/{lid}",
                    "mirror_x_err_px": round(mir_err, 1),
                    "y_err_px": round(y_err, 1),
                    "w_diff_pct": round(w_diff * 100.0, 1),
                    "h_diff_pct": round(h_diff * 100.0, 1),
                }
            )

    if asym_items:
        worst = max(asym_items, key=lambda d: max(d["mirror_x_err_px"], d["y_err_px"], d["w_diff_pct"]))
        sev = "fail" if (worst["mirror_x_err_px"] > 35 * S or worst["y_err_px"] > 30 * S or worst["w_diff_pct"] > 22.0) else "warn"
        return [
            Finding(
                category="symmetry",
                check_id="bilateral_symmetry",
                severity=sev,
                outfit_state="default",
                title="Bilateral L/R Symmetry Drift",
                summary=f"{len(asym_items)} L/R pair(s) show asymmetry (worst: {worst['pair']} mirX={worst['mirror_x_err_px']}px, dY={worst['y_err_px']}px, dW={worst['w_diff_pct']}%).",
                metrics={"pair_count": len(pairs), "asymmetric": asym_items[:8]},
                fix_hint=f"Mirror L/R placement coordinates symmetrically around x={cx_ref:.0f} for {[a['pair'] for a in asym_items[:4]]}.",
            )
        ]

    return [
        Finding(
            category="symmetry",
            check_id="bilateral_symmetry",
            severity="pass",
            outfit_state="default",
            title="Bilateral L/R Symmetry",
            summary=f"All {len(pairs)} paired L/R layers are mirrored symmetrically around x={cx_ref:.0f}.",
            metrics={"pair_count": len(pairs), "asymmetric": []},
        )
    ]


def _audit_rig_kinematics(model: ModelBundle) -> list[Finding]:
    """Verify kinematic rig anchors (elbowHalf, elbow/wrist coords, tail pivot) match baked geometry."""
    findings: list[Finding] = []
    W, H = model.canvas
    S = H / 4800.0
    m = model.motion or {}

    # 1. elbowHalf coverage across ALL arm & sleeve layers (only when rig explicitly uses elbowHalf!)
    elbow = m.get("elbow")
    elbow_half = m.get("elbowHalf") or {}
    if elbow and elbow_half:
        arm_layers = [L for L in model.layers if L.kind in ("arm", "sleeve", "hand") and L.side in ("R", "L")]
        missing_eh = [L.lid for L in arm_layers if L.lid not in elbow_half]
        if missing_eh:
            findings.append(
                Finding(
                    category="rig_alignment",
                    check_id="elbow_half_coverage",
                    severity="fail",
                    outfit_state="all",
                    title="Missing elbowHalf Mapping on Arm/Sleeve Layers",
                    summary=f"{len(missing_eh)} arm/sleeve layer(s) missing from body.elbowHalf: {missing_eh} — arms will shear or detach when bent with sleeves toggled OFF!",
                    metrics={"missing_layers": missing_eh},
                    fix_hint=f"Add {missing_eh} to rig.json body.elbowHalf (1=upper arm/sleeve, 2=forearm/lower sleeve/cuff/hand).",
                )
            )
        else:
            findings.append(
                Finding(
                    category="rig_alignment",
                    check_id="elbow_half_coverage",
                    severity="pass",
                    outfit_state="all",
                    title="Complete elbowHalf Kinematic Coverage",
                    summary=f"All {len(arm_layers)} arm/sleeve/hand layers are mapped in elbowHalf.",
                    metrics={"mapped_layers": len(arm_layers)},
                )
            )

        # 2. Check elbow anchor coordinates vs actual upper/lower arm & sleeve overlap centers
        for s in ("R", "L"):
            e_pt = elbow.get(f"arm{s}")
            if not e_pt:
                continue
            for p_id, c_id in ((f"B03{s}", f"B04{s}"), (f"J04{s}", f"J05{s}")):
                if p_id in model.by_id and c_id in model.by_id:
                    P, C = model.by_id[p_id], model.by_id[c_id]
                    ix0, iy0 = max(P.box[0], C.box[0]), max(P.box[1], C.box[1])
                    ix1, iy1 = min(P.box[2], C.box[2]), min(P.box[3], C.box[3])
                    if ix1 > ix0 and iy1 > iy0:
                        jcx, jcy = 0.5 * (ix0 + ix1), 0.5 * (iy0 + iy1)
                        dist = math.hypot(e_pt[0] - jcx, e_pt[1] - jcy)
                        if dist > 95 * S:
                            findings.append(
                                Finding(
                                    category="rig_alignment",
                                    check_id=f"elbow_anchor_{p_id}_{c_id}",
                                    severity="warn",
                                    outfit_state="all",
                                    title=f"Elbow Pivot Drift ({p_id}/{c_id})",
                                    summary=f"rig elbow arm{s}={e_pt} is {dist:.0f}px away from actual {p_id}/{c_id} join ({jcx:.0f}, {jcy:.0f}).",
                                    metrics={"dist_px": round(dist, 1), "rig_xy": e_pt, "join_xy": [round(jcx, 1), round(jcy, 1)]},
                                    fix_hint=f"Update rig.json body.elbow.arm{s} closer to [{jcx:.0f}, {jcy:.0f}].",
                                )
                            )

    # 3. Check tail pivot lies near tail root pixels
    if model.tail and model.tail.get("id") in model.by_id:
        tl = model.by_id[model.tail["id"]]
        piv = model.tail.get("pivot")
        if piv:
            pts = _layer_points(tl, stride=2)
            if len(pts):
                dists = np.hypot(pts[:, 0] - piv[0], pts[:, 1] - piv[1])
                min_d = float(dists.min())
                if min_d > 55 * S:
                    findings.append(
                        Finding(
                            category="rig_alignment",
                            check_id="tail_pivot_anchor",
                            severity="fail" if min_d > 110 * S else "warn",
                            outfit_state="all",
                            title="Tail Pivot Off Tail Root",
                            summary=f"tail.pivot={piv} is {min_d:.0f}px away from {tl.lid}'s pixels.",
                            metrics={"min_dist_px": round(min_d, 1), "pivot": piv},
                            fix_hint=f"Move tail.pivot onto {tl.lid}'s tucked root point inside the skirt.",
                        )
                    )

    # 4. Default 3D Side-of-Face / Side-of-Jaw Planes (`F01sR`/`F01sL`) on multi-sample production rigs
    if len(model.all_layer_ids) >= 25 and model.sample_count >= 9 and "F01" in model.all_layer_ids:
        has_sr = "F01sR" in model.all_layer_ids
        has_sl = "F01sL" in model.all_layer_ids
        if has_sr and has_sl:
            findings.append(
                Finding(
                    category="rig_alignment",
                    check_id="side_face_3d",
                    severity="pass",
                    outfit_state="all",
                    title="3D Side-of-Face Turn Planes (F01sR/F01sL)",
                    summary="Camera-parallel 3D side-of-face / side-of-jaw layers (F01sR & F01sL) are wired for seamless 3/4 head turns.",
                    metrics={"has_F01sR": True, "has_F01sL": True},
                )
            )
        else:
            missing_sf = [k for k, ok in (("F01sR", has_sr), ("F01sL", has_sl)) if not ok]
            findings.append(
                Finding(
                    category="rig_alignment",
                    check_id="side_face_3d",
                    severity="warn",
                    outfit_state="all",
                    title="Missing 3D Side-of-Face Planes (F01sR/F01sL)",
                    summary=f"Multi-sample head-turn rig is missing side-of-face unfolding layer(s) {missing_sf} — head turns will look flat along the cheek & jawline.",
                    metrics={"missing": missing_sf},
                    fix_hint=(
                        "Generate camera-parallel 3D side-of-face / side-of-jaw layers (`F01sL`/`F01sR`) using "
                        "`rigkit.face3d.synthesize_side_jaw_layers()` + `fit_side_jaw_3d()`, place them immediately after `F01` "
                        "with `visible=False` (op=0 at Front), and add them to `pitchGroups` as `'face'`."
                    ),
                )
            )

    # 5. Crisp Non-Rubbery Body Motion check (prefer Test-Signal-style clean bodyFollowLR, avoid heavy turnChain torso warp!)
    if m:
        has_tc = bool(m.get("turnChain"))
        bf_lr = m.get("bodyFollowLR")
        bf = m.get("bodyFollow")
        bf0 = float(bf[0]) if isinstance(bf, list) and len(bf) >= 1 else 0.0
        warp_issues = []
        if has_tc:
            warp_issues.append("`motion.turnChain` is active (causes rubbery cylinder-squish warping across torso/hips/legs on head turns)")
        if bf_lr is None and bf0 > 0.12:
            warp_issues.append(f"`bodyFollowLR` is unset so body twists with heavy fallback `bodyFollow[0]={bf0:.2f}`")
        if warp_issues:
            findings.append(
                Finding(
                    category="rig_alignment",
                    check_id="body_motion_warp",
                    severity="warn",
                    outfit_state="all",
                    title="Rubbery Full-Body Turn Warping",
                    summary="; ".join(warp_issues) + ".",
                    metrics={"has_turnChain": has_tc, "bodyFollowLR": bf_lr, "bodyFollow0": round(bf0, 3)},
                    fix_hint=(
                        "Remove `body.turnChain` and match Test Signal's solid body motion: set `bodyFollowLR: [0.025, 0.025]`, "
                        "`bodyFollow: [0.12, 0.25]`, `bodyR: 920`, `limit.bend: 0.25`, and subtle `turnBody` (`±8px`)."
                    ),
                )
            )
        elif bf_lr is not None:
            findings.append(
                Finding(
                    category="rig_alignment",
                    check_id="body_motion_warp",
                    severity="pass",
                    outfit_state="all",
                    title="Solid Non-Rubbery Body Motion",
                    summary=f"Clean body motion configured (turnChain=off, bodyFollowLR={bf_lr}).",
                    metrics={"has_turnChain": False, "bodyFollowLR": bf_lr},
                )
            )

    return findings


def _audit_ref_fidelity(model: ModelBundle, composites: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[Finding]:
    """Optional silhouette IoU check when canonical reference PNGs with matching aspect ratio exist."""
    findings: list[Finding] = []
    W, H = model.canvas
    for state_name, ref_path in model.ref_paths.items():
        if state_name not in composites or not ref_path.exists():
            continue
        ref_im = Image.open(ref_path).convert("RGBA")
        rw, rh = ref_im.size
        # Skip refs with mismatched aspect ratio (e.g. 2:3 concept sheet vs 1:2 canonical canvas)
        if abs((rw / max(1, rh)) / (W / max(1, H)) - 1.0) > 0.12:
            continue
        if (rw, rh) != (W, H):
            ref_im = ref_im.resize((W, H), Image.LANCZOS)
        ref_a = np.asarray(ref_im.getchannel("A")) > ALPHA_THRESH
        mod_a = composites[state_name][1] > ALPHA_THRESH
        inter = int((ref_a & mod_a).sum())
        union = max(1, int((ref_a | mod_a).sum()))
        iou = inter / union
        sev = "pass" if iou >= 0.88 else ("warn" if iou >= 0.80 else "fail")
        findings.append(
            Finding(
                category="ref_fidelity",
                check_id=f"{state_name}:canon_iou",
                severity=sev,
                outfit_state=state_name,
                title=f"Canonical Silhouette Fidelity [{state_name}]",
                summary=f"Full-body silhouette IoU vs {ref_path.name} is {iou*100:.1f}%.",
                metrics={"silhouette_iou": round(iou, 4), "ref": ref_path.name},
                fix_hint=("Adjust limb/clothing proportions to match canonical reference silhouette closer (>=88% IoU)." if sev != "pass" else ""),
            )
        )
    return findings


def _compute_score(findings: list[Finding]) -> tuple[float, str]:
    weights = {
        "connectivity": {"fail": 14.0, "warn": 4.0},
        "limb_chains": {"fail": 7.5, "warn": 2.5},
        "partial_masking": {"fail": 8.0, "warn": 3.0},
        "garment_fit": {"fail": 8.0, "warn": 3.0},
        "symmetry": {"fail": 8.0, "warn": 3.0},
        "rig_alignment": {"fail": 6.0, "warn": 2.0},
        "ref_fidelity": {"fail": 6.0, "warn": 2.0},
    }
    cat_caps = {
        "connectivity": 30.0,
        "limb_chains": 34.0,
        "partial_masking": 24.0,
        "garment_fit": 26.0,
        "symmetry": 12.0,
        "rig_alignment": 12.0,
        "ref_fidelity": 10.0,
    }
    penalties: dict[str, float] = {}
    for f in findings:
        if f.severity == "pass":
            continue
        p = weights.get(f.category, {"fail": 6.0, "warn": 2.0})[f.severity]
        penalties[f.category] = min(cat_caps.get(f.category, 25.0), penalties.get(f.category, 0.0) + p)
    score = max(0.0, round(100.0 - sum(penalties.values()), 1))
    grade = "A+" if score >= 94 else ("A" if score >= 88 else ("B" if score >= 78 else ("C" if score >= 65 else ("D" if score >= 50 else "F"))))
    return score, grade


def _render_diagnostic_sheet(
    model: ModelBundle,
    composites: dict[str, tuple[np.ndarray, np.ndarray]],
    findings: list[Finding],
    score: float,
    grade: str,
    out_path: Path,
) -> None:
    """Render annotated full-body overview + auto-cropped close-ups of worst/key attachment joints."""
    W, H = model.canvas
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick up to 8 DISTINCT anatomical close-up tiles prioritizing fails -> warns -> key garment/limb checks
    sev_rank = {"fail": 0, "warn": 1, "pass": 2}
    with_roi = [f for f in findings if f.roi is not None]
    with_roi.sort(key=lambda f: (sev_rank[f.severity], f.category, f.check_id))
    selected: list[Finding] = []
    seen_keys: set[str] = set()
    for f in with_roi:
        if len(selected) >= 8:
            break
        # Deduplicate same anatomical check across multiple outfit states (keep worst severity first!)
        base_key = f.check_id.split(":")[-1]
        if base_key in seen_keys:
            continue
        seen_keys.add(base_key)
        selected.append(f)

    tw, th = 420, 420
    lbl_h = 34
    cols = 4
    rows = max(1, math.ceil(len(selected) / cols))
    ov_w = 340
    header_h = 54
    sheet_w = ov_w * len(composites) + cols * tw
    sheet_h = header_h + rows * (th + lbl_h)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)

    col_map = {"pass": (74, 222, 128), "warn": (251, 191, 36), "fail": (244, 63, 94)}
    n_fail = sum(1 for f in findings if f.severity == "fail")
    n_warn = sum(1 for f in findings if f.severity == "warn")
    n_pass = sum(1 for f in findings if f.severity == "pass")
    hdr_col = col_map["pass"] if n_fail == 0 and n_warn == 0 else (col_map["warn"] if n_fail == 0 else col_map["fail"])
    draw.text(
        (18, 16),
        f"Layer2D Polish Audit — {model.name}   |   SCORE: {score:.1f}/100 ({grade})   |   FAIL: {n_fail}  WARN: {n_warn}  PASS: {n_pass}",
        fill=hdr_col,
    )

    # Left column(s): annotated full-body composites for each outfit state
    for s_idx, (st_name, (rgb, _)) in enumerate(composites.items()):
        im = Image.fromarray(rgb, "RGB")
        # Draw ROI boxes for fails/warns in this state
        ov_draw = ImageDraw.Draw(im)
        for f in findings:
            if f.outfit_state == st_name and f.roi and f.severity != "pass":
                c = col_map[f.severity]
                ov_draw.rectangle(f.roi, outline=c, width=max(4, int(round(H / 480))))
        scale = min((ov_w - 16) / W, (sheet_h - header_h - 40) / H)
        rw, rh = int(round(W * scale)), int(round(H * scale))
        im_r = im.resize((rw, rh), Image.LANCZOS)
        ox = s_idx * ov_w + (ov_w - rw) // 2
        oy = header_h + 28 + (sheet_h - header_h - 40 - rh) // 2
        draw.text((s_idx * ov_w + 14, header_h + 6), f"State: {st_name}", fill=(220, 224, 232))
        sheet.paste(im_r, (ox, oy))

    # Right grid: zoomed close-up diagnostic tiles
    gx0 = ov_w * len(composites)
    for idx, f in enumerate(selected):
        c, r = idx % cols, idx // cols
        x0 = gx0 + c * tw
        y0 = header_h + r * (th + lbl_h)
        st_rgb = composites.get(f.outfit_state, next(iter(composites.values())))[0]
        rx0, ry0, rx1, ry1 = f.roi  # type: ignore[misc]
        crop = Image.fromarray(st_rgb[ry0:ry1, rx0:rx1], "RGB")
        # Draw guide dots if available
        if f.guides:
            cd = ImageDraw.Draw(crop)
            for key, col in (("p_end_xy", (0, 200, 255)), ("c_start_xy", (255, 40, 180))):
                if key in f.guides:
                    px, py = f.guides[key][0] - rx0, f.guides[key][1] - ry0
                    rad = max(5, int(round(max(crop.width, crop.height) * 0.018)))
                    cd.ellipse([px - rad, py - rad, px + rad, py + rad], fill=col, outline=(255, 255, 255), width=2)
        crop.thumbnail((tw - 8, th - 8), Image.LANCZOS)
        tile = Image.new("RGB", (tw, th), (242, 244, 248))
        tile.paste(crop, ((tw - crop.width) // 2, (th - crop.height) // 2))
        # Border color by severity
        td = ImageDraw.Draw(tile)
        td.rectangle([0, 0, tw - 1, th - 1], outline=col_map[f.severity], width=3)
        sheet.paste(tile, (x0, y0 + lbl_h))
        tag = f"[{f.severity.upper()}] {f.title}"
        draw.text((x0 + 8, y0 + 9), tag[:56], fill=col_map[f.severity])

    sheet.save(out_path)


def build_agent_fix_prompt(report: dict[str, Any]) -> str:
    """Build a ready-to-paste, pixel-exact remediation prompt for an AI coding agent."""
    issues = [f for f in report.get("findings", []) if f.get("severity") in ("fail", "warn")]
    if not issues:
        return (
            f"Polish audit PASSED ({report['score']}/100, {report['grade']}) with 0 FAIL and 0 WARN. "
            "No attachment or garment-fit fixes required."
        )
    fails = [f for f in issues if f["severity"] == "fail"]
    warns = [f for f in issues if f["severity"] == "warn"]
    lines = [
        f"Fix the following Layer2D polish audit defects on `{report['model']}` "
        f"(current score: {report['score']}/100 ({report['grade']}), FAIL={len(fails)}, WARN={len(warns)}):",
        "",
    ]
    for idx, f in enumerate(fails + warns, 1):
        sev = f["severity"].upper()
        lines.append(f"{idx}. [{sev}] {f['title']}")
        lines.append(f"   - Problem: {f['summary']}")
        if f.get("fix_hint"):
            lines.append(f"   - Required fix: {f['fix_hint']}")
    lines.extend(
        [
            "",
            "After patching the sprites/layout/rig:",
            "1. Rebuild & validate: `python3 l2d_tools.py validate <avatar>.l2d`",
            "2. Re-run this polish audit: `python3 polish_audit.py <avatar>.l2d --out polish_out/ --prompt`",
            "3. Inspect `polish_out/polish_audit.png` at 1:1 zoom (both jacket ON and jacket OFF states) and iterate until FAIL=0 and score >= 92 (A/A+).",
        ]
    )
    return "\n".join(lines)


def audit_model(source: Path, *, out_dir: Path | None = None, min_score: float = 0.0) -> dict[str, Any]:
    model = load_model(source)

    # Determine outfit states to audit: default + turning OFF each default-ON body-covering outfit
    default_hidden: set[str] = set()
    for o_cfg in model.outfits.values():
        if not o_cfg.get("default", True):
            default_hidden.update(o_cfg.get("hide") or [])

    states: list[tuple[str, set[str]]] = [("default", set(default_hidden))]
    for o_key, o_cfg in model.outfits.items():
        if o_cfg.get("default", True) and len(o_cfg.get("hide") or []) >= 3:
            st_hidden = set(default_hidden) | set(o_cfg.get("hide") or [])
            states.append((f"without_{o_key}", st_hidden))

    composites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    findings: list[Finding] = []
    for st_name, st_hidden in states:
        rgb, alpha = _composite_state(model, st_hidden)
        composites[st_name] = (rgb, alpha)
        findings.extend(_audit_connectivity(model, st_name, st_hidden, alpha))
        findings.extend(_audit_limb_chains(model, st_name, st_hidden, rgb))
        findings.extend(_audit_partial_masking(model, st_name, st_hidden))
        findings.extend(_audit_garment_fit(model, st_name, st_hidden, rgb))
        if st_name == "default":
            findings.extend(_audit_face_and_eyes(model, st_name, st_hidden))

    findings.extend(_audit_symmetry(model, default_hidden))
    findings.extend(_audit_rig_kinematics(model))
    findings.extend(_audit_ref_fidelity(model, composites))

    score, grade = _compute_score(findings)
    n_fail = sum(1 for f in findings if f.severity == "fail")
    n_warn = sum(1 for f in findings if f.severity == "warn")
    n_pass = sum(1 for f in findings if f.severity == "pass")

    sheet_path: str | None = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_p = out_dir / "polish_audit.png"
        _render_diagnostic_sheet(model, composites, findings, score, grade, img_p)
        sheet_path = str(img_p)

    report: dict[str, Any] = {
        "model": model.name,
        "source": model.source,
        "canvas": list(model.canvas),
        "states_audited": [s for s, _ in states],
        "score": score,
        "grade": grade,
        "ok": (n_fail == 0 and score >= min_score),
        "counts": {"pass": n_pass, "warn": n_warn, "fail": n_fail},
        "diagnostic_sheet": sheet_path,
        "findings": [
            {
                "category": f.category,
                "check_id": f.check_id,
                "severity": f.severity,
                "outfit_state": f.outfit_state,
                "title": f.title,
                "summary": f.summary,
                "metrics": f.metrics,
                "fix_hint": f.fix_hint,
            }
            for f in findings
        ],
    }
    report["agent_fix_prompt"] = build_agent_fix_prompt(report)
    if out_dir is not None:
        (out_dir / "polish_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (out_dir / "polish_fix_prompt.txt").write_text(report["agent_fix_prompt"] + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit 2D avatar polish, attachment alignment, clothing fit & symmetry")
    ap.add_argument("source", type=Path, help="Model directory (with placed/) or standalone .l2d archive")
    ap.add_argument("--out", type=Path, default=None, help="Output directory for polish_report.json, polish_audit.png & polish_fix_prompt.txt")
    ap.add_argument("--min-score", type=float, default=80.0, help="Minimum score for exit code 0 (default 80)")
    ap.add_argument("--json", action="store_true", help="Print full JSON report to stdout")
    ap.add_argument("--prompt", action="store_true", help="Print a ready-to-paste AI agent remediation prompt for all FAIL/WARN issues")
    args = ap.parse_args()

    rep = audit_model(args.source, out_dir=args.out, min_score=args.min_score)
    if args.json:
        print(json.dumps(rep, indent=2))
    elif args.prompt:
        print(rep["agent_fix_prompt"])
    else:
        print(f"=== Layer2D Polish Audit: {rep['model']} ===")
        print(
            f"Score: {rep['score']}/100 ({rep['grade']})  |  "
            f"PASS={rep['counts']['pass']}  WARN={rep['counts']['warn']}  FAIL={rep['counts']['fail']}"
        )
        if rep["diagnostic_sheet"]:
            print(f"Diagnostic Sheet: {rep['diagnostic_sheet']}")
        for f in rep["findings"]:
            if f["severity"] != "pass":
                icon = "FAIL" if f["severity"] == "fail" else "WARN"
                print(f"  [{icon}] {f['title']}: {f['summary']}")
                if f["fix_hint"]:
                    print(f"         Fix: {f['fix_hint']}")
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
