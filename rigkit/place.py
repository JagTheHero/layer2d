"""Place cut parts onto a 4K commercial Live2D canvas, fit to canon, and bake aligned RGBA layers.

Supports:
- Exact affine placement (cx, cy, w, h, rot)
- Thin-plate spline (`warp_mode: "tps"`) or piecewise-affine (`warp_mode: "affine"`) landmark warps
- Polygon erase (`cut`) with soft anti-aliased edge
- Straight-cut joint feathering (`fade`: {"top"/"bottom"/"left"/"right": px})
- Optional drawn outline reinforcement (`ink`), RGB gain (`gain`), opacity (`alpha`), and solid-alpha hardening
- Differentiable PyTorch coarse-to-fine alignment (`fit`) with follower ties (`ties`) so hidden toggle under-layers
  stay locked to their covering garment
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

from rigkit.prep import bleed_rgb_edges, harden_solid_alpha


def start_pose(L: dict, im: Image.Image) -> tuple[float, float, float, float, float]:
    pw, ph = im.size
    w, h = L.get("w"), L.get("h")
    sx = w / pw if w else (h / ph if h else 1.0)
    sy = h / ph if h else sx
    return float(L["cx"]), float(L["cy"]), float(sx), float(sy), math.radians(float(L.get("rot") or 0.0))


def _load_parts_map(model_dir: Path) -> dict[str, Path]:
    idx = json.loads((model_dir / "parts/index.json").read_text())
    return {e["id"]: model_dir / "parts" / e["file"] for e in idx}


def _apply_warp(tex: Image.Image, x0: int, y0: int, warp: list, mode: str) -> tuple[Image.Image, int, int, int, int]:
    pad = int(math.ceil(max(math.dist(p, q) for p, q in warp))) + 6
    grown = Image.new("RGBA", (tex.width + 2 * pad, tex.height + 2 * pad), (0, 0, 0, 0))
    grown.paste(tex, (pad, pad))
    tex = grown
    x0 -= pad
    y0 -= pad
    x1, y1 = x0 + tex.width, y0 + tex.height

    src = np.array([p for p, _ in warp], dtype=np.float64) - (x0, y0)
    dst = np.array([q for _, q in warp], dtype=np.float64) - (x0, y0)
    th, tw = tex.height, tex.width
    step = 8
    gy, gx = np.mgrid[0 : th + step : step, 0 : tw + step : step]

    if mode == "affine":
        box = np.array([[-16, -16], [tw + 16, -16], [-16, th + 16], [tw + 16, th + 16]], dtype=np.float64)
        D, S = np.vstack([dst, box]), np.vstack([src, box])
        tri = Delaunay(D)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
        k = tri.find_simplex(pts)
        b = np.einsum("nij,nj->ni", tri.transform[k, :2], pts - tri.transform[k, 2])
        bc = np.c_[b, 1.0 - b.sum(1)]
        off = (np.einsum("ni,nij->nj", bc, S[tri.simplices[k]]) - pts).reshape(gy.shape + (2,))
    else:
        back = RBFInterpolator(dst, src - dst, kernel="thin_plate_spline")
        off = back(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(gy.shape + (2,))

    yy, xx = np.mgrid[0:th, 0:tw]
    ox = ndimage.map_coordinates(off[..., 0], [yy / step, xx / step], order=1)
    oy = ndimage.map_coordinates(off[..., 1], [yy / step, xx / step], order=1)

    a = np.asarray(tex, dtype=np.float32).copy()
    a[..., :3] *= a[..., 3:] / 255.0
    out = np.stack([ndimage.map_coordinates(a[..., c], [yy + oy, xx + ox], order=1) for c in range(4)], axis=-1)
    alpha = out[..., 3:]
    out[..., :3] = np.where(alpha > 1e-3, out[..., :3] / np.maximum(alpha, 1e-3) * 255.0, 0.0)
    tex = Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8), "RGBA")
    return tex, x0, y0, x1, y1


def bake(model_dir: Path, *, layout_file: str = "layout.json", out_subdir: str = "placed") -> dict:
    layout = json.loads((model_dir / layout_file).read_text())
    W, H = layout["canvas"]
    files = _load_parts_map(model_dir)
    params_path = model_dir / "work/place.json"
    params = (
        {k: tuple(v) for k, v in json.loads(params_path.read_text()).items()}
        if params_path.exists()
        else None
    )

    out_dir = model_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir = model_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    canvas_default = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    clip_alphas: dict[str, np.ndarray] = {}

    for L in layout["layers"]:
        lid = L["id"]
        src_path = model_dir / L["file"] if "file" in L else files[L.get("src", lid)]
        im = Image.open(src_path).convert("RGBA")
        if L.get("harden", True):
            im = harden_solid_alpha(im)
        if L.get("bleed", True):
            im = bleed_rgb_edges(im)

        cx, cy, sx, sy, r = params[lid] if (params and lid in params and not L.get("lock_pose")) else start_pose(L, im)
        pw, ph = im.size
        if abs(sx - 1.0) < 1e-5 and abs(sy - 1.0) < 1e-5 and abs(r) < 1e-5:
            # Exact lossless 1:1 pixel placement (zero uint8 premultiply/bicubic quantization rings on soft shading/blush layers!)
            x0 = int(round(cx - pw / 2.0))
            y0 = int(round(cy - ph / 2.0))
            x1, y1 = x0 + pw, y0 + ph
            tex = im.copy()
        else:
            c, s = math.cos(r), math.sin(r)
            hw, hh = pw * sx / 2.0, ph * sy / 2.0
            ex, ey = abs(c) * hw + abs(s) * hh, abs(s) * hw + abs(c) * hh
            x0, y0 = int(math.floor(cx - ex)) - 2, int(math.floor(cy - ey)) - 2
            x1, y1 = int(math.ceil(cx + ex)) + 2, int(math.ceil(cy + ey)) + 2

            ox, oy = x0 - cx, y0 - cy
            a_, b_ = c / sx, -s / sx
            d_, e_ = s / sy, c / sy
            coeffs = (a_, b_, a_ * ox + b_ * oy + pw / 2.0, d_, e_, d_ * ox + e_ * oy + ph / 2.0)

            # Transform premultiplied to prevent dark/white edge fringes
            arr_pm = np.asarray(im, dtype=np.float32)
            arr_pm[..., :3] *= arr_pm[..., 3:] / 255.0
            im_pm = Image.fromarray(np.clip(arr_pm, 0, 255).round().astype(np.uint8), "RGBA")
            tex_pm = im_pm.transform((x1 - x0, y1 - y0), Image.AFFINE, coeffs, Image.BICUBIC)
            t_arr = np.asarray(tex_pm, dtype=np.float32)
            ta = t_arr[..., 3:]
            t_arr[..., :3] = np.where(ta > 1.0, t_arr[..., :3] / np.maximum(ta, 1.0) * 255.0, 0.0)
            tex = Image.fromarray(np.clip(t_arr, 0, 255).round().astype(np.uint8), "RGBA")

        if L.get("warp"):
            tex, x0, y0, x1, y1 = _apply_warp(tex, x0, y0, L["warp"], L.get("warp_mode", "tps"))

        if L.get("cut"):
            hole = Image.new("L", tex.size, 0)
            polys = L["cut"] if isinstance(L["cut"][0][0], (list, tuple)) else [L["cut"]]
            for poly in polys:
                ImageDraw.Draw(hole).polygon([(x - x0, y - y0) for x, y in poly], fill=255)
            a = np.asarray(tex, dtype=np.float32).copy()
            soft = float(L.get("cut_soft", 1.5))
            mask = ndimage.gaussian_filter(np.asarray(hole, dtype=np.float32) / 255.0, soft) if soft > 0 else np.asarray(hole, dtype=np.float32) / 255.0
            a[..., 3] *= 1.0 - mask
            tex = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8), "RGBA")

        if L.get("ink"):
            ink, k = L["ink"], 4
            line = Image.new("L", (tex.width * k, tex.height * k), 0)
            for pl in ink["lines"]:
                ImageDraw.Draw(line).line(
                    [((x - x0) * k, (y - y0) * k) for x, y in pl],
                    fill=255,
                    width=round(ink.get("width", 3) * k),
                    joint="curve",
                )
            m = np.asarray(line.resize(tex.size, Image.LANCZOS), dtype=np.float32)[..., None] / 255.0
            a = np.asarray(tex, dtype=np.float32).copy()
            a[..., :3] = a[..., :3] * (1.0 - m) + np.array(ink.get("color", [35, 30, 38]), dtype=np.float32) * m
            a[..., 3:] = np.maximum(a[..., 3:], 255.0 * m)
            tex = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8), "RGBA")

        if L.get("gain"):
            a = np.asarray(tex, dtype=np.float32).copy()
            a[..., :3] *= np.array(L["gain"], dtype=np.float32)
            tex = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8), "RGBA")

        if L.get("fade"):
            a = np.asarray(tex, dtype=np.float32).copy()
            nz_y = np.nonzero(a[..., 3].max(1) > 8)[0]
            nz_x = np.nonzero(a[..., 3].max(0) > 8)[0]
            if len(nz_y) and len(nz_x):
                yy = np.arange(a.shape[0])[:, None]
                xx = np.arange(a.shape[1])[None, :]
                if "bottom" in L["fade"]:
                    a[..., 3] *= np.clip((nz_y.max() - yy) / L["fade"]["bottom"], 0, 1)
                if "top" in L["fade"]:
                    a[..., 3] *= np.clip((yy - nz_y.min()) / L["fade"]["top"], 0, 1)
                if "left" in L["fade"]:
                    a[..., 3] *= np.clip((xx - nz_x.min()) / L["fade"]["left"], 0, 1)
                if "right" in L["fade"]:
                    a[..., 3] *= np.clip((nz_x.max() - xx) / L["fade"]["right"], 0, 1)
            tex = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8), "RGBA")

        if L.get("alpha") is not None:
            a = np.asarray(tex, dtype=np.float32).copy()
            a[..., 3] *= float(L["alpha"])
            tex = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8), "RGBA")

        # Tight trim transparent padding & bleed RGB cleanly for WebGL/Live2D
        arr = np.asarray(tex, dtype=np.uint8)
        nz_y, nz_x = np.nonzero(arr[..., 3] > 2)
        if len(nz_y) > 0:
            ty0, ty1 = max(0, int(nz_y.min()) - 2), min(arr.shape[0], int(nz_y.max()) + 3)
            tx0, tx1 = max(0, int(nz_x.min()) - 2), min(arr.shape[1], int(nz_x.max()) + 3)
            tex = Image.fromarray(arr[ty0:ty1, tx0:tx1], "RGBA")
            x0, y0 = x0 + tx0, y0 + ty0
            x1, y1 = x0 + tex.width, y0 + tex.height
        if L.get("bleed", True):
            tex = bleed_rgb_edges(tex, solid_thresh=180, max_dist=10)
        tex.save(out_dir / f"{lid}.png")

        hires_path = out_dir / f"{lid}_hires.png"
        if L.get("hires_tex") and abs(r) < 1e-5 and not L.get("warp"):
            # Preserve full native source resolution aligned 1:1 to trimmed canonical box [x0, y0, x1, y1]
            # (prevents blocky/jagged upscaling when a rest-tucked layer expands 2-3x on turns!)
            u0 = (x0 - (cx - pw * sx / 2.0)) / sx
            v0 = (y0 - (cy - ph * sy / 2.0)) / sy
            u1 = (x1 - (cx - pw * sx / 2.0)) / sx
            v1 = (y1 - (cy - ph * sy / 2.0)) / sy
            hr_w = max(tex.width, int(round(u1 - u0)))
            hr_h = max(tex.height, int(round(v1 - v0)))
            hr_pm = np.asarray(im, dtype=np.float32)
            hr_pm[..., :3] *= hr_pm[..., 3:] / 255.0
            hr_src = Image.fromarray(np.clip(hr_pm, 0, 255).round().astype(np.uint8), "RGBA")
            hr_tr = hr_src.transform((hr_w, hr_h), Image.AFFINE, ((u1 - u0) / hr_w, 0.0, u0, 0.0, (v1 - v0) / hr_h, v0), Image.BICUBIC)
            hr_a = np.asarray(hr_tr, dtype=np.float32)
            ha = hr_a[..., 3:]
            hr_a[..., :3] = np.where(ha > 1.0, hr_a[..., :3] / np.maximum(ha, 1.0) * 255.0, 0.0)
            hr_im = Image.fromarray(np.clip(hr_a, 0, 255).round().astype(np.uint8), "RGBA")
            if L.get("bleed", True):
                hr_im = bleed_rgb_edges(hr_im, solid_thresh=140, max_dist=10)
            hr_im.save(hires_path)
        elif hires_path.exists():
            hires_path.unlink()

        skip_keys = {"cx", "cy", "w", "h", "rot", "cut", "cut_soft", "fade", "alpha", "warp", "warp_mode", "ink", "gain", "harden", "bleed", "lock_pose", "hires_tex"}
        entry = {k: v for k, v in L.items() if k not in skip_keys}
        entry["box"] = [int(x0), int(y0), int(x1), int(y1)]
        entry["sheer"] = bool(L.get("sheer", False))
        meta.append(entry)

        # Composite visible default layers for review
        if L.get("visible", True):
            full_im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dx, dy = max(0, x0), max(0, y0)
            sx0, sy0 = max(0, -x0), max(0, -y0)
            if dx < W and dy < H and sx0 < tex.width and sy0 < tex.height:
                crop = tex.crop((sx0, sy0, min(tex.width, sx0 + W - dx), min(tex.height, sy0 + H - dy)))
                full_im.paste(crop, (dx, dy))
            if L.get("clip") in clip_alphas:
                fa = np.asarray(full_im, dtype=np.float32).copy()
                fa[..., 3] *= clip_alphas[L["clip"]]
                full_im = Image.fromarray(np.clip(fa, 0, 255).round().astype(np.uint8), "RGBA")
            if any(M.get("clip") == lid for M in layout["layers"]):
                clip_alphas[lid] = np.asarray(full_im.getchannel("A"), dtype=np.float32) / 255.0
            canvas_default.alpha_composite(full_im)

    (out_dir / "layers.json").write_text(json.dumps({"canvas": [W, H], "layers": meta}, indent=1))
    bg = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    bg.alpha_composite(canvas_default)
    bg.convert("RGB").resize((W // 2, H // 2), Image.LANCZOS).save(review_dir / "place_baked.png")
    canvas_default.save(review_dir / "place_baked_rgba.png")
    print(f"Baked {len(meta)} layers -> {out_dir}")
    return {"layers": len(meta), "canvas": [W, H]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--layout", default="layout.json")
    a = ap.parse_args()
    bake(a.model_dir, layout_file=a.layout)


if __name__ == "__main__":
    main()
