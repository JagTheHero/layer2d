"""1:1 pixel & ink-line comparison of a baked model against its canonical reference.

Usage:
    python -m rigkit.compare <model_dir> [region ...] [--ref refs/canon_full.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

DIFF_THRESH = 0.12
GRID_STEP = 50


def ink_mask(rgb: np.ndarray) -> np.ndarray:
    lum = rgb.mean(-1)
    return (lum < 0.30) | (ndimage.gaussian_filter(lum, 3.0) - lum > 0.12)


def draw_grid(im: Image.Image, box: tuple[int, int, int, int], k: float) -> Image.Image:
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = box
    for x in range((x0 // GRID_STEP + 1) * GRID_STEP, x1, GRID_STEP):
        col = (220, 0, 0) if x % 200 == 0 else (255, 165, 165)
        d.line([((x - x0) * k, 0), ((x - x0) * k, im.height)], fill=col)
        if x % 200 == 0:
            d.text(((x - x0) * k + 2, 2), str(x), fill=(200, 0, 0))
    for y in range((y0 // GRID_STEP + 1) * GRID_STEP, y1, GRID_STEP):
        col = (0, 0, 200) if y % 200 == 0 else (165, 165, 255)
        d.line([(0, (y - y0) * k), (im.width, (y - y0) * k)], fill=col)
        if y % 200 == 0:
            d.text((2, (y - y0) * k + 2), str(y), fill=(0, 0, 180))
    return im


def compare_regions(
    model_dir: Path,
    regions: dict[str, tuple[int, int, int, int]],
    ref_rel: str = "refs/canon_full.png",
    baked_rel: str = "review/place_baked_rgba.png",
    out_subdir: str = "review/compare",
    scale: float | None = None,
) -> dict[str, dict]:
    ref_im = Image.open(model_dir / ref_rel).convert("RGBA")
    W, H = ref_im.size
    bg_r = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    bg_r.alpha_composite(ref_im)
    C = np.asarray(bg_r.convert("RGB"), dtype=np.float32) / 255.0

    mod_im = Image.open(model_dir / baked_rel).convert("RGBA")
    bg_m = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    bg_m.alpha_composite(mod_im)
    M = np.asarray(bg_m.convert("RGB"), dtype=np.float32) / 255.0

    out_dir = model_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for name, box in regions.items():
        x0, y0, x1, y1 = box
        c, m = C[y0:y1, x0:x1], M[y0:y1, x0:x1]
        ic, im_ = ink_mask(c), ink_mask(m)
        ov = np.ones_like(c)
        ov[ic & ~im_] = (0.9, 0.1, 0.1)
        ov[im_ & ~ic] = (0.1, 0.3, 0.95)
        ov[ic & im_] = (0.35, 0.05, 0.4)

        d = np.abs(c - m).max(-1) > DIFF_THRESH
        darker = m.mean(-1) < c.mean(-1)
        bl = 0.5 * (c + m)
        bl[d & darker] = bl[d & darker] * 0.4 + np.array([0.9, 0.1, 0.1]) * 0.6
        bl[d & ~darker] = bl[d & ~darker] * 0.4 + np.array([0.0, 0.8, 0.9]) * 0.6

        near = ndimage.binary_dilation(im_, iterations=4)
        hit = float((ic & near).sum() / max(1, ic.sum()))
        diff_pct = float(d.mean() * 100.0)
        stats[name] = {"diff_pct": round(diff_pct, 2), "ink_hit_pct": round(hit * 100.0, 2)}

        k = scale or min(1.0, 750.0 / max(x1 - x0, y1 - y0))
        pw, ph = round((x1 - x0) * k), round((y1 - y0) * k)
        panels = [
            draw_grid(
                Image.fromarray((p * 255).round().astype(np.uint8)).resize(
                    (pw, ph), Image.LANCZOS if k < 1 else Image.NEAREST
                ),
                box,
                k,
            )
            for p in (c, m, ov, bl)
        ]
        S = Image.new("RGB", (pw * 2 + 8, ph * 2 + 8), "white")
        for i, p in enumerate(panels):
            S.paste(p, ((i % 2) * (pw + 8), (i // 2) * (ph + 8)))
        S.save(out_dir / f"{name}.png")
        print(f"{name:10s}  diff={diff_pct:5.1f}%   ink@4px={hit*100:5.1f}%")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--ref", default="refs/canon_full.png")
    ap.add_argument("--scale", type=float, default=None)
    a = ap.parse_args()
    cfg = json.loads((a.model_dir / "model.json").read_text())
    regs = cfg["compare_regions"]
    if a.names:
        regs = {n: regs[n] for n in a.names}
    compare_regions(a.model_dir, regs, ref_rel=a.ref, scale=a.scale)


if __name__ == "__main__":
    main()
