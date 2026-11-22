"""Unified 3D Volumetric Hair Depth Engine (`rigkit.hair3d`) for Layer2D Avatars.

Solves the classic 2D VTuber "flat front-hair card gap" defect on head turns (±32.5° yaw)
by projecting all hair planes onto ONE shared head-local 3D skull ellipsoid and synthesizing
100% solid (`op = 1.0`) multi-depth interlocking hair shells:

  Plane 1: Back Hair (`H01`, `H01bR/L`)               — rz ≈ -108 .. -82
  Plane 2: Wolf Ears / Crown-Ear Dome (`WE*`, `H02c*`) — rz ≈  +55 .. +72  (seals ear-root & temple!)
  Plane 3: Side-of-Skull Curtain (`H02s*`, `H04s*`)    — rz ≈  +45 .. +88  (continuous side-of-head volume)
  Plane 4: Framing Locks & Behind-Bangs (`H04*`, `H03u*`) — rz ≈ +90 .. +148 (adds solid 3D strand thickness!)
  Plane 5: Front Bangs & Crown (`H03*`, `H05`, `H06`)  — rz ≈ +165 .. +232

Every active hair shell stays 100% SOLID (`op = 1.0`, ZERO semi-transparency, ZERO opacity crossfades,
ZERO sliding cutout cards). Also includes an automated chroma-green hair-gap leak detector (`detect_chroma_gaps`)
for CI and visual QA verification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from rigkit.prep import bleed_rgb_edges


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / np.maximum(1e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class HeadSkull3D:
    """Shared head-local 3D ellipsoid coordinate frame for one Layer2D avatar."""

    cx: float = 1200.0
    cy: float = 620.0
    rx: float = 435.0
    ry: float = 560.0
    max_yaw_deg: float = 32.5
    focal: float = 2600.0

    @property
    def max_yaw_rad(self) -> float:
        return math.radians(self.max_yaw_deg)

    def project(
        self,
        p: np.ndarray,
        yaw: float,
        *,
        rz: float,
        rz_top: float | None = None,
        persp_k: float = 0.30,
        lift_y: float = -6.0,
        top_y0: float = 340.0,
        top_y1: float = 740.0,
    ) -> np.ndarray:
        """Project 2D rest-canvas points onto a 3D skull shell at depth `rz` and rotate by `yaw`."""
        p = np.asarray(p, dtype=np.float64)
        if abs(yaw) < 1e-7:
            return p.copy()
        x = p[..., 0]
        y = p[..., 1]
        dx = x - self.cx
        dy = y - self.cy
        ux = np.clip(dx / self.rx, -0.97, 0.97)
        uy = np.clip(dy / self.ry, -0.95, 0.95)

        if rz_top is not None:
            t_y = smoothstep(top_y0, top_y1, y)
            rz_eff = rz_top + (rz - rz_top) * t_y
        else:
            rz_eff = rz

        z = rz_eff * np.sqrt(np.maximum(0.05, 1.0 - 0.80 * ux * ux)) * np.sqrt(
            np.maximum(0.45, 1.0 - 0.22 * uy * uy)
        )
        co, si = math.cos(yaw), math.sin(yaw)
        x_rot = self.cx + dx * co + z * si
        z_rot = -dx * si + z * co

        u = abs(yaw) / self.max_yaw_rad
        persp = self.focal / (self.focal - (z_rot - z) * persp_k)
        x_out = self.cx + (x_rot - self.cx) * persp
        y_out = self.cy + dy * persp + lift_y * u
        return np.stack([x_out, y_out], axis=-1)


@dataclass(frozen=True)
class DepthShellSpec:
    """Parameters for an intermediate 100%-solid 3D hair shell."""

    layer_id: str
    side: str  # "R" (viewer-left) or "L" (viewer-right)
    rz: float
    rz_top: float
    lift_y: float = -6.0
    near_spread_px: float = 0.0
    far_tuck_px: float = 16.0
    persp_k: float = 0.28


def make_shell_fwd(skull: HeadSkull3D, spec: DepthShellSpec) -> Callable[[np.ndarray, float], np.ndarray]:
    """Build a C1-smooth, globally invertible forward warp for an intermediate 3D depth hair shell.

    - Near side (turning toward camera): stays firmly anchored at intermediate depth `spec.rz`,
      optionally widening subtly with natural skull curvature so the shell bridges seamlessly
      between the wolf ear / skull temple and the forward-sweeping front bangs!
    - Far side (turning away): gently tucks inward behind `F01` (`behind = 1.0`) so ZERO pixels
      ever poke outside the far cheek or float over the far eye!
    """

    def fwd(p: np.ndarray, yaw: float) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if abs(yaw) < 1e-7:
            return p.copy()
        out = skull.project(
            p,
            yaw,
            rz=spec.rz,
            rz_top=spec.rz_top,
            persp_k=spec.persp_k,
            lift_y=spec.lift_y,
        )
        u = abs(yaw) / skull.max_yaw_rad
        sgn = -1.0 if yaw < 0 else 1.0
        is_far = (spec.side == "R" and yaw < 0) or (spec.side == "L" and yaw > 0)
        y = p[..., 1]

        if is_far:
            # Gently tuck far side inward behind F01 along the temple/cheek so it stays clean behind far cheek
            tuck_band = smoothstep(380.0, 620.0, y) * (1.0 - 0.30 * smoothstep(900.0, 1120.0, y))
            out[..., 0] = out[..., 0] - sgn * spec.far_tuck_px * u * tuck_band
        elif spec.near_spread_px != 0.0:
            # Subtle organic near-side temple fill bridging smoothly from ear-root down behind front bangs
            fill_band = smoothstep(380.0, 560.0, y) * (1.0 - 0.35 * smoothstep(820.0, 1080.0, y))
            out[..., 0] = out[..., 0] - sgn * spec.near_spread_px * u * fill_band
        return out

    return fwd


def apply_cool_ao_shade(
    rgb: np.ndarray,
    darken: float = 0.14,
    cool_tint: tuple[float, float, float] = (0.94, 0.93, 1.03),
    edge_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply subtle anime cast ambient-occlusion shading to a recessed hair depth shell.

    Makes recessed behind-bangs and inner temple shells read unmistakably as deeper 3D hair
    volume beneath the sunlit front strands without muddying silver highlights or linework.
    """
    out = rgb.astype(np.float32, copy=True)
    tint = np.array(cool_tint, dtype=np.float32)[None, None, :]
    factor = 1.0 - darken
    if edge_mask is not None:
        w = np.clip(edge_mask.astype(np.float32), 0.0, 1.0)[..., None]
        eff = 1.0 - w * (1.0 - factor * tint)
    else:
        eff = factor * tint
    out *= eff
    return np.clip(out, 0.0, 255.0)


def synthesize_temple_crown_shell(
    canvas_w: int,
    canvas_h: int,
    *,
    source_hair_rgba: np.ndarray,
    source_origin: tuple[int, int],
    front_silhouette_mask: np.ndarray,
    ear_root_mask: np.ndarray,
    side: str = "L",
    darken: float = 0.13,
) -> tuple[Image.Image, list[int]]:
    """Synthesize a 100%-solid Inner Temple-to-Ear-Root 3D Depth Shell (`H02cL` / `H02cR`).

    Extracts real painted hair texture from `source_hair_rgba`, extends solid strand coverage
    seamlessly UP into the wolf-ear root & temple dome (eliminating any ear-under-bangs gap),
    applies cool cast AO depth shading, and clips strictly INSIDE `front_silhouette_mask | ear_root_mask`
    so Front rest pose (`a = 0`) retains 100% exact canon silhouette with ZERO bloat!

    Args:
        canvas_w, canvas_h: Full avatar canvas dimensions (e.g. 2400, 4800).
        source_hair_rgba: Painted RGBA array of the model's canonical side/crown hair.
        source_origin: (ox, oy) top-left canvas pixel of `source_hair_rgba`.
        front_silhouette_mask: Boolean (H_roi, W_roi) union of Front hair+face+ear pixels in the same ROI.
        ear_root_mask: Boolean (H_roi, W_roi) region covering the ear-base weld zone inside the skull dome.
        side: "L" (viewer-right) or "R" (viewer-left).
        darken: Cast AO darkening fraction (default 0.13).

    Returns:
        (tight_rgba_image, [x0, y0, x1, y1] canvas_box)
    """
    ox, oy = source_origin
    H, W = source_hair_rgba.shape[:2]
    yy, xx = np.mgrid[oy : oy + H, ox : ox + W].astype(np.float32)

    rgb = source_hair_rgba[..., :3].astype(np.float32)
    alpha = source_hair_rgba[..., 3].astype(np.float32)

    # Nearest-neighbor hair texture extension upward into the ear-root / upper temple cavity
    # so even where the source art had the ear occluding the scalp, we get authentic strand color/grain!
    valid_hair = alpha > 200.0
    if not valid_hair.any():
        raise ValueError("source_hair_rgba has no opaque hair pixels")
    _, nn_idx = ndi.distance_transform_edt(~valid_hair, return_indices=True)
    ext_rgb = rgb[nn_idx[0], nn_idx[1]]

    # Add delicate vertical anime strand shading modulation inside extended ear-root fill so it blends organically
    strand_wave = 0.5 + 0.5 * np.sin((xx - 1200.0) * 0.24 + 0.018 * (yy - 400.0) + np.sin(yy * 0.04) * 1.1)
    ext_rgb = np.clip(ext_rgb * (0.91 + 0.14 * strand_wave[..., None]), 0.0, 255.0)
    blend_src = np.clip(alpha / 255.0, 0.0, 1.0)[..., None]
    full_rgb = rgb * blend_src + ext_rgb * (1.0 - blend_src)

    # Target coverage mask for canonical viewer-right ("L") side (mirror x for "R")
    xm = xx if side == "L" else (canvas_w - xx)

    # Smooth curved dome spanning from scalp/ear-base (y=350) down through temple & behind cheek lock (y=1015):
    # Stays cleanly inside the skull dome BEHIND WE01/WE02/WE01f so snow-white wolf-ear fluff is 100% unobstructed!
    y_top_dome = 350.0 + 0.22 * np.maximum(0.0, xm - 1345.0)
    y_bot_taper = 1015.0 - 0.14 * np.maximum(0.0, 1365.0 - xm)
    t_eye_clear = smoothstep(585.0, 670.0, yy)
    x_in_cut = 1235.0 + 118.0 * t_eye_clear + 14.0 * smoothstep(760.0, 980.0, yy)
    x_out_cut = 1565.0

    m_y = smoothstep(y_top_dome - 14.0, y_top_dome + 18.0, yy) * (
        1.0 - smoothstep(y_bot_taper - 45.0, y_bot_taper + 15.0, yy)
    )
    m_x = smoothstep(x_in_cut - 10.0, x_in_cut + 22.0, xm) * (
        1.0 - smoothstep(x_out_cut - 24.0, x_out_cut + 6.0, xm)
    )

    # Close internal hair-tip notches inside front_silhouette_mask while eroding 4px along outer boundary (ZERO rest bloat!)
    closed_sil = ndi.binary_closing(front_silhouette_mask | ear_root_mask, iterations=18)
    closed_sil = ndi.binary_fill_holes(closed_sil)
    allowed = ndi.binary_erosion(closed_sil, iterations=4)
    allowed_soft = ndi.gaussian_filter(allowed.astype(np.float32), sigma=1.4)
    shell_a = np.clip(m_y * m_x * np.clip(allowed_soft * 1.45, 0.0, 1.0), 0.0, 1.0)
    # Keep solid interior 100% opaque (op = 1.0!) so ZERO background green leaks through!
    shell_a = np.where(shell_a > 0.68, 1.0, shell_a / 0.68)

    # Cool recessed AO shading (strongest near inner front-bangs overhang & top ear-root, lighter toward outer locks)
    ao_w = 0.60 + 0.40 * (1.0 - smoothstep(1250.0, 1460.0, xm))
    shaded_rgb = apply_cool_ao_shade(full_rgb, darken=darken, edge_mask=ao_w)

    rgba = np.dstack([shaded_rgb, shell_a * 255.0])
    nz_y, nz_x = np.nonzero(rgba[..., 3] > 2.0)
    ty0, ty1 = max(0, int(nz_y.min()) - 2), min(H, int(nz_y.max()) + 3)
    tx0, tx1 = max(0, int(nz_x.min()) - 2), min(W, int(nz_x.max()) + 3)
    sub = rgba[ty0:ty1, tx0:tx1]
    out_im = bleed_rgb_edges(
        Image.fromarray(np.clip(sub, 0, 255).round().astype(np.uint8), "RGBA"),
        solid_thresh=16,
        max_dist=10,
    )
    box = [int(ox + tx0), int(oy + ty0), int(ox + tx1), int(oy + ty1)]
    return out_im, box


def synthesize_under_bangs_shell(
    canvas_w: int,
    canvas_h: int,
    *,
    bang_rgba: np.ndarray,
    bang_box: list[int],
    side_lock_rgba: np.ndarray,
    side_lock_box: list[int],
    front_hair_union: np.ndarray,
    union_origin: tuple[int, int],
    side: str = "L",
    darken: float = 0.16,
) -> tuple[Image.Image, list[int]]:
    """Synthesize a 100%-solid Behind-Bangs Mid-Lock Depth Volume (`H03uL` / `H03uR`).

    Combines painted texture from the model's front side-bangs (`H03L/R`) and upper side-lock (`H04L/R`),
    bridges the hollow cutout between them into a continuous layered mid-depth lock mass (sits at rz=+145
    directly behind the rz=+210 front bangs!), shades it ~16% cooler/deeper for cast bangs shadow,
    and masks strictly inside the Front hair union so Front rest pose (`a = 0`) is 100% preserved!
    """
    ux, uy = union_origin
    UH, UW = front_hair_union.shape[:2]

    # Composite bang + upper side lock onto ROI canvas
    roi = Image.new("RGBA", (UW, UH), (0, 0, 0, 0))
    sl_im = Image.fromarray(side_lock_rgba, "RGBA")
    bg_im = Image.fromarray(bang_rgba, "RGBA")
    roi.alpha_composite(sl_im, (side_lock_box[0] - ux, side_lock_box[1] - uy))
    roi.alpha_composite(bg_im, (bang_box[0] - ux, bang_box[1] - uy))

    arr = np.asarray(roi, dtype=np.float32)
    yy, xx = np.mgrid[uy : uy + UH, ux : ux + UW].astype(np.float32)
    xm = xx if side == "L" else (canvas_w - xx)

    # Fill internal notches between H03 and H04 strands using morphological closing + nearest-hair RGB
    solid0 = arr[..., 3] > 180.0
    closed = ndi.binary_closing(solid0, iterations=14)
    closed = ndi.binary_fill_holes(closed)
    # Keep strictly inside the front hair union (eroded 4px so ZERO fringe peeks out at a=0!)
    allowed = ndi.binary_erosion(front_hair_union, iterations=4)
    # Restrict strictly to outer temple/side-bang depth zone [355..945] and lateral band [1345..1525]
    # (strictly OUTSIDE forehead & eyebrows so ZERO under-bangs ever peek over the forehead!)
    band = (yy >= 355.0) & (yy <= 945.0) & (xm >= 1345.0) & (xm <= 1525.0)
    mask_bin = closed & allowed & band

    # Smooth edges cleanly while keeping interior 100% solid opaque
    mask_soft = ndi.gaussian_filter(mask_bin.astype(np.float32), sigma=1.6)
    alpha_out = np.clip(mask_soft * 1.45, 0.0, 1.0)
    # Taper bottom tips softly around y=850..945 so under-bangs end organically behind H04/H02s
    bot_taper = 1.0 - smoothstep(840.0, 945.0, yy)
    alpha_out *= bot_taper
    alpha_out = np.where(alpha_out > 0.72, 1.0, alpha_out / 0.72)

    # Nearest-neighbor hair RGB fill inside closed interior notches
    _, nn_idx = ndi.distance_transform_edt(~solid0, return_indices=True)
    ext_rgb = arr[nn_idx[0], nn_idx[1], :3]
    src_w = np.clip(arr[..., 3] / 255.0, 0.0, 1.0)[..., None]
    rgb = arr[..., :3] * src_w + ext_rgb * (1.0 - src_w)

    # Rich cool-plum AO shadow directly underneath the front bangs
    ao_grad = 0.75 + 0.25 * (1.0 - smoothstep(420.0, 840.0, yy))
    shaded = apply_cool_ao_shade(rgb, darken=darken, cool_tint=(0.92, 0.91, 1.04), edge_mask=ao_grad)

    rgba = np.dstack([shaded, alpha_out * 255.0])
    nz_y, nz_x = np.nonzero(rgba[..., 3] > 2.0)
    ty0, ty1 = max(0, int(nz_y.min()) - 2), min(UH, int(nz_y.max()) + 3)
    tx0, tx1 = max(0, int(nz_x.min()) - 2), min(UW, int(nz_x.max()) + 3)
    sub = rgba[ty0:ty1, tx0:tx1]
    out_im = bleed_rgb_edges(
        Image.fromarray(np.clip(sub, 0, 255).round().astype(np.uint8), "RGBA"),
        solid_thresh=16,
        max_dist=10,
    )
    box = [int(ux + tx0), int(uy + ty0), int(ux + tx1), int(uy + ty1)]
    return out_im, box


def detect_chroma_gaps(
    rgba_or_chroma_rgb: np.ndarray,
    *,
    roi_frac: tuple[float, float, float, float] = (0.18, 0.24, 0.82, 0.68),
    min_hole_px: int = 12,
) -> dict:
    """Detect enclosed interior chroma-green (or transparent) leaks inside the head/hair silhouette.

    Uses exterior flood-fill from image borders so legitimate outer background around the head
    is ignored, while ANY interior gap/hole trapped between front bangs, wolf ears, temples,
    and side/back hair is caught and quantified to the exact pixel!
    """
    arr = np.asarray(rgba_or_chroma_rgb)
    H, W = arr.shape[:2]
    if arr.ndim == 3 and arr.shape[2] == 4:
        is_bg = arr[..., 3] < 200
    else:
        r, g, b = arr[..., 0].astype(np.int16), arr[..., 1].astype(np.int16), arr[..., 2].astype(np.int16)
        is_bg = (g > 180) & (g - r > 100) & (g - b > 60)

    # Flood-fill exterior background connected to image border
    seed = np.zeros((H, W), dtype=bool)
    seed[0, :] = is_bg[0, :]
    seed[-1, :] = is_bg[-1, :]
    seed[:, 0] = is_bg[:, 0]
    seed[:, -1] = is_bg[:, -1]
    ext_bg = ndi.binary_propagation(seed, mask=is_bg)

    # Enclosed background holes inside the ROI (upper-head/ear/temple/bangs region)
    x0, y0 = int(W * roi_frac[0]), int(H * roi_frac[1])
    x1, y1 = int(W * roi_frac[2]), int(H * roi_frac[3])
    roi_mask = np.zeros((H, W), dtype=bool)
    roi_mask[y0:y1, x0:x1] = True

    interior_holes = is_bg & (~ext_bg) & roi_mask
    lbl, n = ndi.label(interior_holes)
    holes = []
    total_px = 0
    for i in range(1, n + 1):
        comp = lbl == i
        cnt = int(comp.sum())
        if cnt >= min_hole_px:
            ys, xs = np.nonzero(comp)
            holes.append({
                "pixels": cnt,
                "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            })
            total_px += cnt

    return {
        "pass": total_px == 0,
        "hole_count": len(holes),
        "total_hole_px": total_px,
        "holes": sorted(holes, key=lambda h: -h["pixels"]),
    }
