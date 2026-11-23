"""Reusable 3D Side-of-Face / Side-of-Jaw + Turn Depth Shading & 3/4 Nose Generator (`rigkit.face3d`).

In 2D VTuber rigs, turning a flat front face (`F01`) sideways looks paper-thin unless BOTH:
  (A) The lateral plane of the skull & mandible unfolds sleekly along the 3/4 jawline (`F01sL` / `F01sR`), AND
  (B) Broad rear-to-front 3D directional form shading (`F02tR` / `F02tL`, `clip="F01"`) and a sculpted 3D 3/4 nose
      (`F03tR` / `F03tL`, `clip="F01"`) emerge smoothly as the head turns!

This module makes full 3D side-of-head realism a zero-guesswork default step for ANY Layer2D avatar:
1. `extract_face_contour_and_palette(face_rgba, face_box)`:
   - Measures the avatar's exact `F01` cheek/jaw silhouette in canonical canvas coordinates, AND
   - Auto-samples `F01`'s exact cheek skin RGB and dark jawline ink RGB.
2. `synthesize_side_jaw_layers(...)` + `fit_side_jaw_3d(...)`:
   - Projects `F01`'s measured near cheek contour through `fwd_face_fn(pts, -max_yaw_rad)` to get
     the exact full-turn hinge curve (`< 0.5px` lock to `F01` at every vertical height `y`).
   - Builds sleek, slender, straight-angled anime mandible / side-of-skull planes (`F01sL`, `F01sR`)
     with feathered inner cheek overlap (erasing `F01`'s old near-side ink outline cleanly) and rich
     rear-to-front dark-to-light 3D plane-turn shading.
3. `synthesize_turn_face_depth_layers(...)` + `fit_turn_face_shade_3d(...)` + `fit_turn_nose_3d(...)`:
   - Renders 4x supersampled float32 premultiplied + sub-LSB dithered:
     - `F02tR` / `F02tL` (`clip="F01"`): Broad rear-to-front directional 3D cheek & hairline shadow
       (`~138px` warm rose-peach gradient + tight `48px` hairline contact shadow), receding far eye-socket
       shadow, silky lower-jawline form shadow, wrapping blush, and subtle lit front-cheek highlight.
     - `F03tR` / `F03tL` (`clip="F01"`): Glossy tilted 3/4 anime nose-bridge specular highlight + subtle
       warm side cast shadow + crisp angled warm-brown 3/4 nose-tip tick (`op = 0` at Front).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage


def _smoothstep(edge0, edge1, x):
    denom = edge1 - edge0
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _r4(v) -> list[float]:
    return [round(float(x), 4) for x in np.asarray(v, dtype=np.float64).ravel()]


def _over_f32(dst_pm: np.ndarray, rgb: tuple[float, float, float], alpha_01: np.ndarray) -> np.ndarray:
    """Exact 32-bit float premultiplied Porter-Duff 'over' (zero uint8 quantization rings)."""
    a = np.clip(alpha_01, 0.0, 1.0).astype(np.float32)
    a3 = a[..., None]
    src_pm = np.empty_like(dst_pm)
    src_pm[..., 0] = float(rgb[0]) * a
    src_pm[..., 1] = float(rgb[1]) * a
    src_pm[..., 2] = float(rgb[2]) * a
    src_pm[..., 3] = a
    return src_pm + dst_pm * (1.0 - a3)


def _pm_to_rgba_u8(pm: np.ndarray, *, dither_seed: int = 101, dither_amp: float = 0.65) -> Image.Image:
    """Convert premultiplied float32 RGBA array to clean uint8 RGBA with sub-LSB triangular dithering."""
    h, w = pm.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.float32)
    a = pm[..., 3:4]
    valid = a > 1e-5
    out[..., :3] = np.where(valid, pm[..., :3] / np.maximum(a, 1e-5), 0.0)
    a_u8 = a * 255.0
    if dither_amp > 0.0:
        rng = np.random.default_rng(dither_seed)
        d_rgb = (rng.random((h, w, 3), dtype=np.float32) - rng.random((h, w, 3), dtype=np.float32)) * float(dither_amp)
        d_a = (rng.random((h, w, 1), dtype=np.float32) - rng.random((h, w, 1), dtype=np.float32)) * float(dither_amp)
        out[..., :3] = np.where(valid, out[..., :3] + d_rgb, 0.0)
        a_u8 = np.where(a_u8 < 0.5, 0.0, a_u8 + d_a)
    out[..., 3:4] = a_u8
    return Image.fromarray(np.clip(out, 0.0, 255.0).round().astype(np.uint8), "RGBA")


@dataclass
class SideJawSpec:
    """Serialized geometry & palette spec linking `F01sL`/`F01sR` sprites to `fit_side_jaw_3d`."""

    canvas_w: int
    max_yaw_deg: float
    box_l: list[int]  # [x0, y0, x1, y1] for F01sL (viewer-right, active on a < 0)
    box_r: list[int]  # [x0, y0, x1, y1] for F01sR (viewer-left, active on a > 0)
    y_front: list[float]
    x_l_front: list[float]
    x_r_front: list[float]
    y_turn: list[float]
    x_r_turn: list[float]
    y_outer: list[float]
    x_outer: list[float]
    skin_rgb: list[float]
    ink_rgb: list[float]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SideJawSpec:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class TurnFaceDepthSpec:
    """Serialized spec linking clipped 3D face shading (`F02tR/L`) & 3/4 nose (`F03tR/L`) to keyform fitters."""

    canvas_w: int
    max_yaw_deg: float
    box_shade_r: list[int]  # F02tR (active on a < 0, right turn)
    box_shade_l: list[int]  # F02tL (active on a > 0, left turn)
    box_nose_r: list[int]   # F03tR (active on a < 0, right turn)
    box_nose_l: list[int]   # F03tL (active on a > 0, left turn)
    nose_front_xy: list[float]
    nose_turn_r_xy: list[float]
    face_front_cx: float
    skin_rgb: list[float]
    ink_rgb: list[float]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TurnFaceDepthSpec:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def extract_face_contour_and_palette(
    face_rgba: np.ndarray,
    face_box: tuple[int, int, int, int] | list[int],
    *,
    canvas_w: int = 2400,
    n_samples: int = 18,
) -> dict:
    """Measure symmetric canonical F01 left/right cheek contour and auto-sample skin & ink RGB."""
    arr = np.asarray(face_rgba, dtype=np.uint8)
    bx0, by0, bx1, by1 = [int(v) for v in face_box]
    fh, fw = arr.shape[:2]
    alpha = arr[..., 3] > 60
    nz_y = np.nonzero(alpha.any(axis=1))[0]
    if len(nz_y) < 12:
        raise ValueError("F01 image has insufficient opaque pixels to extract face contour")

    y0_loc, y1_loc = int(nz_y.min()), int(nz_y.max())
    ts = np.array(
        [0.02, 0.12, 0.24, 0.36, 0.48, 0.56, 0.63, 0.69, 0.75, 0.80, 0.85, 0.89, 0.925, 0.955, 0.978, 0.992, 1.0],
        dtype=np.float64,
    )
    if n_samples != len(ts) and n_samples >= 10:
        ts = np.linspace(0.02, 1.0, n_samples, dtype=np.float64)

    cx = 0.5 * canvas_w
    bw, bh = max(1.0, float(bx1 - bx0)), max(1.0, float(by1 - by0))
    y_list, xl_list, xr_list = [], [], []
    for t in ts:
        ry = int(round(y0_loc + t * (y1_loc - y0_loc)))
        ry = max(0, min(fh - 1, ry))
        r0, r1 = max(0, ry - 1), min(fh, ry + 2)
        xs = np.nonzero(alpha[r0:r1].any(axis=0))[0]
        if len(xs) < 4:
            continue
        x_l = bx0 + float(xs[0]) * bw / float(fw)
        x_r = bx0 + float(xs[-1]) * bw / float(fw)
        half_w = max(0.0, 0.5 * ((x_r - cx) + (cx - x_l)))
        y_list.append(by0 + float(ry) * bh / float(fh))
        xl_list.append(cx - half_w)
        xr_list.append(cx + half_w)

    y_front = np.array(y_list, dtype=np.float64)
    x_l_front = np.array(xl_list, dtype=np.float64)
    x_r_front = np.array(xr_list, dtype=np.float64)

    if len(y_front) >= 2:
        x_r_front[-1] = min(x_r_front[-1], cx + 8.0)
        x_l_front[-1] = 2.0 * cx - x_r_front[-1]

    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    yy_l, xx_l = np.mgrid[0:fh, 0:fw]
    cheek_band = (
        (arr[..., 3] > 235)
        & (yy_l >= y0_loc + int(0.42 * (y1_loc - y0_loc)))
        & (yy_l <= y0_loc + int(0.84 * (y1_loc - y0_loc)))
        & (lum > 180.0)
    )
    if cheek_band.sum() >= 50:
        p55 = float(np.percentile(lum[cheek_band], 55))
        lit = cheek_band & (lum >= p55)
        skin_rgb = np.median(arr[lit, :3], axis=0).astype(np.float64)
    else:
        skin_rgb = np.array([248.0, 222.0, 214.0], dtype=np.float64)

    edge_dist = ndimage.distance_transform_edt(alpha)
    jaw_border = (
        alpha
        & (edge_dist <= 6.0)
        & (yy_l >= y0_loc + int(0.55 * (y1_loc - y0_loc)))
        & (arr[..., 3] > 170)
        & (lum < 155.0)
    )
    if jaw_border.sum() >= 20:
        ink_rgb = np.median(arr[jaw_border, :3], axis=0).astype(np.float64)
    else:
        ink_rgb = np.array([96.0, 64.0, 62.0], dtype=np.float64)

    return {
        "y_front": y_front,
        "x_l_front": x_l_front,
        "x_r_front": x_r_front,
        "skin_rgb": skin_rgb,
        "ink_rgb": ink_rgb,
        "y0_can": by0 + float(y0_loc) * bh / float(fh),
        "y1_can": by0 + float(y1_loc) * bh / float(fh),
    }


def _side_head_shadow_components(
    d_from_outer: np.ndarray,
    skin_rgb: np.ndarray,
    *,
    shade_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    """Shared analytical rear-to-front 3D side-of-head shadow profile used IDENTICALLY by BOTH
    `F02tR/L` (clipped to `F01`) and `F01sL/R` (placed immediately after `F02tR/L`) so the
    feather transition across `F01`'s edge is mathematically seamless (zero double-darkening or step!).
    """
    st = float(shade_strength)
    d = np.maximum(0.0, d_from_outer).astype(np.float32)
    g_broad = (np.clip(1.0 - d / 138.0, 0.0, 1.0) ** 1.55) * (0.50 * st)
    g_deep = (np.clip(1.0 - d / 48.0, 0.0, 1.0) ** 1.82) * (0.34 * st)
    broad_rgb = (
        max(0.0, float(skin_rgb[0]) - 58.0),
        max(0.0, float(skin_rgb[1]) - 92.0),
        max(0.0, float(skin_rgb[2]) - 86.0),
    )
    deep_rgb = (
        max(0.0, float(skin_rgb[0]) - 76.0),
        max(0.0, float(skin_rgb[1]) - 108.0),
        max(0.0, float(skin_rgb[2]) - 96.0),
    )
    return g_broad, g_deep, broad_rgb, deep_rgb


def synthesize_side_jaw_layers(
    face_rgba: np.ndarray | Image.Image,
    face_box: tuple[int, int, int, int] | list[int],
    fwd_face_fn: Callable[[np.ndarray, float], np.ndarray],
    *,
    canvas_w: int = 2400,
    max_yaw_deg: float = 32.5,
    jaw_depth_px: float = 18.0,
    temple_depth_px: float = 20.0,
    inner_overlap_px: float = 34.0,
    ink_half_width_px: float = 1.55,
    shade_strength: float = 1.0,
    supersample: int = 4,
) -> tuple[Image.Image, Image.Image, SideJawSpec]:
    """Synthesize sleek, slender 3D Camera-Parallel Side-of-Face / Side-of-Jaw RGBA layers (`F01sL`, `F01sR`).

    Uses sleek anime mandible proportions (`jaw_depth_px ~ 14..22px`) so the turned jawline forms ONE
    slender straight-angled 3/4 anime mandible sweeping cleanly into the side hair (NEVER a puffy cheek bulge!),
    with a rich rear-to-front dark-to-light 3D directional shading gradient matched seamlessly to `F01` & `F02t`.
    """
    arr = np.asarray(face_rgba.convert("RGBA") if isinstance(face_rgba, Image.Image) else face_rgba, dtype=np.uint8)
    max_yaw_rad = math.radians(max_yaw_deg)

    info = extract_face_contour_and_palette(arr, face_box, canvas_w=canvas_w)
    y_front = info["y_front"]
    x_l_front = info["x_l_front"]
    x_r_front = info["x_r_front"]
    skin_rgb = info["skin_rgb"]
    ink_rgb = info["ink_rgb"]

    pts_front_r = np.stack([x_r_front, y_front], axis=-1)
    pts_turn_r = np.asarray(fwd_face_fn(pts_front_r, -max_yaw_rad), dtype=np.float64)
    x_r_turn = pts_turn_r[:, 0]
    y_turn = pts_turn_r[:, 1]

    order = np.argsort(y_turn)
    y_turn = y_turn[order]
    x_r_turn = x_r_turn[order]
    y_front = y_front[order]
    x_l_front = x_l_front[order]
    x_r_front = x_r_front[order]

    y_face_top = float(y_turn[0])
    y_chin = float(y_turn[-1])
    face_h = max(100.0, y_chin - y_face_top)

    y_top_layer = y_face_top + 0.18 * face_h
    y_bot_layer = y_chin + 6.0

    # Sleek, slender anime mandible outer silhouette `x_outer(y)`:
    # Dense 48-point sampling guarantees convex `x_outer >= x_hinge` everywhere along the lower jaw (zero chord undercut!).
    n_out = 48
    y_outer = np.linspace(y_top_layer, y_chin, n_out, dtype=np.float64)
    t_out = (y_outer - y_top_layer) / max(1.0, y_chin - y_top_layer)
    x_hinge_at_out = np.interp(y_outer, y_turn, x_r_turn)

    # Broad smooth convex ramp from 0px at chin apex -> sleek mandible -> temple hairline (zero lower-jaw kink!)
    t_chin_ramp = _smoothstep(0.97, 0.56, t_out) ** 0.90
    corner_bump = np.exp(-0.5 * ((t_out - 0.68) / 0.22) ** 2)
    base_depth = temple_depth_px + (jaw_depth_px - temple_depth_px) * corner_bump
    depth_profile = base_depth * t_chin_ramp
    x_outer_raw = x_hinge_at_out + depth_profile

    x_outer = ndimage.gaussian_filter1d(x_outer_raw, sigma=2.2, mode="nearest")
    x_outer[-1] = x_hinge_at_out[-1]
    x_outer[-2] = x_hinge_at_out[-2]
    x_outer = np.maximum(x_outer, x_hinge_at_out)

    chin_x = float(x_r_turn[-1])
    bx0 = int(math.floor(min(chin_x - 32.0, float(np.min(x_r_turn)) - inner_overlap_px - 14.0)))
    bx1 = int(math.ceil(float(np.max(x_outer)) + 16.0))
    by0 = int(math.floor(y_top_layer - 6.0))
    by1 = int(math.ceil(y_bot_layer + 6.0))
    bx0 = max(0, bx0)
    bx1 = min(canvas_w, bx1)

    W, H = bx1 - bx0, by1 - by0
    SS = max(2, int(supersample))
    Ws, Hs = W * SS, H * SS

    xg = np.linspace(bx0 + 0.5 / SS, bx1 - 0.5 / SS, Ws, dtype=np.float32)
    yg = np.linspace(by0 + 0.5 / SS, by1 - 0.5 / SS, Hs, dtype=np.float32)
    xx, yy = np.meshgrid(xg, yg)

    x_hinge = np.interp(yy, y_turn, x_r_turn).astype(np.float32)
    x_out_2d = np.interp(yy, y_outer, x_outer).astype(np.float32)

    ys_dense = np.linspace(y_top_layer, y_chin, 320, dtype=np.float32)
    xs_dense = np.interp(ys_dense, y_outer, x_outer).astype(np.float32)
    dx_dy = np.gradient(xs_dense, ys_dense)
    slope_y = np.interp(yy, ys_dense, dx_dy).astype(np.float32)
    inv_norm = 1.0 / np.sqrt(1.0 + slope_y * slope_y)
    d_outer = (xx - x_out_2d) * inv_norm
    d_bot = yy - float(y_chin - 2.0)
    d_edge = np.maximum(d_outer, d_bot)

    m_outer = _smoothstep(0.75, -0.75, d_edge).astype(np.float32)

    xs_h_dense = np.interp(ys_dense, y_turn, x_r_turn).astype(np.float32)
    dh_dy = np.gradient(xs_h_dense, ys_dense)
    slope_h = np.interp(yy, ys_dense, dh_dy).astype(np.float32)
    inv_norm_h = 1.0 / np.sqrt(1.0 + slope_h * slope_h)
    d_hinge = (xx - x_hinge) * inv_norm_h

    m_inner = _smoothstep(-float(inner_overlap_px), -9.0, d_hinge).astype(np.float32)
    top_fade_h = max(36.0, 0.10 * face_h)
    m_top = _smoothstep(float(y_top_layer), float(y_top_layer + top_fade_h), yy).astype(np.float32)
    # Taper F01s smoothly out before reaching the bottom chin apex so F01 owns the chin tip cleanly at ALL angles (0, +-0.5, +-1.0)!
    m_join = (
        _smoothstep(float(chin_x + 20.0), float(chin_x + 68.0), xx)
        * _smoothstep(float(y_chin - 16.0), float(y_chin - 50.0), yy)
    ).astype(np.float32)
    skin_alpha = m_outer * m_inner * m_top * m_join

    # ---- Dimensional Rear-to-Front Dark-to-Light 3D Form Shading EXACTLY matched to F02t ----
    # Evaluates the IDENTICAL `_side_head_shadow_components(x_out_2d - xx)` Porter-Duff composite over `skin_rgb`
    # as `F02tR/L`, so when `F01sL/R` sits immediately after `F02tR/L` the inner feather matches `F01 OVER F02t`
    # to <1/255 LSB -> ZERO seam cliff, ZERO double-darkening, 100% seamless rear-to-front 3D depth!
    g_broad, g_deep, broad_rgb, deep_rgb = _side_head_shadow_components(
        x_out_2d - xx, skin_rgb, shade_strength=shade_strength
    )
    # Exact sequential Porter-Duff composite over opaque `skin_rgb`
    r = np.full_like(xx, float(skin_rgb[0]))
    g = np.full_like(xx, float(skin_rgb[1]))
    b = np.full_like(xx, float(skin_rgb[2]))
    r = broad_rgb[0] * g_broad + r * (1.0 - g_broad)
    g = broad_rgb[1] * g_broad + g * (1.0 - g_broad)
    b = broad_rgb[2] * g_broad + b * (1.0 - g_broad)
    r = deep_rgb[0] * g_deep + r * (1.0 - g_deep)
    g = deep_rgb[1] * g_deep + g * (1.0 - g_deep)
    b = deep_rgb[2] * g_deep + b * (1.0 - g_deep)

    # Matching gentle under-bangs temple shadow smoothly continuing F02 onto temple
    bang_y = float(y_top_layer + 0.36 * (y_chin - y_top_layer))
    t_bang = _smoothstep(bang_y, bang_y - max(60.0, 0.22 * (y_chin - y_top_layer)), yy)
    r -= 12.0 * t_bang
    g -= 18.0 * t_bang
    b -= 12.0 * t_bang

    pm = np.zeros((Hs, Ws, 4), dtype=np.float32)
    pm[..., 0] = np.clip(r, 0.0, 255.0) * skin_alpha
    pm[..., 1] = np.clip(g, 0.0, 255.0) * skin_alpha
    pm[..., 2] = np.clip(b, 0.0, 255.0) * skin_alpha
    pm[..., 3] = skin_alpha

    # Crisp antialiased anime jawline ink stroke matching F01's exact sampled outline RGB
    ink_top_y = float(y_top_layer + 0.44 * (y_chin - y_top_layer))
    ink_y_top = np.clip((yy - ink_top_y) / max(32.0, 0.12 * (y_chin - y_top_layer)), 0.0, 1.0)
    ink_y_bot = _smoothstep(float(y_chin - 16.0), float(y_chin - 42.0), yy)
    half_w_ink = float(ink_half_width_px) - 0.26 * np.clip((float(y_chin - 0.22 * face_h) - yy) / max(45.0, 0.18 * face_h), 0.0, 1.0)
    d_ink = np.abs(d_outer + 0.5)
    ink_a = _smoothstep(half_w_ink + 0.7, half_w_ink - 0.7, d_ink) * ink_y_top * ink_y_bot * m_inner * m_join * 0.94

    pm[..., 0] = float(ink_rgb[0]) * ink_a + pm[..., 0] * (1.0 - ink_a)
    pm[..., 1] = float(ink_rgb[1]) * ink_a + pm[..., 1] * (1.0 - ink_a)
    pm[..., 2] = float(ink_rgb[2]) * ink_a + pm[..., 2] * (1.0 - ink_a)
    pm[..., 3] = ink_a + pm[..., 3] * (1.0 - ink_a)

    pm_1x = pm.reshape(H, SS, W, SS, 4).mean(axis=(1, 3))
    out_rgba = np.zeros((H, W, 4), dtype=np.float32)
    a1 = pm_1x[..., 3:4]
    valid = a1 > 1e-5
    rng = np.random.default_rng(101)
    dither_rgb = (rng.random((H, W, 3), dtype=np.float32) - rng.random((H, W, 3), dtype=np.float32)) * 0.6
    for c in range(3):
        ch = np.where(valid, pm_1x[..., c : c + 1] / np.maximum(a1, 1e-5), float(skin_rgb[c]))
        out_rgba[..., c : c + 1] = np.where(valid, ch + dither_rgb[..., c : c + 1], ch)
    out_rgba[..., 3:4] = a1 * 255.0

    im_l = Image.fromarray(np.clip(out_rgba, 0.0, 255.0).round().astype(np.uint8), "RGBA")
    im_r = im_l.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    spec = SideJawSpec(
        canvas_w=int(canvas_w),
        max_yaw_deg=float(max_yaw_deg),
        box_l=[bx0, by0, bx1, by1],
        box_r=[canvas_w - bx1, by0, canvas_w - bx0, by1],
        y_front=[round(float(v), 2) for v in y_front],
        x_l_front=[round(float(v), 2) for v in x_l_front],
        x_r_front=[round(float(v), 2) for v in x_r_front],
        y_turn=[round(float(v), 2) for v in y_turn],
        x_r_turn=[round(float(v), 2) for v in x_r_turn],
        y_outer=[round(float(v), 2) for v in y_outer],
        x_outer=[round(float(v), 2) for v in x_outer],
        skin_rgb=[round(float(v), 1) for v in skin_rgb],
        ink_rgb=[round(float(v), 1) for v in ink_rgb],
    )
    return im_l, im_r, spec


def synthesize_turn_face_depth_layers(
    face_rgba: np.ndarray | Image.Image,
    face_box: tuple[int, int, int, int] | list[int],
    fwd_face_fn: Callable[[np.ndarray, float], np.ndarray],
    *,
    side_jaw_spec: SideJawSpec | None = None,
    nose_front_xy: tuple[float, float] | None = None,
    eye_y: float | None = None,
    cheek_y: float | None = None,
    canvas_w: int = 2400,
    max_yaw_deg: float = 32.5,
    shade_strength: float = 1.0,
    nose_strength: float = 1.0,
    supersample: int = 4,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, TurnFaceDepthSpec]:
    """Synthesize clipped 3D head-turn face-plane shading (`F02tR`/`F02tL`) and 3/4 nose (`F03tR`/`F03tL`).

    Both pairs are clipped cleanly to `F01` (`clip="F01"`), hidden at Front (`op=0`), and emerge smoothly on turns:
    - `F02tR` (active on `a < 0`) & exact mirror `F02tL` (active on `a > 0`):
      Broad back-to-forward directional 3D side-of-head shadow sweeping ~138px forward from the turned hairline
      & jawbone, tight 48px rear hairline contact occlusion shadow, receding far eye-socket depth shadow,
      silky lower-jawline shadow, delicate wrapping cheek blush, and subtle lit front-cheek highlight.
    - `F03tR` (active on `a < 0`) & exact mirror `F03tL` (active on `a > 0`):
      Glossy tilted 3/4 anime nose-bridge teardrop specular highlight + delicate warm side cast shadow +
      crisp angled warm-brown 3/4 anime nose-tip tick.
    """
    arr = np.asarray(face_rgba.convert("RGBA") if isinstance(face_rgba, Image.Image) else face_rgba, dtype=np.uint8)
    max_yaw_rad = math.radians(max_yaw_deg)

    info = extract_face_contour_and_palette(arr, face_box, canvas_w=canvas_w)
    y_front = info["y_front"]
    x_l_front = info["x_l_front"]
    x_r_front = info["x_r_front"]
    skin_rgb = info["skin_rgb"]
    ink_rgb = info["ink_rgb"]
    cx = 0.5 * canvas_w

    # Project BOTH far (L) and near (R) F01 contours at full right-turn (yaw = -max_yaw_rad, a = -1.0)
    pts_turn_l = np.asarray(fwd_face_fn(np.stack([x_l_front, y_front], axis=-1), -max_yaw_rad), dtype=np.float64)
    pts_turn_r = np.asarray(fwd_face_fn(np.stack([x_r_front, y_front], axis=-1), -max_yaw_rad), dtype=np.float64)
    ord_l = np.argsort(pts_turn_l[:, 1])
    ord_r = np.argsort(pts_turn_r[:, 1])
    yl_t, xl_t = pts_turn_l[ord_l, 1], pts_turn_l[ord_l, 0]
    yr_t, xr_t = pts_turn_r[ord_r, 1], pts_turn_r[ord_r, 0]

    y_top = float(min(yl_t[0], yr_t[0]))
    y_chin = float(max(yl_t[-1], yr_t[-1]))
    face_h = max(120.0, y_chin - y_top)
    eye_y_val = float(eye_y) if eye_y is not None else (y_top + 0.57 * face_h)
    cheek_y_val = float(cheek_y) if cheek_y is not None else (y_top + 0.71 * face_h)
    if nose_front_xy is None:
        nose_front_xy = (cx, y_top + 0.73 * face_h)

    # Project nose front coordinate onto the turned 3/4 face plane + natural 3D nose-bridge forward projection
    nose_proj = np.asarray(fwd_face_fn(np.array([nose_front_xy], dtype=np.float64), -max_yaw_rad), dtype=np.float64)[0]
    nose_turn_x = float(nose_proj[0]) - 22.0
    nose_turn_y = float(nose_proj[1]) + 4.0

    # ---- 1. 3D 3/4 Nose (`F03tR` active on a < 0, exact horizontal mirror `F03tL` active on a > 0) ----
    nw, nh = 70, 70
    nx0 = int(round(nose_turn_x - nw / 2.0))
    ny0 = int(round(nose_turn_y - nh / 2.0))
    nx1, ny1 = nx0 + nw, ny0 + nh

    SS = max(2, int(supersample))
    Ws, Hs = nw * SS, nh * SS
    xg = np.linspace(nx0 + 0.5 / SS, nx1 - 0.5 / SS, Ws, dtype=np.float32)
    yg = np.linspace(ny0 + 0.5 / SS, ny1 - 0.5 / SS, Hs, dtype=np.float32)
    xx, yy = np.meshgrid(xg, yg)

    nose_pm = np.zeros((Hs, Ws, 4), dtype=np.float32)
    ns = float(nose_strength)
    sh_nose_rgb = (
        max(0.0, float(skin_rgb[0]) - 22.0),
        max(0.0, float(skin_rgb[1]) - 48.0),
        max(0.0, float(skin_rgb[2]) - 46.0),
    )
    tip_rgb = (
        min(170.0, max(95.0, float(ink_rgb[0]) * 1.35)),
        min(115.0, max(52.0, float(ink_rgb[1]) * 1.15)),
        min(110.0, max(50.0, float(ink_rgb[2]) * 1.15)),
    )

    # a) Soft warm peach-rose cast shadow tucked subtly on shaded side (viewer-right) of 3/4 nose tip
    dx_s = (xx - (nose_turn_x + 3.5)) + 0.24 * (yy - (nose_turn_y + 7.5))
    dy_s = yy - (nose_turn_y + 7.5)
    g_sh = np.exp(-0.5 * ((dx_s / 5.2) ** 2 + (dy_s / 8.5) ** 2)) * np.clip((xx - (nose_turn_x - 3.5)) / 4.0, 0.0, 1.0)
    nose_pm = _over_f32(nose_pm, sh_nose_rgb, g_sh * (74.0 / 255.0) * ns)

    # b) Delicate glossy tilted anime teardrop nose-bridge highlight tapering cleanly down to nose tip
    co_n, si_n = math.cos(math.radians(-26.0)), math.sin(math.radians(-26.0))
    u_n = (xx - (nose_turn_x + 2.2)) * co_n + (yy - (nose_turn_y - 2.5)) * si_n
    v_n = -(xx - (nose_turn_x + 2.2)) * si_n + (yy - (nose_turn_y - 2.5)) * co_n
    t_tap = np.clip((v_n + 10.0) / 20.0, 0.0, 1.0)
    rx_drop = 5.8 * (1.0 - 0.56 * t_tap)
    r_drop = np.sqrt((u_n / np.maximum(rx_drop, 1.4)) ** 2 + (v_n / 10.4) ** 2)
    shine_core = _smoothstep(1.0, 0.80, r_drop)
    shine_halo = np.exp(-0.5 * (r_drop / 1.05) ** 2) * 0.22
    crease = np.exp(-0.5 * ((u_n + 1.0) / 0.85) ** 2) * np.clip(1.0 - np.abs(v_n) / 10.0, 0.0, 1.0)
    shine_a = np.clip((shine_core * 0.72 + shine_halo) * (1.0 - 0.42 * crease), 0.0, 0.78) * ns
    nose_pm = _over_f32(nose_pm, (255.0, 253.0, 252.0), shine_a)

    # c) Crisp delicate angled warm-brown 3/4 nose-tip tick (4x supersampled polyline SDF)
    def _seg_dist(px, py, ax, ay, bx, by):
        abx, aby = bx - ax, by - ay
        t = np.clip(((px - ax) * abx + (py - ay) * aby) / max(1e-6, abx * abx + aby * aby), 0.0, 1.0)
        return np.hypot(px - (ax + t * abx), py - (ay + t * aby))

    d_tick = np.minimum(
        _seg_dist(xx, yy, nose_turn_x - 2.6, nose_turn_y + 4.8, nose_turn_x - 3.8, nose_turn_y + 9.4),
        _seg_dist(xx, yy, nose_turn_x - 3.8, nose_turn_y + 9.4, nose_turn_x - 1.8, nose_turn_y + 13.0),
    )
    tick_a = np.clip((1.22 - d_tick) / 0.70, 0.0, 1.0) * 0.85 * ns
    nose_pm = _over_f32(nose_pm, tip_rgb, tick_a)

    nose_pm_1x = nose_pm.reshape(nh, SS, nw, SS, 4).mean(axis=(1, 3))
    f03t_r = _pm_to_rgba_u8(nose_pm_1x, dither_seed=203, dither_amp=0.25)
    f03t_l = f03t_r.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # ---- 2. Clipped 3D Face-Plane Form Shading (`F02tR` active on a < 0, mirror `F02tL` active on a > 0) ----
    fx0 = max(0, int(math.floor(float(np.min(xl_t)) - 22.0)))
    fx1 = min(canvas_w, int(math.ceil(float(np.max(xr_t)) + 48.0)))
    fy0 = max(0, int(math.floor(eye_y_val - 0.16 * face_h)))
    fy1 = int(math.ceil(y_chin + 14.0))
    fw, fh = fx1 - fx0, fy1 - fy0

    fyy, fxx = np.mgrid[fy0:fy1, fx0:fx1].astype(np.float32)
    shade_pm = np.zeros((fh, fw, 4), dtype=np.float32)
    st = float(shade_strength)

    def _add_blob(cx_b: float, cy_b: float, rx_b: float, ry_b: float, rgb_b: tuple[float, float, float], max_a_255: float, tilt: float = 0.0) -> None:
        nonlocal shade_pm
        co_b, si_b = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
        ub = (fxx - cx_b) * co_b + (fyy - cy_b) * si_b
        vb = -(fxx - cx_b) * si_b + (fyy - cy_b) * co_b
        gb = np.exp(-0.5 * ((ub / rx_b) ** 2 + (vb / ry_b) ** 2))
        shade_pm = _over_f32(shade_pm, rgb_b, gb * (max_a_255 / 255.0) * st)

    # Harmonize shadow palette directly from F01's sampled skin RGB (silky warm peach-rose anime shadows!)
    sock_rgb = (
        max(0.0, float(skin_rgb[0]) - 26.0),
        max(0.0, float(skin_rgb[1]) - 44.0),
        max(0.0, float(skin_rgb[2]) - 36.0),
    )
    jaw_rgb = (
        max(0.0, float(skin_rgb[0]) - 24.0),
        max(0.0, float(skin_rgb[1]) - 48.0),
        max(0.0, float(skin_rgb[2]) - 44.0),
    )
    broad_rgb = (
        max(0.0, float(skin_rgb[0]) - 60.0),
        max(0.0, float(skin_rgb[1]) - 94.0),
        max(0.0, float(skin_rgb[2]) - 88.0),
    )
    deep_rgb = (
        max(0.0, float(skin_rgb[0]) - 78.0),
        max(0.0, float(skin_rgb[1]) - 110.0),
        max(0.0, float(skin_rgb[2]) - 98.0),
    )
    blush_rgb = (
        min(255.0, float(skin_rgb[0]) * 1.005),
        max(0.0, float(skin_rgb[1]) * 0.74),
        max(0.0, float(skin_rgb[2]) * 0.76),
    )

    x_far_eye = float(np.interp(eye_y_val, yl_t, xl_t))
    x_far_cheek = float(np.interp(cheek_y_val, yl_t, xl_t))
    x_near_cheek = float(np.interp(cheek_y_val, yr_t, xr_t))
    x_chin_t = float(xr_t[-1])

    # a) Subtle receding far-temple / far eye-socket depth shadow (sinks far eye cleanly into 3D skull socket!)
    _add_blob(x_far_eye + 14.0, eye_y_val + 8.0, 22.0, 30.0, sock_rgb, 54.0, tilt=-8.0)
    # b) Delicate wrapping peach-pink blush on the bulging far cheekbone edge
    _add_blob(x_far_cheek + 22.0, cheek_y_val + 6.0, 28.0, 20.0, blush_rgb, 44.0, tilt=8.0)
    # c) Soft broad rosy blush wrap across prominent camera-facing near cheek
    _add_blob(0.5 * (nose_turn_x + x_near_cheek) + 18.0, cheek_y_val + 6.0, 64.0, 30.0, blush_rgb, 46.0, tilt=-10.0)
    # d) Ultra-silky broad lower-jawline form shadow along the turned near mandible
    jaw_mid_x = 0.55 * x_near_cheek + 0.45 * x_chin_t + 16.0
    jaw_mid_y = y_chin - 0.10 * face_h
    _add_blob(jaw_mid_x, jaw_mid_y, 92.0, 34.0, jaw_rgb, 44.0, tilt=-26.0)

    # e) TRUE BACK-TO-FORWARD 3D SIDE-OF-HEAD DIRECTIONAL GRADIENT SHADOW:
    #    Anchored to `side_jaw_spec.x_outer` (when present) or `F01`'s turned edge `xr_t`, using the EXACT SAME
    #    `_side_head_shadow_components` formula as `F01sL/R` so `F02tR/L` -> `F01sL/R` is 100% seamless!
    if side_jaw_spec is not None:
        x_hairline_curve = np.interp(
            fyy,
            np.asarray(side_jaw_spec.y_outer, dtype=np.float32),
            np.asarray(side_jaw_spec.x_outer, dtype=np.float32),
        ).astype(np.float32)
    else:
        x_hairline_curve = np.interp(fyy, yr_t, xr_t).astype(np.float32) + 8.0
    g_broad, g_deep, broad_rgb, deep_rgb = _side_head_shadow_components(
        x_hairline_curve - fxx, skin_rgb, shade_strength=st
    )
    shade_pm = _over_f32(shade_pm, broad_rgb, g_broad)
    shade_pm = _over_f32(shade_pm, deep_rgb, g_deep)

    # f) Subtle broad lit cheek-plane lift on forward mid-cheek so light visibly wraps around a 3D sphere!
    lit_cx = nose_turn_x + 0.30 * (x_near_cheek - nose_turn_x)
    _add_blob(lit_cx, cheek_y_val - 6.0, 54.0, 32.0, (255.0, 249.0, 246.0), 26.0, tilt=-8.0)

    f02t_r = _pm_to_rgba_u8(shade_pm, dither_seed=202, dither_amp=0.75)
    f02t_l = f02t_r.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    spec = TurnFaceDepthSpec(
        canvas_w=int(canvas_w),
        max_yaw_deg=float(max_yaw_deg),
        box_shade_r=[fx0, fy0, fx1, fy1],
        box_shade_l=[canvas_w - fx1, fy0, canvas_w - fx0, fy1],
        box_nose_r=[nx0, ny0, nx1, ny1],
        box_nose_l=[canvas_w - nx1, ny0, canvas_w - nx0, ny1],
        nose_front_xy=[round(float(nose_front_xy[0]), 2), round(float(nose_front_xy[1]), 2)],
        nose_turn_r_xy=[round(float(nose_turn_x), 2), round(float(nose_turn_y), 2)],
        face_front_cx=round(float(cx), 2),
        skin_rgb=[round(float(v), 1) for v in skin_rgb],
        ink_rgb=[round(float(v), 1) for v in ink_rgb],
    )
    return f02t_r, f02t_l, f03t_r, f03t_l, spec


def fit_turn_face_shade_3d(
    spec: TurnFaceDepthSpec,
    rest_t: np.ndarray,
    a: float,
    rig: dict,
    *,
    active_sign: float,
    max_op: float = 1.0,
    head_ride_inv_fn: Callable[[np.ndarray, float, dict], np.ndarray] | None = None,
) -> dict:
    """Zero-pop keyform for clipped 3D directional face shading (`F02tR`: active_sign=-1, `F02tL`: +1).

    Reaches `max_op = 1.0` smoothly on full turn so its rear-to-front shadow intensity locks 1:1 onto
    `F01sL/R` (zero visible intensity step at `F01`'s clip edge!).
    """
    if head_ride_inv_fn is None:
        head_ride_inv_fn = lambda p, _a, _r: np.array(p, dtype=np.float64, copy=True)
    u_s = max(0.0, float(a * active_sign))
    op_s = float(max_op * _smoothstep(0.04, 0.42, u_s))
    k_s = 1.0 - u_s
    p_scr = np.asarray(rest_t, dtype=np.float64) + np.array(
        [(float(spec.face_front_cx) - float(rest_t[0])) * k_s * 0.65, 6.0 * k_s],
        dtype=np.float64,
    )
    p_hr = head_ride_inv_fn(p_scr, a, rig)
    return {"A": [1.0, 0.0, 0.0, 1.0], "t": _r4(p_hr), "op": round(op_s, 4)}


def fit_turn_nose_3d(
    spec: TurnFaceDepthSpec,
    rest_t: np.ndarray,
    a: float,
    rig: dict,
    *,
    active_sign: float,
    head_ride_inv_fn: Callable[[np.ndarray, float, dict], np.ndarray] | None = None,
) -> dict:
    """Zero-pop keyform for clipped 3D 3/4 nose (`F03tR`: active_sign=-1, `F03tL`: +1)."""
    if head_ride_inv_fn is None:
        head_ride_inv_fn = lambda p, _a, _r: np.array(p, dtype=np.float64, copy=True)
    u_n = max(0.0, float(a * active_sign))
    op_n = float(_smoothstep(0.12, 0.56, u_n))
    k_n = 1.0 - u_n
    nf_x, nf_y = float(spec.nose_front_xy[0]), float(spec.nose_front_xy[1])
    p_scr = np.asarray(rest_t, dtype=np.float64) + np.array(
        [(nf_x - float(rest_t[0])) * k_n, (nf_y + 2.0 - float(rest_t[1])) * k_n],
        dtype=np.float64,
    )
    p_hr = head_ride_inv_fn(p_scr, a, rig)
    sx_n = 0.78 + 0.22 * u_n
    return {"A": _r4([[sx_n, 0.0], [0.0, 1.0]]), "t": _r4(p_hr), "op": round(op_n, 4)}


def _invert_map(target_screen: np.ndarray, fwd_fn: Callable[[np.ndarray, float], np.ndarray], n_iter: int = 14) -> np.ndarray:
    p = np.array(target_screen, dtype=np.float64, copy=True)
    for _ in range(n_iter):
        err = fwd_fn(p, 1.0) - target_screen
        eps = 1.0
        px = p.copy()
        px[..., 0] += eps
        py = p.copy()
        py[..., 1] += eps
        jx = (fwd_fn(px, 1.0) - fwd_fn(p, 1.0)) / eps
        jy = (fwd_fn(py, 1.0) - fwd_fn(p, 1.0)) / eps
        det = jx[..., 0] * jy[..., 1] - jy[..., 0] * jx[..., 1]
        det = np.where(np.abs(det) < 1e-4, 1.0, det)
        dp_x = (jy[..., 1] * err[..., 0] - jy[..., 0] * err[..., 1]) / det
        dp_y = (-jx[..., 1] * err[..., 0] + jx[..., 0] * err[..., 1]) / det
        p[..., 0] -= np.clip(dp_x, -120.0, 120.0)
        p[..., 1] -= np.clip(dp_y, -120.0, 120.0)
        if float(np.max(np.abs(err))) < 1e-3:
            break
    return p


def fit_side_jaw_3d(
    spec: SideJawSpec,
    w: float,
    h: float,
    rest_t: np.ndarray,
    a: float,
    rig: dict,
    *,
    active_sign: float,
    head_ride_inv_fn: Callable[[np.ndarray, float, dict], np.ndarray] | None = None,
    head_ride_fwd_fn: Callable[[np.ndarray, float, dict], np.ndarray] | None = None,
    gd: int = 7,
) -> dict:
    """Compute exact 3D camera-parallel unfolding keyform (`A, t, op, D`) for `F01sL` / `F01sR`.

    - `active_sign = -1.0` for `F01sL` (viewer-right side-of-face, unfolds when head turns R `a < 0`).
    - `active_sign = +1.0` for `F01sR` (viewer-left side-of-face, unfolds when head turns L `a > 0`).
    """
    half = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    s = float(a * active_sign)
    is_r = active_sign > 0.0
    cw = float(spec.canvas_w)
    max_yaw_rad = math.radians(spec.max_yaw_deg)

    if head_ride_inv_fn is None:
        head_ride_inv_fn = lambda p, _a, _r: np.array(p, dtype=np.float64, copy=True)
    if head_ride_fwd_fn is None:
        head_ride_fwd_fn = lambda p, _a, _r: np.array(p, dtype=np.float64, copy=True)

    y_front = np.asarray(spec.y_front, dtype=np.float64)
    x_l_front = np.asarray(spec.x_l_front, dtype=np.float64)
    x_r_front = np.asarray(spec.x_r_front, dtype=np.float64)
    y_turn = np.asarray(spec.y_turn, dtype=np.float64)
    x_r_turn = np.asarray(spec.x_r_turn, dtype=np.float64)

    if s <= 1e-6:
        x_mid_edge = float(np.median(x_l_front)) if is_r else float(np.median(x_r_front))
        p_hr = head_ride_inv_fn(np.array([x_mid_edge, rest_t[1]], dtype=np.float64), a, rig)
        return {"A": [0.05, 0.0, 0.0, 1.0], "t": _r4(p_hr), "op": 0.0, "D": [0.0] * (gd * gd * 2)}

    u = math.sin(s * max_yaw_rad) / math.sin(max_yaw_rad)
    # Match F02t's smooth ramp so mid-yaw (a=+-0.5) shading intensity stays 100% seamless with zero dark edge!
    op = float(_smoothstep(0.04, 0.42, s))

    max_jaw_w = max(16.0, float(np.max(np.asarray(spec.x_outer, dtype=np.float64) - np.interp(spec.y_outer, y_turn, x_r_turn))) + 8.0)

    def fwd_side_jaw(pts: np.ndarray, _yw: float) -> np.ndarray:
        x = pts[..., 0]
        y = pts[..., 1]
        xm = (cw - x) if is_r else x

        x_h_turn = np.interp(y, y_turn, x_r_turn)
        y_f = np.interp(y, y_turn, y_front)
        x_h_front = (cw - np.interp(y_f, y_front, x_l_front)) if is_r else np.interp(y_f, y_front, x_r_front)

        x_h_u = x_h_front + u * (x_h_turn - x_h_front)
        dx = xm - x_h_turn

        # Bounded, strictly monotonic C1 hinge foreshortening (derivative ALWAYS in [sx_side, 1.0] > 0!):
        #   dx < 0 (inner cheek overlap): slope = 1.0 (keeps F01's old ink 100% erased at ALL angles!)
        #   0 <= dx <= max_jaw_w (protruding side-of-jaw): slope = sx_side = max(0.08, u) (exact rotational foreshortening!)
        #   dx > max_jaw_w (transparent box corner): slope returns cleanly to 1.0 -> bounded shift <= max_jaw_w,
        #   eliminating 1/u corner D-grid blowup at small/mid yaw angles (a = +-0.25, +-0.5)!
        sx_side = max(0.08, u)
        k_sm = 2.0
        w_jaw = max_jaw_w
        dx_out = dx - 0.5 * (1.0 - sx_side) * (
            np.sqrt(dx * dx + k_sm * k_sm) - np.sqrt((dx - w_jaw) ** 2 + k_sm * k_sm) + w_jaw
            - (math.sqrt(k_sm * k_sm) - math.sqrt(w_jaw * w_jaw + k_sm * k_sm) + w_jaw)
        )

        x_out_l = x_h_u + dx_out
        x_out = (cw - x_out_l) if is_r else x_out_l
        y_out = y_f + u * (y - y_f)
        return np.stack([x_out, y_out], axis=-1)

    s5 = np.linspace(-0.85, 0.85, 5)
    gx5, gy5 = np.meshgrid(s5, s5)
    g5 = np.stack([gx5.ravel(), gy5.ravel()], axis=1)
    p_rest_5 = rest_t + g5 * half
    p_scr_5 = fwd_side_jaw(p_rest_5, 0.0)
    p_lay_5 = head_ride_inv_fn(p_scr_5, a, rig)

    X = np.hstack([p_rest_5 - rest_t, np.ones((len(p_rest_5), 1))])
    coef, *_ = np.linalg.lstsq(X, p_lay_5, rcond=None)
    A_mat = coef[:2, :].T
    t_new = coef[2, :]

    gs = np.linspace(-1.0, 1.0, gd)
    gx, gy = np.meshgrid(gs, gs)
    g = np.stack([gx.ravel(), gy.ravel()], axis=1)
    p_layer = t_new + (g * half) @ A_mat.T
    p_screen = head_ride_fwd_fn(p_layer, a, rig)
    p_rest = _invert_map(p_screen, fwd_side_jaw, n_iter=14)
    D = (p_rest - rest_t) / half - g
    return {"A": _r4(A_mat), "t": _r4(t_new), "op": round(op, 4), "D": _r4(D)}
