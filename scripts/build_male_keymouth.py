#!/usr/bin/env python3
"""Generate the studio-grade masculine anime Key-Shape Mouth preset (`assets/mouth_keyshape/male/`).

Produces 17 cohesive, 4x-supersampled shonen/seinen masculine anime mouth keyshapes + exact 33-point
U/Lo contour metadata (`meta.json`) compatible with Layer2D's Keshin2D contour-warped WebGL2 mouth
engine (`mouth_keyshape.js`).

Design goals (vs feminine kurage02/chiwa):
- Flat, restrained, stoic neutral line (zero cute U-smile curl)
- Angular/masculine tapered charcoal-espresso ink outlines (heavier upper lip, crisp lighter lower lip)
- Muted masculine terracotta-plum oral interior & tongue (zero bright bubblegum pink)
- Crisp off-white anime teeth with subtle corner shading and masculine clenched-teeth grimace/grin
- Asymmetric confident smirk (`smug`), fierce battle shout (`tri`), and gritted teeth (`square`)
- Strict single-contour topology with zero detached chin dashes so 27x7 Keshin2D UV warping stays
  100% artifact-free across all phoneme, webcam, and expression crossfades.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "mouth_keyshape" / "male"

# Shared canonical mouth coordinate space (same scale convention as kurage02/chiwa: ref.w = 160.0)
CX = 1500.0
CY = 1006.0
REF_W = 160.0
SS = 4  # 4x supersampling for razor-sharp anti-aliased anime linework & cel gradients

# Local render canvas in canonical px around (CX, CY): [CX-140 .. CX+140], [CY-90 .. CY+110]
CAN_X0, CAN_Y0 = CX - 140.0, CY - 90.0
CAN_W, CAN_H = 280, 200
HI_W, HI_H = CAN_W * SS, CAN_H * SS

# Masculine shonen/seinen anime palette (muted terracotta-plum cavity, charcoal-espresso ink, warm off-white teeth)
COL_INK = np.array([36.0, 20.0, 22.0], dtype=np.float32)
COL_INK_LOWER = np.array([64.0, 38.0, 38.0], dtype=np.float32)
COL_CAVITY_TOP = np.array([66.0, 34.0, 36.0], dtype=np.float32)
COL_CAVITY_MID = np.array([106.0, 58.0, 56.0], dtype=np.float32)
COL_CAVITY_BOT = np.array([124.0, 72.0, 68.0], dtype=np.float32)
COL_TONGUE_DARK = np.array([142.0, 84.0, 80.0], dtype=np.float32)
COL_TONGUE_BASE = np.array([170.0, 106.0, 100.0], dtype=np.float32)
COL_TONGUE_HI = np.array([194.0, 128.0, 120.0], dtype=np.float32)
COL_TEETH_MAIN = np.array([240.0, 237.0, 236.0], dtype=np.float32)
COL_TEETH_SHADE = np.array([198.0, 190.0, 192.0], dtype=np.float32)
COL_TEETH_EDGE = np.array([92.0, 58.0, 58.0], dtype=np.float32)
COL_LIP_SHADOW = np.array([138.0, 92.0, 74.0], dtype=np.float32)


def _eval_curve(u: np.ndarray, knots_u: list[float], knots_y: list[float]) -> np.ndarray:
    """Smooth cubic-like interpolation along u in [0, 1] via monotone PCHIP-style Hermite or interp + smooth."""
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(np.asarray(knots_u, dtype=np.float64), np.asarray(knots_y, dtype=np.float64))(u)


def _dist_to_polyline(gx: np.ndarray, gy: np.ndarray, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact minimum Euclidean distance from grid (gx, gy) to polyline (px, py), plus parametric u in [0, 1]."""
    min_d2 = np.full(gx.shape, 1e18, dtype=np.float32)
    best_u = np.zeros(gx.shape, dtype=np.float32)
    n_seg = len(px) - 1
    for i in range(n_seg):
        ax, ay = float(px[i]), float(py[i])
        bx, by = float(px[i + 1]), float(py[i + 1])
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy + 1e-12
        t = np.clip(((gx - ax) * dx + (gy - ay) * dy) / l2, 0.0, 1.0)
        cx_p = ax + t * dx
        cy_p = ay + t * dy
        d2 = (gx - cx_p) ** 2 + (gy - cy_p) ** 2
        better = d2 < min_d2
        min_d2 = np.where(better, d2, min_d2)
        best_u = np.where(better, (i + t) / float(n_seg), best_u)
    return np.sqrt(min_d2), best_u


def _blend_over(dst_rgb: np.ndarray, dst_a: np.ndarray, src_rgb: np.ndarray, src_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sa = np.clip(src_a, 0.0, 1.0)[..., None]
    da = np.clip(dst_a, 0.0, 1.0)[..., None]
    oa = sa + da * (1.0 - sa)
    out_rgb = np.where(oa > 1e-5, (src_rgb * sa + dst_rgb * da * (1.0 - sa)) / np.maximum(oa, 1e-5), 0.0)
    return out_rgb, oa[..., 0]


def _render_closed_line(
    xa_rel: float,
    xb_rel: float,
    knots_u: list[float],
    knots_y: list[float],
    *,
    w_mid: float = 2.0,
    w_corner: float = 0.75,
    center_break: float = 0.0,
    center_dip_w: float = 1.0,
    tilt_shadow: float = 0.0,
) -> Image.Image:
    """Render a crisp 4x-supersampled masculine closed anime mouth stroke (zero moe curl, sharp tapered corners)."""
    xs = CAN_X0 + (np.arange(HI_W, dtype=np.float32) + 0.5) / SS
    ys = CAN_Y0 + (np.arange(HI_H, dtype=np.float32) + 0.5) / SS
    gx, gy = np.meshgrid(xs, ys)

    xa, xb = CX + xa_rel, CX + xb_rel
    us = np.linspace(0.0, 1.0, 48, dtype=np.float64)
    px = (xa + us * (xb - xa)).astype(np.float32)
    py = (CY + _eval_curve(us, knots_u, knots_y)).astype(np.float32)

    dist, bu = _dist_to_polyline(gx, gy, px, py)

    # Tapered masculine anime pen weight: sharp thin corners, firm center, optional subtle center dip/break
    taper = np.maximum(0.0, np.sin(np.pi * np.clip(bu, 0.0, 1.0))) ** 0.55
    # Slight thickening near outer third (masculine upper-lip shadow weight) & crisp center definition
    mid_dip = 1.0 - center_break * np.exp(-((bu - 0.5) / 0.08) ** 2)
    thick = (w_corner + (w_mid - w_corner) * taper) * mid_dip * center_dip_w
    # Taper ends smoothly to a clean point at exact u=0 and u=1
    end_tip = np.clip(np.minimum(bu, 1.0 - bu) / 0.045, 0.0, 1.0)
    thick = np.maximum(0.35, thick * (0.38 + 0.62 * end_tip))

    # Crisp AA edge (0.45 canonical px)
    stroke_a = np.clip((thick - dist) / 0.50 + 0.5, 0.0, 1.0)
    # Prevent round end-caps extending far past xa..xb
    end_cap = np.clip((gx - (xa - 1.2)) / 1.2, 0.0, 1.0) * np.clip(((xb + 1.2) - gx) / 1.2, 0.0, 1.0)
    stroke_a *= end_cap

    # Very subtle warm lip-crease tint immediately under the ink line (strictly <= 3.5px below line so it stays INSIDE PADY=7!)
    py_at_x = np.interp(gx, px, py)
    dy_under = gy - py_at_x
    span_u = (gx - xa) / max(1.0, xb - xa)
    under_x = np.exp(-((span_u - 0.5 - tilt_shadow) / 0.24) ** 2)
    under_y = np.exp(-((dy_under - 2.4) / 1.55) ** 2) * (dy_under > 0.2).astype(np.float32)
    sh_a = 0.28 * under_x * under_y * end_cap

    rgb = np.zeros((HI_H, HI_W, 3), dtype=np.float32)
    a = np.zeros((HI_H, HI_W), dtype=np.float32)
    rgb, a = _blend_over(rgb, a, np.broadcast_to(COL_LIP_SHADOW, rgb.shape), sh_a)
    # Slight warm color shift at thin tapered tips
    ink_col = COL_INK[None, None, :] * (0.85 + 0.15 * taper[..., None]) + np.array([20.0, 12.0, 10.0], dtype=np.float32) * (1.0 - taper[..., None])
    rgb, a = _blend_over(rgb, a, ink_col, stroke_a)

    rgba = np.zeros((HI_H, HI_W, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).round().astype(np.uint8)
    rgba[..., 3] = np.clip(a * 255.0, 0, 255).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA").resize((CAN_W, CAN_H), Image.LANCZOS)


def _render_open_mouth(
    xa_rel: float,
    xb_rel: float,
    u_knots: tuple[list[float], list[float]],
    lo_knots: tuple[list[float], list[float]],
    *,
    upper_teeth_frac: float = 0.24,
    lower_teeth_frac: float = 0.0,
    clenched_teeth: bool = False,
    bite_curve_y: float = 0.0,
    teeth_ticks: bool = False,
    tongue_frac: float = 0.44,
    tongue_shift_x: float = 0.0,
    upper_stroke: float = 2.5,
    lower_stroke: float = 1.65,
    corner_taper: float = 0.72,
    asymmetry_tilt: float = 0.0,
    lower_gap_sides: float = 0.0,
) -> Image.Image:
    """Render a crisp 4x-supersampled masculine open anime mouth with clean cel-shaded interior, teeth & tapered ink."""
    xs = CAN_X0 + (np.arange(HI_W, dtype=np.float32) + 0.5) / SS
    ys = CAN_Y0 + (np.arange(HI_H, dtype=np.float32) + 0.5) / SS
    gx, gy = np.meshgrid(xs, ys)

    xa, xb = CX + xa_rel, CX + xb_rel
    span = xb - xa
    us = np.linspace(0.0, 1.0, 48, dtype=np.float64)
    px = (xa + us * span).astype(np.float32)
    uy_pts = (CY + _eval_curve(us, u_knots[0], u_knots[1])).astype(np.float32)
    ly_pts = (CY + _eval_curve(us, lo_knots[0], lo_knots[1])).astype(np.float32)
    # Guarantee corners meet cleanly at u=0 and u=1
    ly_pts = np.maximum(ly_pts, uy_pts + 0.4)

    # Interior signed vertical coordinate at each x
    u_x = (gx - xa) / max(1.0, span)
    in_x = (gx >= xa - 0.5) & (gx <= xb + 0.5)
    uy_g = np.interp(gx, px, uy_pts)
    ly_g = np.interp(gx, px, ly_pts)
    h_g = np.maximum(0.5, ly_g - uy_g)

    # Smooth interior mask (using signed distance to top/bottom + crisp corner clipping)
    d_top_in = gy - uy_g
    d_bot_in = ly_g - gy
    # Horizontal taper near xa / xb so corners don't bulge
    d_left = gx - xa
    d_right = xb - gx
    in_mask_hard = (d_top_in >= 0.0) & (d_bot_in >= 0.0) & (d_left >= 0.0) & (d_right >= 0.0)
    # Distance transform for silky AA interior edge
    sd_in = np.minimum(np.minimum(d_top_in, d_bot_in), np.minimum(d_left, d_right))
    interior_a = np.clip(sd_in / 0.45 + 0.5, 0.0, 1.0) * in_x.astype(np.float32)

    # Normalized vertical & horizontal coordinates inside the mouth cavity: vn=0 (top lip), vn=1 (bottom lip)
    vn = np.clip((gy - uy_g) / h_g, 0.0, 1.0)
    hn = np.clip(u_x * 2.0 - 1.0, -1.0, 1.0)

    # 1. Base oral cavity gradient (dark masculine plum-brown roof shadow -> warm muted mid-cavity)
    t_cav = np.clip(vn / 0.55, 0.0, 1.0)
    cav_rgb = COL_CAVITY_TOP[None, None, :] * (1.0 - t_cav[..., None]) + COL_CAVITY_MID[None, None, :] * t_cav[..., None]
    # Darken outer corners subtly for authentic 3D anime mouth depth!
    corner_ao = 1.0 - 0.30 * (np.abs(hn) ** 2.2)
    cav_rgb = cav_rgb * corner_ao[..., None]

    # 2. Tongue (muted terracotta-rose with subtle twin-lobe masculine contour + cel highlight)
    if tongue_frac > 0.01 and not clenched_teeth:
        # Tongue dome rises smoothly from lower lip interior
        hn_t = hn - tongue_shift_x
        dome = (1.0 - 0.52 * (hn_t ** 2)) * (1.0 - 0.07 * np.exp(-((hn_t) / 0.16) ** 2))
        y_tongue_top = ly_g - h_g * tongue_frac * np.clip(dome, 0.0, 1.05)
        t_mask = np.clip((gy - y_tongue_top) / 0.9 + 0.5, 0.0, 1.0)
        # Subtle shadow band right behind tongue top rim + warm highlight lower-center
        t_vn = np.clip((gy - y_tongue_top) / np.maximum(4.0, ly_g - y_tongue_top), 0.0, 1.0)
        t_hi = np.exp(-((hn_t) / 0.52) ** 2) * np.exp(-((t_vn - 0.32) / 0.32) ** 2) * 0.65
        t_col = (
            COL_TONGUE_DARK[None, None, :] * (1.0 - np.clip(t_vn * 1.8, 0.0, 1.0))[..., None]
            + COL_TONGUE_BASE[None, None, :] * np.clip(t_vn * 1.8, 0.0, 1.0)[..., None]
        )
        t_col = t_col * (1.0 - t_hi[..., None]) + COL_TONGUE_HI[None, None, :] * t_hi[..., None]
        t_col = t_col * (1.0 - 0.22 * (np.abs(hn) ** 2.4))[..., None]
        cav_rgb = cav_rgb * (1.0 - t_mask[..., None]) + t_col * t_mask[..., None]

    # 3. Teeth (crisp anime upper teeth arch, optional lower teeth, or clenched-teeth grin/grimace!)
    if clenched_teeth:
        # Full crisp off-white clenched teeth filling cavity with subtle corner cavity shadow & clean bite seam!
        tooth_side_fade = np.clip((0.92 - np.abs(hn)) / 0.14, 0.0, 1.0)
        # Shade teeth slightly darker toward outer corners & top gum shadow
        gum_sh = np.exp(-((vn) / 0.16) ** 2) * 0.28 + np.exp(-((1.0 - vn) / 0.14) ** 2) * 0.22
        side_sh = (np.abs(hn) ** 2.4) * 0.42
        sh_f = np.clip(gum_sh + side_sh, 0.0, 0.72)
        teeth_rgb = COL_TEETH_MAIN[None, None, :] * (1.0 - sh_f[..., None]) + COL_TEETH_SHADE[None, None, :] * sh_f[..., None]
        cav_rgb = cav_rgb * (1.0 - tooth_side_fade[..., None]) + teeth_rgb * tooth_side_fade[..., None]

        # Crisp masculine horizontal bite line across middle of clenched teeth
        bite_y = (uy_g + ly_g) * 0.5 + bite_curve_y * h_g * (0.25 - hn * hn) + asymmetry_tilt * hn * 4.5
        d_bite = np.abs(gy - bite_y)
        bite_w = 0.85 * (1.0 - 0.35 * np.abs(hn))
        bite_a = np.clip((bite_w - d_bite) / 0.45 + 0.5, 0.0, 1.0) * tooth_side_fade * 0.80
        cav_rgb = cav_rgb * (1.0 - bite_a[..., None]) + COL_TEETH_EDGE[None, None, :] * bite_a[..., None]

        if teeth_ticks:
            # Subtle vertical anime canine/incisor accent ticks (sleek & restrained at avatar scale!)
            for tx in (-0.42, 0.42, -0.70, 0.70):
                dt = np.abs(hn - tx) * (span * 0.5)
                ta = np.clip((0.55 - dt) / 0.45 + 0.5, 0.0, 1.0) * np.clip(1.0 - np.abs(vn - 0.5) * 1.5, 0.0, 1.0) * 0.26
                cav_rgb = cav_rgb * (1.0 - ta[..., None]) + COL_TEETH_EDGE[None, None, :] * ta[..., None]
    else:
        if upper_teeth_frac > 0.01:
            # Crisp straight masculine upper teeth bar (subtle gentle arch, crisp shaded lower edge)
            arch_u = 1.0 - 0.14 * (hn ** 2)
            # Slight lip overhang on far corners so teeth recess naturally inside corners
            teeth_inset = np.clip((0.90 - np.abs(hn)) / 0.16, 0.0, 1.0)
            y_teeth_bot = uy_g + h_g * upper_teeth_frac * arch_u
            ut_mask = np.clip((y_teeth_bot - gy) / 0.55 + 0.5, 0.0, 1.0) * teeth_inset
            sh_u = np.clip((np.abs(hn) ** 2.2) * 0.38 + np.clip((uy_g + 4.0 - gy) / 4.0, 0.0, 1.0) * 0.18, 0.0, 0.65)
            ut_rgb = COL_TEETH_MAIN[None, None, :] * (1.0 - sh_u[..., None]) + COL_TEETH_SHADE[None, None, :] * sh_u[..., None]
            # Crisp thin dark edge along bottom rim of upper teeth
            edge_u = np.exp(-((gy - y_teeth_bot) / 0.75) ** 2) * teeth_inset * 0.68
            cav_rgb = cav_rgb * (1.0 - ut_mask[..., None]) + ut_rgb * ut_mask[..., None]
            cav_rgb = cav_rgb * (1.0 - edge_u[..., None]) + COL_TEETH_EDGE[None, None, :] * edge_u[..., None]
            # Subtle cast shadow from upper teeth onto cavity directly underneath
            sh_cast = np.exp(-((gy - (y_teeth_bot + 2.4)) / 2.2) ** 2) * (gy > y_teeth_bot).astype(np.float32) * teeth_inset * 0.42
            cav_rgb = cav_rgb * (1.0 - sh_cast[..., None]) + (COL_CAVITY_TOP * 0.75)[None, None, :] * sh_cast[..., None]

        if lower_teeth_frac > 0.01:
            arch_l = 1.0 - 0.22 * (hn ** 2)
            teeth_in_l = np.clip((0.76 - np.abs(hn)) / 0.16, 0.0, 1.0)
            y_lt_top = ly_g - h_g * lower_teeth_frac * arch_l
            lt_mask = np.clip((gy - y_lt_top) / 0.55 + 0.5, 0.0, 1.0) * teeth_in_l
            sh_l = np.clip((np.abs(hn) ** 2.0) * 0.42 + 0.15, 0.0, 0.68)
            lt_rgb = COL_TEETH_MAIN[None, None, :] * (1.0 - sh_l[..., None]) + COL_TEETH_SHADE[None, None, :] * sh_l[..., None]
            edge_l = np.exp(-((gy - y_lt_top) / 0.70) ** 2) * teeth_in_l * 0.55
            cav_rgb = cav_rgb * (1.0 - lt_mask[..., None]) + lt_rgb * lt_mask[..., None]
            cav_rgb = cav_rgb * (1.0 - edge_l[..., None]) + COL_TEETH_EDGE[None, None, :] * edge_l[..., None]

    # 4. Crisp variable-weight anime ink outlines (stronger upper lip, slightly finer lower lip, sharp corners!)
    d_up, bu_up = _dist_to_polyline(gx, gy, px, uy_pts)
    d_lo, bu_lo = _dist_to_polyline(gx, gy, px, ly_pts)

    taper_up = corner_taper + (1.0 - corner_taper) * (np.maximum(0.0, np.sin(np.pi * np.clip(bu_up, 0.0, 1.0))) ** 0.50)
    w_up = upper_stroke * taper_up
    # Sharp crisp corners right at u=0 and u=1
    end_up = np.clip(np.minimum(bu_up, 1.0 - bu_up) / 0.04, 0.0, 1.0)
    w_up = np.maximum(0.6, w_up * (0.55 + 0.45 * end_up))
    up_a = np.clip((w_up - d_up) / 0.45 + 0.5, 0.0, 1.0)

    taper_lo = 0.55 + 0.45 * (np.maximum(0.0, np.sin(np.pi * np.clip(bu_lo, 0.0, 1.0))) ** 0.55)
    w_lo = lower_stroke * taper_lo
    # Optional subtle anime lower-lip side break (thinning slightly near corners)
    if lower_gap_sides > 0.0:
        side_thin = 1.0 - lower_gap_sides * np.exp(-((np.abs(bu_lo - 0.5) - 0.36) / 0.09) ** 2)
        w_lo *= side_thin
    lo_a = np.clip((w_lo - d_lo) / 0.45 + 0.5, 0.0, 1.0)

    # Composite layers: cavity -> lower ink -> upper ink
    rgb = np.zeros((HI_H, HI_W, 3), dtype=np.float32)
    a = np.zeros((HI_H, HI_W), dtype=np.float32)
    rgb, a = _blend_over(rgb, a, cav_rgb, interior_a)
    rgb, a = _blend_over(rgb, a, np.broadcast_to(COL_INK_LOWER, rgb.shape), lo_a)
    rgb, a = _blend_over(rgb, a, np.broadcast_to(COL_INK, rgb.shape), up_a)

    rgba = np.zeros((HI_H, HI_W, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).round().astype(np.uint8)
    rgba[..., 3] = np.clip(a * 255.0, 0, 255).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA").resize((CAN_W, CAN_H), Image.LANCZOS)


def _extract_meta_and_crop(im: Image.Image, is_open: bool, pad: int = 3) -> tuple[Image.Image, dict]:
    """Tight-crop with `pad` px margin and compute exact Keshin2D 33-point U/Lo silhouette contours."""
    arr = np.asarray(im.convert("RGBA"))
    al = arr[..., 3].astype(np.float32) / 255.0
    mask = al > 0.25
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, "Empty mouth sprite rendered!"
    x0_p = max(0, int(xs.min()) - pad)
    y0_p = max(0, int(ys.min()) - pad)
    x1_p = min(im.width, int(xs.max()) + pad + 1)
    y1_p = min(im.height, int(ys.max()) + pad + 1)

    cropped = im.crop((x0_p, y0_p, x1_p, y1_p))

    # Exact subpixel horizontal span [xa, xb] in canonical coordinates
    col_has = mask.any(axis=0)
    nz_cols = np.nonzero(col_has)[0]
    xa = CAN_X0 + float(nz_cols[0])
    xb = CAN_X0 + float(nz_cols[-1])

    # Sample 33 evenly spaced U/Lo silhouette contours across [xa .. xb]
    N = 33
    U_vals = np.zeros(N, dtype=np.float64)
    Lo_vals = np.zeros(N, dtype=np.float64)
    for k in range(N):
        u = k / float(N - 1)
        x_f = nz_cols[0] + u * (nz_cols[-1] - nz_cols[0])
        c0 = int(np.clip(round(x_f), nz_cols[0], nz_cols[-1]))
        # Use 3-column neighborhood at extreme corners for robust vertical bounds
        lo_c = max(int(nz_cols[0]), c0 - 1)
        hi_c = min(int(nz_cols[-1]), c0 + 1)
        col_slice = al[:, lo_c : hi_c + 1].max(axis=1)
        nz_y = np.nonzero(col_slice > 0.25)[0]
        if len(nz_y):
            # Subpixel interpolation on top & bottom edge
            yt = float(nz_y[0])
            yb = float(nz_y[-1])
            if yt > 0:
                a0, a1 = col_slice[int(yt) - 1], col_slice[int(yt)]
                if a1 > a0 + 1e-4:
                    yt = yt - 1.0 + (0.25 - a0) / (a1 - a0)
            if yb + 1 < len(col_slice):
                a0, a1 = col_slice[int(yb)], col_slice[int(yb) + 1]
                if a0 > a1 + 1e-4:
                    yb = yb + (a0 - 0.25) / (a0 - a1)
            U_vals[k] = CAN_Y0 + yt
            Lo_vals[k] = CAN_Y0 + max(yb, yt + 1.5)
        else:
            U_vals[k] = CY
            Lo_vals[k] = CY + 2.0

    # Gentle 3-tap smoothing on interior points to keep UV warp mesh butter-smooth
    if N >= 5:
        U_s = U_vals.copy()
        Lo_s = Lo_vals.copy()
        U_s[1:-1] = 0.22 * U_vals[:-2] + 0.56 * U_vals[1:-1] + 0.22 * U_vals[2:]
        Lo_s[1:-1] = 0.22 * Lo_vals[:-2] + 0.56 * Lo_vals[1:-1] + 0.22 * Lo_vals[2:]
        U_vals, Lo_vals = U_s, np.maximum(Lo_s, U_s + 1.5)

    meta = {
        "x": int(round(CAN_X0 + x0_p)),
        "y": int(round(CAN_Y0 + y0_p)),
        "w": cropped.width,
        "h": cropped.height,
        "xa": round(float(xa), 1),
        "xb": round(float(xb), 1),
        "U": [round(float(v), 1) for v in U_vals],
        "Lo": [round(float(v), 1) for v in Lo_vals],
        "open": bool(is_open),
    }
    return cropped, meta


def build_all_keys() -> dict[str, tuple[Image.Image, dict]]:
    """Build all 17 cohesive masculine anime mouth keyshapes."""
    out: dict[str, tuple[Image.Image, dict]] = {}

    def reg(name: str, im: Image.Image, is_open: bool) -> None:
        out[name] = _extract_meta_and_crop(im, is_open)

    # 1. closed: flat stoic confident masculine line (subtle central lip cleft, ZERO cute upward smile!)
    reg("closed", _render_closed_line(
        -80.0, 80.0,
        [0.0, 0.18, 0.38, 0.50, 0.62, 0.82, 1.0],
        [0.2, -0.4, 0.3, 1.3, 0.3, -0.4, 0.2],
        w_mid=2.1, w_corner=0.85, center_break=0.14,
    ), False)

    # 2. a: masculine open "A" (firm straight upper lip, squared-masculine lower jaw curve, muted terracotta tongue)
    reg("a", _render_open_mouth(
        -76.0, 76.0,
        ([0.0, 0.16, 0.50, 0.84, 1.0], [-14.0, -21.0, -22.5, -21.0, -14.0]),
        ([0.0, 0.14, 0.32, 0.50, 0.68, 0.86, 1.0], [-11.0, 22.0, 44.0, 50.0, 44.0, 22.0, -11.0]),
        upper_teeth_frac=0.25, tongue_frac=0.44, upper_stroke=2.7, lower_stroke=1.8,
    ), True)

    # 3. i: masculine wide "I" (taut straight horizontal opening showing crisp clenched teeth + bite seam)
    reg("i", _render_open_mouth(
        -79.0, 79.0,
        ([0.0, 0.18, 0.50, 0.82, 1.0], [-6.0, -13.0, -14.0, -13.0, -6.0]),
        ([0.0, 0.18, 0.50, 0.82, 1.0], [-3.0, 15.0, 18.5, 15.0, -3.0]),
        clenched_teeth=True, teeth_ticks=False, upper_stroke=2.5, lower_stroke=1.7,
    ), True)

    # 4. u: masculine compact structured "U" (smooth masculine rounded oval, zero side pinch!)
    reg("u", _render_open_mouth(
        -29.0, 29.0,
        ([0.0, 0.05, 0.16, 0.34, 0.50, 0.66, 0.84, 0.95, 1.0], [4.0, -3.5, -10.5, -15.5, -17.0, -15.5, -10.5, -3.5, 4.0]),
        ([0.0, 0.05, 0.16, 0.34, 0.50, 0.66, 0.84, 0.95, 1.0], [5.0, 12.5, 19.5, 25.0, 27.0, 25.0, 19.5, 12.5, 5.0]),
        upper_teeth_frac=0.0, tongue_frac=0.34, upper_stroke=2.5, lower_stroke=1.85,
    ), True)

    # 5. e: masculine open "E" (confident medium-wide mouth with straight upper teeth + muted cavity)
    reg("e", _render_open_mouth(
        -73.0, 73.0,
        ([0.0, 0.18, 0.50, 0.82, 1.0], [-10.0, -17.5, -18.5, -17.5, -10.0]),
        ([0.0, 0.16, 0.34, 0.50, 0.66, 0.84, 1.0], [-7.0, 16.0, 30.0, 34.0, 30.0, 16.0, -7.0]),
        upper_teeth_frac=0.29, tongue_frac=0.38, upper_stroke=2.6, lower_stroke=1.75,
    ), True)

    # 6. o: masculine structured tall "O" (smooth structured oval, upper teeth rim, deep cavity, subtle tongue)
    reg("o", _render_open_mouth(
        -45.0, 45.0,
        ([0.0, 0.05, 0.16, 0.34, 0.50, 0.66, 0.84, 0.95, 1.0], [2.0, -13.0, -25.5, -33.5, -36.0, -33.5, -25.5, -13.0, 2.0]),
        ([0.0, 0.05, 0.16, 0.34, 0.50, 0.66, 0.84, 0.95, 1.0], [4.0, 21.0, 35.5, 44.5, 48.0, 44.5, 35.5, 21.0, 4.0]),
        upper_teeth_frac=0.14, tongue_frac=0.36, upper_stroke=2.6, lower_stroke=1.9,
    ), True)

    # 7. small: masculine half-parted murmur / soft speech opening
    reg("small", _render_open_mouth(
        -66.0, 66.0,
        ([0.0, 0.20, 0.50, 0.80, 1.0], [-3.0, -10.5, -11.5, -10.5, -3.0]),
        ([0.0, 0.20, 0.50, 0.80, 1.0], [0.0, 11.5, 15.0, 11.5, 0.0]),
        upper_teeth_frac=0.28, tongue_frac=0.30, upper_stroke=2.5, lower_stroke=1.7,
    ), True)

    # 8. laugh: confident wide masculine laugh (sharp lifted corners, strong upper teeth, rich cavity)
    reg("laugh", _render_open_mouth(
        -93.0, 93.0,
        ([0.0, 0.15, 0.50, 0.85, 1.0], [-24.0, -22.5, -20.5, -22.5, -24.0]),
        ([0.0, 0.14, 0.30, 0.50, 0.70, 0.86, 1.0], [-20.0, 22.0, 50.0, 58.0, 50.0, 22.0, -20.0]),
        upper_teeth_frac=0.26, lower_teeth_frac=0.07, tongue_frac=0.45, upper_stroke=2.8, lower_stroke=1.85,
    ), True)

    # 9. he: masculine stoic scowl / intense frown (sharp downward corners)
    reg("he", _render_closed_line(
        -71.0, 71.0,
        [0.0, 0.20, 0.50, 0.80, 1.0],
        [8.5, 0.0, -4.5, 0.0, 8.5],
        w_mid=2.3, w_corner=0.85, center_break=0.05,
    ), False)

    # 10. omega: restrained masculine dry-wit / sly half-smirk curve (NEVER moe :3 cat mouth!)
    reg("omega", _render_closed_line(
        -74.0, 74.0,
        [0.0, 0.22, 0.50, 0.78, 1.0],
        [2.5, 1.0, 0.5, -2.5, -8.5],
        w_mid=2.2, w_corner=0.85, center_break=0.08, tilt_shadow=0.08,
    ), False)

    # 11. wavy: masculine tense uneasy / grimacing quiver line
    reg("wavy", _render_closed_line(
        -76.0, 76.0,
        [0.0, 0.18, 0.35, 0.50, 0.65, 0.82, 1.0],
        [4.5, -1.5, 2.5, -2.0, 2.5, -1.5, 4.5],
        w_mid=2.1, w_corner=0.85, center_break=0.0,
    ), False)

    # 12. grin: confident masculine clenched-teeth grin (sharp lifted corners + subtle tooth ticks!)
    reg("grin", _render_open_mouth(
        -87.0, 87.0,
        ([0.0, 0.16, 0.50, 0.84, 1.0], [-16.0, -14.5, -12.5, -14.5, -16.0]),
        ([0.0, 0.16, 0.34, 0.50, 0.66, 0.84, 1.0], [-12.0, 12.0, 21.5, 24.5, 21.5, 12.0, -12.0]),
        clenched_teeth=True, bite_curve_y=0.22, teeth_ticks=True, upper_stroke=2.6, lower_stroke=1.75,
    ), True)

    # 13. tri: fierce masculine battle shout / intense excited yell (angular trapezoidal shout!)
    reg("tri", _render_open_mouth(
        -89.0, 89.0,
        ([0.0, 0.18, 0.50, 0.82, 1.0], [18.0, -14.0, -26.0, -14.0, 18.0]),
        ([0.0, 0.20, 0.50, 0.80, 1.0], [22.0, 56.0, 62.0, 56.0, 22.0]),
        upper_teeth_frac=0.22, lower_teeth_frac=0.18, tongue_frac=0.38, upper_stroke=2.8, lower_stroke=1.9,
    ), True)

    # 14. round: masculine surprised / shocked gasp oval (smooth oval, zero side pinch!)
    reg("round", _render_open_mouth(
        -36.0, 36.0,
        ([0.0, 0.05, 0.16, 0.34, 0.50, 0.66, 0.84, 0.95, 1.0], [2.0, -11.5, -22.5, -29.5, -32.0, -29.5, -22.5, -11.5, 2.0]),
        ([0.0, 0.05, 0.16, 0.34, 0.50, 0.66, 0.84, 0.95, 1.0], [4.0, 17.5, 28.5, 35.0, 37.5, 35.0, 28.5, 17.5, 4.0]),
        upper_teeth_frac=0.11, tongue_frac=0.34, upper_stroke=2.5, lower_stroke=1.85,
    ), True)

    # 15. smug: signature masculine cocky asymmetric smirk (lifted sharply on viewer-right, M09 vibe!)
    reg("smug", _render_open_mouth(
        -83.0, 85.0,
        ([0.0, 0.22, 0.50, 0.78, 1.0], [2.0, -5.0, -11.5, -18.5, -24.0]),
        ([0.0, 0.20, 0.45, 0.72, 1.0], [5.0, 18.5, 22.5, 13.0, -19.0]),
        upper_teeth_frac=0.34, tongue_frac=0.38, tongue_shift_x=-0.12, upper_stroke=2.7, lower_stroke=1.75,
    ), True)

    # 16. pout: masculine stubborn short tight frown (NO cute girl triangle!)
    reg("pout", _render_closed_line(
        -46.0, 46.0,
        [0.0, 0.25, 0.50, 0.75, 1.0],
        [6.5, -1.0, -4.0, -1.0, 6.5],
        w_mid=2.4, w_corner=0.95, center_break=0.0,
    ), False)

    # 17. square: masculine gritted-teeth grimace (M12 vibe — angled clenched teeth + canine ticks!)
    reg("square", _render_open_mouth(
        -75.0, 75.0,
        ([0.0, 0.14, 0.50, 0.86, 1.0], [3.0, -13.0, -16.5, -18.5, -7.0]),
        ([0.0, 0.14, 0.50, 0.86, 1.0], [8.0, 24.5, 22.0, 23.0, -2.0]),
        clenched_teeth=True, bite_curve_y=-0.15, teeth_ticks=True, asymmetry_tilt=-0.45, upper_stroke=2.7, lower_stroke=1.85,
    ), True)

    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys_data = build_all_keys()
    meta_keys: dict[str, dict] = {}
    for nm, (im, km) in keys_data.items():
        im.save(OUT_DIR / f"{nm}.png", optimize=True)
        meta_keys[nm] = km

    c_km = meta_keys["closed"]
    a_km = meta_keys["a"]
    ref = {
        "cx": round(0.5 * (c_km["xa"] + c_km["xb"]), 1),
        "cy": round(float(np.mean(c_km["U"][14:19])), 1),
        "w": round(c_km["xb"] - c_km["xa"], 1),
        "h_open": round(max(a_km["Lo"]) - min(a_km["U"]), 1),
    }
    meta = {
        "source": "Layer2D Masculine Anime KeyMouth (MIT License, contour-warp engine adapted from Keshin2D)",
        "sharp": 2.8,
        "ref": ref,
        "keys": meta_keys,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote 17 masculine mouth keys + meta.json -> {OUT_DIR} (ref={ref})")


if __name__ == "__main__":
    main()
