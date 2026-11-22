"""Part cleanup utilities for layered 2D avatar textures.

Fixes common parts-sheet alpha & edge defects:
1. Semi-transparent interior pixels (soft fills sometimes sit at alpha 225-250):
   solid interiors are snapped cleanly to alpha=255 while keeping smooth anti-aliased outer edges.
2. Edge color decontamination (RGB bleeding): transparent / low-alpha edge pixels get their RGB
   filled from the nearest solid pixel so WebGL2 bilinear sampling and mesh warping never
   produce dark or white fringes.
3. Optional white-halo removal & color-tone matching to the canonical artwork.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def harden_solid_alpha(im: Image.Image, core_thresh: int = 160, margin_px: int = 3) -> Image.Image:
    """Snap interior pixels of a solid painted part to alpha=255, keeping smooth AA border."""
    arr = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
    a = arr[..., 3]
    solid = a >= core_thresh
    if not solid.any():
        return im
    # Fill enclosed holes inside the solid shape (e.g. faint translucent paint spots)
    filled = ndimage.binary_fill_holes(solid)
    interior = ndimage.binary_erosion(filled, iterations=margin_px)
    # Only harden pixels that actually have some paint (avoid filling intentional cut-outs)
    harden = interior & (a >= 90)
    a[harden] = 255
    # Clean near-zero alpha dust
    a[a < 6] = 0
    arr[..., 3] = a
    return Image.fromarray(arr, "RGBA")


def bleed_rgb_edges(im: Image.Image, solid_thresh: int = 200, max_dist: int = 16) -> Image.Image:
    """Propagate solid-pixel RGB into soft/transparent edge pixels to eliminate matte halos."""
    arr = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
    a = arr[..., 3]
    solid = a >= solid_thresh
    if not solid.any() or solid.all():
        return im
    dist, (iy, ix) = ndimage.distance_transform_edt(~solid, return_distances=True, return_indices=True)
    fix = (~solid) & (dist <= max_dist)
    for c in range(3):
        ch = arr[..., c]
        ch[fix] = ch[iy[fix], ix[fix]]
    return Image.fromarray(arr, "RGBA")


def clean_part(im: Image.Image, *, solid: bool = True, bleed: bool = True) -> Image.Image:
    out = im.convert("RGBA")
    if solid:
        out = harden_solid_alpha(out)
    if bleed:
        out = bleed_rgb_edges(out)
    return out


def match_rgb_gain(im: Image.Image, gain: tuple[float, float, float]) -> Image.Image:
    arr = np.asarray(im.convert("RGBA"), dtype=np.float32).copy()
    for c in range(3):
        arr[..., c] = np.clip(arr[..., c] * gain[c], 0, 255)
    return Image.fromarray(arr.round().astype(np.uint8), "RGBA")
