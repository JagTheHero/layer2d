"""Commercial-grade Adobe Photoshop (.psd) writer tailored for Live2D Cubism 4/5.

Produces standard 8-bit RGB+Alpha PSD v1 files with:
- Nested folder hierarchy (Adobe `lsct` section dividers, bottom-to-top PSD order)
- Clean UTF-16BE layer names (`luni`) + ASCII Pascal names
- Native clipping mask flags (`clipping=1`, recognized by Live2D Cubism, Photoshop, CSP)
- Per-layer opacity, visibility (hidden alternate expressions / toggles), blend modes
- PackBits RLE channel compression (huge size savings + universal reader support)
- Full merged composite image data (for thumbnails & previewers)

Spec reference: Adobe Photoshop File Formats Specification (PSD v1).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


BLEND_MODES = {
    "normal": b"norm",
    "multiply": b"mul ",
    "screen": b"scrn",
    "overlay": b"over",
    "add": b"lddg",
    "darken": b"dark",
    "lighten": b"lite",
}


@dataclass
class PsdLayer:
    name: str
    image: Image.Image  # RGBA cropped image
    x: int              # left on canvas
    y: int              # top on canvas
    visible: bool = True
    opacity: int = 255
    clipping: bool = False
    blend_mode: str = "normal"


@dataclass
class PsdGroup:
    name: str
    children: list[PsdLayer | PsdGroup] = field(default_factory=list)  # bottom-to-top render order!
    visible: bool = True
    opened: bool = True
    opacity: int = 255


def packbits_encode(row: bytes) -> bytes:
    """Adobe PackBits RLE encoder for a single scanline."""
    n = len(row)
    out = bytearray()
    i = 0
    while i < n:
        # Check for run of identical bytes (length >= 3)
        run_end = i + 1
        while run_end < n and row[run_end] == row[i] and (run_end - i) < 128:
            run_end += 1
        run_len = run_end - i
        if run_len >= 3:
            out.append((257 - run_len) & 0xFF)  # -(run_len - 1) as signed int8
            out.append(row[i])
            i = run_end
        else:
            lit_start = i
            i += 1
            while i < n and (i - lit_start) < 128:
                if i + 2 < n and row[i] == row[i + 1] == row[i + 2]:
                    break
                i += 1
            lit_len = i - lit_start
            out.append(lit_len - 1)
            out.extend(row[lit_start:i])
    return bytes(out)


def _pad4(data: bytes) -> bytes:
    rem = len(data) % 4
    return data if rem == 0 else data + b"\x00" * (4 - rem)


def _pascal_str(name: str, pad_multiple: int = 4) -> bytes:
    raw = name.encode("ascii", errors="replace")[:255]
    b = bytes([len(raw)]) + raw
    rem = len(b) % pad_multiple
    return b if rem == 0 else b + b"\x00" * (pad_multiple - rem)


def _luni_block(name: str) -> bytes:
    """Unicode layer name additional info block (`luni`)."""
    utf16 = name.encode("utf-16-be")
    # PSD spec: uint32 count of UTF-16 code units (not Python codepoints)
    body = struct.pack(">I", len(utf16) // 2) + utf16
    if len(body) % 2:
        body += b"\x00"
    # Additional layer info blocks in PSD v1 must be padded to even (4-byte safe)
    padded = _pad4(body)
    return b"8BIMluni" + struct.pack(">I", len(padded)) + padded


def _lsct_block(divider_type: int) -> bytes:
    """Section divider setting (`lsct`): 1=open folder, 2=closed folder, 3=bounding section divider."""
    # Include '8BIMpass' subtype for folder headers (pass-through blend)
    if divider_type in (1, 2):
        body = struct.pack(">I4s4s", divider_type, b"8BIM", b"pass")
    else:
        body = struct.pack(">I", divider_type)
    return b"8BIMlsct" + struct.pack(">I", len(body)) + body


def _encode_channel_rle(arr2d: np.ndarray) -> bytes:
    """Encode a 2D uint8 array as PSD RLE channel data (compression=1 + row byte counts + packed rows)."""
    h, w = arr2d.shape
    if h == 0 or w == 0:
        return struct.pack(">H", 0)
    rows = [packbits_encode(arr2d[r].tobytes()) for r in range(h)]
    counts = b"".join(struct.pack(">H", len(r)) for r in rows)
    return struct.pack(">H", 1) + counts + b"".join(rows)


def _flatten_records(
    items: list[PsdLayer | PsdGroup], canvas_w: int, canvas_h: int
) -> list[tuple[bytes, list[bytes]]]:
    """Flatten bottom-to-top groups/layers into PSD layer records & channel byte lists.

    In Photoshop PSD layer lists (bottom-to-top order):
    - A group starts with a hidden bounding divider (`lsct` type 3, name '</Layer group>')
    - Then all children inside the group (bottom-to-top)
    - Then the folder header (`lsct` type 1 or 2, name = group.name)
    """
    records: list[tuple[bytes, list[bytes]]] = []

    def add_divider(name: str, divider_type: int, visible: bool = True, opacity: int = 255) -> None:
        # 4 empty channels: A (-1), R (0), G (1), B (2)
        ch_ids = [-1, 0, 1, 2]
        ch_data = [struct.pack(">H", 0) for _ in ch_ids]  # raw 0-byte empty channel
        ch_info = b"".join(struct.pack(">hI", cid, len(cd)) for cid, cd in zip(ch_ids, ch_data))

        flags = 0x00
        if not visible:
            flags |= 0x02  # bit 1 = hidden
        flags |= 0x08 | 0x10  # irrelevant pixel data marker for section dividers

        extra_blocks = _luni_block(name) + _lsct_block(divider_type)
        # Mask data (0), blending ranges (0), pascal name, additional blocks
        extra = struct.pack(">II", 0, 0) + _pascal_str(name, 4) + extra_blocks

        rec = (
            struct.pack(">iiii", 0, 0, 0, 0)  # top, left, bottom, right
            + struct.pack(">H", len(ch_ids))
            + ch_info
            + b"8BIM"
            + b"pass"
            + struct.pack(">BBBB", opacity, 0, flags, 0)
            + struct.pack(">I", len(extra))
            + extra
        )
        records.append((rec, ch_data))

    def add_layer(layer: PsdLayer) -> None:
        im = layer.image.convert("RGBA")
        # Crop strictly to canvas bounds
        x0, y0 = layer.x, layer.y
        x1, y1 = x0 + im.width, y0 + im.height

        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(canvas_w, x1), min(canvas_h, y1)

        if cx1 <= cx0 or cy1 <= cy0:
            arr = np.zeros((0, 0, 4), dtype=np.uint8)
            top, left, bottom, right = 0, 0, 0, 0
        else:
            crop = im.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
            arr = np.asarray(crop, dtype=np.uint8)
            # Tight-trim completely transparent margins inside canvas bounds
            nz_y, nz_x = np.nonzero(arr[..., 3] > 0)
            if len(nz_y) == 0:
                arr = np.zeros((0, 0, 4), dtype=np.uint8)
                top, left, bottom, right = 0, 0, 0, 0
            else:
                ty0, ty1 = int(nz_y.min()), int(nz_y.max()) + 1
                tx0, tx1 = int(nz_x.min()), int(nz_x.max()) + 1
                arr = arr[ty0:ty1, tx0:tx1]
                top, left = cy0 + ty0, cx0 + tx0
                bottom, right = cy0 + ty1, cx0 + tx1

        # Channel order: Alpha (-1), Red (0), Green (1), Blue (2)
        ch_ids = [-1, 0, 1, 2]
        if arr.size == 0:
            ch_data = [struct.pack(">H", 0) for _ in ch_ids]
        else:
            ch_data = [
                _encode_channel_rle(arr[..., 3]),  # A
                _encode_channel_rle(arr[..., 0]),  # R
                _encode_channel_rle(arr[..., 1]),  # G
                _encode_channel_rle(arr[..., 2]),  # B
            ]

        ch_info = b"".join(struct.pack(">hI", cid, len(cd)) for cid, cd in zip(ch_ids, ch_data))
        blend = BLEND_MODES.get(layer.blend_mode, b"norm")
        flags = 0x00 if layer.visible else 0x02
        clip = 1 if layer.clipping else 0

        extra_blocks = _luni_block(layer.name)
        extra = struct.pack(">II", 0, 0) + _pascal_str(layer.name, 4) + extra_blocks

        rec = (
            struct.pack(">iiii", top, left, bottom, right)
            + struct.pack(">H", len(ch_ids))
            + ch_info
            + b"8BIM"
            + blend
            + struct.pack(">BBBB", max(0, min(255, layer.opacity)), clip, flags, 0)
            + struct.pack(">I", len(extra))
            + extra
        )
        records.append((rec, ch_data))

    def walk(nodes: list[PsdLayer | PsdGroup]) -> None:
        for node in nodes:
            if isinstance(node, PsdGroup):
                add_divider("</Layer group>", 3, visible=True, opacity=255)
                walk(node.children)
                add_divider(node.name, 1 if node.opened else 2, visible=node.visible, opacity=node.opacity)
            else:
                add_layer(node)

    walk(items)
    return records


def _srgb_icc_profile() -> bytes:
    """Return standard sRGB IEC61966-2.1 ICC profile bytes via PIL.ImageCms if available."""
    try:
        from PIL import ImageCms
        prof = ImageCms.createProfile("sRGB")
        return ImageCms.ImageCmsProfile(prof).tobytes()
    except Exception:
        return b""


def _render_composite(
    items: list[PsdLayer | PsdGroup], canvas_w: int, canvas_h: int
) -> Image.Image:
    """Render all visible layers (respecting group visibility, clipping & Cubism blend modes) onto transparent RGBA."""
    dst = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)  # straight RGBA in [0..1]
    last_base_alpha: np.ndarray | None = None

    def walk(nodes: list[PsdLayer | PsdGroup], parent_vis: bool) -> None:
        nonlocal last_base_alpha
        for node in nodes:
            if isinstance(node, PsdGroup):
                prev_base = last_base_alpha
                last_base_alpha = None
                walk(node.children, parent_vis and node.visible)
                last_base_alpha = prev_base
                continue
            if not (parent_vis and node.visible):
                if not node.clipping:
                    last_base_alpha = None
                continue
            im = node.image.convert("RGBA")
            x0, y0 = node.x, node.y
            x1, y1 = x0 + im.width, y0 + im.height
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(canvas_w, x1), min(canvas_h, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                if not node.clipping:
                    last_base_alpha = None
                continue

            crop = im.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
            src = np.asarray(crop, dtype=np.float32) / 255.0
            if node.opacity < 255:
                src[..., 3] *= node.opacity / 255.0

            if node.clipping and last_base_alpha is not None:
                src[..., 3] *= last_base_alpha[cy0:cy1, cx0:cx1]
            else:
                base_a = np.zeros((canvas_h, canvas_w), dtype=np.float32)
                base_a[cy0:cy1, cx0:cx1] = src[..., 3]
                last_base_alpha = base_a

            d_sub = dst[cy0:cy1, cx0:cx1]
            sa = src[..., 3:4]
            da = d_sub[..., 3:4]
            srgb = src[..., :3]
            drgb = d_sub[..., :3]

            if node.blend_mode == "multiply":
                blended = drgb * srgb
                rgb_mix = (1.0 - da) * srgb + da * blended
            elif node.blend_mode == "add":
                blended = np.minimum(1.0, drgb + srgb)
                rgb_mix = (1.0 - da) * srgb + da * blended
            else:
                rgb_mix = srgb

            out_a = sa + da * (1.0 - sa)
            out_rgb_pm = sa * rgb_mix + da * (1.0 - sa) * drgb
            d_sub[..., :3] = np.where(out_a > 1e-6, out_rgb_pm / np.maximum(out_a, 1e-6), 0.0)
            d_sub[..., 3:4] = out_a

    walk(items, True)
    return Image.fromarray(np.clip(dst * 255.0, 0.0, 255.0).round().astype(np.uint8), "RGBA")


def write_psd(
    out_path: Path,
    canvas_w: int,
    canvas_h: int,
    items: list[PsdLayer | PsdGroup],
    composite: Image.Image | None = None,
) -> dict:
    """Write a commercial Live2D-ready PSD v1 file (8-bit RGB + per-layer alpha, RLE compressed, sRGB ICC)."""
    if canvas_w > 30000 or canvas_h > 30000:
        raise ValueError("PSD v1 supports max 30000x30000")

    records = _flatten_records(items, canvas_w, canvas_h)
    layer_count = len(records)

    # 1. File Header (26 bytes): standard 3 channels (R, G, B), 8-bit, RGB color mode (3).
    #    Live2D Cubism Editor strictly expects a 3-channel RGB document header (no extra global alpha channel
    #    in the Channels panel; per-layer transparency lives in each layer's channel -1!).
    header = struct.pack(">4sH6sHIIHH", b"8BPS", 1, b"\x00" * 6, 3, canvas_h, canvas_w, 8, 3)

    # 2. Color Mode Data (empty)
    color_mode = struct.pack(">I", 0)

    # 3. Image Resources: 0x03ED ResolutionInfo (300 DPI) + 0x040F sRGB IEC61966-2.1 ICC profile
    res_payload = struct.pack(">IHHIHH", 300 << 16, 1, 1, 300 << 16, 1, 1)
    res_blocks = b"8BIM" + struct.pack(">H", 0x03ED) + b"\x00\x00" + struct.pack(">I", len(res_payload)) + res_payload
    icc = _srgb_icc_profile()
    if icc:
        icc_pad = b"\x00" if (len(icc) % 2) else b""
        res_blocks += b"8BIM" + struct.pack(">H", 0x040F) + b"\x00\x00" + struct.pack(">I", len(icc)) + icc + icc_pad
    image_resources = struct.pack(">I", len(res_blocks)) + res_blocks

    # 4. Layer and Mask Information Section
    layer_records_bytes = b"".join(rec for rec, _ in records)
    channel_data_bytes = b"".join(cd for _, cds in records for cd in cds)
    # Positive layer_count matches standard 3-channel RGB composite (no 4th global alpha channel)
    layer_info_body = struct.pack(">h", layer_count) + layer_records_bytes + channel_data_bytes
    if len(layer_info_body) % 2:
        layer_info_body += b"\x00"
    layer_info = struct.pack(">I", len(layer_info_body)) + layer_info_body
    global_mask = struct.pack(">I", 0)
    layer_mask_body = layer_info + global_mask
    layer_mask_section = struct.pack(">I", len(layer_mask_body)) + layer_mask_body

    # 5. Merged Composite Image Data (R, G, B planar RLE, white-matted preview matching Photoshop 3-ch RGB)
    comp = composite.convert("RGBA") if composite is not None else _render_composite(items, canvas_w, canvas_h)
    if comp.size != (canvas_w, canvas_h):
        comp = comp.resize((canvas_w, canvas_h), Image.LANCZOS)
    bg_white = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    bg_white.alpha_composite(comp)
    comp_arr = np.asarray(bg_white.convert("RGB"), dtype=np.uint8)
    # Planar order: R, G, B — all scanline byte counts first, then all packed scanlines
    all_counts = bytearray()
    all_rows = bytearray()
    for ch in (0, 1, 2):
        plane = comp_arr[..., ch]
        for r in range(canvas_h):
            packed = packbits_encode(plane[r].tobytes())
            all_counts.extend(struct.pack(">H", len(packed)))
            all_rows.extend(packed)
    merged_section = struct.pack(">H", 1) + bytes(all_counts) + bytes(all_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(color_mode)
        f.write(image_resources)
        f.write(layer_mask_section)
        f.write(merged_section)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    return {
        "path": str(out_path),
        "canvas": [canvas_w, canvas_h],
        "records": layer_count,
        "size_mb": round(size_mb, 2),
    }
