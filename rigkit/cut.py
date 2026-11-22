"""Segment transparent parts-sheet pages into clean individual RGBA parts.

Usage:
    python -m rigkit.cut find <model_dir> <page> [--merge 4] [--core 128] [--erode 0]
    python -m rigkit.cut split <model_dir> <page> <comp_n> <x0> <y0> <x1> <y1>
    python -m rigkit.cut export <model_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ALPHA_MIN, AREA_MIN, PAD = 8, 120, 8


def checker(w: int, h: int, s: int = 32) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    v = np.where(((yy // s) + (xx // s)) % 2 == 0, 205, 235).astype(np.uint8)
    return Image.fromarray(np.dstack([v, v, v]), "RGB")


def find_components(
    model_dir: Path, page: str, merge: int = 4, core_min: int = 128, erode: int = 0
) -> list[dict]:
    work = model_dir / "work"
    review = model_dir / "review"
    work.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)

    im = Image.open(model_dir / f"out/{page}.png").convert("RGBA")
    alpha = np.asarray(im)[..., 3]
    mask = alpha > ALPHA_MIN
    core = alpha > core_min if core_min else mask
    if erode > 0:
        core = ndimage.binary_erosion(core, iterations=erode)
    grown = ndimage.binary_dilation(core, iterations=merge) if merge > 0 else core
    lab, n = ndimage.label(grown)
    lab[~core] = 0
    if erode > 0:
        sizes = ndimage.sum(core, lab, range(n + 1))
        lab[sizes[lab] < AREA_MIN] = 0
    if core_min or erode > 0:
        _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(mask, lab[iy, ix], 0)

    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if area < AREA_MIN:
            lab[lab == i] = 0
            continue
        comps.append(
            {"label": i, "box": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)], "area": area}
        )

    comps.sort(key=lambda c: (c["box"][1] // 220, c["box"][0]))
    for k, c in enumerate(comps):
        c["n"] = k

    np.save(work / f"{page}.labels.npy", lab.astype(np.int32))
    (work / f"{page}.comps.json").write_text(json.dumps(comps, indent=1))
    _save_label_preview(im, comps, review / f"labels_{page}.png")
    print(f"{page}: {len(comps)} components -> review/labels_{page}.png")
    return comps


def split_component(
    model_dir: Path, page: str, comp_n: int, x0: int, y0: int, x1: int, y1: int
) -> list[dict]:
    """Split pixels of component `comp_n` inside rectangle [x0,y0,x1,y1] into a new component."""
    work = model_dir / "work"
    review = model_dir / "review"
    im = Image.open(model_dir / f"out/{page}.png").convert("RGBA")
    lab = np.load(work / f"{page}.labels.npy")
    comps = json.loads((work / f"{page}.comps.json").read_text())
    target = next(c for c in comps if c["n"] == comp_n)
    old_id = target["label"]
    new_id = int(lab.max()) + 1

    sel = lab == old_id
    rect = np.zeros_like(sel)
    rect[y0:y1, x0:x1] = True
    lab[sel & rect] = new_id

    new_comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if area < AREA_MIN:
            lab[lab == i] = 0
            continue
        new_comps.append(
            {"label": i, "box": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)], "area": area}
        )
    new_comps.sort(key=lambda c: (c["box"][1] // 220, c["box"][0]))
    for k, c in enumerate(new_comps):
        c["n"] = k

    np.save(work / f"{page}.labels.npy", lab.astype(np.int32))
    (work / f"{page}.comps.json").write_text(json.dumps(new_comps, indent=1))
    _save_label_preview(im, new_comps, review / f"labels_{page}.png")
    print(f"{page}: split comp {comp_n} -> now {len(new_comps)} components")
    return new_comps


def _save_label_preview(im: Image.Image, comps: list[dict], out_path: Path) -> None:
    bg = checker(*im.size)
    bg.paste(im, (0, 0), im)
    d = ImageDraw.Draw(bg)
    f = ImageFont.truetype(FONT, 44)
    for c in comps:
        x0, y0, x1, y1 = c["box"]
        d.rectangle((x0, y0, x1, y1), outline=(235, 30, 40), width=3)
        d.text((x0 + 6, y0 + 2), str(c["n"]), fill=(235, 30, 40), font=f, stroke_width=4, stroke_fill=(255, 255, 255))
    bg.thumbnail((2400, 2400), Image.LANCZOS)
    bg.save(out_path)


def export_parts(model_dir: Path) -> list[dict]:
    work = model_dir / "work"
    parts_dir = model_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    mp = json.loads((model_dir / "parts_map.json").read_text())

    index = []
    for page, entries in mp.items():
        im = np.asarray(Image.open(model_dir / f"out/{page}.png").convert("RGBA"))
        lab = np.load(work / f"{page}.labels.npy")
        comps = {c["n"]: c for c in json.loads((work / f"{page}.comps.json").read_text())}
        used = [n for p in entries for n in p["comps"]]
        dup = {n for n in used if used.count(n) > 1}
        if dup:
            sys.exit(f"{page}: duplicate components: {sorted(dup)}")
        missing = sorted(set(comps) - set(used))
        if missing:
            print(f"  [note] {page}: unassigned components {missing}")
        folder = page.split("_v")[0]
        out_dir = parts_dir / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in entries:
            labels = [comps[n]["label"] for n in p["comps"]]
            sel = np.isin(lab, labels)
            ys, xs = np.nonzero(sel)
            if len(ys) == 0:
                print(f"  [warn] {page} {p['id']}: empty!")
                continue
            x0, y0 = max(int(xs.min()) - PAD, 0), max(int(ys.min()) - PAD, 0)
            x1, y1 = min(int(xs.max()) + 1 + PAD, im.shape[1]), min(int(ys.max()) + 1 + PAD, im.shape[0])
            crop = im[y0:y1, x0:x1].copy()
            crop[..., 3] = np.where(sel[y0:y1, x0:x1], crop[..., 3], 0)
            fname = f"{p['id']}_{p['name']}.png"
            Image.fromarray(crop, "RGBA").save(out_dir / fname)
            index.append(
                {
                    "id": p["id"],
                    "name": p["name"],
                    "file": f"{folder}/{fname}",
                    "page": page,
                    "box_on_page": [x0, y0, x1, y1],
                    "size": [x1 - x0, y1 - y0],
                }
            )
    (parts_dir / "index.json").write_text(json.dumps(index, indent=1))
    _build_sheets(parts_dir, index)
    print(f"Exported {len(index)} parts -> {parts_dir}")
    return index


def _build_sheets(parts_dir: Path, index: list[dict]) -> None:
    f = ImageFont.truetype(FONT, 17)
    for folder in sorted({e["file"].split("/")[0] for e in index}):
        items = [e for e in index if e["file"].startswith(folder + "/")]
        cell, cols = 300, 8
        rows = -(-len(items) // cols)
        sheet = checker(cols * cell, rows * (cell + 34))
        d = ImageDraw.Draw(sheet)
        for k, e in enumerate(items):
            x, y = (k % cols) * cell, (k // cols) * (cell + 34)
            part = Image.open(parts_dir / e["file"])
            part.thumbnail((cell - 16, cell - 16), Image.LANCZOS)
            sheet.paste(part, (x + (cell - part.width) // 2, y + (cell - part.height) // 2), part)
            d.text((x + 6, y + cell + 4), f"{e['id']} {e['name']}"[:32], fill=(20, 20, 20), font=f)
        sheet.save(parts_dir / f"sheet_{folder}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_find = sub.add_parser("find")
    p_find.add_argument("model_dir", type=Path)
    p_find.add_argument("page")
    p_find.add_argument("--merge", type=int, default=4)
    p_find.add_argument("--core", type=int, default=128)
    p_find.add_argument("--erode", type=int, default=0)

    p_split = sub.add_parser("split")
    p_split.add_argument("model_dir", type=Path)
    p_split.add_argument("page")
    p_split.add_argument("comp_n", type=int)
    p_split.add_argument("x0", type=int)
    p_split.add_argument("y0", type=int)
    p_split.add_argument("x1", type=int)
    p_split.add_argument("y1", type=int)

    p_exp = sub.add_parser("export")
    p_exp.add_argument("model_dir", type=Path)

    a = ap.parse_args()
    if a.cmd == "find":
        find_components(a.model_dir, a.page, a.merge, a.core, a.erode)
    elif a.cmd == "split":
        split_component(a.model_dir, a.page, a.comp_n, a.x0, a.y0, a.x1, a.y1)
    elif a.cmd == "export":
        export_parts(a.model_dir)


if __name__ == "__main__":
    main()
