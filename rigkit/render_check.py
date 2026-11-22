"""Headless Chrome (Playwright) automated visual QA renderer for layer2d WebGL2 viewer.

Renders contact sheets of front, head-turn sweep (-1.0..+1.0), outfit toggles, and expressions.
Usage:
    python -m rigkit.render_check <viewer_dir> <out_png> <poses_json> <rect> <scale> '<json_spec>' ...
"""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def render_sheet(
    viewer_dir: Path,
    out_path: Path,
    rect: list[float],
    scale: float,
    specs: list[dict],
    base_pose: dict | None = None,
    cols: int = 4,
    transparent: bool = False,
) -> Path:
    handler = partial(_QuietHandler, directory=str(viewer_dir.resolve()))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    url = f"http://127.0.0.1:{port}/index.html" + ("?obs=1" if transparent else "")
    W = round((rect[2] - rect[0]) * scale)
    H = round((rect[3] - rect[1]) * scale)
    default_front = base_pose or {
        "a": 0,
        "pitch": 0.0,
        "vowel": None,
        "roll": 0,
        "body": 0,
        "sway": 0,
        "breath": 1,
        "lean": 0,
        "bounce": 0,
        "drop": 0,
        "armR": 0,
        "armL": 0,
        "raiseR": 0,
        "raiseL": 0,
        "elbowR": 0,
        "elbowL": 0,
        "wristR": 0,
        "wristL": 0,
        "legR": 0,
        "legL": 0,
        "gaze": [0, 0],
        "eyeOpen": 1.0,
        "hair": {},
        "skirt": [0, 0],
        "flare": [0, 0],
    }

    ims = []
    logs = []
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(
                channel="chrome",
                headless=True,
                args=[
                    "--use-angle=swiftshader",
                    "--enable-unsafe-swiftshader",
                    "--ignore-gpu-blocklist",
                ],
            )
            page = b.new_page(viewport={"width": 1000, "height": 1000})
            page.on("console", lambda msg: logs.append(f"{msg.type}: {msg.text}"))
            page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
            page.goto(url)
            page.wait_for_function("window.__ready === true", timeout=120000)

            for raw_sp in specs:
                sp = dict(raw_sp)
                label = sp.pop("_label", "")
                m = {**default_front, **sp}
                if "only" in sp:
                    page.evaluate(
                        "(keep) => { P.expr.holds.__dbg = { hide: P.layers.map((l) => l.id).filter((i) => !keep.includes(i)), add: [], still: [] }; }",
                        sp["only"],
                    )
                    m["hold"] = "__dbg"
                page.evaluate(
                    "([m, r, W, H, n]) => window.__pose(m, r, W, H, n)",
                    [m, rect, W, H, sp.get("face")],
                )
                data = page.evaluate("document.getElementById('gl').toDataURL('image/png')")
                mode = "RGBA" if transparent else "RGB"
                im = Image.open(io.BytesIO(base64.b64decode(data.split(",")[1]))).convert(mode)
                txt = label or json.dumps(sp)[:70]
                d = ImageDraw.Draw(im)
                d.rectangle((0, 0, min(im.width, len(txt) * 7 + 16), 22), fill=(20, 22, 28, 230) if transparent else (20, 22, 28))
                d.text((6, 4), txt, fill=(245, 245, 250, 255) if transparent else (245, 245, 250))
                ims.append(im)
            b.close()
    finally:
        srv.shutdown()

    errs = [l for l in logs if "pageerror" in l or l.startswith("error:")]
    if errs:
        print("Browser errors:\n" + "\n".join(errs[-20:]), file=sys.stderr)

    cols = min(cols, len(ims))
    rows = (len(ims) + cols - 1) // cols
    sheet_mode = "RGBA" if transparent else "RGB"
    sheet_bg = (0, 0, 0, 0) if transparent else (235, 236, 242)
    sheet = Image.new(sheet_mode, (cols * W + (cols - 1) * 6, rows * H + (rows - 1) * 6), sheet_bg)
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * (W + 6), (i // cols) * (H + 6)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"Rendered {len(ims)} views -> {out_path} ({sheet.size})")
    return out_path
