#!/usr/bin/env python3
"""Offline Layer2D (.l2d) Agent Visual & Runtime QA Runner.

Validates structure + SHA-256 digests, boots the bundled standalone WebGL2 Player in
headless Chromium, checks zero console/page errors and OBS ?obs=1 transparency, and
captures visual contact sheets for agent inspection:
  - qa_turns_wide.png   (9-angle yaw sweep -1.0 .. +1.0)
  - qa_turns_center.png (9-angle fine center sweep -0.40 .. +0.40)
  - qa_pitch_grid.png   (3x3 Yaw x Pitch grid)
  - qa_expressions.png  (up to 16 expressions)
  - qa_report.json      (machine-readable summary; exits 1 on failure)
"""
from __future__ import annotations

import argparse
import functools
import io
import json
import math
import shutil
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageDraw
from l2d_tools import validate_l2d


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass


def _compose_grid(tiles: list[tuple[str, Image.Image]], cols: int, out_path: Path) -> None:
    if not tiles:
        return
    tw, th = tiles[0][1].size
    lbl_h = 26
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + lbl_h)), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, im) in enumerate(tiles):
        c, r = idx % cols, idx // cols
        x0, y0 = c * tw, r * (th + lbl_h)
        sheet.paste(im.convert("RGB"), (x0, y0 + lbl_h))
        draw.text((x0 + 8, y0 + 6), label, fill=(242, 243, 245))
    sheet.save(out_path)


def run_qa(l2d_path: Path, out_dir: Path) -> dict:
    from playwright.sync_api import sync_playwright

    l2d_path = l2d_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"l2d": str(l2d_path), "ok": True, "errors": []}

    val = validate_l2d(l2d_path)
    report["validate"] = val
    if not val["ok"]:
        report["ok"] = False
        report["errors"].extend(val["errors"])
        (out_dir / "qa_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    player_src = Path(__file__).resolve().parent / "player.html"
    if not player_src.exists():
        raise FileNotFoundError(f"Missing bundled player.html next to {__file__}")

    tmp = Path(tempfile.mkdtemp(prefix="l2d_qa_"))
    try:
        shutil.copy2(player_src, tmp / "index.html")
        arm3d_src = Path(__file__).resolve().parent / "arm3d.js"
        if arm3d_src.exists():
            shutil.copy2(arm3d_src, tmp / "arm3d.js")
        shutil.copy2(l2d_path, tmp / "model.l2d")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(tmp)))
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        console_errors: list[str] = []
        try:
            with sync_playwright() as pw:
                try:
                    br = pw.chromium.launch(
                        channel="chrome",
                        headless=True,
                        args=["--use-gl=angle", "--use-angle=gl-egl", "--enable-webgl", "--ignore-gpu-blocklist"],
                    )
                except Exception:
                    br = pw.chromium.launch(
                        headless=True,
                        args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
                    )
                # 1. OBS ?obs=1 alpha + hidden UI check
                page = br.new_page(viewport={"width": 640, "height": 800})
                page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
                page.on("console", lambda m: console_errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
                page.goto(f"http://127.0.0.1:{port}/?model=model.l2d&obs=1&auto=0", wait_until="networkidle")
                page.wait_for_function("() => window.__ready === true", timeout=30000)
                page.wait_for_timeout(250)

                obs_png = page.screenshot(omit_background=True)
                obs_im = Image.open(io.BytesIO(obs_png)).convert("RGBA")
                tl_a = obs_im.getpixel((4, 4))[3]
                tr_a = obs_im.getpixel((obs_im.width - 5, 4))[3]
                bar_disp = page.evaluate("() => getComputedStyle(document.getElementById('bar')).display")
                report["obs_alpha"] = {"top_left": tl_a, "top_right": tr_a, "bar_display": bar_disp}
                if tl_a != 0 or tr_a != 0 or bar_disp != "none":
                    report["ok"] = False
                    report["errors"].append(f"OBS transparency check failed: {report['obs_alpha']}")

                # Extract head center & face names for framing
                meta = page.evaluate("""() => {
                    window.__rigApi.init();
                    const P = window.__rigApi && window.P ? window.P : null;
                    return {
                        title: document.title,
                        faces: Array.from(document.querySelectorAll('#faces button')).map(b => b.dataset.face).filter(Boolean),
                    };
                }""")
                # Read pack.json from archive for exact head framing
                import zipfile
                with zipfile.ZipFile(l2d_path) as zf:
                    pack = json.loads(zf.read("pack.json").decode("utf-8"))
                hc = pack["head"]["centre"]
                hr = pack["head"]["radius"]
                head_rect = [hc[0] - hr * 1.65, hc[1] - hr * 1.45, hc[0] + hr * 1.65, hc[1] + hr * 1.65]
                face_rect = [hc[0] - hr * 1.15, hc[1] - hr * 0.95, hc[0] + hr * 1.15, hc[1] + hr * 1.15]

                def snap_state(st: dict, rect: list[float], w: int = 360, h: int = 360) -> Image.Image:
                    base = {
                        "a": 0.0, "pitch": 0.0, "roll": 0.0, "face": None, "hold": None, "vowel": None,
                        "eyes": 1.0, "wear": {}, "bodySway": 0.0, "bodyLean": 0.0, "lift": 0.0,
                        "raiseR": 0.0, "elbowR": 0.0, "raiseL": 0.0, "elbowL": 0.0, "tailWag": 0.0, "bounce": False,
                    }
                    base.update(st)
                    page.set_viewport_size({"width": w, "height": h})
                    page.evaluate("""([st, rect, w, h]) => {
                        window.__rigApi.init();
                        for (let i = 0; i < 3; i++) window.__rigApi.render(st, rect, w, h, 1/30, i === 0);
                    }""", [base, rect, w, h])
                    raw = page.screenshot(type="png", timeout=90000)
                    return Image.open(io.BytesIO(raw)).convert("RGB")

                # 2. Wide yaw sweep (-1.0 .. +1.0)
                wide_tiles = [(f"a={a:+.2f}", snap_state({"a": a}, head_rect)) for a in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)]
                _compose_grid(wide_tiles, 5, out_dir / "qa_turns_wide.png")

                # 3. Fine center yaw sweep (-0.40 .. +0.40) to detect under-bangs / behind-mask popping!
                ctr_tiles = [(f"a={a:+.2f}", snap_state({"a": a}, face_rect)) for a in (-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40)]
                _compose_grid(ctr_tiles, 5, out_dir / "qa_turns_center.png")

                # 4. 3x3 Yaw x Pitch grid
                pitch_tiles = [
                    (f"X={a:+.1f} Y={p:+.1f}", snap_state({"a": a, "pitch": p}, head_rect))
                    for p in (0.8, 0.0, -0.8) for a in (-0.8, 0.0, 0.8)
                ]
                _compose_grid(pitch_tiles, 3, out_dir / "qa_pitch_grid.png")

                # 5. Expressions sheet (up to 16)
                faces = list((pack.get("expr") or {}).get("faces", {}).keys())[:16]
                if faces:
                    expr_tiles = [(fn, snap_state({"face": fn}, face_rect)) for fn in faces]
                    _compose_grid(expr_tiles, 4, out_dir / "qa_expressions.png")

                br.close()
        finally:
            srv.shutdown()

        real_errs = [e for e in console_errors if "favicon" not in e]
        report["console_errors"] = real_errs
        if real_errs:
            report["ok"] = False
            report["errors"].extend(real_errs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    (out_dir / "qa_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline Layer2D .l2d Visual & Runtime QA Runner")
    ap.add_argument("l2d", type=Path, help="Path to .l2d archive")
    ap.add_argument("--out", type=Path, default=Path("qa_out"), help="Output directory for PNG sheets & report")
    args = ap.parse_args()
    res = run_qa(args.l2d, args.out)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
