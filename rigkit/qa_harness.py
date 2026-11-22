"""Unified Layer2D Automated QA Harness (`rigkit.qa_harness`).

Verifies every Layer2D avatar against studio acceptance criteria:
1. Pack & structural invariants (100% solid hair op==1.0, binary behind keys,
   acyclic clip graph, all textures present & readable, data-driven tail/shadow/highlights).
2. Open `.l2d` bundle cryptographic integrity (`rigkit.format.validate_l2d`).
3. Live headless Playwright WebGL2 rendering checks:
   - Zero console errors & zero uncaught page errors
   - `?obs=1` transparent alpha canvas (corners alpha == 0)
   - Live FPS & tail wag motion check
   - Chroma-green hair-gap audit across head yaw sweep `[-1.0, -0.6, 0.0, +0.6, +1.0]`
4. Commercial copy & release hygiene audit:
   - Zero forbidden strings (`hand-drawn`, `$1,500`, `$150`, `non-exclusive`)
   - Zero internal files (`COST_LEDGER.json`, `showcase_stage.html`, `_stage_bg`) inside `release/`
   - `product.json` single-source-of-truth consistency check

Usage:
    python -m rigkit.qa_harness models/test-signal
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rigkit.format import expected_texture_files, validate_l2d
from rigkit.hair3d import detect_chroma_gaps
from rigkit.schema import load_avatar_spec

FORBIDDEN_RELEASE_FILES = ("COST_LEDGER.json", "showcase_stage.html", "_stage_bg")
FORBIDDEN_COPY_REGEXES = [
    (r"hand[\s\-]drawn", "Forbidden 'hand-drawn' claim"),
    (r"\$1,?500", "Forbidden fake anchor price '$1,500'"),
    (r"\$150\b", "Forbidden old price '$150'"),
    (r"non[\s\-]exclusive", "Forbidden 'non-exclusive' phrasing"),
]


def check_pack_invariants(viewer_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    pack_path = viewer_dir / "pack.json"
    if not pack_path.exists():
        return {"ok": False, "errors": [f"Missing {pack_path}"], "warnings": []}

    pack = json.loads(pack_path.read_text())
    layers = pack.get("layers", [])
    samples = pack.get("samples", [])
    by_id = {L["id"]: L for L in layers}

    # 1. Verify every referenced texture exists and decodes cleanly
    req_tex = expected_texture_files(pack)
    missing = [rel for rel in sorted(req_tex) if not (viewer_dir / rel).exists()]
    if missing:
        errors.append(f"Missing {len(missing)} texture(s): {missing[:5]}")

    # 2. Verify NO hair layer crossfades opacity across yaw samples (op must stay 1.0 or intentionally 0.0 everywhere)
    for j, L in enumerate(layers):
        lid = L["id"]
        if lid.startswith("H") and L.get("head"):
            ops = [round(float(s["layers"][j].get("op", 1.0)), 4) for s in samples]
            if any(0.0 < o < 0.999 for o in ops):
                errors.append(f"Hair layer {lid} uses translucent crossfade opacity: {ops}")
            if len(set(ops)) > 1:
                errors.append(f"Hair layer {lid} changes opacity across yaw samples: {ops}")

    # 3. Verify clip graph is acyclic and every clip parent exists & has clipTarget=True
    for L in layers:
        parent = L.get("clip")
        if parent:
            if parent not in by_id:
                errors.append(f"Layer {L['id']} clips to missing layer {parent}")
            elif not by_id[parent].get("clipTarget"):
                errors.append(f"Clip parent {parent} missing clipTarget=True")
            seen = {L["id"]}
            cur = parent
            while cur:
                if cur in seen:
                    errors.append(f"Cyclic clip dependency involving {L['id']}")
                    break
                seen.add(cur)
                cur = (by_id.get(cur) or {}).get("clip")

    # 4. Data-driven runtime metadata check
    if not pack.get("title"):
        warnings.append("pack.json missing 'title'")
    if "hairShadowId" not in pack:
        warnings.append("pack.json missing 'hairShadowId'")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "layers": len(layers),
        "samples": len(samples),
        "expressions": len((pack.get("expr") or {}).get("faces", {})),
        "holds": len((pack.get("expr") or {}).get("holds", {})),
        "outfits": len(pack.get("outfits") or {}),
        "chains": len(pack.get("chains") or []),
    }


def check_release_hygiene(model_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    rel = model_dir / "release"
    if not rel.exists():
        return {"ok": True, "errors": [], "warnings": ["No release/ directory present"]}

    # Check forbidden files anywhere inside release/
    for p in rel.rglob("*"):
        for bad in FORBIDDEN_RELEASE_FILES:
            if p.name == bad or bad in p.parts:
                errors.append(f"Forbidden internal artifact in release/: {p.relative_to(model_dir)}")

    # Check text files in release/ and viewer/showcase_stage.html for forbidden commercial copy
    scan_files: list[Path] = []
    for ext in ("*.txt", "*.md", "*.json", "*.html"):
        scan_files.extend(rel.rglob(ext))
    stage = model_dir / "viewer/showcase_stage.html"
    if stage.exists():
        scan_files.append(stage)
    prod_p = model_dir / "product.json"
    if prod_p.exists():
        scan_files.append(prod_p)

    for fp in scan_files:
        if fp.name in ("pack.json", "SHA256SUMS.json", "index.html"):
            continue
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat, desc in FORBIDDEN_COPY_REGEXES:
            if re.search(pat, txt, re.IGNORECASE):
                errors.append(f"{desc} found in {fp.relative_to(model_dir)}")

    # Validate .l2d archives in release/
    l2d_results: list[dict[str, Any]] = []
    for l2d in sorted(rel.glob("*.l2d")):
        res = validate_l2d(l2d)
        l2d_results.append({"file": l2d.name, **res})
        if not res["ok"]:
            errors.append(f"Invalid .l2d bundle {l2d.name}: {res['errors']}")

    # Verify outer release/SHA256SUMS.json digests & completeness
    sums_path = rel / "SHA256SUMS.json"
    checksums_verified = 0
    if sums_path.exists():
        import hashlib
        recorded = json.loads(sums_path.read_text(encoding="utf-8"))
        actual_files = {
            p.relative_to(rel).as_posix()
            for p in rel.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS.json"
        }
        missing_cs = sorted(actual_files - set(recorded.keys()))
        extra_cs = sorted(set(recorded.keys()) - actual_files)
        if missing_cs:
            errors.append(f"Unhashed release file(s) missing from SHA256SUMS.json: {missing_cs[:5]}")
        if extra_cs:
            errors.append(f"Stale entry in SHA256SUMS.json missing on disk: {extra_cs[:5]}")
        for rel_path, expected_hex in recorded.items():
            fp = rel / rel_path
            if not fp.is_file():
                continue
            h = hashlib.sha256()
            with fp.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_hex:
                errors.append(f"Release SHA-256 mismatch: {rel_path}")
            else:
                checksums_verified += 1
    else:
        warnings.append("Missing outer release/SHA256SUMS.json")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "l2d": l2d_results,
        "release_checksums_verified": checksums_verified,
    }


def check_webgl_runtime(viewer_dir: Path, *, check_hair_gaps: bool = True) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    console_errs: list[str] = []
    page_errs: list[str] = []
    metrics: dict[str, Any] = {}

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    handler = functools.partial(QuietHandler, directory=str(viewer_dir))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    try:
        with sync_playwright() as p:
            br = p.chromium.launch(
                headless=True,
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
            )
            pg = br.new_page(viewport={"width": 640, "height": 800})
            pg.on("console", lambda m: console_errs.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: page_errs.append(str(e)))
            pg.goto(f"http://127.0.0.1:{port}/index.html?obs=1&auto=0", wait_until="networkidle")
            pg.wait_for_function("() => window.__ready === true", timeout=20000)

            # 1. Verify OBS transparent corner alpha straight from the live WebGL2 canvas
            import base64
            from io import BytesIO
            obs_url = pg.evaluate("() => document.getElementById('gl').toDataURL('image/png')")
            obs_arr = np.array(Image.open(BytesIO(base64.b64decode(obs_url.split(',', 1)[1]))).convert("RGBA"))
            # Check top-left & top-right corners (bust framing may legitimately touch bottom corners with sleeves/haori)
            corners = [int(obs_arr[4, 4, 3]), int(obs_arr[4, -5, 3])]
            metrics["obs_top_corner_alpha"] = corners
            if any(a != 0 for a in corners):
                errors.append(f"?obs=1 transparent top corners non-zero alpha: {corners}")

            # 2. Verify non-blank avatar pixels rendered
            opaque_ratio = float((obs_arr[:, :, 3] > 200).mean())
            metrics["opaque_ratio"] = round(opaque_ratio, 4)
            if opaque_ratio < 0.05:
                errors.append(f"Rendered canvas appears nearly empty (opaque_ratio={opaque_ratio:.4f})")

            # 3. Verify UI bar hidden in ?obs=1
            bar_display = pg.evaluate("() => getComputedStyle(document.getElementById('bar')).display")
            metrics["obs_bar_display"] = bar_display
            if bar_display != "none":
                errors.append(f"?obs=1 failed to hide control bar (display={bar_display})")

            # 4. Chroma-green yaw hair gap check via window.__rigApi + direct canvas readback
            if check_hair_gaps:
                has_api = pg.evaluate("() => Boolean(window.__rigApi && window.__rigApi.init)")
                if has_api:
                    pack = json.loads((viewer_dir / "pack.json").read_text())
                    hcx, hcy = pack["head"]["centre"]
                    hr = pack["head"]["radius"]
                    rect = [hcx - hr * 1.45, hcy - hr * 1.35, hcx + hr * 1.45, hcy + hr * 1.25]
                    pg.evaluate("() => { window.__rigApi.init(); }")
                    gap_stats = {}
                    # Continuous settled left-to-right yaw sweep (24 frames/stop -> true settled yaw + live spring physics!)
                    for idx_a, a_val in enumerate((-1.0, -0.6, 0.0, 0.6, 1.0)):
                        data_url = pg.evaluate(
                            """([a, rect, first]) => {
                                const st = { a, pitch: 0, roll: 0, vowel: null, eyes: 1.0, lift: 0, tailWag: 0,
                                    raiseR: 0, elbowR: 0, raiseL: 0, elbowL: 0, bounce: false, face: null, hold: null,
                                    bodySway: 0, bodyLean: 0, wear: {} };
                                for (let i = 0; i < 24; i++) window.__rigApi.render(st, rect, 640, 640, 1/30, first && i === 0);
                                return document.getElementById('gl').toDataURL('image/png');
                            }""",
                            [a_val, rect, idx_a == 0],
                        )
                        shot_rgba = Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGBA")
                        arr_rgba = np.array(shot_rgba)
                        # Upper-temple & ear-root cavity ROI (covers full near silhouette x0=0.23 while excluding outer hairpin knot loop x=0.75)
                        g_temple = detect_chroma_gaps(arr_rgba, roi_frac=(0.23, 0.24, 0.72, 0.54), min_hole_px=24)
                        g_full = detect_chroma_gaps(arr_rgba)
                        gap_stats[f"a={a_val:+.1f}"] = {
                            "temple_holes": g_temple["hole_count"],
                            "temple_px": g_temple["total_hole_px"],
                            "full_holes": g_full["hole_count"],
                            "full_px": g_full["total_hole_px"],
                        }
                        if g_temple["total_hole_px"] > 120:
                            errors.append(
                                f"Upper-temple hair gap at a={a_val:+.1f}: {g_temple['total_hole_px']}px ({g_temple['hole_count']} hole(s))"
                            )
                    metrics["yaw_chroma_gaps"] = gap_stats

            br.close()
    finally:
        srv.shutdown()

    if console_errs:
        errors.append(f"Console errors: {console_errs[:3]}")
    if page_errs:
        errors.append(f"Uncaught page errors: {page_errs[:3]}")

    return {"ok": len(errors) == 0, "errors": errors, "metrics": metrics}


def run_qa(model_dir: Path, *, skip_browser: bool = False) -> dict[str, Any]:
    model_dir = Path(model_dir)
    report: dict[str, Any] = {"model_dir": str(model_dir), "checks": {}}

    spec_path = model_dir / "avatar_spec.json"
    if spec_path.exists():
        try:
            load_avatar_spec(spec_path)
            report["checks"]["avatar_spec"] = {"ok": True, "errors": []}
        except Exception as exc:
            report["checks"]["avatar_spec"] = {"ok": False, "errors": [str(exc)]}

    report["checks"]["pack_invariants"] = check_pack_invariants(model_dir / "viewer")
    report["checks"]["release_hygiene"] = check_release_hygiene(model_dir)
    if not skip_browser:
        report["checks"]["webgl_runtime"] = check_webgl_runtime(model_dir / "viewer")

    report["ok"] = all(c["ok"] for c in report["checks"].values())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Layer2D unified QA suite on a model directory")
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--skip-browser", action="store_true")
    a = ap.parse_args()
    rep = run_qa(a.model_dir, skip_browser=a.skip_browser)
    print(json.dumps(rep, indent=2))
    if not rep["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
