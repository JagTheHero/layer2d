"""Layer2D Open Avatar Container Format (.l2d / layer2d/v1).

A portable, royalty-free, ZIP-based archive format for 2D rigged avatars that
rivals proprietary ecosystems without licensing lock-in.

Archive layout:
    layer2d.json        # Open manifest (format: "layer2d/v1", metadata, counts, entrypoints)
    pack.json           # Runtime mesh/keyform/physics/expression pack
    tex/<id>.<ext>      # Layer textures, eye masks, expression pieces, shared flipbook FX
    preview.png         # Optional portrait thumbnail (recommended 1200x1600 RGBA)
    SHA256SUMS.json     # Deterministic SHA-256 checksums for every non-checksum member

CLI usage:
    python -m rigkit.format pack models/test-signal/viewer models/test-signal/test-signal.l2d
    python -m rigkit.format inspect models/test-signal/test-signal.l2d
    python -m rigkit.format validate models/test-signal/test-signal.l2d
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT_VERSION = "layer2d/v1"
MANIFEST_NAME = "layer2d.json"
CHECKSUMS_NAME = "SHA256SUMS.json"
FORBIDDEN_ARCHIVE_PATTERNS = (
    "COST_LEDGER.json",
    "showcase_stage.html",
    "_stage_bg",
    ".DS_Store",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_arcname(name: str) -> bool:
    if not name or name.startswith("/") or name.startswith("\\"):
        return False
    parts = Path(name).parts
    if ".." in parts:
        return False
    return True


def expected_texture_files(pack: dict[str, Any]) -> set[str]:
    """Return every relative `tex/...` path required by a Layer2D pack.json."""
    ext = pack.get("texExt", "png")
    req: set[str] = set()
    for L in pack.get("layers", []):
        lid = L["id"]
        req.add(f"tex/{lid}.{ext}")
        if L.get("sheer"):
            req.add(f"tex/{lid}_sheer.png")
        if L.get("hole"):
            req.add(f"tex/{lid}_hole.png")
        if L.get("up"):
            req.add(f"tex/{lid}_up.png")
        if L.get("palm"):
            req.add(f"tex/{lid}_palm.png")
        if L.get("posed"):
            req.add(f"tex/{lid}_posed.png")
        if L.get("wipe"):
            req.add(f"tex/{lid}_wipe.png")
        if L.get("eye") and L["eye"].get("mode") == 1:
            req.add(f"tex/{lid}_open.png")
    for st in (pack.get("hairStyles") or {}).values():
        for L in st.get("layers", []):
            req.add(f"tex/{L['id']}.{ext}")
    expr = pack.get("expr") or {}
    for pid, pc in (expr.get("pieces") or {}).items():
        if pc.get("tex"):
            req.add(f"tex/{pc['tex']}.png")
        else:
            req.add(f"tex/expr_{pid}.png")
    arm3d_tex = ((pack.get("arm3d") or {}).get("tex") or {}).get("file")
    if arm3d_tex:
        req.add(str(arm3d_tex))
    return req


def build_manifest(pack: dict[str, Any], meta: dict[str, Any] | None = None, has_preview: bool = False) -> dict[str, Any]:
    meta = meta or {}
    expr = pack.get("expr") or {}
    samples = pack.get("samples") or []
    sample_angles = [round(float(s.get("angle", 0.0)), 5) for s in samples]
    manifest: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "id": str(meta.get("id") or meta.get("slug") or pack.get("title", "avatar")).lower().replace(" ", "-"),
        "name": str(meta.get("name") or pack.get("title") or "Layer2D Avatar"),
        "title": str(pack.get("title") or meta.get("name") or "Layer2D Avatar"),
        "version": str(meta.get("version") or "1.0.0"),
        "author": str(meta.get("author") or "Layer2D"),
        "license": str(meta.get("license") or "See accompanying license"),
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entry": "pack.json",
        "texExt": str(pack.get("texExt", "webp")),
        "canvas": list(pack.get("canvas", [2400, 4800])),
        "stats": {
            "layers": len(pack.get("layers", [])),
            "nodes": len(pack.get("nodes", [])),
            "chains": len(pack.get("chains", [])),
            "samples": len(samples),
            "pitchSamples": len(pack.get("pitchSamples", [])),
            "blendShapes": len(pack.get("blendShapes", [])),
            "expressions": len(expr.get("faces", {})),
            "holds": len(expr.get("holds", {})),
            "outfits": len(pack.get("outfits", {})),
            "exprPieces": len(expr.get("pieces", {})),
            "vowels": sorted(list((expr.get("vowels") or {}).keys())),
        },
        "capabilities": {
            "headYawSamples": len(sample_angles),
            "headPitch2D": True,
            "headPitchKeyforms": len(pack.get("pitchSamples", [])) > 0,
            "blendShapeDeform": len(pack.get("blendShapes", [])) > 0,
            "sceneGraph": len(pack.get("nodes", [])) > 0 or any(bool(L.get("parent")) for L in pack.get("layers", [])),
            "springChains": len(pack.get("chains", [])) > 0,
            "tailRig": bool(pack.get("tail")),
            "obsAlpha": True,
            "micLipSync": bool(expr.get("vowels")),
            "webcamTracking": True,
        },
    }
    if has_preview:
        manifest["preview"] = "preview.png"
    if meta.get("description"):
        manifest["description"] = str(meta["description"])
    return manifest


def pack_l2d(
    viewer_dir: Path,
    out_path: Path,
    *,
    meta: dict[str, Any] | None = None,
    preview_path: Path | None = None,
    compress: bool = True,
) -> dict[str, Any]:
    """Package a viewer directory (`pack.json` + `tex/`) into a self-contained `.l2d` archive."""
    viewer_dir = Path(viewer_dir)
    out_path = Path(out_path)
    pack_path = viewer_dir / "pack.json"
    if not pack_path.exists():
        raise FileNotFoundError(f"Missing pack.json in {viewer_dir}")

    pack_raw = pack_path.read_bytes()
    pack = json.loads(pack_raw.decode("utf-8"))
    req_tex = sorted(expected_texture_files(pack))
    missing = [rel for rel in req_tex if not (viewer_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} texture(s) referenced by pack.json: {missing[:6]}")

    has_preview = bool(preview_path and Path(preview_path).exists())
    manifest = build_manifest(pack, meta=meta, has_preview=has_preview)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    # Gather all archive payloads deterministically
    payloads: list[tuple[str, bytes, int]] = []
    comp_json = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    payloads.append((MANIFEST_NAME, manifest_bytes, comp_json))
    payloads.append(("pack.json", pack_raw, comp_json))
    if has_preview and preview_path is not None:
        payloads.append(("preview.png", Path(preview_path).read_bytes(), zipfile.ZIP_STORED))

    # Include all required textures (stored uncompressed since WebP/PNG are already compressed, fast load in JS!)
    for rel in req_tex:
        payloads.append((rel, (viewer_dir / rel).read_bytes(), zipfile.ZIP_STORED))

    checksums = {name: _sha256_bytes(data) for name, data, _ in payloads}
    checksum_bytes = (json.dumps(checksums, indent=2, sort_keys=True) + "\n").encode("utf-8")
    payloads.append((CHECKSUMS_NAME, checksum_bytes, comp_json))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w") as zf:
        for name, data, ctype in payloads:
            zi = zipfile.ZipInfo(filename=name, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = ctype
            zf.writestr(zi, data)

    size_bytes = out_path.stat().st_size
    return {
        "path": str(out_path),
        "format": FORMAT_VERSION,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1e6, 2),
        "files": len(payloads),
        "stats": manifest["stats"],
    }


def inspect_l2d(l2d_path: Path) -> dict[str, Any]:
    l2d_path = Path(l2d_path)
    with zipfile.ZipFile(l2d_path, "r") as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        return {
            "path": str(l2d_path),
            "size_bytes": l2d_path.stat().st_size,
            "file_count": len(names),
            "manifest": manifest,
        }


def validate_l2d(l2d_path: Path) -> dict[str, Any]:
    """Validate structure, safety, manifest, required textures, and SHA-256 digests of a `.l2d` bundle."""
    l2d_path = Path(l2d_path)
    errors: list[str] = []
    if not l2d_path.exists():
        return {"ok": False, "errors": [f"Archive not found: {l2d_path}"]}

    with zipfile.ZipFile(l2d_path, "r") as zf:
        names = zf.namelist()
        name_set = set(names)
        if len(names) != len(name_set):
            errors.append("Duplicate entries inside .l2d archive")
        for n in names:
            if not _safe_arcname(n):
                errors.append(f"Unsafe path in archive: {n}")
            for bad in FORBIDDEN_ARCHIVE_PATTERNS:
                if bad in n:
                    errors.append(f"Forbidden internal artifact in archive: {n}")

        for req in (MANIFEST_NAME, "pack.json", CHECKSUMS_NAME):
            if req not in name_set:
                errors.append(f"Missing required entry: {req}")
        if errors:
            return {"ok": False, "errors": errors}

        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        if manifest.get("format") != FORMAT_VERSION:
            errors.append(f"Unsupported format: {manifest.get('format')!r} (expected {FORMAT_VERSION!r})")

        pack = json.loads(zf.read("pack.json").decode("utf-8"))
        req_tex = expected_texture_files(pack)
        missing_tex = sorted(req_tex - name_set)
        if missing_tex:
            errors.append(f"Missing {len(missing_tex)} texture(s): {missing_tex[:5]}")

        # Validate optional Scene Graph (`pack.nodes[]` and `parent` pointers: no missing refs or cycles)
        scene_parents: dict[str, str | None] = {}
        for nd in pack.get("nodes") or []:
            nid = nd.get("id")
            if not nid:
                errors.append("Scene graph node missing required id")
            elif nid in scene_parents:
                errors.append(f"Duplicate scene graph node id: {nid}")
            else:
                scene_parents[nid] = nd.get("parent") or None
        for L in pack.get("layers") or []:
            lid = L.get("id")
            if lid in scene_parents:
                errors.append(f"Scene graph id collision between node and layer: {lid}")
            elif lid:
                scene_parents[lid] = L.get("parent") or None
        for sid, pid in scene_parents.items():
            if pid and pid not in scene_parents:
                errors.append(f"Scene graph item {sid!r} references missing parent {pid!r}")
                continue
            seen = {sid}
            cur = pid
            while cur:
                if cur in seen:
                    errors.append(f"Scene graph cycle detected involving {sid!r}")
                    break
                seen.add(cur)
                cur = scene_parents.get(cur)

        # Verify SHA-256 checksums for every non-checksum file
        checksums = json.loads(zf.read(CHECKSUMS_NAME).decode("utf-8"))
        expected_members = {n for n in names if n != CHECKSUMS_NAME}
        if set(checksums.keys()) != expected_members:
            extra = sorted(expected_members - set(checksums.keys()))
            missing_cs = sorted(set(checksums.keys()) - expected_members)
            errors.append(f"Checksum key mismatch: unhashed={extra[:4]}, missing={missing_cs[:4]}")
        else:
            for member, expected_hash in checksums.items():
                actual_hash = _sha256_bytes(zf.read(member))
                if actual_hash != expected_hash:
                    errors.append(f"SHA-256 mismatch for {member}")

        # Verify manifest stats match pack.json truth
        stats = manifest.get("stats") or {}
        if stats.get("layers") != len(pack.get("layers", [])):
            errors.append(f"Manifest layers={stats.get('layers')} != pack layers={len(pack.get('layers', []))}")
        if stats.get("expressions") != len((pack.get("expr") or {}).get("faces", {})):
            errors.append("Manifest expressions count != pack.json")

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "format": manifest.get("format"),
            "name": manifest.get("name"),
            "files": len(names),
            "stats": stats,
            "size_mb": round(l2d_path.stat().st_size / 1e6, 2),
        }


def unpack_l2d(l2d_path: Path, out_dir: Path, *, verify: bool = True) -> dict[str, Any]:
    """Unpack a `.l2d` archive into `out_dir`, validating checksums and path safety first."""
    l2d_path = Path(l2d_path)
    out_dir = Path(out_dir)
    if verify:
        res = validate_l2d(l2d_path)
        if not res["ok"]:
            raise ValueError(f"Invalid .l2d archive: {res['errors']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(l2d_path, "r") as zf:
        for info in zf.infolist():
            if not _safe_arcname(info.filename):
                raise ValueError(f"Unsafe path in .l2d: {info.filename}")
            dest = out_dir / info.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(info.filename))
    return inspect_l2d(l2d_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer2D Open Avatar Format (.l2d) CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack", help="Pack viewer dir into a .l2d archive")
    p_pack.add_argument("viewer_dir", type=Path)
    p_pack.add_argument("out_path", type=Path)
    p_pack.add_argument("--meta", type=Path, default=None, help="Optional JSON metadata (avatar_spec.json / product.json)")
    p_pack.add_argument("--preview", type=Path, default=None, help="Optional preview.png portrait")

    p_ins = sub.add_parser("inspect", help="Print .l2d manifest summary")
    p_ins.add_argument("l2d_path", type=Path)

    p_val = sub.add_parser("validate", help="Validate .l2d archive integrity & checksums")
    p_val.add_argument("l2d_path", type=Path)

    p_unp = sub.add_parser("unpack", help="Unpack a .l2d archive safely")
    p_unp.add_argument("l2d_path", type=Path)
    p_unp.add_argument("out_dir", type=Path)

    args = ap.parse_args()
    if args.cmd == "pack":
        meta = json.loads(args.meta.read_text()) if (args.meta and args.meta.exists()) else None
        res = pack_l2d(args.viewer_dir, args.out_path, meta=meta, preview_path=args.preview)
        print(json.dumps(res, indent=2))
    elif args.cmd == "inspect":
        print(json.dumps(inspect_l2d(args.l2d_path), indent=2))
    elif args.cmd == "validate":
        res = validate_l2d(args.l2d_path)
        print(json.dumps(res, indent=2))
        if not res["ok"]:
            raise SystemExit(1)
    elif args.cmd == "unpack":
        res = unpack_l2d(args.l2d_path, args.out_dir)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
