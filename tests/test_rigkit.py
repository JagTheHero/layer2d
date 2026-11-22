"""Automated test suite for the Layer2D unified avatar engine (`rigkit`)."""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rigkit.expr_library import AvatarFaceAnchors, build_default_timelines, build_standard_faces
from rigkit.format import inspect_l2d, pack_l2d, unpack_l2d, validate_l2d
from rigkit.hair3d import DepthShellSpec, HeadSkull3D, detect_chroma_gaps, make_shell_fwd, smoothstep
from rigkit.model_scaffold import scaffold_avatar
from rigkit.qa_harness import check_pack_invariants
from rigkit.schema import load_avatar_spec, validate_avatar_spec

ROOT = Path(__file__).resolve().parent.parent
TEST_SIGNAL_L2D = ROOT / "models/test-signal/test-signal.l2d"


def test_hair3d_smoothstep_and_skull_wrap() -> None:
    arr = np.array([-10.0, 0.0, 50.0, 100.0, 150.0])
    out = smoothstep(0.0, 100.0, arr)
    assert np.allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])

    skull = HeadSkull3D(cx=1200.0, cy=620.0, rx=435.0, ry=560.0)
    pts = np.array([[1200.0, 620.0], [880.0, 620.0]])
    yaw = math.radians(32.5)
    proj_front = skull.project(pts, yaw, rz=204.0, rz_top=160.0)
    proj_temple = skull.project(pts, yaw, rz=68.0, rz_top=56.0)
    # Deeper front plane shifts more horizontally than shallow temple shell
    assert (proj_front[0, 0] - 1200.0) > (proj_temple[0, 0] - 1200.0)

    spec = DepthShellSpec(layer_id="H02cR", side="R", rz=68.0, rz_top=56.0)
    fwd = make_shell_fwd(skull, spec)
    warped = fwd(pts, yaw)
    assert warped.shape == (2, 2)
    assert np.all(np.isfinite(warped))


def test_detect_chroma_gaps() -> None:
    # Solid gray image with one enclosed chroma-green 12x12 hole inside ROI
    img = np.full((120, 120, 3), 180, dtype=np.uint8)
    img[0:5, :, :] = (0, 255, 0)  # exterior green border (must NOT count as interior hole)
    img[50:62, 50:62, :] = (0, 255, 0)  # interior 144px hole
    res = detect_chroma_gaps(img)
    assert res["hole_count"] == 1
    assert res["total_hole_px"] == 144
    assert res["pass"] is False


def test_expr_library_builds_37_expressions() -> None:
    anchors = AvatarFaceAnchors(
        eye_r=(1085.0, 682.0),
        eye_l=(1315.0, 682.0),
        mouth=(1200.0, 825.0),
        face_center=(1200.0, 680.0),
        forehead_y=535.0,
        cheek_y=755.0,
    )
    available = {
        "M02", "M03", "M04", "M05", "M06", "M09", "M10", "M11", "M12",
        "E09R", "E09L", "E10R", "E10L",
        "X01R", "X01L", "X02", "X03", "X04", "X05R", "X05L", "X06R", "X06L",
        "X07R", "X07L", "X08R", "X08L", "X09R", "X09L", "X10R", "X10L",
        "FX_sp1", "FX_sp2", "FX_sp3", "FX_question", "FX_heart_L", "FX_heart_R",
        "FX_tear_R", "FX_tear_L", "FX_sweat_R", "FX_sweat_L", "FX_scribble",
        "FX_puff", "FX_drool", "FX_shade_blue", "FX_shade_pale", "FX_shade_purple",
        "FX_shadow", "FX_blush", "FX_glint_R", "FX_glint_L",
    }
    faces = build_standard_faces(anchors, available)
    assert len(faces) == 37
    face_tl, hold_tl = build_default_timelines(faces, {"Wave": {}, "Peace": {}})
    assert len(face_tl) >= 4
    assert len(hold_tl) >= 2


def _sample_spec() -> dict:
    return {
        "spec_version": "layer2d-spec/v1",
        "slug": "sample_avatar",
        "name": "Sample Avatar",
        "title": "Sample Avatar — Layer2D Avatar",
        "version": "1.0.0",
        "author": "Layer2D",
        "canvas": [1200, 2000],
        "head": {
            "centre": [600.0, 440.0],
            "radius": 240.0,
            "chin": 590.0,
            "neck": 640.0,
            "neck_pivot": [600.0, 610.0],
            "turn": {"L": 0.40, "R": -0.40},
        },
        "commercial": {
            "price_usd": 0,
            "license": "MIT",
            "ai_assisted_disclosure": "Created with Layer2D open rigging toolkit.",
        },
    }


def test_schema_validation_and_commercial_guard(tmp_path: Path) -> None:
    spec_path = tmp_path / "avatar_spec.json"
    spec_path.write_text(json.dumps(_sample_spec(), indent=2))
    spec = load_avatar_spec(spec_path)
    assert spec["slug"] == "sample_avatar"

    bad = json.loads(json.dumps(spec))
    bad["commercial"]["ai_assisted_disclosure"] = "100% hand-drawn art!"
    errs = validate_avatar_spec(bad)
    assert any("hand-drawn" in e for e in errs)


def test_scaffold_second_avatar_and_l2d_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "kitsune_test"
    res = scaffold_avatar("kitsune_test", name="Kitsune Test", out_dir=out_dir)
    l2d_path = Path(res["l2d"]["path"])
    assert l2d_path.exists()
    assert res["validation"]["ok"] is True

    # Inspect & unpack round-trip
    info = inspect_l2d(l2d_path)
    assert info["manifest"]["format"] == "layer2d/v1"
    assert info["manifest"]["id"] == "kitsune_test"

    unpacked = tmp_path / "unpacked"
    unpack_l2d(l2d_path, unpacked)
    assert (unpacked / "pack.json").exists()
    assert (unpacked / "layer2d.json").exists()
    assert (unpacked / "preview.png").exists()

    # Verify pack invariants on scaffolded second avatar
    inv = check_pack_invariants(out_dir / "viewer")
    assert inv["ok"], f"Scaffolded pack invariant errors: {inv['errors']}"

    # Tamper test: corrupting a byte in pack.json must fail SHA-256 validation
    tampered = tmp_path / "tampered.l2d"
    with zipfile.ZipFile(l2d_path, "r") as zin, zipfile.ZipFile(tampered, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "pack.json":
                data = data + b" "
            zout.writestr(item, data)
    val_bad = validate_l2d(tampered)
    assert not val_bad["ok"]
    assert any("SHA-256 mismatch" in e for e in val_bad["errors"])


def test_scaffold_from_authored_spec(tmp_path: Path) -> None:
    spec_path = tmp_path / "custom_spec.json"
    spec_data = _sample_spec()
    spec_data["slug"] = "authored_fox"
    spec_data["name"] = "Authored Fox"
    spec_data["title"] = "Authored Fox — Layer2D Avatar"
    spec_path.write_text(json.dumps(spec_data, indent=2))

    out_dir = tmp_path / "authored_fox"
    res = scaffold_avatar("ignored", name="Ignored", out_dir=out_dir, spec_path=spec_path)
    assert res["slug"] == "authored_fox"
    assert res["validation"]["ok"] is True
    pack = json.loads((out_dir / "viewer/pack.json").read_text())
    assert "pitchGroups" in pack and pack["pitchGroups"]["H03C"] == "bangs"


def test_test_signal_demo_l2d_valid(tmp_path: Path) -> None:
    assert TEST_SIGNAL_L2D.exists(), f"Missing MIT demo avatar: {TEST_SIGNAL_L2D}"
    val = validate_l2d(TEST_SIGNAL_L2D)
    assert val["ok"], f"Test Signal .l2d validation errors: {val['errors']}"
    assert val["stats"]["layers"] >= 90
    assert val["stats"]["expressions"] >= 40
    assert val["stats"]["samples"] == 17
    assert val["stats"]["vowels"] == ["A", "E", "I", "O", "U"]
    unpacked = tmp_path / "test_signal_unpacked"
    unpack_l2d(TEST_SIGNAL_L2D, unpacked)
    pack = json.loads((unpacked / "pack.json").read_text())
    assert "pitchGroups" in pack and len(pack["pitchGroups"]) >= 20
    assert "views" in pack and "bust" in pack["views"]
