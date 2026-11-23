"""Automated test suite for the Layer2D unified avatar engine (`rigkit`)."""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rigkit.expr_library import AvatarFaceAnchors, build_default_timelines, build_standard_faces
from rigkit.face3d import (
    fit_side_jaw_3d,
    fit_turn_face_shade_3d,
    fit_turn_nose_3d,
    synthesize_side_jaw_layers,
    synthesize_turn_face_depth_layers,
)
from rigkit.format import inspect_l2d, pack_l2d, unpack_l2d, validate_l2d
from rigkit.hair3d import DepthShellSpec, HeadSkull3D, detect_chroma_gaps, make_shell_fwd, smoothstep
from rigkit.model_scaffold import scaffold_avatar
from rigkit.polish import audit_model
from rigkit.qa_harness import check_pack_invariants
from rigkit.schema import load_avatar_spec, validate_avatar_spec

ROOT = Path(__file__).resolve().parent.parent
TEST_SIGNAL_L2D = ROOT / "models/test-signal/test-signal.l2d"
NORI_L2D = ROOT / "models/nori/nori.l2d"


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


def test_polish_auditor_catches_overhanging_choker_and_detached_limb(tmp_path: Path) -> None:
    """Verify rigkit.polish detects oversized choker overhang AND disconnected limb joints."""
    from PIL import Image, ImageDraw

    model_dir = tmp_path / "bad_rig"
    placed = model_dir / "placed"
    placed.mkdir(parents=True)
    W, H = 1200, 2400

    def save_rect(lid: str, w: int, h: int, color: tuple[int, int, int, int]) -> None:
        im = Image.new("RGBA", (w, h), color)
        im.save(placed / f"{lid}.png")

    # Neck 80px wide, Torso 300px wide, Face 240px wide, Oversized choker 170px wide (>2x neck!)
    save_rect("B02", 300, 600, (245, 220, 210, 255))
    save_rect("B01", 80, 180, (245, 220, 210, 255))
    save_rect("F01", 240, 280, (250, 225, 215, 255))
    save_rect("A03", 170, 40, (25, 25, 30, 255))
    # Detached arm: B03R upper arm -> 50px air gap + 45px lateral shift -> B04R forearm!
    save_rect("B03R", 60, 260, (245, 220, 210, 255))
    save_rect("B04R", 60, 260, (245, 220, 210, 255))

    layers = [
        {"id": "B02", "kind": "torso", "box": [450, 620, 750, 1220]},
        {"id": "B03R", "kind": "arm", "side": "R", "box": [410, 650, 470, 910]},
        {"id": "B04R", "kind": "arm", "side": "R", "box": [355, 960, 415, 1220]},
        {"id": "B01", "kind": "torso", "box": [560, 480, 640, 660]},
        {"id": "A03", "kind": "accessory", "box": [515, 530, 685, 570]},
        {"id": "F01", "kind": "head", "box": [480, 230, 720, 510]},
    ]
    (placed / "layers.json").write_text(json.dumps({"canvas": [W, H], "layers": layers}))
    (model_dir / "rig.json").write_text(json.dumps({"name": "BadRig", "centre": [600.0, 370.0], "radius": 180.0}))

    rep = audit_model(model_dir, out_dir=tmp_path / "polish_out", min_score=80.0)
    assert rep["ok"] is False
    assert rep["score"] < 75.0
    fail_ids = {f["check_id"] for f in rep["findings"] if f["severity"] == "fail"}
    assert any("choker_A03" in cid for cid in fail_ids), f"Expected choker overhang fail, got {fail_ids}"
    assert any("B03R_B04R" in cid or "floating_parts" in cid for cid in fail_ids), f"Expected detached arm fail, got {fail_ids}"
    assert (tmp_path / "polish_out/polish_audit.png").exists()
    assert (tmp_path / "polish_out/polish_fix_prompt.txt").exists()
    assert "A03" in rep["agent_fix_prompt"]


def test_polish_auditor_v2_partial_masking_shoulders_neck_and_shoes(tmp_path: Path) -> None:
    """Verify v2 polish auditor catches:
    1. Unmasked hollow back-rim opening drawn ON TOP of shin (partial_masking)
    2. Bare shoulder V-notch + translucent alpha hole when jacket is OFF (shoulder_attach)
    3. Flared upper jaw stub on neck B01 (neck_chin)
    4. Unfilled open-arch shoe / internal sock outline inside Mary Jane arch (sock_to_shoe)
    """
    from PIL import Image, ImageDraw

    model_dir = tmp_path / "v2_bad_rig"
    placed = model_dir / "placed"
    placed.mkdir(parents=True)
    W, H = 1200, 2400

    # 1. Flared jaw-stub neck B01 (wide 110px at top y=0..32, slender 70px below y=35..160)
    b01 = Image.new("RGBA", (120, 160), (0, 0, 0, 0))
    d = ImageDraw.Draw(b01)
    d.rectangle([5, 0, 115, 34], fill=(245, 220, 210, 255))
    d.rectangle([25, 30, 95, 160], fill=(245, 220, 210, 255))
    b01.save(placed / "B01.png")

    f01 = Image.new("RGBA", (240, 280), (250, 225, 215, 255))
    f01.save(placed / "F01.png")

    # 2. Torso B02 + bare shoulder B03R with a big V-notch gap & translucent alpha=90 hole at shoulder join!
    b02 = Image.new("RGBA", (300, 560), (245, 220, 210, 255))
    b02.save(placed / "B02.png")
    b03r = Image.new("RGBA", (140, 280), (0, 0, 0, 0))
    d = ImageDraw.Draw(b03r)
    d.rectangle([0, 0, 110, 280], fill=(245, 220, 210, 255))
    # Translucent dip & notch on inner shoulder overlap
    d.rectangle([80, 22, 139, 280], fill=(245, 220, 210, 95))
    b03r.save(placed / "B03R.png")
    b04r = Image.new("RGBA", (80, 260), (245, 220, 210, 255))
    b04r.save(placed / "B04R.png")

    # 3. Thigh B05R -> Shin B06R -> Unmasked hollow-opening Sock S01R -> Open-arch Mary Jane Shoe S02R
    Image.new("RGBA", (110, 320), (245, 220, 210, 255)).save(placed / "B05R.png")
    Image.new("RGBA", (100, 340), (245, 220, 210, 255)).save(placed / "B06R.png")

    # Unmasked hollow-rim sock S01R: curved top outer back rim + gray hollow + dark front lip + narrow dark side outlines
    s01 = Image.new("RGBA", (130, 380), (0, 0, 0, 0))
    ds = ImageDraw.Draw(s01)
    # Hollow ellipse opening at top (back wall curves UPWARD away from sock body!)
    ds.ellipse([10, 4, 120, 46], fill=(200, 200, 205, 255), outline=(40, 40, 45, 255), width=3)
    # Main white sock body with dark outline narrower at bottom so it cuts inside the shoe arch!
    ds.rectangle([10, 26, 120, 378], fill=(248, 248, 250, 255), outline=(35, 35, 40, 255), width=3)
    # Re-draw crisp front lip arc (curving downward from sides to center!)
    ds.arc([10, 4, 120, 46], start=0, end=180, fill=(35, 35, 40, 255), width=4)
    s01.save(placed / "S01R.png")

    # Open-arch Mary Jane shoe S02R: outer black shell + top buckle strap creating a wide enclosed arch window wider than S01R!
    s02 = Image.new("RGBA", (180, 220), (0, 0, 0, 0))
    dsh = ImageDraw.Draw(s02)
    dsh.rectangle([10, 115, 170, 215], fill=(28, 28, 32, 255))  # lower vamp/sole
    dsh.rectangle([10, 20, 28, 120], fill=(28, 28, 32, 255))   # left wall
    dsh.rectangle([152, 20, 170, 120], fill=(28, 28, 32, 255)) # right wall
    dsh.rectangle([10, 18, 170, 42], fill=(28, 28, 32, 255))   # Mary Jane ankle strap closing top of arch!
    s02.save(placed / "S02R.png")

    layers = [
        {"id": "B05R", "kind": "leg", "side": "R", "box": [450, 1200, 560, 1520]},
        {"id": "B06R", "kind": "leg", "side": "R", "box": [455, 1480, 555, 1820]},
        {"id": "S01R", "kind": "leg", "side": "R", "box": [440, 1650, 570, 2030]},
        {"id": "S02R", "kind": "leg", "side": "R", "box": [415, 1920, 595, 2140]},
        {"id": "B02", "kind": "torso", "box": [450, 650, 750, 1210]},
        {"id": "B04R", "kind": "arm", "side": "R", "box": [375, 860, 455, 1120]},
        {"id": "B03R", "kind": "arm", "side": "R", "box": [360, 622, 500, 902]},
        {"id": "B01", "kind": "torso", "box": [540, 500, 660, 660]},
        {"id": "F01", "kind": "head", "box": [480, 250, 720, 530]},
    ]
    (placed / "layers.json").write_text(json.dumps({"canvas": [W, H], "layers": layers}))
    (model_dir / "rig.json").write_text(json.dumps({"name": "V2BadRig", "centre": [600.0, 390.0], "radius": 180.0}))

    rep = audit_model(model_dir, out_dir=tmp_path / "v2_out", min_score=80.0)
    fail_ids = {f["check_id"] for f in rep["findings"] if f["severity"] == "fail"}
    assert any("mask_S01R" in cid for cid in fail_ids), f"Expected partial masking fail on S01R, got {fail_ids}"
    assert any("B02_B03R" in cid for cid in fail_ids), f"Expected bare shoulder V-notch/alpha fail, got {fail_ids}"
    assert any("neck_chin" in cid for cid in fail_ids), f"Expected jaw-stub neck_chin fail, got {fail_ids}"
    assert any("S01R_S02R" in cid for cid in fail_ids), f"Expected open-arch shoe fill fail, got {fail_ids}"


def test_polish_auditor_v3_face_stem_double_ears_and_walleyed_gaze(tmp_path: Path) -> None:
    """Verify v3 polish auditor catches:
    1. Neck stem baked onto face F01 below the dark chin outline (face_neck_stem)
    2. Human ears baked onto F01 while separate F05/F06 ears also exist (face_double_ears)
    3. Wall-eyed / outward-diverging unfocused irises & pupils (eye_gaze_focus)
    """
    from PIL import Image, ImageDraw

    model_dir = tmp_path / "v3_bad_face"
    placed = model_dir / "placed"
    placed.mkdir(parents=True)
    W, H = 1200, 2400

    # F01 with dark cheek/chin outline + baked-on side ears outside outline + hanging neck stem below chin!
    f01 = Image.new("RGBA", (320, 360), (0, 0, 0, 0))
    d = ImageDraw.Draw(f01)
    # Baked-on side ears outside cheek oval
    d.ellipse([2, 160, 48, 250], fill=(248, 222, 212, 255), outline=(60, 40, 35, 255), width=4)
    d.ellipse([272, 160, 318, 250], fill=(248, 222, 212, 255), outline=(60, 40, 35, 255), width=4)
    # Hanging pink neck stem below chin
    d.rectangle([118, 298, 202, 358], fill=(248, 222, 212, 255), outline=(60, 40, 35, 255), width=4)
    # Main face oval with continuous dark ink outline
    d.ellipse([36, 14, 284, 318], fill=(252, 228, 218, 255), outline=(60, 40, 35, 255), width=5)
    f01.save(placed / "F01.png")

    # Separate duplicate ear layers F05/F06
    Image.new("RGBA", (42, 70), (248, 222, 212, 255)).save(placed / "F05.png")
    Image.new("RGBA", (42, 70), (248, 222, 212, 255)).save(placed / "F06.png")
    Image.new("RGBA", (84, 180), (248, 222, 212, 255)).save(placed / "B01.png")
    Image.new("RGBA", (300, 560), (248, 222, 212, 255)).save(placed / "B02.png")

    # Wall-eyed sclera/iris: R iris shifted 14px LEFT, L iris shifted 14px RIGHT (+28px outward divergence!)
    Image.new("RGBA", (90, 56), (255, 255, 255, 255)).save(placed / "E01R.png")
    Image.new("RGBA", (90, 56), (255, 255, 255, 255)).save(placed / "E01L.png")
    Image.new("RGBA", (46, 46), (40, 180, 90, 255)).save(placed / "E02R.png")
    Image.new("RGBA", (46, 46), (40, 180, 90, 255)).save(placed / "E02L.png")

    layers = [
        {"id": "B02", "kind": "torso", "box": [450, 650, 750, 1210]},
        {"id": "B01", "kind": "torso", "box": [558, 520, 642, 700]},
        {"id": "F05", "kind": "ear", "side": "R", "box": [430, 390, 472, 460]},
        {"id": "F06", "kind": "ear", "side": "L", "box": [728, 390, 770, 460]},
        {"id": "F01", "kind": "head", "box": [440, 220, 760, 580]},
        {"id": "E01R", "kind": "eye", "side": "R", "box": [485, 360, 575, 416]},
        {"id": "E02R", "kind": "iris", "side": "R", "box": [493, 365, 539, 411]},
        {"id": "E01L", "kind": "eye", "side": "L", "box": [625, 360, 715, 416]},
        {"id": "E02L", "kind": "iris", "side": "L", "box": [661, 365, 707, 411]},
    ]
    (placed / "layers.json").write_text(json.dumps({"canvas": [W, H], "layers": layers}))
    (model_dir / "rig.json").write_text(json.dumps({"name": "V3BadFace", "centre": [600.0, 390.0], "radius": 180.0}))

    rep = audit_model(model_dir, out_dir=tmp_path / "v3_out", min_score=80.0)
    fail_ids = {f["check_id"] for f in rep["findings"] if f["severity"] == "fail"}
    assert any("face_neck_stem" in cid for cid in fail_ids), f"Expected face_neck_stem fail, got {fail_ids}"
    assert any("face_double_ears" in cid for cid in fail_ids), f"Expected face_double_ears fail, got {fail_ids}"
    assert any("eye_gaze_focus" in cid for cid in fail_ids), f"Expected eye_gaze_focus fail, got {fail_ids}"


def test_face3d_side_jaw_generator_and_unfolding(tmp_path: Path) -> None:
    """Verify reusable rigkit.face3d synthesizes mirrored F01sL/F01sR and unfolds cleanly on head turns."""
    from PIL import Image, ImageDraw

    f01 = Image.new("RGBA", (480, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(f01)
    d.ellipse([20, 10, 460, 588], fill=(248, 222, 214, 255), outline=(80, 32, 28, 255), width=6)
    f01_box = [960, 320, 1440, 920]

    def simple_fwd(pts: np.ndarray, yaw: float) -> np.ndarray:
        out = np.array(pts, dtype=np.float64, copy=True)
        out[..., 0] = 1200.0 + (out[..., 0] - 1200.0) * math.cos(yaw) + 120.0 * math.sin(yaw)
        return out

    im_l, im_r, spec = synthesize_side_jaw_layers(f01, f01_box, simple_fwd, canvas_w=2400, max_yaw_deg=32.5)
    assert im_l.size == im_r.size and im_l.width > 80 and im_l.height > 200
    # Verify exact horizontal mirroring of bounding boxes about canvas center x=1200
    assert spec.box_l[0] + spec.box_r[2] == 2400
    assert spec.box_l[2] + spec.box_r[0] == 2400
    # Verify sampled skin & ink tones match F01 AND rear-to-front dark-to-light 3D depth shading gradient exists!
    assert spec.skin_rgb[0] > 230 and spec.ink_rgb[0] < 120
    arr_l = np.asarray(im_l)
    mid_row = arr_l[arr_l.shape[0] // 2]
    op_xs = np.nonzero(mid_row[:, 3] > 220)[0]
    assert len(op_xs) > 20
    front_lum = float(mid_row[op_xs[len(op_xs) // 5], :3].mean())
    rear_lum = float(mid_row[op_xs[-max(3, len(op_xs) // 10)], :3].mean())
    assert front_lum - rear_lum >= 12.0, f"Expected rear-to-front dark-to-light 3D shading gradient, got front={front_lum:.1f} vs rear={rear_lum:.1f}"

    # Verify camera-parallel unfolding keyforms: op=0 at Front (a=0) & far turn (a=+1), op=1 on active turn (a=-1)
    # AND bounded strictly-monotonic D-grid at mid-yaw (a=-0.5 -> max|D| < 0.35, zero corner blowup!)
    lw, lh = float(spec.box_l[2] - spec.box_l[0]), float(spec.box_l[3] - spec.box_l[1])
    rest_t = np.array([(spec.box_l[0] + spec.box_l[2]) / 2.0, (spec.box_l[1] + spec.box_l[3]) / 2.0])
    k_front = fit_side_jaw_3d(spec, lw, lh, rest_t, 0.0, {}, active_sign=-1.0)
    k_far = fit_side_jaw_3d(spec, lw, lh, rest_t, +1.0, {}, active_sign=-1.0)
    k_mid = fit_side_jaw_3d(spec, lw, lh, rest_t, -0.5, {}, active_sign=-1.0)
    k_active = fit_side_jaw_3d(spec, lw, lh, rest_t, -1.0, {}, active_sign=-1.0)
    assert k_front["op"] == 0.0 and k_far["op"] == 0.0
    assert k_active["op"] == 1.0 and len(k_active["D"]) == 7 * 7 * 2
    assert float(np.max(np.abs(k_mid["D"]))) < 0.35, f"Mid-yaw D-grid blowup: {np.max(np.abs(k_mid['D']))}"

    # Verify clipped 3D directional turn face shading (F02tR/L) & 3/4 nose (F03tR/L) synthesis & zero-pop keyforms
    f02t_r, f02t_l, f03t_r, f03t_l, dspec = synthesize_turn_face_depth_layers(
        f01, f01_box, simple_fwd, side_jaw_spec=spec, nose_front_xy=(1200.0, 760.0), canvas_w=2400
    )
    assert f02t_r.size == f02t_l.size and f03t_r.size == f03t_l.size
    assert dspec.box_shade_r[0] + dspec.box_shade_l[2] == 2400
    assert dspec.box_nose_r[0] + dspec.box_nose_l[2] == 2400
    sh_rt = np.array([(dspec.box_shade_r[0] + dspec.box_shade_r[2]) / 2.0, (dspec.box_shade_r[1] + dspec.box_shade_r[3]) / 2.0])
    ns_rt = np.array([(dspec.box_nose_r[0] + dspec.box_nose_r[2]) / 2.0, (dspec.box_nose_r[1] + dspec.box_nose_r[3]) / 2.0])
    assert fit_turn_face_shade_3d(dspec, sh_rt, 0.0, {}, active_sign=-1.0)["op"] == 0.0
    assert fit_turn_face_shade_3d(dspec, sh_rt, -1.0, {}, active_sign=-1.0)["op"] == 1.0
    assert fit_turn_nose_3d(dspec, ns_rt, 0.0, {}, active_sign=-1.0)["op"] == 0.0
    assert fit_turn_nose_3d(dspec, ns_rt, -1.0, {}, active_sign=-1.0)["op"] == 1.0


def test_nori_l2d_valid_and_polished(tmp_path: Path) -> None:
    if not NORI_L2D.exists():
        pytest.skip("models/nori/nori.l2d not built")
    val = validate_l2d(NORI_L2D)
    assert val["ok"], f"Nori .l2d validation errors: {val['errors']}"
    unpacked = tmp_path / "nori_unpacked"
    unpack_l2d(NORI_L2D, unpacked)
    pack = json.loads((unpacked / "pack.json").read_text())
    lids = {L["id"] for L in pack["layers"]}
    assert {"F01sR", "F01sL", "F02tR", "F02tL", "F03tR", "F03tL"}.issubset(lids), (
        "Nori must include full 3D side-of-head stack (F01sR/L, F02tR/L, F03tR/L)"
    )
    assert not pack["motion"].get("turnChain"), "Nori must omit rubbery turnChain for solid Test-Signal-style body motion"
    assert pack["motion"].get("bodyFollowLR") == [0.025, 0.025]
    rep = audit_model(NORI_L2D, out_dir=tmp_path / "nori_polish", min_score=95.0)
    assert rep["counts"]["fail"] == 0, f"Nori polish fails: {[f for f in rep['findings'] if f['severity'] == 'fail']}"
    assert rep["counts"]["warn"] == 0, f"Nori polish warns: {[f for f in rep['findings'] if f['severity'] == 'warn']}"
    assert rep["score"] >= 98.0, f"Expected Nori polish score >= 98, got {rep['score']}"


def test_scene_graph_l2d_validation(tmp_path: Path) -> None:
    """Verify rigkit.format packs & validates Scene Graph nodes/parents and rejects cycles & dangling parents."""
    from PIL import Image

    vdir = tmp_path / "sg_viewer"
    (vdir / "tex").mkdir(parents=True)
    for lid in ("F01", "E02R", "E03R"):
        Image.new("RGBA", (64, 64), (200, 120, 140, 255)).save(vdir / "tex" / f"{lid}.png")

    def make_pack(nodes, layers):
        return {
            "canvas": [1200, 2000],
            "texExt": "png",
            "head": {"centre": [600, 500], "radius": 200, "eu": [1, 0], "ev": [0, 1], "chin": 180, "neck": 280, "jawNear": 120, "pivot": [600, 680], "turn": {"R": -0.22, "L": 0.22}},
            "motion": {"waist": [600, 1100], "footY": 1900, "bodyC": 600, "bodyR": 220, "locks": []},
            "nodes": nodes,
            "layers": layers,
            "samples": [
                {
                    "angle": ang,
                    "jaw": 0.0,
                    "chin": 0.0,
                    "chinAt": [600, 680],
                    "layers": [{"A": [1, 0, 0, 1], "t": [600, 500], "op": 1.0} for _ in layers],
                    "nodes": {"N_EyeR": {"dt": [ang * 40, 0], "dRot": 0, "dScale": [0, 0], "dOp": 0}},
                }
                for ang in (-0.22, 0.0, 0.22)
            ],
            "blendShapes": [
                {
                    "id": "bs_eye_x",
                    "name": "Eye Look X",
                    "min": -1,
                    "max": 1,
                    "default": 0,
                    "bind": "eyeX",
                    "layers": {},
                    "nodes": {"N_EyeR": {"dt": [18, 0], "dRot": 0, "dScale": [0, 0], "dOp": 0}},
                }
            ],
            "expr": {"pieces": {}, "faces": {"Neutral": {}}, "holds": {}},
        }

    # 1. Valid multi-level hierarchy: N_Head -> N_EyeR -> E02R -> E03R
    good_nodes = [
        {"id": "N_Head", "name": "Head Rig", "pivot": [600, 500]},
        {"id": "N_EyeR", "name": "Eye R", "parent": "N_Head", "pivot": [540, 480]},
    ]
    good_layers = [
        {"id": "F01", "w": 64, "h": 64, "kind": 0, "head": True, "parent": "N_Head"},
        {"id": "E02R", "w": 64, "h": 64, "kind": 0, "head": True, "parent": "N_EyeR"},
        {"id": "E03R", "w": 64, "h": 64, "kind": 0, "head": True, "parent": "E02R"},
    ]
    (vdir / "pack.json").write_text(json.dumps(make_pack(good_nodes, good_layers)))
    out_good = tmp_path / "good_sg.l2d"
    pack_l2d(vdir, out_good)
    res_good = validate_l2d(out_good)
    manifest_good = inspect_l2d(out_good)["manifest"]
    assert res_good["ok"], f"Valid scene graph failed: {res_good['errors']}"
    assert res_good["stats"]["nodes"] == 2
    assert manifest_good["capabilities"]["sceneGraph"] is True
    assert manifest_good["capabilities"]["blendShapeDeform"] is True

    # 2. Cycle detection: N_Head -> N_EyeR -> N_Head
    bad_nodes = [
        {"id": "N_Head", "name": "Head Rig", "parent": "N_EyeR", "pivot": [600, 500]},
        {"id": "N_EyeR", "name": "Eye R", "parent": "N_Head", "pivot": [540, 480]},
    ]
    (vdir / "pack.json").write_text(json.dumps(make_pack(bad_nodes, good_layers)))
    out_cycle = tmp_path / "cycle_sg.l2d"
    pack_l2d(vdir, out_cycle)
    res_cycle = validate_l2d(out_cycle)
    assert not res_cycle["ok"], "Expected cycle scene graph to fail validation"
    assert any("cycle" in e.lower() for e in res_cycle["errors"]), res_cycle["errors"]

    # 3. Missing parent rejection
    dangling_layers = [
        {"id": "F01", "w": 64, "h": 64, "kind": 0, "head": True, "parent": "N_DoesNotExist"},
        {"id": "E02R", "w": 64, "h": 64, "kind": 0, "head": True},
        {"id": "E03R", "w": 64, "h": 64, "kind": 0, "head": True},
    ]
    (vdir / "pack.json").write_text(json.dumps(make_pack([], dangling_layers)))
    out_dang = tmp_path / "dangling_sg.l2d"
    pack_l2d(vdir, out_dang)
    res_dang = validate_l2d(out_dang)
    assert not res_dang["ok"], "Expected missing parent to fail validation"
    assert any("missing parent" in e.lower() for e in res_dang["errors"]), res_dang["errors"]
