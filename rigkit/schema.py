"""Unified Layer2D Avatar Specification Schema (`layer2d-spec/v1`).

Single declarative blueprint that lets any new avatar plug straight into the
Layer2D pipeline (`hair3d`, `expr_library`, `meshpack`, `format`, `qa_harness`)
with zero viewer code modifications.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

SPEC_VERSION = "layer2d-spec/v1"


@dataclass
class HeadAnatomy:
    centre: list[float]            # [cx, cy] in canvas px
    radius: float                  # skull radius in canvas px
    chin: float                    # offset from centre y to chin
    neck: float                    # offset from centre y to neck
    neck_pivot: list[float]        # [x, y] head rotation pivot
    yaw_max_deg: float = 32.5      # max turn angle in degrees
    eye_r: list[float] = field(default_factory=lambda: [1080.0, 680.0])
    eye_l: list[float] = field(default_factory=lambda: [1320.0, 680.0])
    mouth: list[float] = field(default_factory=lambda: [1200.0, 825.0])
    forehead_y: float = 540.0


@dataclass
class HairDepthConfig:
    enabled: bool = True
    rz_forehead: float = 204.0
    rz_temple: float = 88.0
    rz_crown: float = 142.0
    rz_under_bangs: float = 148.0
    rz_temple_shell: float = 68.0
    solid_alpha_only: bool = True   # Never crossfade hair opacity on turns


@dataclass
class CommercialMeta:
    exclusivity: str = "1-of-1 Exclusive"
    price_usd: int = 200
    ai_assisted_disclosure: str = "AI-assisted digital illustration, Layer2D volumetric rigging, and custom post-production."
    license_summary: str = "Exclusive commercial streaming, content creation, and merchandising rights transferred once; model retired upon sale."


@dataclass
class AvatarSpec:
    spec_version: str
    slug: str
    name: str
    title: str
    version: str
    author: str
    canvas: list[int]
    head: HeadAnatomy
    hair_depth: HairDepthConfig = field(default_factory=HairDepthConfig)
    commercial: CommercialMeta | None = None
    outfits: list[str] = field(default_factory=list)
    holds: list[str] = field(default_factory=list)
    expression_pack: str = "standard-37"
    has_kemonomimi_ears: bool = False
    has_tail: bool = False
    qa_gates: dict[str, Any] = field(
        default_factory=lambda: {
            "max_hair_gap_px_center": 400,
            "zero_hair_opacity_crossfade": True,
            "min_expressions": 10,
            "require_obs_alpha": True,
            "require_l2d_valid": True,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def validate_avatar_spec(data: dict[str, Any]) -> list[str]:
    """Validate an `avatar_spec.json` dictionary and return a list of error strings (empty == valid)."""
    errors: list[str] = []
    if data.get("spec_version") != SPEC_VERSION:
        errors.append(f"spec_version must be {SPEC_VERSION!r}, got {data.get('spec_version')!r}")
    for key in ("slug", "name", "title", "canvas", "head"):
        if key not in data or not data[key]:
            errors.append(f"Missing required field: {key}")
    canvas = data.get("canvas")
    if not (isinstance(canvas, list) and len(canvas) == 2 and all(isinstance(v, (int, float)) and v >= 256 for v in canvas)):
        errors.append(f"canvas must be [width, height] >= 256, got {canvas!r}")
    head = data.get("head") or {}
    for hk in ("centre", "radius", "chin", "neck", "neck_pivot"):
        if hk not in head:
            errors.append(f"head.{hk} is required")
    if isinstance(head.get("centre"), list) and len(head["centre"]) != 2:
        errors.append("head.centre must be [x, y]")
    if isinstance(head.get("neck_pivot"), list) and len(head["neck_pivot"]) != 2:
        errors.append("head.neck_pivot must be [x, y]")
    comm = data.get("commercial")
    if comm is not None:
        txt = json.dumps(comm).lower()
        if "hand-drawn" in txt or "hand drawn" in txt:
            errors.append("Commercial copy must NOT claim 'hand-drawn'")
        if "$1,500" in txt or "$1500" in txt:
            errors.append("Commercial copy must NOT reference fake anchor pricing ($1,500)")
    return errors


def load_avatar_spec(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    errs = validate_avatar_spec(data)
    if errs:
        raise ValueError(f"Invalid avatar_spec.json ({path}): {errs}")
    return data
