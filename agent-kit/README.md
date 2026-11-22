# Layer2D Agent Testing Kit (`layer2d/v1`)

Self-contained offline toolkit for AI agents and developers to build, validate, and visually QA `.l2d` avatars without installing Layer2D Studio or proprietary SDKs.

## Files
- `SKILL.md` — Full agent skill guide, `pack.json` contract, and hard-won rigging rules.
- `l2d_tools.py` — Pure stdlib Python CLI (`pack`, `validate`, `inspect`, `unpack`).
- `starter_builder.py` — Minimal runnable builder that creates a valid `starter.l2d` from scratch.
- `qa_runner.py` — Headless Playwright WebGL2 visual & runtime QA suite.
- `player.html` — Unmodified standalone Layer2D WebGL2 Player used offline by `qa_runner.py`.

## Quickstart
```bash
# 1. Build a valid starter .l2d archive
python3 starter_builder.py starter.l2d

# 2. Cryptographically & structurally validate any .l2d archive (exits non-zero on error)
python3 l2d_tools.py validate starter.l2d
python3 l2d_tools.py inspect starter.l2d

# 3. Run offline WebGL2 runtime & visual QA (requires playwright + pillow)
python3 qa_runner.py starter.l2d --out qa_out/
```

Outputs in `qa_out/`:
- `qa_report.json` — machine-readable pass/fail report (validation, console errors, OBS alpha)
- `qa_turns_wide.png` — 9-angle yaw sweep (`-1.0 .. +1.0`)
- `qa_turns_center.png` — 9-angle fine center sweep (`-0.40 .. +0.40`, catches popping under-bangs)
- `qa_pitch_grid.png` — 3x3 head Yaw x Pitch tracking grid
- `qa_expressions.png` — expression contact sheet
