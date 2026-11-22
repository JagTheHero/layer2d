<p align="left">
  <img src="assets/brand/logo-wordmark.svg" alt="Layer2D" height="44" />
</p>

# Layer2D

**An open, royalty-free 2D layered avatar format (`.l2d`), pure-browser WebGL2 Player, client-side Rigging & Animation Studio, and AI Agent Testing Kit.**

- **Live Website:** https://layer2d.com
- **Layer2D Studio (100% client-side):** https://layer2d.com/studio/
- **Layer2D Player (OBS-ready WebGL2):** https://layer2d.com/player/?model=test-signal.l2d
- **Format & Agent Docs:** https://layer2d.com/docs.html
- **Agent Skill (`SKILL.md`):** https://layer2d.com/skills/layer2d-avatar-builder/SKILL.md
- **Downloadable Agent Testing Kit:** https://layer2d.com/downloads/layer2d-agent-kit.zip

---

## What's Included

| Path | Description |
| --- | --- |
| `studio/` | Pure client-side WebGL2 **Layer2D Studio** — open `.l2d` bundles or import layered `.psd` files directly, sculpt inverse warp grids (`D`) and lattice cages on yaw keyforms, edit draw order, clipping masks, `behind` depth occlusion, spring physics chains, expressions, flipbook FX, and keyframe animations, then export verified `.l2d` archives or WebM clips. |
| `viewer/index.html` | Standalone zero-dependency **Layer2D WebGL2 Player** — drop any `.l2d` onto the stage or load via `?model=…`, with `?obs=1` premultiplied alpha transparency, real-time Web Audio AEIOU mic lip-sync, and MediaPipe webcam head tracking. |
| `models/test-signal/test-signal.l2d` | MIT-licensed **Test Signal** reference avatar (`95` layers, `17` yaw keyforms, `41` expressions, `4` pose-arm holds, `AEIOU` vowels). |
| `rigkit/` | Open-source Python rigging, packing, PSD export, format validation (`rigkit.format`), volumetric hair-shell synthesis (`rigkit.hair3d`), expression library (`rigkit.expr_library`), scaffolding (`rigkit.model_scaffold`), and automated QA (`rigkit.qa_harness`). |
| `agent-kit/` | Standalone Agent Testing Kit (`SKILL.md`, stdlib `l2d_tools.py`, `starter_builder.py`, offline Playwright `qa_runner.py`, bundled `player.html`). |
| `.agents/skills/layer2d-avatar-builder/SKILL.md` | Agent Skill describing how AI agents can build, validate, visually QA, and fix `.l2d` avatars end to end. |
| `docs/` | Open `layer2d/v1` specification ([LAYER2D_FORMAT_SPEC.md](docs/LAYER2D_FORMAT_SPEC.md)), system architecture ([BASE_SYSTEM.md](docs/BASE_SYSTEM.md)), standalone agent guide ([AGENT_GUIDE.md](docs/AGENT_GUIDE.md)), and ecosystem analysis ([INOCHI2D_ANALYSIS.md](docs/INOCHI2D_ANALYSIS.md)). |
| `assets/fx/` | Reusable animated expression flipbook sprite sheets (`sparkles`, `hearts`, `tears`, `blush`, mood shades, etc.). |
| `tests/test_rigkit.py` | Automated unit, schema, scaffolding, `.l2d` SHA-256 tamper, and Test Signal validation tests. |

---

## Quickstart

### 1. Run Studio & Player Locally (Zero Build Step)

```bash
python3 -m http.server 8765
```

- **Layer2D Studio:** http://127.0.0.1:8765/studio/ (auto-loads `models/test-signal/test-signal.l2d`)
- **Layer2D Player:** http://127.0.0.1:8765/viewer/?model=../models/test-signal/test-signal.l2d
- **OBS Browser Source:** `http://127.0.0.1:8765/viewer/?model=../models/test-signal/test-signal.l2d&obs=1&cam=1&mic=1`

### 2. Inspect, Validate, or Pack `.l2d` Archives (Python Stdlib Only)

```bash
python3 -m rigkit.format validate models/test-signal/test-signal.l2d
python3 -m rigkit.format inspect  models/test-signal/test-signal.l2d
python3 -m rigkit.format unpack   models/test-signal/test-signal.l2d /tmp/test_signal
python3 -m rigkit.format pack     /tmp/test_signal /tmp/repacked.l2d
```

### 3. Scaffold a New Avatar & Run Tests

```bash
pip install numpy pillow scipy pytest
python3 -m pytest -v tests/test_rigkit.py
python3 -m rigkit.model_scaffold my_avatar --name "My Avatar" --out models/my_avatar
```

---

## License

Released under the [MIT License](LICENSE). The included **Test Signal** demo avatar (`models/test-signal/test-signal.l2d`) and reusable FX sheets (`assets/fx/`) are also licensed under MIT.
