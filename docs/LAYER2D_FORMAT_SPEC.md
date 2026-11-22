# Layer2D Open Avatar Format Specification (`layer2d/v1`)

**Status:** Draft 1.0 (Open, Royalty-Free Specification)  
**File Extension:** `.l2d`  
**MIME Type:** `application/vnd.layer2d+zip`

---

## 1. Purpose

Layer2D (`.l2d`) is a free, open, self-contained container format and WebGL2 runtime specification for 2D rigged avatars. It is designed as an independent, zero-licensing-fee alternative to proprietary 2D VTuber ecosystems, optimized for browser runtimes, OBS Browser Source capture, and automated multi-avatar pipelines.

Anyone may author, pack, parse, play, or distribute `.l2d` models without runtime royalties, proprietary SDK binaries, or per-seat licensing restrictions.

---

## 2. Container Layout

A `.l2d` file is a standard ZIP archive containing UTF-8 JSON descriptors, pre-encoded raster textures, an optional portrait preview, and a deterministic SHA-256 integrity manifest:

```text
model.l2d
├── layer2d.json          # Open format manifest (required)
├── pack.json             # Runtime rig, keyforms, springs, expressions (required)
├── preview.png           # Optional portrait thumbnail (recommended 1200x1600 RGBA)
├── tex/                  # Referenced layer, eye, mask, and flipbook textures (required)
│   ├── F01.webp
│   ├── H03C.webp
│   ├── expr_M04.png
│   └── S_fx_sparkles.png
└── SHA256SUMS.json       # SHA-256 hex digest for every non-checksum member (required)
```

### Container Safety Rules
- All archive paths MUST be relative POSIX paths with no leading `/` and no `..` traversal segments.
- Duplicate filenames are forbidden.
- Binary texture files SHOULD use `ZIP_STORED` (method 0) so browser runtimes can decode images directly without redundant decompression overhead; JSON files MAY use `ZIP_DEFLATED` (method 8).
- Internal build logs, cost ledgers, and staging files MUST NOT be bundled inside `.l2d`.

---

## 3. `layer2d.json` Manifest Schema

```json
{
  "format": "layer2d/v1",
  "id": "test-signal",
  "name": "Test Signal",
  "title": "Test Signal — Layer2D Demo Avatar",
  "version": "1.0.0",
  "author": "Layer2D",
  "license": "MIT",
  "createdAt": "2026-11-22T00:00:00Z",
  "entry": "pack.json",
  "texExt": "webp",
  "canvas": [1296, 1728],
  "preview": "preview.png",
  "stats": {
    "layers": 95,
    "chains": 0,
    "samples": 17,
    "expressions": 41,
    "holds": 4,
    "outfits": 0,
    "exprPieces": 81,
    "vowels": ["A", "E", "I", "O", "U"]
  },
  "capabilities": {
    "headYawSamples": 17,
    "headPitch2D": true,
    "springChains": false,
    "tailRig": false,
    "obsAlpha": true,
    "micLipSync": true,
    "webcamTracking": true
  }
}
```

---

## 4. Runtime `pack.json` Contract

A conforming Layer2D runtime (`viewer/index.html`) evaluates each frame using pure GPU inverse-grid fragment deformation in WebGL2:

1. **Multi-Sample Continuous Turn Keyforms (`samples[]`)**: Each sample defines `angle` and per-layer affine transform `{A, t}`, optional inverse displacement grid `D`, binary depth occlusion `behind` (`>= 0.5 ? 1 : 0`), and `op`. Hair layers remain `op = 1.0` across all yaw samples — volumetric depth is achieved through multi-depth physical shells (`H01`, `H02c`, `H02`, `H02s`, `H03u`, `H04s`, `H04`, `H03`) and two-zone skull-wrap geometry rather than translucent crossfades.
2. **Topological Clipping Masks (`clip`, `clipTarget`)**: Offscreen alpha mask framebuffers are sorted topologically so parent masks (e.g., face `F01`) always render before nested child masks (e.g., sclera `E01R/L`).
3. **2D Head Pitch Groups (`pitchGroups`)**: Layers group into `ear`, `crown`, `back`, `humanEar`, `side`, `bangs`, or `face` for perspective-consistent vertical look.
4. **Spring Physics Chains (`chains[]`) & Pelvis-Locked Tail (`tail`)**: Multi-joint damped Verlet/spring chains drive hair locks, kemonomimi ears, ribbons, tassels, and accessories; optional `tail` config specifies `{id, pivot, amp, freq, happyFaces}` for crease-free pelvis-local sway.
5. **Expressions, Flipbook FX & Lip-Sync (`expr`)**:
   - `pieces`: Static swaps (mouths, brows, eye overlays, pose arms) and sprite-sheet flipbooks (`tex`, `frames`, `fps`, `loop`, `at`, `base`, `scale`).
   - `faces`: Declarative emotion definitions combining `hide`, `add`, `move`, `eyes`, `shake`, and `gaze`.
   - `holds`: Interactive full-arm pose holds with automatic resting-arm suppression.
   - `vowels`: Direct `A/E/I/O/U` mouth piece mapping for Web Audio microphone lip-sync and MediaPipe webcam blendshapes.
6. **Data-Driven Runtime Identifiers**: `hairShadowId`, `highlightIds`, `tail`, `pitchGroups`, `expr.faceTimeline`, and `expr.holdTimeline` are declared in `pack.json`, allowing a single unmodified Player binary to run any conforming avatar.

---

## 5. Browser & OBS Runtime Interface

A conforming Layer2D Player supports:
- Direct `.l2d` loading via `?model=<url-to-file.l2d>`, file picker, or drag-and-drop onto the viewport, in addition to unpacked `pack.json` directories.
- Transparent OBS Browser Source mode via `?obs=1` (hides UI chrome, outputs premultiplied alpha).
- Automation query parameters: `?obs=1&cam=1&mic=1&auto=0&face=<ExpressionName>&view=bust|full`.
- Standard hotkeys: `M` (Mic), `C` (Cam), `A` (Auto performance), `B` (Background cycle), `0` (Clear expression), `1`–`9` (Quick expressions).

---

## 6. Reference CLI (`rigkit.format`)

```bash
python -m rigkit.format pack models/test-signal/viewer models/test-signal/test-signal.l2d
python -m rigkit.format validate models/test-signal/test-signal.l2d
python -m rigkit.format inspect models/test-signal/test-signal.l2d
python -m rigkit.format unpack models/test-signal/test-signal.l2d /tmp/test_signal_unpacked
```
