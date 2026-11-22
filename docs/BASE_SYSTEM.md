# Layer2D Unified Base Avatar System

Layer2D turns avatar creation from bespoke one-off scripting into a modular, repeatable pipeline capable of producing studio-grade `.l2d` avatars rapidly.

---

## Architecture Overview

```text
avatar_spec.json (rigkit.schema)
        │
        ├──► Art & Volumetric Depth Shells (rigkit.hair3d)
        │       - HeadSkull3D ellipsoid + two-zone skull-wrap (forehead -> temple)
        │       - 100% solid temple-crown & under-bangs depth shells
        │
        ├──► Expression & Animated Flipbook Library (rigkit.expr_library + assets/fx/)
        │       - AvatarFaceAnchors -> 37 reusable expressions & 20 animated FX
        │       - Clipped forehead mood shades, floating hearts, sparkles, tears
        │
        ├──► WebGL2 Runtime Packer (rigkit.meshpack)
        │       - Multi-sample turn keyforms, WebP texture encoding, spring chains
        │       - Data-driven tail, hairShadowId, highlightIds, pitchGroups, timelines
        │
        ├──► Open .l2d Container Builder (rigkit.format)
        │       - Deterministic ZIP archive + layer2d.json + SHA256SUMS.json
        │
        ├──► Free Layer2D WebGL2 OBS Player (viewer/index.html)
        │       - Zero model hardcoding; runs unpacked packs OR .l2d bundles directly
        │       - Webcam MediaPipe tracking, Web Audio AEIOU lip-sync, ?obs=1 alpha
        │
        └──► Automated QA Harness (rigkit.qa_harness + tests/test_rigkit.py)
                - Pack invariants, .l2d SHA-256 verification, WebGL2 chroma gap check,
                  OBS transparency check, and commercial copy hygiene linter
```

---

## Rapidly Creating a New Avatar

1. **Scaffold a runnable project in one command:**
   ```bash
   python -m rigkit.model_scaffold kitsune_nova --name "Kitsune Nova" --out models/kitsune_nova
   ```
   This produces `avatar_spec.json`, `rig.json`, starter `placed/` layers, `viewer/pack.json`, `viewer/index.html`, and a validated `kitsune_nova.l2d` bundle.

2. **Drop in final character art & synthesize 3D volumetric hair depth:**
   - Replace starter RGBA layers in `placed/` with character art sheets.
   - Use `rigkit.hair3d.HeadSkull3D`, `synthesize_temple_crown_shell`, and `synthesize_under_bangs_shell` to build 100% solid intermediate hair depth shells positioned behind kemonomimi ears (if any) and behind front bangs.

3. **Attach the 37-expression + animated flipbook FX library:**
   - Instantiate `rigkit.expr_library.AvatarFaceAnchors` from your `avatar_spec.json` head coordinates and call `build_flipbook_fx_pieces`, `build_standard_faces`, and `build_default_timelines`.

4. **Pack, bundle into `.l2d`, and run full automated QA:**
   ```bash
   python -m rigkit.meshpack models/kitsune_nova
   python -m rigkit.format pack models/kitsune_nova/viewer models/kitsune_nova/kitsune_nova.l2d --meta models/kitsune_nova/avatar_spec.json
   python -m rigkit.qa_harness models/kitsune_nova
   ```

5. **Run in the Layer2D Player or OBS:**
   - Drop `kitsune_nova.l2d` directly onto the Layer2D Player (`viewer/index.html`) or open `index.html?model=kitsune_nova.l2d&obs=1&cam=1&mic=1`.
