# Inochi2D vs. Layer2D — Architecture Analysis & Features Worth Adopting

**Inochi2D** (BSD-2-Clause permissive license — `inochi2d` core in D, `inochi-creator` editor, `inochi-session` tracker, plus forks like `nijigenerate`) is the leading open-source 2D puppet ecosystem. Because it is BSD-2-Clause, we can study its design patterns freely (and borrow concepts cleanly into Layer2D's `layer2d/v1` / `v2` spec and browser Studio).

---

## 1. Architectural Comparison

| Dimension | Inochi2D (`.inp` / `.inx`) | Layer2D (`.l2d` / `layer2d/v1`) |
|---|---|---|
| **License** | BSD-2-Clause | Open, royalty-free spec + MIT runtime/tools |
| **Container** | Custom big-endian binary (`TRNSRTS\0` + JSON + `TEX_SECT` raw blobs + optional `EXT_SECT` editor metadata) | Standard ZIP (`layer2d.json` + `pack.json` + `ZIP_STORED` WebP/PNG textures + optional `studio.json` + `SHA256SUMS.json`) |
| **Runtime Target** | Native D + OpenGL desktop (`inochi-session`, Godot/Unity bindings) | Zero-install Browser WebGL2 + OBS Browser Source (`?obs=1`) + pure-browser Studio |
| **Deformation Model** | Explicit triangle meshes (`vertices`, `uvs`, `indices`), `MeshGroup` parent cages, path deformers, vertex welding | GPU fragment-shader inverse displacement grid `D[gd×gd]` + cylindrical head-turn + multi-depth shells + topological alpha FBO clipping |
| **Parameters** | Arbitrary 1D sliders & 2D XY parameter grids with per-node keyframe interpolation | Multi-sample yaw keyforms (`samples[]`), 2D pitch groups, continuous roll/eyes/vowels/expressions/holds |
| **Physics** | Standardized in-spec `SimplePhysics` nodes (pendulum / spring-pendulum outputting to parameters) | Standardized in-spec multi-joint Verlet spring chains (`chains[]`) + rigid pelvis-locked tail (`tail`) |
| **Compositing** | Blend modes (Multiply, Screen, Add/Linear Dodge, Clip Studio modes), multi-texturing (Albedo + **Emissive** + **Bumpmap**), **Mask & DodgeMask** (inverted cutout), `DynamicComposite` | Premultiplied alpha, topological `clip` masks, binary `behind` occlusion masks, `sheer` thin-film + procedural `sky` aurora, OKLab hair/eye re-tinting |

---

## 2. High-Value Inochi2D Tricks & Features Worth Adopting in Layer2D

### Priority 1 — High Impact, Clean Fit into `layer2d/v1` + Studio
1. **Per-Layer Blend Modes (`blend`: `normal | multiply | screen | add | overlay`)**
   - *Why it matters:* Right now shadows (`F02` hair shadow, cheek blush, glow sparkles, starry eye gleams) bake their color math into alpha PNGs. Adding native WebGL2 blend-mode switching (`gl.blendFuncSeparate` / shader blend) lets artists import Photoshop Multiply shadows and Add/Glow eye highlights straight from PSD with zero color muddying!
   - *Effort:* Small (1 field on `layer` + 10 lines in `viewer/index.html` / `engine.js` + read `rec.blendMode` already parsed in `psd_import.js`!).
2. **Inverted Cutout Mask (`DodgeMask` — mask-out / hole-punch by another layer)**
   - *Why it matters:* Inochi2D's `MaskMode.DodgeMask` lets any layer punch an exact inverse stencil out of another layer dynamically. Layer2D already has `behind: [maskA, maskB]` for face/neck occlusion — generalizing `clipMode: "mask" | "dodge"` lets any accessory, bangs-over-eyebrows translucency ("anime eye-through-hair" effect!), or mouth interior mask cleanly without custom baked `_hole.png` textures!
   - *Effort:* Tiny (already have `clipFbo` in topological order; 2 lines in `FS`!).
3. **Emissive Glow Map Slot (`emissive`: optional `tex/<id>_emissive.webp`)**
   - *Why it matters:* Inochi2D's 3-slot multi-texturing (Albedo + Emissive + Bumpmap) makes glowing cyber accents, starry eyes, and shrine lantern talismans pop dramatically under dark/ambient lighting tint at nearly zero GPU cost. Adding an optional `emissive` texture + ambient tint uniform gives streamers instant scene-reactive lighting in OBS!
   - *Effort:* Small.
4. **Point / Nearest-Neighbor Filtering Toggle (`filter: "linear" | "nearest"`)**
   - *Why it matters:* Inochi2D is famous in the indie/VTuber community for crisp **pixel-art avatars** because Live2D blurs pixel art unless scaled 8×. Adding a single per-pack or per-layer `filter: "nearest"` flag (`gl.NEAREST` without mipmaps) instantly unlocks the entire pixel-art VTuber market for Layer2D!
   - *Effort:* Tiny (~6 lines in texture upload!).
5. **2D XY Parameter Pad in Studio (Arming & Keying arbitrary 2D parameters)**
   - *Why it matters:* Inochi Creator's 2D XY parameter box (dragging a dot inside a `[-1..+1] × [-1..+1]` grid to sculpt Yaw × Pitch corners simultaneously!) is vastly faster than scrubbing two separate 1D sliders when fine-tuning diagonal head turns (`Yaw=-1, Pitch=+1`).
   - *Effort:* Medium (pure Studio UI widget + optional 2D keyform grid in `layer2d/v2`).

### Priority 2 — Strong `layer2d/v2` / Studio Upgrades
6. **`MeshGroup` / Shared Parent Warp Cage (Deform multiple layers together)**
   - *Why it matters:* Right now sculpting bangs (`H03C`) + front hair highlights + hairpin (`HP01`) means sculpting each layer separately. A `MeshGroup` (or Studio parent warp cage that propagates its lattice displacement `D` onto all child layers cleanly!) lets riggers sculpt an entire hair cluster or eye group in one stroke — just like Cubism Warp Deformers & Inochi2D `MeshGroup`!
7. **Path Deformers in Studio (Spline-Guided Eyebrows, Mouths & Hair Strands)**
   - *Why it matters:* Letting a rigger draw a 4-point Bézier spline along a hair lock or eyebrow and bend the spline handles to automatically populate the inverse warp grid `D` makes curved expressions and hair turns effortless!
8. **DynamicComposite (Render-to-Texture Folder Deformation)**
   - *Why it matters:* Deforming an entire PSD folder (e.g., whole face + eyes + mouth + blush) as a single warped FBO layer — ideal for squash-and-stretch chibi bounce, smear frames, or unified head tilt without inter-layer seams.
9. **Vertex / Seam Welding (`weld`)**
   - *Why it matters:* Binding seam vertices between upper arm/shoulder or neck/torso prevents micro-gaps during extreme poses.
10. **Standardized External Tracking Protocols (VMC / OSC / VTube Studio API WebSocket)**
    - *Why it matters:* Inochi Session's plugin tracking architecture accepts iPhone ARKit / VMC / OpenSeeFace streams over WebSocket/OSC. Adding a lightweight WebSocket receiver option to Layer2D Player lets iPhone ARKit users drive 52-blendshape tracking right in OBS Browser Source alongside built-in webcam + mic!

---

## 3. Where Layer2D Already Has Distinct Advantages Over Inochi2D

- **Zero-install browser runtime & browser Studio:** Inochi2D requires native desktop binaries (D + OpenGL / ImGui). Layer2D runs natively in any WebGL2 browser and drops straight into OBS Browser Source with zero install.
- **Standard ZIP + cryptographic SHA-256 integrity (`SHA256SUMS.json`):** Easy for any script, CI pipeline, or AI agent to inspect, diff, pack, and validate using standard ZIP libraries — whereas `.inp` is a custom big-endian binary format.
- **Smooth inverse-grid GPU deformation (`D`) + analytical 3D head-cylinder math:** Eliminates jagged low-poly triangle silhouettes on high-res 4K anime lineart and automatically gives continuous parallax turns from far fewer manual mesh vertices.
- **Built-in Web Audio AEIOU lip-sync, multi-depth zero-pop hair architecture, and rigid pelvis-locked tails** out of the box.
