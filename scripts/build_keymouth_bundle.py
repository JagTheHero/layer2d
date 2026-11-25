#!/usr/bin/env python3
"""Build self-contained Layer2D WebGL2 Key-Shape Mouth module from assets/mouth_keyshape/.

Assets adapted from Keshin2D (https://github.com/rotejin/Keshin2D, MIT License).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "mouth_keyshape"
KEYS = [
    "closed", "a", "i", "u", "e", "o", "small", "laugh", "he",
    "omega", "wavy", "grin", "tri", "round", "smug", "pout", "square",
]

RUNTIME_JS = r'''/* Layer2D Key-Shape Contour-Warped Mouth Engine (WebGL2)
 * Adapted from Keshin2D's keyMouth algorithm & MIT-licensed mouth key assets
 * (https://github.com/rotejin/Keshin2D).
 *
 * Features:
 *  - 17 drawn anime mouth keys with 33-point U/Lo silhouette contours
 *  - Stroke-preserving 27x7 UV warp mesh (constant outline weight during stretch)
 *  - Decoupled geometry vs texture blending (smoothstep open switch + mouthKeySharp)
 *  - Premultiplied additive offscreen FBO (zero skin bleed-through on crossfades)
 *  - Continuous driving from LPC phoneme probabilities, mic volume, webcam blendshapes, & face expressions
 */
(function (root) {
  "use strict";

  const NC = 24, MR = 7, PADX = 9, PADY = 7, STROKE = 6;
  const OUT_W = 320, OUT_H = 256, OUT_CX = 160, OUT_CY = 108, REF_CLOSED_W = 160;
  const VOWEL_KEYS = ["a", "i", "u", "e", "o"];

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const sstep = (a, b, v) => { const t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

  const KM_VS = `#version 300 es
  in vec2 aPos;
  in vec2 aUV;
  out vec2 vUV;
  uniform vec4 uView;
  void main() {
    // Rendering into an offscreen FBO sampled by Layer2D (where tc.y=0 is top of image):
    // NDC y=-1 maps to tc.y=0, so positive (aPos.y - uView.y) puts smaller Y (top) at tc.y=0.
    vec2 p = vec2((aPos.x - uView.x) * uView.z, (aPos.y - uView.y) * uView.w);
    vUV = aUV;
    gl_Position = vec4(p, 0.0, 1.0);
  }`;

  const KM_FS = `#version 300 es
  precision highp float;
  in vec2 vUV;
  out vec4 oCol;
  uniform sampler2D uTex;
  uniform float uAlpha;
  void main() {
    oCol = texture(uTex, vUV) * uAlpha;
  }`;

  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error("KeyMouth shader error: " + gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  function mapFaceNameToMouth(faceName) {
    if (!faceName) return { closed: 1 };
    const s = String(faceName).toLowerCase();
    if (/laugh|giggle|ecstatic|joy|cheer/.test(s)) return { laugh: 1 };
    if (/grin|happy|bright|star|spark|bliss/.test(s)) return { grin: 0.8, closed: 0.2 };
    if (/smile|wink|warm|gentle/.test(s)) return { closed: 0.85, smug: 0.15 };
    if (/smug|tease|mischief|playful|wink/.test(s)) return { smug: 1 };
    if (/cat|uwu|omega|nya|cute|blush|shy/.test(s)) return { omega: 1 };
    if (/pout|sulky|tsun|hmph|puff/.test(s)) return { pout: 1 };
    if (/angry|mad|furious|glare|annoy|scowl|disgust/.test(s)) return { he: 1 };
    if (/sad|cry|tear|worry|nervous|sweat|dizzy|pale|scare|fear/.test(s)) return { wavy: 1 };
    if (/shock|surpris|gasp|wow|stun|panic/.test(s)) return { round: 1 };
    if (/awkward|blank|eh|what/.test(s)) return { square: 1 };
    if (/excited|yay|sing|hype/.test(s)) return { tri: 1 };
    return { closed: 1 };
  }

  function createKeyMouth(gl) {
    const prog = gl.createProgram();
    gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER, KM_VS));
    gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, KM_FS));
    gl.bindAttribLocation(prog, 0, "aPos");
    gl.bindAttribLocation(prog, 1, "aUV");
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error("KeyMouth link error: " + gl.getProgramInfoLog(prog));
    }
    const uView = gl.getUniformLocation(prog, "uView");
    const uTex = gl.getUniformLocation(prog, "uTex");
    const uAlpha = gl.getUniformLocation(prog, "uAlpha");

    // Output FBO & premultiplied RGBA texture (320x256)
    const prevFbInit = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    const prevTexInit = gl.getParameter(gl.TEXTURE_BINDING_2D);
    const outTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, outTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, OUT_W, OUT_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const outFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, outFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbInit);
    gl.bindTexture(gl.TEXTURE_2D, prevTexInit);

    const sets = Object.create(null);
    let activePreset = null;
    let volEnv = 0;

    function buildPreset(presetName) {
      const src = PRESETS[presetName];
      if (!src) return null;
      const prevVaoBuild = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
      const prevBufBuild = gl.getParameter(gl.ARRAY_BUFFER_BINDING);
      const prevTexBuild = gl.getParameter(gl.TEXTURE_BINDING_2D);
      const K = Object.create(null);
      const names = [];
      const pending = [];
      let loadedCount = 0;
      const totalCount = Object.keys(src.keys).length;

      for (const nm of Object.keys(src.keys)) {
        const km = src.keys[nm];
        const N = km.U.length;
        const U = Float32Array.from(km.U);
        const Lo = Float32Array.from(km.Lo);
        const cols = NC + 3;
        const nv = cols * MR;
        const pos = new Float32Array(nv * 2);
        const uv = new Float32Array(nv * 2);
        const uu = new Float32Array(cols);
        const idx = [];

        for (let c = 0; c < cols; c++) {
          const u = clamp((c - 1) / NC, 0, 1);
          const f = u * (N - 1);
          const i0 = Math.min(Math.floor(f), N - 2);
          const t = f - i0;
          const x = c === 0 ? km.xa - PADX : c === cols - 1 ? km.xb + PADX : km.xa + (km.xb - km.xa) * u;
          const Uy = lerp(U[i0], U[i0 + 1], t);
          const Ly = Math.max(lerp(Lo[i0], Lo[i0 + 1], t), Uy + 1);
          const sw = Math.min(STROKE, (Ly - Uy) / 2);
          const ys = [Uy - PADY, Uy, Uy + sw, (Uy + Ly) / 2, Ly - sw, Ly, Ly + PADY];
          uu[c] = c === 0 ? -1 : c === cols - 1 ? 2 : u;
          for (let r = 0; r < MR; r++) {
            const v = c * MR + r;
            pos[v * 2] = x;
            pos[v * 2 + 1] = ys[r];
            uv[v * 2] = (x - km.x) / km.w;
            uv[v * 2 + 1] = (ys[r] - km.y) / km.h;
            if (c < cols - 1 && r < MR - 1) {
              idx.push(v, v + MR, v + 1, v + 1, v + MR, v + MR + 1);
            }
          }
        }

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const pb = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, pb);
        gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        const ub = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, ub);
        gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
        const ib = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx), gl.STATIC_DRAW);
        gl.bindVertexArray(null);

        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        const entry = {
          nm, km, xa: km.xa, xb: km.xb, U, Lo, N, open: !!km.open,
          cols, uu, pos, vao, pb, ub, ib, count: idx.length, tex, wi: 0, ready: false,
        };
        K[nm] = entry;
        names.push(nm);

        const im = new Image();
        im.onload = () => {
          const prevFlip = gl.getParameter(gl.UNPACK_FLIP_Y_WEBGL);
          const prevPre = gl.getParameter(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL);
          const prevTex = gl.getParameter(gl.TEXTURE_BINDING_2D);
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
          gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, im);
          gl.generateMipmap(gl.TEXTURE_2D);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, prevFlip);
          gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, prevPre);
          gl.bindTexture(gl.TEXTURE_2D, prevTex);
          entry.ready = true;
          loadedCount++;
          setObj.ready = loadedCount >= totalCount;
        };
        pending.push([im, "data:image/png;base64," + km.png]);
      }

      gl.bindVertexArray(prevVaoBuild);
      gl.bindBuffer(gl.ARRAY_BUFFER, prevBufBuild);
      gl.bindTexture(gl.TEXTURE_2D, prevTexBuild);

      const N = K.closed.N;
      const W = Object.create(null);
      const exW = Object.create(null);
      for (const nm of names) {
        W[nm] = nm === "closed" ? 1 : 0;
        exW[nm] = nm === "closed" ? 1 : 0;
      }
      const setObj = {
        name: presetName,
        sharp: src.sharp || 0,
        ref: src.ref,
        K, names, N, W, exW,
        cur: { xa: K.closed.xa, xb: K.closed.xb, U: new Float32Array(N), Lo: new Float32Array(N) },
        ready: false,
      };
      sets[presetName] = setObj;
      for (const [im, dataUrl] of pending) im.src = dataUrl;
      return setObj;
    }

    function ensurePreset(mode) {
      const pname = mode === "keyshape_fang" ? "chiwa" : "kurage02";
      if (!sets[pname]) buildPreset(pname);
      activePreset = sets[pname];
      return activePreset;
    }

    function stepAndDraw(mode, input, dt) {
      const S = ensurePreset(mode);
      if (!S || !S.ready) return null;
      const safeDt = clamp(dt || 0.016, 0.001, 0.1);
      const { K, names, N, W, exW, cur } = S;

      // 1. Base expression weights (smoothed at 13/s)
      const em = (input && input.exprWeights) || mapFaceNameToMouth(input && input.faceName);
      const ke = 1 - Math.exp(-safeDt * 13);
      let se = 0;
      const kmT = Object.create(null);
      for (const nm of names) {
        exW[nm] += ((em[nm] || 0) - exW[nm]) * ke;
        se += exW[nm];
      }
      for (const nm of names) kmT[nm] = exW[nm] / (se || 1);

      // 2. Live speech (continuous LPC probs OR discrete vowel OR webcam blendshapes)
      const hasProbs = input && input.probs && typeof input.probs === "object";
      const rawVol = clamp((input && typeof input.volume === "number") ? input.volume : 0, 0, 1);
      const envTc = rawVol > volEnv ? 0.03 : 0.075;
      volEnv += (rawVol - volEnv) * (1 - Math.exp(-safeDt / envTc));

      let speechApplied = false;
      if (hasProbs && input.open) {
        const p = input.probs;
        const wA = p.A || 0, wI = p.I || 0, wU = p.U || 0, wE = p.E || 0, wO = p.O || 0;
        const vs = wA + wI + wU + wE + wO;
        const v = clamp(Math.pow(Math.max(volEnv, 0.35), 0.55) * (1 - (p.X || 0) * 0.85), 0, 1);
        if (v > 0.02 && vs > 0.01) {
          for (const nm of names) kmT[nm] *= 1 - v;
          kmT.a = (kmT.a || 0) + v * (wA / vs);
          kmT.i = (kmT.i || 0) + v * (wI / vs);
          kmT.u = (kmT.u || 0) + v * (wU / vs);
          kmT.e = (kmT.e || 0) + v * (wE / vs);
          kmT.o = (kmT.o || 0) + v * (wO / vs);
          speechApplied = true;
        }
      } else if (input && input.vowel) {
        const vk = String(input.vowel).toLowerCase();
        if (K[vk]) {
          const v = input.volume !== undefined ? clamp(Math.pow(Math.max(volEnv, 0.55), 0.55), 0, 1) : 1.0;
          for (const nm of names) kmT[nm] *= 1 - v;
          kmT[vk] = (kmT[vk] || 0) + v;
          speechApplied = true;
        }
      }

      if (!speechApplied && input && input.cam) {
        const c = input.cam;
        const jaw = clamp((c.jaw || 0) * 1.65, 0, 1);
        const funnel = clamp(c.funnel || 0, 0, 1);
        const pucker = clamp(c.pucker || 0, 0, 1);
        const smile = clamp(c.smile || 0, 0, 1);
        const frown = clamp(c.frown || 0, 0, 1);
        if (jaw > 0.05 || funnel > 0.2 || pucker > 0.28) {
          let wA = jaw, wO = funnel * 1.1, wU = pucker * 1.25, wI = smile * Math.max(0, 0.65 - jaw) * 1.2;
          let wE = (c.stretch || 0) * 0.9;
          const vs = wA + wO + wU + wI + wE;
          const v = clamp(Math.max(jaw, funnel * 0.85, pucker * 0.85), 0, 1);
          if (vs > 0.01 && v > 0.04) {
            for (const nm of names) kmT[nm] *= 1 - v;
            kmT.a = (kmT.a || 0) + v * (wA / vs);
            kmT.o = (kmT.o || 0) + v * (wO / vs);
            kmT.u = (kmT.u || 0) + v * (wU / vs);
            kmT.i = (kmT.i || 0) + v * (wI / vs);
            kmT.e = (kmT.e || 0) + v * (wE / vs);
            speechApplied = true;
          }
        } else if (frown > 0.25 && K.he) {
          const nf = sstep(0.2, 0.7, frown);
          for (const nm of names) kmT[nm] *= 1 - nf;
          kmT.he = (kmT.he || 0) + nf;
        } else if (smile > 0.45 && K.grin) {
          const ns = sstep(0.4, 0.85, smile) * 0.55;
          for (const nm of names) kmT[nm] *= 1 - ns;
          kmT.grin = (kmT.grin || 0) + ns;
        }
      }

      if (!speechApplied && input && typeof input.mouthOpen === "number" && input.mouthOpen > 0.01) {
        const O = clamp(input.mouthOpen, 0, 1);
        const wS = O < 0.4 ? O / 0.4 : O < 0.75 ? 1 - (O - 0.4) / 0.35 : 0;
        const wA = O < 0.4 ? 0 : O < 0.75 ? (O - 0.4) / 0.35 : 1 - (O - 0.75) / 0.25;
        const wL = O > 0.75 ? (O - 0.75) / 0.25 : 0;
        const tot = Math.min(wS + wA + wL, 1);
        for (const nm of names) kmT[nm] *= 1 - tot;
        kmT.small = (kmT.small || 0) + wS;
        kmT.a = (kmT.a || 0) + wA;
        kmT.laugh = (kmT.laugh || 0) + wL;
      }

      // 3. Smoothly step weights W[nm] (30/s) and normalize
      let sTarget = 0;
      for (const nm of names) sTarget += Math.max(kmT[nm] || 0, 0);
      const kw = 1 - Math.exp(-safeDt * 30);
      let sw = 0;
      for (const nm of names) {
        const tg = sTarget > 1e-4 ? Math.max(kmT[nm] || 0, 0) / sTarget : (nm === "closed" ? 1 : 0);
        W[nm] += (tg - W[nm]) * kw;
        sw += W[nm];
      }
      for (const nm of names) W[nm] /= sw || 1;

      // 4. Linearly blend contour geometry (xa, xb, U[33], Lo[33])
      cur.xa = 0; cur.xb = 0; cur.U.fill(0); cur.Lo.fill(0);
      let wo = 0, wl = 0;
      for (const nm of names) {
        const w = W[nm], q = K[nm];
        if (w < 1e-4) continue;
        cur.xa += w * q.xa;
        cur.xb += w * q.xb;
        for (let i = 0; i < N; i++) {
          cur.U[i] += w * q.U[i];
          cur.Lo[i] += w * q.Lo[i];
        }
        if (q.open) wo += w; else wl += w;
      }

      // 5. Decoupled texture weights: open/closed smoothstep + mouthKeySharp
      const fOpen = sstep(0.03, 0.20, wo);
      for (const nm of names) {
        const q = K[nm];
        q.wi = q.open ? (wo > 1e-4 ? (W[nm] / wo) * fOpen : 0) : (wl > 1e-4 ? (W[nm] / wl) * (1 - fOpen) : 0);
      }
      if (wl <= 1e-4 && wo > 1e-4) for (const nm of names) if (K[nm].open) K[nm].wi = W[nm] / wo;
      if (wo <= 1e-4) for (const nm of names) if (!K[nm].open) K[nm].wi = W[nm] / (wl || 1);
      if (S.sharp > 0) {
        let so = 0, sp = 0;
        for (const nm of names) {
          const q = K[nm];
          if (q.open && q.wi > 0) { so += q.wi; sp += Math.pow(q.wi, S.sharp); }
        }
        if (sp > 0) {
          for (const nm of names) {
            const q = K[nm];
            if (q.open && q.wi > 0) q.wi = so * Math.pow(q.wi, S.sharp) / sp;
          }
        }
      }

      // 6. Render into offscreen premultiplied FBO (ONE, ONE additive blend -> zero skin bleed)
      const prevFbo = gl.getParameter(gl.FRAMEBUFFER_BINDING);
      const prevProg = gl.getParameter(gl.CURRENT_PROGRAM);
      const prevVp = gl.getParameter(gl.VIEWPORT);
      const prevVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
      const prevBlendSrcRgb = gl.getParameter(gl.BLEND_SRC_RGB);
      const prevBlendDstRgb = gl.getParameter(gl.BLEND_DST_RGB);
      const prevBlendSrcA = gl.getParameter(gl.BLEND_SRC_ALPHA);
      const prevBlendDstA = gl.getParameter(gl.BLEND_DST_ALPHA);
      const prevActive = gl.getParameter(gl.ACTIVE_TEXTURE);

      const scaleToOut = REF_CLOSED_W / Math.max(1, S.ref.w);
      const halfW = (OUT_W * 0.5) / scaleToOut;
      const halfH = (OUT_H * 0.5) / scaleToOut;
      const viewCx = S.ref.cx;
      const viewCy = S.ref.cy + (OUT_H * 0.5 - OUT_CY) / scaleToOut;

      gl.bindFramebuffer(gl.FRAMEBUFFER, outFbo);
      gl.viewport(0, 0, OUT_W, OUT_H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(prog);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.uniform4f(uView, viewCx, viewCy, 1 / halfW, 1 / halfH);
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1i(uTex, 0);

      for (const nm of names) {
        const q = K[nm];
        if (q.wi <= 0.004 || !q.ready) continue;
        const d = q.pos;
        for (let c = 0; c < q.cols; c++) {
          const u = clamp(q.uu[c], 0, 1);
          const ff = u * (N - 1);
          const i0 = Math.min(Math.floor(ff), N - 2);
          const t = ff - i0;
          const x = q.uu[c] < 0 ? cur.xa - PADX : q.uu[c] > 1 ? cur.xb + PADX : cur.xa + (cur.xb - cur.xa) * u;
          const Uy = lerp(cur.U[i0], cur.U[i0 + 1], t);
          const Ly = Math.max(lerp(cur.Lo[i0], cur.Lo[i0 + 1], t), Uy + 1);
          const swBand = Math.min(STROKE, (Ly - Uy) / 2);
          const ys = [Uy - PADY, Uy, Uy + swBand, (Uy + Ly) / 2, Ly - swBand, Ly, Ly + PADY];
          for (let r = 0; r < MR; r++) {
            const v = c * MR + r;
            d[v * 2] = x;
            d[v * 2 + 1] = ys[r];
          }
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, q.pb);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, d);
        gl.uniform1f(uAlpha, q.wi);
        gl.bindTexture(gl.TEXTURE_2D, q.tex);
        gl.bindVertexArray(q.vao);
        gl.drawElements(gl.TRIANGLES, q.count, gl.UNSIGNED_SHORT, 0);
      }

      // Restore caller WebGL2 state cleanly
      gl.bindVertexArray(prevVao);
      gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbo);
      gl.viewport(prevVp[0], prevVp[1], prevVp[2], prevVp[3]);
      gl.useProgram(prevProg);
      gl.blendFuncSeparate(prevBlendSrcRgb, prevBlendDstRgb, prevBlendSrcA, prevBlendDstA);
      gl.activeTexture(prevActive);
      return outTex;
    }

    function makePiece(m01Layer, id) {
      const mw = Math.max(24, (m01Layer && m01Layer.w) || 90);
      // M01 textures include transparent padding around the drawn mouth (~34%), so target ~0.66 * M01.w
      const scale = REF_CLOSED_W / (mw * 0.66);
      // Align the closed-mouth center (OUT_CX, OUT_CY) onto M01's center (0, 0).
      const relTy = OUT_CY - OUT_H * 0.5;
      return {
        id: id || "__keymouth__",
        base: (m01Layer && m01Layer.id) || "M01",
        w: OUT_W,
        h: OUT_H,
        relA: [scale, 0, 0, scale],
        relT: [0, relTy],
        op: 1,
        reveal: 1,
      };
    }

    function dispose() {
      gl.deleteTexture(outTex);
      gl.deleteFramebuffer(outFbo);
      gl.deleteProgram(prog);
      for (const s of Object.values(sets)) {
        for (const q of Object.values(s.K)) {
          gl.deleteTexture(q.tex);
          gl.deleteBuffer(q.pb);
          gl.deleteBuffer(q.ub);
          gl.deleteBuffer(q.ib);
          gl.deleteVertexArray(q.vao);
        }
      }
    }

    return {
      outTex,
      ensurePreset,
      stepAndDraw,
      makePiece,
      dispose,
      isReady: (mode) => {
        const pname = mode === "keyshape_fang" ? "chiwa" : "kurage02";
        return !!(sets[pname] && sets[pname].ready);
      },
    };
  }

  const api = {
    MODES: ["pack", "keyshape", "keyshape_fang"],
    LABELS: { pack: "Mouth: Pack", keyshape: "Mouth: Keyshape", keyshape_fang: "Mouth: Fang" },
    normalizeMode(m) {
      const s = String(m || "").trim().toLowerCase();
      return api.MODES.includes(s) ? s : "pack";
    },
    createKeyMouth,
    mapFaceNameToMouth,
    PRESETS,
  };

  root.Layer2DKeyMouth = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
'''

def main() -> None:
    presets = {}
    for model_name in ("kurage02", "chiwa"):
        d = ASSETS / model_name
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        out_keys = {}
        for k in KEYS:
            km = dict(meta["keys"][k])
            km["png"] = base64.b64encode((d / f"{k}.png").read_bytes()).decode("ascii")
            out_keys[k] = km
        presets[model_name] = {
            "source": meta["source"],
            "sharp": meta["sharp"],
            "ref": meta["ref"],
            "keys": out_keys,
        }
    presets_json = json.dumps(presets, separators=(",", ":"))
    bundle = RUNTIME_JS.replace(
        "  const VOWEL_KEYS",
        f"  const PRESETS = {presets_json};\n\n  const VOWEL_KEYS",
        1,
    )
    targets = [
        ROOT / "viewer" / "mouth_keyshape.js",
        ROOT / "site" / "player" / "mouth_keyshape.js",
        ROOT / "site" / "studio" / "js" / "mouth_keyshape.js",
    ]
    for t in targets:
        t.write_text(bundle, encoding="utf-8")
        print(f"Wrote {t} ({len(bundle) / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
