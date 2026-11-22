"""Standalone OpenAI Image API wrapper for layer2d with cost ledger tracking.

Uses OPENAI_API_KEY from environment (or an optional file specified by LAYER2D_SECRETS_FILE).
Never prints or logs API keys.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Sequence

import httpx

API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = os.environ.get("LAYER2D_IMAGE_MODEL", "gpt-image-1")

USD_TEXT_IN = 5.0 / 1e6
USD_IMAGE_IN = 8.0 / 1e6
USD_IMAGE_OUT = 30.0 / 1e6


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    secrets_env = os.environ.get("LAYER2D_SECRETS_FILE", "").strip()
    if secrets_env and Path(secrets_env).exists():
        for line in Path(secrets_env).read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)$", line)
            if m and m.group(1).strip().strip("'\""):
                return m.group(1).strip().strip("'\"")
    raise RuntimeError("OPENAI_API_KEY is not set in environment")


def cost_usd(usage: dict | None) -> float:
    u = usage or {}
    det = u.get("input_tokens_details") or {}
    img = det.get("image_tokens", 0) or 0
    txt = det.get("text_tokens", (u.get("input_tokens", 0) or 0) - img) or 0
    return txt * USD_TEXT_IN + img * USD_IMAGE_IN + (u.get("output_tokens", 0) or 0) * USD_IMAGE_OUT


class ImageClient:
    def __init__(self, ledger_path: Path | None = None, model: str = DEFAULT_MODEL, timeout: float = 600.0):
        self.model = model
        self.ledger_path = ledger_path
        self.http = httpx.Client(timeout=timeout)
        self._key = get_api_key()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"}

    def _log(self, rec: dict) -> None:
        if not self.ledger_path:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def generate(
        self,
        prompt: str,
        out_path: Path,
        *,
        size: str = "1024x1536",
        quality: str = "high",
        background: str = "opaque",
        model: str | None = None,
        tag: str = "",
        retries: int = 2,
    ) -> dict:
        """Generate an image from text only (/images/generations)."""
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": 1,
            "output_format": "png",
            "background": background,
        }
        t0 = time.time()
        last_err = None
        for attempt in range(retries + 1):
            r = self.http.post(f"{API_BASE}/images/generations", headers=self._headers(), json=payload)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(15 * (attempt + 1))
                continue
            if not r.is_success:
                last_err = f"HTTP {r.status_code}: {r.text[:400]}"
                raise RuntimeError(last_err)
            j = r.json()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(base64.b64decode(j["data"][0]["b64_json"]))
            usage = j.get("usage") or {}
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "op": "generate",
                "tag": tag or out_path.stem,
                "model": payload["model"],
                "size": size,
                "quality": quality,
                "background": background,
                "out": str(out_path),
                "seconds": round(time.time() - t0, 1),
                "cost_usd": round(cost_usd(usage), 4),
                "usage": usage,
            }
            self._log(rec)
            return rec
        raise RuntimeError(last_err or "generate failed")

    def edit(
        self,
        prompt: str,
        ref_paths: Sequence[Path],
        out_path: Path,
        *,
        size: str = "3840x2160",
        quality: str = "xhigh",
        background: str = "transparent",
        model: str | None = None,
        tag: str = "",
        retries: int = 2,
    ) -> dict:
        """Edit / generate from reference images (/images/edits)."""
        fields = {
            "model": model or self.model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": "1",
            "output_format": "png",
            "background": background,
        }
        files = [("image[]", (Path(p).name, Path(p).read_bytes(), "image/png")) for p in ref_paths]
        t0 = time.time()
        last_err = None
        for attempt in range(retries + 1):
            r = self.http.post(f"{API_BASE}/images/edits", headers=self._headers(), data=fields, files=files)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(15 * (attempt + 1))
                continue
            if not r.is_success:
                last_err = f"HTTP {r.status_code}: {r.text[:400]}"
                raise RuntimeError(last_err)
            j = r.json()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(base64.b64decode(j["data"][0]["b64_json"]))
            usage = j.get("usage") or {}
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "op": "edit",
                "tag": tag or out_path.stem,
                "model": fields["model"],
                "refs": [str(p) for p in ref_paths],
                "size": size,
                "quality": quality,
                "background": background,
                "out": str(out_path),
                "seconds": round(time.time() - t0, 1),
                "cost_usd": round(cost_usd(usage), 4),
                "usage": usage,
            }
            self._log(rec)
            return rec
        raise RuntimeError(last_err or "edit failed")
