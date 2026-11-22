"""Simple local HTTP server for testing a layer2d model's WebGL2 viewer."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("viewer_dir", type=Path)
    ap.add_argument("--port", type=int, default=7880)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    handler = partial(QuietHandler, directory=str(a.viewer_dir.resolve()))
    srv = ThreadingHTTPServer((a.host, a.port), handler)
    print(f"Serving {a.viewer_dir} at http://{a.host}:{a.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
