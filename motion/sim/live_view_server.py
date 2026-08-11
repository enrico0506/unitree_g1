#!/usr/bin/env python3
"""Serve the live sim2sim render as an MJPEG stream in a browser tab.

Reads motion/sim/live/frame.jpg (written in real time by
eval_mujoco_sim2sim.py's live-view hook during any headless run) and serves
it as a classic MJPEG multipart stream -- browsers render that natively in
an <img> tag, no JS/websockets/player needed. Standalone, no dependencies
beyond the stdlib, runs on the HOST (not in the container) since it just
reads a file that's already bind-mounted out.

Usage:
    python motion/sim/live_view_server.py [--port 8098]

run_holomotion.sh / app.sh start this automatically if it isn't already
running -- you normally don't need to launch it by hand.
"""

from __future__ import annotations

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAME_PATH = HERE / "live" / "frame.jpg"

PLACEHOLDER_JPEG_HTML = """
<!doctype html>
<html>
<head>
<title>motion/sim -- live view</title>
<style>
  body { margin: 0; background: #111; color: #ddd; font-family: system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; }
  h1 { font-size: 1rem; font-weight: 500; color: #999; margin: 1rem; }
  img { max-width: 100vw; max-height: 90vh; }
</style>
</head>
<body>
<h1>motion/sim -- live view (auto-refreshes while a run is active)</h1>
<img src="/stream">
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout quiet -- this runs unattended in the background

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = PLACEHOLDER_JPEG_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            last_mtime = None
            try:
                while True:
                    if FRAME_PATH.is_file():
                        mtime = FRAME_PATH.stat().st_mtime
                        if mtime != last_mtime:
                            try:
                                data = FRAME_PATH.read_bytes()
                            except OSError:
                                time.sleep(0.05)
                                continue
                            last_mtime = mtime
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(data)}\r\n\r\n".encode()
                            )
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                    time.sleep(0.05)  # ~20Hz poll, matches typical sim frame rate
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_response(404)
        self.end_headers()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8098)
    args = ap.parse_args()

    FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"motion/sim live view at http://<jetson-ip>:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
