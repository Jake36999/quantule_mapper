#!/usr/bin/env python
"""Run gallery + viewer (local web page).

    .venv/Scripts/python.exe tools/serve_viewer.py [--port 8766]   ->  http://127.0.0.1:8766/

GALLERY  every run in the results index (docs/runs/_index.sqlite) as a card: thumbnail, badges
         (current / revalidated / stale / unaffected), what it can show (final fields, recorded
         history), filters by substrate and branch.
VIEWER   one tab per measured field; 2-D centre-plane slices and a 3-D isosurface of any quantity
         (|psi|^2, |psi|, phase, Re, Im); every time series (er(t), telemetry) with tolerance lines and
         SUGGESTED WINDOWS where the dynamics change fastest; recorded histories play frame by frame.
RE-RUN   for a run whose spec is known (spec.json, or a reconstructable a* probe cell), choose the whole
         run at a low frame rate or a selected time window at a high frame rate, see the cost estimate,
         and QUEUE it. Queuing writes specs/queue/<job>.json and nothing else. Jobs run only when a person
         starts `tools/render_queue_worker.py` -- the page has no control path into a simulation (the
         HUD rule; agents still cannot launch anything).

Bound to 127.0.0.1. Standard library only on the server; the page loads plotly from a CDN.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import viewer_data as vd  # noqa: E402

PAGE = os.path.join(ROOT, "web", "viewer", "index.html")
STATIC_ROOTS = (os.path.join(ROOT, "docs"), os.path.join(ROOT, "sweep_runs"))


class Handler(BaseHTTPRequestHandler):
    server_version = "irer-viewer/1"

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass

    def _vol(self, hdr, raw):
        return self._send(200, {**hdr, "data": base64.b64encode(raw).decode("ascii")})

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
            if u.path.startswith("/files/"):
                rel = u.path[len("/files/"):]
                p = os.path.abspath(os.path.join(ROOT, rel))
                if not p.lower().endswith((".png", ".gif", ".jpg")) or not any(vd._inside(p, r) for r in STATIC_ROOTS) \
                        or not os.path.exists(p):
                    return self._send(404, {"status": "NOT_FOUND"})
                return self._send(200, open(p, "rb").read(), mimetypes.guess_type(p)[0] or "application/octet-stream")
            if u.path == "/api/runs":
                return self._send(200, vd.list_runs(q.get("substrate"), q.get("branch"), q.get("q")))
            if u.path == "/api/run":
                return self._send(200, vd.run_detail(q["id"]))
            if u.path == "/api/field":
                return self._vol(*vd.field_volume(q["id"], q["file"], q["key"], q.get("quantity", "abs2"),
                                                  int(q.get("n", 48)), int(q.get("t", -1))))
            if u.path == "/api/series":
                return self._send(200, vd.time_series(q["id"]))
            if u.path == "/api/group_series":
                return self._send(200, vd.group_series(q["id"], q["group"], q.get("quantity", "abs2")))
            if u.path == "/api/history":
                return self._send(200, vd.history_index(q["id"]))
            if u.path == "/api/frame":
                return self._vol(*vd.history_frame(q["id"], q["i"], q["field"], q.get("quantity", "abs2"),
                                                   q.get("kind", "vol")))
            if u.path == "/api/queue":
                return self._send(200, vd.queue_status())
        except (KeyError, ValueError, IndexError, FileNotFoundError) as exc:
            return self._send(400, {"status": "BAD_REQUEST", "error": str(exc)[:300]})
        return self._send(404, {"status": "NOT_FOUND"})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(n, 2_000_000)) or b"{}")
        except ValueError:
            return self._send(400, {"status": "BAD_JSON"})
        try:
            if u.path == "/api/rerun_preview":
                r = vd.rerun_spec(body["id"], body.get("cell"), record=body.get("record"))
                if r["status"] == "OK":
                    r["estimate"] = vd.estimate(r["spec"])
                return self._send(200, r)
            if u.path == "/api/queue":
                return self._send(200, vd.queue_render(body["spec"]))
        except (KeyError, ValueError, FileNotFoundError) as exc:
            return self._send(400, {"status": "BAD_REQUEST", "error": str(exc)[:300]})
        return self._send(404, {"status": "NOT_FOUND"})


def make_server(port=8766):
    return ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    srv = make_server(a.port)
    print("run viewer on http://127.0.0.1:%d/  (queues re-runs; never runs them)" % a.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
