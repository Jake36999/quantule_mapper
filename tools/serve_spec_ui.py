#!/usr/bin/env python
"""Local spec editor (IMPLEMENTATION_PLAN_2026-10 Phase E3).

    python tools/serve_spec_ui.py [--port 8765]      ->  http://127.0.0.1:8765/

A schema-driven form over irer_specs.SCHEMA (react-jsonschema-form in web/spec_editor/index.html), so the
interface adapts to the schema instead of being hand-built per experiment.

WHAT IT CAN DO: list specs (drafts / proposed / approved), load one, edit it in the form or as raw JSON,
validate it (schema + registered component names), save it to specs/drafts/, and promote a draft or an
agent's proposal to specs/approved/.

WHAT IT CANNOT DO: launch a run. Same rule as tools/hud_monitor.py -- no control path into the
simulation. For an approved spec the page shows the `tools/run_spec.py` command to copy.

Bound to 127.0.0.1 only. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import irer_specs  # noqa: E402
from mcp_server import research_tools as rt  # noqa: E402

UI_FILE = os.path.join(ROOT, "web", "spec_editor", "index.html")
FOLDERS = ("drafts", "proposed", "approved")


def spec_path(folder, sid):
    return os.path.join(ROOT, "specs", folder, sid + ".json")


def full_validate(spec):
    errs = irer_specs.validate(spec)
    if errs:
        return errs
    comps = rt.list_components(ROOT)
    if comps.get("status") == "OK":
        known = {k: {r["name"] for r in comps[k]} for k in ("substrates", "ics", "observers")}
        if spec["substrate"]["name"] not in known["substrates"]:
            errs.append("unknown substrate '%s'" % spec["substrate"]["name"])
        if spec["protocol"]["ic"]["name"] not in known["ics"]:
            errs.append("unknown ic '%s'" % spec["protocol"]["ic"]["name"])
        errs += ["unknown observer '%s'" % o["name"] for o in spec["observers"] if o["name"] not in known["observers"]]
    return errs


def save_draft(spec):
    errs = full_validate(spec)
    if errs:
        return 400, {"status": "INVALID", "errors": errs}
    sid = spec["id"]
    for f in ("proposed", "approved"):
        if os.path.exists(spec_path(f, sid)):
            return 409, {"status": "EXISTS", "errors": ["id '%s' is already in specs/%s/; choose a new id" % (sid, f)]}
    os.makedirs(os.path.dirname(spec_path("drafts", sid)), exist_ok=True)
    with open(spec_path("drafts", sid), "w", encoding="utf-8") as fh:      # drafts may be re-saved
        json.dump(spec, fh, indent=2)
        fh.write("\n")
    return 200, {"status": "SAVED", "path": "specs/drafts/%s.json" % sid}


def promote(sid, src):
    if src not in ("drafts", "proposed") or not rt._safe_id(sid):
        return 400, {"status": "BAD_INPUT"}
    p = spec_path(src, sid)
    if not os.path.exists(p):
        return 404, {"status": "NOT_FOUND"}
    spec = irer_specs.load(p)
    errs = full_validate(spec)
    if errs:
        return 400, {"status": "INVALID", "errors": errs}
    if os.path.exists(spec_path("approved", sid)):
        return 409, {"status": "EXISTS", "errors": ["already approved"]}
    os.makedirs(os.path.dirname(spec_path("approved", sid)), exist_ok=True)
    shutil.move(p, spec_path("approved", sid))
    return 200, {"status": "APPROVED", "path": "specs/approved/%s.json" % sid,
                 "run_command": "python tools/run_spec.py specs/approved/%s.json" % sid}


class Handler(BaseHTTPRequestHandler):
    server_version = "irer-spec-ui/1"

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):            # quiet
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, open(UI_FILE, "rb").read(), "text/html")
        if u.path == "/api/schema":
            return self._send(200, irer_specs.SCHEMA)
        if u.path == "/api/components":
            return self._send(200, rt.list_components(ROOT))
        if u.path == "/api/specs":
            return self._send(200, rt.list_specs(ROOT))
        if u.path == "/api/spec":
            sid = (q.get("id") or [""])[0]
            return self._send(200, rt.get_spec(ROOT, sid))
        return self._send(404, {"status": "NOT_FOUND"})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                return self._send(413, {"status": "TOO_LARGE"})
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, OSError):
            return self._send(400, {"status": "BAD_JSON"})
        spec = irer_specs.prune_empty(body.get("spec") or {})
        if u.path == "/api/validate":
            errs = full_validate(spec)
            return self._send(200, {"status": "VALID" if not errs else "INVALID", "errors": errs,
                                    "points": len(irer_specs.expand(spec)) if not errs else 0})
        if u.path == "/api/save":
            code, out = save_draft(spec)
            return self._send(code, out)
        if u.path == "/api/promote":
            code, out = promote(body.get("id", ""), body.get("from", "drafts"))
            return self._send(code, out)
        return self._send(404, {"status": "NOT_FOUND"})


def make_server(port=8765):
    return ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = make_server(args.port)
    print("spec editor on http://127.0.0.1:%d/  (Ctrl-C to stop; it cannot launch runs)" % args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
