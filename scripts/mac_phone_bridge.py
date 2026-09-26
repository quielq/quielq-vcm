#!/usr/bin/env python
"""Run on your Mac: lets the VCM (on the Pi or this Mac) send iMessages and
start phone calls through your iPhone. Standard library only.

    python scripts/mac_phone_bridge.py --token <shared secret> [--port 8765] [--service iMessage|SMS]

Then in configs/settings.toml on the device:
    [phone]
    bridge_url = "http://<mac-name>.local:8765"     # http://127.0.0.1:8765 when testing on the Mac
    bridge_token = "<shared secret>"
    default_contact = "Mom"
    contacts = { Mom = "+639171234567" }

Mac setup (once):
- Messages app signed in to your Apple ID. For SMS to non-iPhones, enable
  iPhone Settings > Messages > Text Message Forwarding for this Mac, and
  run with --service SMS.
- Calls: iPhone Settings > Phone > Calls on Other Devices > allow this Mac;
  FaceTime on the Mac > Settings > "Calls from iPhone".
- The first message triggers a macOS prompt allowing Terminal (or Python)
  to control Messages: click Allow.
- Placing a call opens macOS's call prompt; click Call. macOS has no fully
  hands-free way to place a call.

Only numbers configured on the device are sent here, and every request
must carry the shared token.
"""

from __future__ import annotations

import argparse
import hmac
import json
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Arguments are passed to AppleScript as argv, never spliced into the
# script text, so a message can't inject AppleScript.
_SEND_SCRIPT = """
on run argv
    set theNumber to item 1 of argv
    set theText to item 2 of argv
    set theServiceType to item 3 of argv
    tell application "Messages"
        if theServiceType is "SMS" then
            set theService to 1st account whose service type = SMS
        else
            set theService to 1st account whose service type = iMessage
        end if
        send theText to participant theNumber of theService
    end tell
end run
"""
_NUMBER_RE = re.compile(r"^\+?[0-9 ()-]{5,20}$")


def message_command(number: str, text: str, service: str) -> list[str]:
    return ["osascript", "-e", _SEND_SCRIPT, number, text, service]


def call_command(number: str) -> list[str]:
    return ["open", "tel:" + re.sub(r"[^0-9+]", "", number)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", required=True, help="shared secret; the same value goes in [phone].bridge_token")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--service", choices=["iMessage", "SMS"], default="iMessage")
    parser.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):  # noqa: N802
            if not hmac.compare_digest(self.headers.get("X-Bridge-Token", ""), args.token):
                return self._reply(403, {"error": "bad token"})
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self._reply(400, {"error": "bad json"})
            number = str(payload.get("number", ""))
            if not _NUMBER_RE.match(number):
                return self._reply(400, {"error": "bad number"})
            if self.path == "/message":
                cmd = message_command(number, str(payload.get("text", ""))[:1000], args.service)
                status = f"message sent via {args.service}"
            elif self.path == "/call":
                cmd = call_command(number)
                status = "call started on the Mac: confirm with the Call button"
            else:
                return self._reply(404, {"error": "unknown path"})
            if args.dry_run:
                print("would run:", cmd[:2] + ["<script>"] + cmd[3:] if cmd[0] == "osascript" else cmd)
                return self._reply(200, {"status": f"dry run: {status}"})
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if result.returncode != 0:
                return self._reply(500, {"error": result.stderr.strip()[:300]})
            print(f"{self.path[1:]} -> {payload.get('contact', number)}: {status}")
            self._reply(200, {"status": status})

        def log_message(self, *a):
            pass

    print(f"Phone bridge on {args.host}:{args.port} ({args.service}{', dry run' if args.dry_run else ''}). Ctrl+C to stop.")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
