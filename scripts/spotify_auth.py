#!/usr/bin/env python
"""One-time Spotify login: prints the refresh_token for configs/settings.toml.

Run it on a computer with a browser (your Mac), not the headless Pi:

1. At https://developer.spotify.com/dashboard create an app, add the
   redirect URI http://127.0.0.1:8888/callback (Spotify no longer accepts
   "localhost"), and tick "Web API".
2. python scripts/spotify_auth.py --client-id <id> --client-secret <secret>
3. Log in when the browser opens. The script prints a [spotify] block with
   the refresh_token to paste into configs/settings.toml (on the Mac and on
   the Pi). The token keeps working until you revoke the app.
"""

from __future__ import annotations

import argparse
import secrets
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

from vcm.home.spotify import SCOPES, TOKEN_URL

REDIRECT = "http://127.0.0.1:8888/callback"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", required=True)
    args = parser.parse_args()

    state = secrets.token_urlsafe(16)
    result: dict[str, str] = {}
    done = threading.Event()

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if query.get("state", [""])[0] == state and "code" in query:
                result["code"] = query["code"][0]
                message = b"Spotify login done. You can close this tab."
            else:
                message = b"Login failed (state mismatch or access denied)."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(message)
            done.set()

        def log_message(self, *a):  # quiet
            pass

    server = HTTPServer(("127.0.0.1", 8888), Callback)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(
        {"client_id": args.client_id, "response_type": "code", "redirect_uri": REDIRECT, "scope": SCOPES, "state": state}
    )
    print(f"Opening the Spotify login page. If it doesn't open, visit:\n{url}\n")
    webbrowser.open(url)
    done.wait()
    server.shutdown()
    if "code" not in result:
        raise SystemExit("No authorization code received.")

    r = requests.post(
        TOKEN_URL,
        data={"grant_type": "authorization_code", "code": result["code"], "redirect_uri": REDIRECT},
        auth=(args.client_id, args.client_secret),
        timeout=10,
    )
    r.raise_for_status()
    refresh = r.json()["refresh_token"]
    print("Add this to configs/settings.toml (on the Mac and on the Pi):\n")
    print("[spotify]")
    print(f'client_id = "{args.client_id}"')
    print(f'client_secret = "{args.client_secret}"')
    print(f'refresh_token = "{refresh}"')
    print('device_name = "kiwi"        # the raspotify device name on the Pi; empty = whichever device is active')
    print('default_uri = ""            # optional playlist/album/track URI to start when nothing is playing')


if __name__ == "__main__":
    main()
