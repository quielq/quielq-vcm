"""CALL and MESSAGE through your phone, by way of your Mac.

The device (Pi, or the Mac itself when testing) sends a small HTTP request
to scripts/mac_phone_bridge.py running on your Mac. The Mac then:
- sends an iMessage (or SMS, with iPhone Text Message Forwarding) from the
  Messages app, automatically;
- starts a phone call through your iPhone ("Calls from iPhone" /
  Continuity). macOS shows a Call button to confirm; there's no fully
  hands-free way to place a call from a Mac.

The bridge needs the Mac awake and signed in to the same Apple ID as the
iPhone. Settings: configs/settings.toml [phone]. If no bridge is
configured, calls and messages are logged on the dashboard as simulated.
Other routes (Bluetooth hands-free on the Pi, a phone notification app)
are in TODO.md.
"""

from __future__ import annotations

import requests


class PhoneNotConfigured(RuntimeError):
    pass


class PhoneBridge:
    def __init__(self, url: str, token: str, contacts: dict[str, str], default_contact: str, default_message: str):
        if not url:
            raise PhoneNotConfigured("no [phone].bridge_url set")
        self.url, self.token = url.rstrip("/"), token
        self.contacts = {k.lower(): v for k, v in contacts.items()}
        self.default_contact, self.default_message = default_contact, default_message

    @classmethod
    def from_settings(cls, section: dict) -> "PhoneBridge":
        return cls(
            section.get("bridge_url", ""),
            section.get("bridge_token", ""),
            section.get("contacts", {}),
            section.get("default_contact", "Mom"),
            section.get("default_message", "Hi! This is a message from my voice assistant."),
        )

    def number(self, contact: str) -> str:
        number = self.contacts.get(contact.lower())
        if not number:
            raise PhoneNotConfigured(f"no number for '{contact}' in [phone].contacts")
        return number

    def _post(self, path: str, payload: dict) -> str:
        r = requests.post(f"{self.url}{path}", json=payload, headers={"X-Bridge-Token": self.token}, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"phone bridge {path} failed ({r.status_code}): {r.text[:200]}")
        return r.json().get("status", "ok")

    def call(self, contact: str | None = None) -> str:
        contact = contact or self.default_contact
        return self._post("/call", {"number": self.number(contact), "contact": contact})

    def message(self, contact: str | None = None, text: str | None = None) -> str:
        contact = contact or self.default_contact
        return self._post("/message", {"number": self.number(contact), "text": text or self.default_message})
