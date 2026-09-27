"""Recognized command (intent + slot value) -> action + spoken reply.

Covers all 19 intents of the class taxonomy plus unknown_background. Every
integration is optional and injected, so a missing one (no Spotify
credentials, no phone bridge, no weather key, no real bulb) degrades to a
clear spoken reply and a dashboard entry instead of an error:

| Intent | What happens |
|---|---|
| PLAY_MUSIC, PAUSE, STOP, NEXT | Spotify Connect; local files via mpv as fallback |
| VOLUME_UP / VOLUME_DOWN | System output volume (USB or Bluetooth speaker), else Spotify's |
| WEATHER, TIME | Weather API / system clock, spoken |
| LIGHT_ON/OFF, BRIGHTNESS, COLOR | Virtual lamp on the dashboard (+ real Xiaomi bulb if configured) |
| TEMPERATURE | Reports the simulated room temperature (seeded from the Sense HAT) + thermostat target (no direction yet: TODO.md) |
| TIMER, ALARM | Scheduled from the slot value; rings when due until STOP / PAUSE |
| CREATE_REMINDER, LIST_REMINDERS | Reminder list (text is edited on the dashboard: TODO.md) |
| CALL, MESSAGE | Your phone via the Mac bridge, else logged as simulated |
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from vcm.home import volume as system_volume
from vcm.home.phone import PhoneBridge, PhoneNotConfigured
from vcm.home.scheduler import duration_seconds, next_alarm_time, spoken_duration
from vcm.home.spotify import SpotifyClient, SpotifyError
from vcm.home.state import HomeState, new_id

VOLUME_STEP = 10
THERMO_MIN_C, THERMO_MAX_C, THERMO_STEP_C = 16, 30, 1
DUCK_PERCENT = 15  # Spotify volume while listening for a command


@dataclass
class Integrations:
    """Optional external services; None means "not set up"."""

    spotify: SpotifyClient | None = None
    phone: PhoneBridge | None = None
    weather: Callable[[], str] | None = None
    temperature: Callable[[], float] | None = None
    local_music: object | None = None  # module with play(), and media_control-style pause/stop/next_track
    local_media: object | None = None
    bulb: object | None = None  # module with turn_on/turn_off/set_brightness
    ringer: object | None = None  # vcm.home.ringer.Ringer: alarm/timer sound
    get_volume: Callable[[], int | None] = system_volume.get_volume
    change_volume: Callable[[int], int | None] = system_volume.change_volume


class Dispatcher:
    def __init__(self, state: HomeState, integrations: Integrations):
        self.state, self.x = state, integrations
        self._duck_lock, self._duck_restore = threading.Lock(), None
        self.handlers = {
            "PLAY_MUSIC": self.play_music, "PAUSE": self.pause, "STOP": self.stop, "NEXT": self.next,
            "VOLUME_UP": lambda s: self.volume(+VOLUME_STEP), "VOLUME_DOWN": lambda s: self.volume(-VOLUME_STEP),
            "WEATHER": self.weather, "TIME": self.time,
            "LIGHT_ON": lambda s: self.lights(on=True), "LIGHT_OFF": lambda s: self.lights(on=False),
            "BRIGHTNESS": self.brightness, "COLOR": self.color, "TEMPERATURE": self.temperature,
            "TIMER": self.timer, "ALARM": self.alarm,
            "CREATE_REMINDER": self.create_reminder, "LIST_REMINDERS": self.list_reminders,
            "CALL": self.call, "MESSAGE": self.message,
        }  # fmt: skip

    def handle(self, intent: str, slot: str | None = None, confidence: float | None = None, source: str = "voice") -> str:
        if intent == "unknown_background":
            reply = ""
        elif intent in self.handlers:
            try:
                reply = self.handlers[intent](slot)
            except Exception as exc:  # an integration failing must not take the server down
                reply = f"Sorry, that didn't work: {exc}"
        else:
            reply = f"I don't know how to do {intent}."
        entry = {"at": time.time(), "intent": intent, "slot": slot, "confidence": confidence, "reply": reply, "source": source}
        self.state.update(lambda d: d["log"].append(entry))
        return reply

    # --- music -----------------------------------------------------------
    def _music(self, spotify_action: str, local_action: str | None, playing: bool, verb: str) -> str:
        source = self.state.snapshot()["music"]["source"]
        if self.x.spotify and source in (None, "spotify"):
            try:
                getattr(self.x.spotify, spotify_action)()
                track = self.x.spotify.now_playing() if playing else None
                self.state.update(lambda d: d["music"].update(source="spotify", playing=playing, track=track or d["music"]["track"]))
                return f"{verb}{f': {track}' if track else ''}."
            except SpotifyError as exc:
                if not self.x.local_media:
                    return f"Spotify isn't available: {exc}"
        if self.x.local_media and local_action:
            getattr(self.x.local_media, local_action)()
            self.state.update(lambda d: d["music"].update(source="local", playing=playing))
            return f"{verb}."
        return "Music isn't set up. Add Spotify or local music in settings."

    def play_music(self, slot):
        source = self.state.snapshot()["music"]["source"]
        if self.x.spotify and source in (None, "spotify"):
            try:
                self.x.spotify.play()
                track = self.x.spotify.now_playing()
                self.state.update(lambda d: d["music"].update(source="spotify", playing=True, track=track))
                return f"Playing {track}." if track else "Playing music."
            except SpotifyError as exc:
                if not self.x.local_music:
                    return f"Spotify isn't available: {exc}"
        if self.x.local_music:
            try:
                track = self.x.local_music.play()
            except FileNotFoundError:
                return "There's no local music to play."
            name = getattr(track, "stem", str(track))
            self.state.update(lambda d: d["music"].update(source="local", playing=True, track=name))
            return f"Playing {name}."
        return "Music isn't set up. Add Spotify or local music in settings."

    def _stop_ringing(self) -> bool:
        return bool(self.x.ringer and self.x.ringer.stop())

    def pause(self, slot):
        if self._stop_ringing():
            return "Okay."
        return self._music("pause", "pause", playing=False, verb="Paused")

    def stop(self, slot):
        if self._stop_ringing():  # "Hey Kiwi, stop" while an alarm rings
            return "Okay."
        # Spotify has no "stop": pausing is the closest.
        return self._music("pause", "stop", playing=False, verb="Stopped")

    def next(self, slot):
        return self._music("next", "next_track", playing=True, verb="Next song")

    def duck(self, on: bool) -> None:
        """Lower Spotify (and pause a ringing alarm) while Kiwi listens after
        the wake word, so the mic hears the command; restore it afterwards."""
        if self.x.ringer:
            self.x.ringer.mute(on)
        if not self.x.spotify:
            return
        with self._duck_lock:
            if on and self._duck_restore is None:
                playback = self.x.spotify.playback()
                if playback["playing"] and (playback["volume"] or 0) > DUCK_PERCENT:
                    self.x.spotify.set_volume(DUCK_PERCENT)
                    self._duck_restore = playback["volume"]
            elif not on and self._duck_restore is not None:
                restore, self._duck_restore = self._duck_restore, None
                self.x.spotify.set_volume(restore)

    def volume(self, step: int) -> str:
        new = self.x.change_volume(step)
        if new is None and self.x.spotify:
            current = self.state.snapshot()["music"]["volume"]
            new = max(0, min(100, current + step))
            with self._duck_lock:
                if self._duck_restore is not None:  # ducked: apply when listening ends
                    self._duck_restore = new
                else:
                    self.x.spotify.set_volume(new)
        if new is None:
            return "I can't change the volume on this device."
        self.state.update(lambda d: d["music"].update(volume=new))
        return f"Volume {new} percent."

    # --- information -----------------------------------------------------
    def weather(self, slot):
        if not self.x.weather:
            return "Weather isn't set up. Add an OpenWeatherMap key in settings."
        text = self.x.weather()
        self.state.update(lambda d: d.update(weather={"text": text, "updated": time.time()}))
        return text

    def time(self, slot):
        now = datetime.now()
        return f"It's {now.strftime('%I:%M %p').lstrip('0')}, {now.strftime('%A, %B %d').replace(' 0', ' ')}."

    # --- lights and thermostat -------------------------------------------
    def _bulb(self, method: str, *args) -> None:
        if self.x.bulb:
            try:
                getattr(self.x.bulb, method)(*args)
            except Exception:  # the virtual lamp still changes; a flaky real bulb shouldn't block it
                pass

    def lights(self, on: bool) -> str:
        self.state.update(lambda d: d["lights"].update(on=on))
        self._bulb("turn_on" if on else "turn_off")
        return "Lights on." if on else "Lights off."

    def brightness(self, slot):
        if not slot:
            return "What brightness? Try 'set the brightness to 60 percent'."
        percent = int(slot.rstrip("%"))
        self.state.update(lambda d: d["lights"].update(on=True, brightness=percent))
        self._bulb("set_brightness", percent)
        return f"Brightness {percent} percent."

    def color(self, slot):
        if not slot:
            return "What color? Try 'change the lights to blue'."
        self.state.update(lambda d: d["lights"].update(on=True, color=slot))
        return f"Lights set to {slot}."

    def read_sensor(self) -> float | None:
        """Read the sensor into current_c; seeds the simulated room_c the first time."""
        try:
            current = self.x.temperature() if self.x.temperature else None
        except Exception:  # no sensor attached
            current = None

        def change(d):
            d["thermostat"]["current_c"] = current
            if d["thermostat"].get("room_c") is None and current is not None:
                d["thermostat"]["room_c"] = round(current, 1)

        self.state.update(change)
        return current

    def temperature(self, slot):
        """No slot: report. A slot sets the (simulated) thermostat: "22 degrees"
        / "22" sets the target, "up" / "down" move it THERMO_STEP_C."""
        self.read_sensor()
        target = self.state.snapshot()["thermostat"]["target_c"]
        if slot:
            word = str(slot).lower().split()[0]
            if word in ("up", "warmer", "increase"):
                target += THERMO_STEP_C
            elif word in ("down", "cooler", "decrease"):
                target -= THERMO_STEP_C
            else:
                target = round(float(word.rstrip("°c")))
            target = max(THERMO_MIN_C, min(THERMO_MAX_C, target))
            self.state.update(lambda d: d["thermostat"].update(target_c=target))
            room = self.state.snapshot()["thermostat"].get("room_c")
            if room is None or abs(room - target) < 0.5:
                return f"Thermostat set to {target} degrees."
            return f"Thermostat set to {target} degrees. {'Cooling' if room > target else 'Heating'} from {room:.0f}."
        thermo = self.state.snapshot()["thermostat"]
        room = thermo.get("room_c")
        if room is None:
            return f"The thermostat is set to {target} degrees."
        if abs(room - target) < 0.5:
            return f"It's {target} degrees, right at the thermostat setting."
        action = "Cooling" if room > target else "Heating"
        return f"It's {room:.0f} degrees in the room. {action} to {target}."

    # --- timers, alarms, reminders ---------------------------------------
    def timer(self, slot):
        if not slot:
            return "How long? Try 'set a timer for five minutes'."
        seconds = duration_seconds(slot)
        entry = {"id": new_id(), "label": slot, "duration_s": seconds, "ends_at": time.time() + seconds}
        self.state.update(lambda d: d["timers"].append(entry))
        return f"Timer set for {spoken_duration(seconds)}."

    def alarm(self, slot):
        if not slot:
            return "For what time? Try 'set an alarm for 6 AM'."
        at = next_alarm_time(slot)
        entry = {"id": new_id(), "time": slot, "next_at": at.timestamp()}
        self.state.update(lambda d: d["alarms"].append(entry))
        day = "today" if at.date() == datetime.now().date() else "tomorrow"
        return f"Alarm set for {slot} {day}."

    def create_reminder(self, slot):
        entry = {"id": new_id(), "text": slot or "New reminder (add details on the dashboard)", "created": time.time()}
        self.state.update(lambda d: d["reminders"].append(entry))
        return "Reminder created." if slot else "Reminder created. You can add the details on the dashboard."

    def list_reminders(self, slot):
        items = self.state.snapshot()["reminders"]
        if not items:
            return "You have no reminders."
        texts = "; ".join(r["text"] for r in items[:5])
        more = f", and {len(items) - 5} more" if len(items) > 5 else ""
        return f"You have {len(items)} reminder{'s' if len(items) > 1 else ''}: {texts}{more}."

    # --- phone -----------------------------------------------------------
    def _phone(self, kind: str) -> str:
        contact = self.x.phone.default_contact if self.x.phone else "Mom"
        try:
            if not self.x.phone:
                raise PhoneNotConfigured("no phone bridge")
            status = self.x.phone.call() if kind == "call" else self.x.phone.message()
            reply = f"Calling {contact}." if kind == "call" else f"Message sent to {contact}."
        except PhoneNotConfigured:
            status = "simulated (no phone bridge configured)"
            reply = f"Calling {contact} (simulated)." if kind == "call" else f"Message to {contact} (simulated)."
        text = None if kind == "call" else (self.x.phone.default_message if self.x.phone else "")
        entry = {"id": new_id(), "kind": kind, "contact": contact, "text": text, "status": status, "at": time.time()}
        self.state.update(lambda d: d["calls"].append(entry))
        return reply

    def call(self, slot):
        return self._phone("call")

    def message(self, slot):
        return self._phone("message")


def integrations_from_settings(settings) -> Integrations:
    """Build whatever integrations configs/settings.toml has credentials for."""
    x = Integrations()
    try:
        x.spotify = SpotifyClient.from_settings(settings.section("spotify"))
    except SpotifyError:
        pass
    try:
        x.phone = PhoneBridge.from_settings(settings.section("phone"))
    except PhoneNotConfigured:
        pass
    if settings.weather.get("api_key"):
        from vcm.actions import weather

        x.weather = weather.get_weather
    from vcm.hal.temperature import read_temperature

    x.temperature = read_temperature
    if settings.music.get("media_dir") and __import__("shutil").which("mpv"):
        from vcm.actions import media_control, music

        x.local_music, x.local_media = music, media_control
    if settings.xiaomi.get("bulb_host") and settings.xiaomi.get("bulb_token"):
        from vcm.actions import lights

        x.bulb = lights
    return x
