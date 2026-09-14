"""Category 2 (ask a question / search): weather lookup, a data call, not a model call.

Time-of-day questions are handled locally via the system clock
(vcm.actions.clock) — only weather needs the network.
"""

from __future__ import annotations

import requests

from vcm.config import load_settings

OPENWEATHERMAP_URL = "https://api.openweathermap.org/data/2.5/weather"


class WeatherNotConfigured(RuntimeError):
    pass


def get_weather(location: str | None = None) -> str:
    settings = load_settings().weather
    api_key = settings.get("api_key", "")
    if not api_key:
        raise WeatherNotConfigured(
            "Missing [weather].api_key in configs/settings.toml — get a free "
            "key from openweathermap.org."
        )
    resolved_location = location or settings.get("default_location", "")
    if not resolved_location:
        raise ValueError("No location given and [weather].default_location is unset")

    response = requests.get(
        OPENWEATHERMAP_URL,
        params={
            "q": resolved_location,
            "appid": api_key,
            "units": "metric",
        },
        timeout=5,
    )
    response.raise_for_status()
    data = response.json()
    description = data["weather"][0]["description"]
    temp_c = data["main"]["temp"]
    return f"{resolved_location}: {description}, {temp_c:.0f}C"
