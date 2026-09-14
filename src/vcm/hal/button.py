"""Push-to-talk trigger.

Real implementation on both platforms (Section 8 only asks to *mock the
trigger signal*, not skip the button entirely): on the Mac, holding the
spacebar stands in for holding the physical pushbutton; on the RPi, the
actual pushbutton wired to GPIO 17 (Section 10) is read via gpiozero,
using the Pi's internal pull-up so a press pulls the pin LOW.

Both backends expose the same interface: `is_pressed` and
`wait_for_release()`, matching gpiozero.Button's real API so the mac
backend is a genuine stand-in, not a lookalike.
"""

from __future__ import annotations

import time
from typing import Protocol

from vcm.config import get_platform


class PushButton(Protocol):
    @property
    def is_pressed(self) -> bool: ...

    def wait_for_press(self) -> None: ...

    def wait_for_release(self) -> None: ...


class _MacSpacebarButton:
    """Spacebar-hold stand-in, per Section 8's laptop-simulation plan.

    Requires the `pynput` keyboard listener, which on macOS needs
    Accessibility/Input Monitoring permission granted to the terminal
    app running this process.
    """

    def __init__(self) -> None:
        from pynput import keyboard

        self._pressed = False
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key) -> None:
        from pynput import keyboard

        if key == keyboard.Key.space:
            self._pressed = True

    def _on_release(self, key) -> None:
        from pynput import keyboard

        if key == keyboard.Key.space:
            self._pressed = False

    @property
    def is_pressed(self) -> bool:
        return self._pressed

    def wait_for_press(self, poll_interval: float = 0.02) -> None:
        while not self._pressed:
            time.sleep(poll_interval)

    def wait_for_release(self, poll_interval: float = 0.02) -> None:
        while self._pressed:
            time.sleep(poll_interval)


class _RpiGpioButton:
    """Real pushbutton on GPIO 17, per Section 10's wiring diagram."""

    def __init__(self, gpio_pin: int = 17) -> None:
        from gpiozero import Button

        self._button = Button(gpio_pin, pull_up=True)

    @property
    def is_pressed(self) -> bool:
        return self._button.is_pressed

    def wait_for_press(self) -> None:
        self._button.wait_for_press()

    def wait_for_release(self) -> None:
        self._button.wait_for_release()


def get_button() -> PushButton:
    """Return the push-to-talk button for the current platform."""
    platform = get_platform()
    if platform == "rpi":
        return _RpiGpioButton()
    if platform == "mac":
        return _MacSpacebarButton()
    raise ValueError(f"Unknown platform: {platform!r}")
