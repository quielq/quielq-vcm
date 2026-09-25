"""The VCM "home": what recognized commands actually do.

vcm_listen (wake word -> intent + slot) posts each command to the home
server (server.py), which runs the matching action (dispatcher.py),
updates the shared state (state.py) and pushes it live to the web
dashboard. Timers, alarms and reminders fire from a background scheduler
(scheduler.py). Integrations: Spotify (spotify.py), the Mac phone bridge
for calls/messages (phone.py), system volume (volume.py), and the existing
weather/lights modules in vcm.actions.

Standard library only (plus `requests` for the web APIs), so it runs next
to the voice pipeline on a 512 MB Raspberry Pi.
"""
