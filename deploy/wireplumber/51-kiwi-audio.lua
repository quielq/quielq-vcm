-- Kiwi (VCM) audio setup on the Raspberry Pi, for WirePlumber 0.4 (Pi OS
-- bookworm). Installed by scripts/deploy_pi.sh to
-- ~/.config/wireplumber/main.lua.d/, then `systemctl --user restart wireplumber`.
--
-- Why (found on the Pi before a demo):
-- * The USB mic dropped for a few seconds; when it came back, the soundbar's
--   own mic had become the default input and Kiwi stopped hearing commands.
-- * WirePlumber saved each app's stream volume, so a music duck that never
--   ended carried over to every new Spotify stream (and to Kiwi's replies).
-- Device names are for this hardware (C-Media "USB PnP" mic, Dell AC511
-- soundbar): `pactl list short sources` / `sinks` shows them.

-- 1. Start every stream at its app's own volume; never restore a saved one.
stream_defaults.properties["restore-props"] = false

-- 2. The USB PnP mic is the input whenever it's plugged in.
table.insert(alsa_monitor.rules, {
  matches = { { { "node.name", "matches", "alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device*" } } },
  apply_properties = { ["priority.session"] = 3000, ["priority.driver"] = 3000 },
})

-- 3. The soundbar's built-in mic is never used, so it can't become the input.
table.insert(alsa_monitor.rules, {
  matches = { { { "node.name", "matches", "alsa_input.usb-Dell_Dell_AC511*" } } },
  apply_properties = { ["node.disabled"] = true },
})

-- 4. The soundbar is the output whenever it's plugged in.
table.insert(alsa_monitor.rules, {
  matches = { { { "node.name", "matches", "alsa_output.usb-Dell_Dell_AC511*" } } },
  apply_properties = { ["priority.session"] = 3000, ["priority.driver"] = 3000 },
})
