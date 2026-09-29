#!/usr/bin/env bash
# Deploy the VCM to a Raspberry Pi over SSH, from the laptop (DEPLOYMENT.md
# steps 3, 4, 6 and optionally 9 in one go). Safe to re-run: it's also how to
# update the Pi after pulling new code or models.
#
#   scripts/deploy_pi.sh raspberrypi.local              # code + models + venv + benchmark
#   scripts/deploy_pi.sh raspberrypi.local --services   # also start on boot (systemd)
#   scripts/deploy_pi.sh raspberrypi.local --settings   # overwrite the Pi's settings.toml with the laptop's
#
# The target is anything `ssh` accepts: a ~/.ssh/config Host (DEPLOYMENT.md
# step 2 sets up raspberrypi.local with User quielq) or user@host.
#
# configs/settings.toml is copied only the first time, so edits made on the Pi
# (Spotify device_name, the Mac bridge URL) survive re-runs; the copy switches
# the voice from macOS `say` to espeak-ng. Needs SSH access (DEPLOYMENT.md
# steps 1-2); sudo on the Pi may ask for its password once for apt.
set -euo pipefail

TARGET="${1:?usage: scripts/deploy_pi.sh raspberrypi.local (or user@host) [--services] [--settings]}"
shift
SERVICES=0
SETTINGS=0
for arg in "$@"; do
  case "$arg" in
    --services) SERVICES=1 ;;
    --settings) SETTINGS=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
for f in models/vcm_intent.onnx models/kiwi_wakeword.onnx requirements-pi.txt; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done

echo "== checking the Pi"
ARCH=$(ssh "$TARGET" uname -m)
[[ "$ARCH" == "aarch64" ]] || { echo "Pi reports $ARCH: needs 64-bit Raspberry Pi OS (aarch64) for onnxruntime" >&2; exit 1; }

echo "== system packages (sudo)"
ssh -t "$TARGET" "sudo apt-get update -qq && sudo apt-get install -y -qq git python3-venv libportaudio2 alsa-utils tmux espeak-ng sense-hat"

echo "== copying code and models"
rsync -az --delete \
  --exclude /.git --exclude /.venv --exclude /data --exclude /checkpoints --exclude /logs \
  --exclude /debug_recordings --exclude /media --exclude /reports --exclude /models/tts --exclude '__pycache__' \
  --exclude .pytest_cache --exclude '*.egg-info' --exclude .DS_Store --exclude /configs/settings.toml \
  ./ "$TARGET:quielq-vcm/"

if [[ -f configs/settings.toml ]]; then
  if [[ $SETTINGS == 1 ]] || ! ssh "$TARGET" test -f quielq-vcm/configs/settings.toml; then
    echo "== copying configs/settings.toml (voice: espeak-ng)"
    sed 's/^backend = "mac_say"/backend = "espeak_ng"/' configs/settings.toml | ssh "$TARGET" "cat > quielq-vcm/configs/settings.toml"
  else
    echo "== keeping the Pi's configs/settings.toml (--settings overwrites it)"
  fi
fi

echo "== Python environment"
ssh "$TARGET" "cd quielq-vcm && mkdir -p data && { [[ -d .venv ]] || python3 -m venv .venv; } \
  && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements-pi.txt \
  && .venv/bin/pip install -q --no-deps -e ."

# The Sense HAT library only ships as a Debian package (the pip one needs a
# hand-built RTIMULib), so link just its modules into the venv instead of
# exposing every system package (whose broken metadata makes pip warn, and
# whose scipy clashes with the venv's numpy).
echo "== Sense HAT library in the venv"
ssh "$TARGET" 'cd quielq-vcm && sed -i "s/include-system-site-packages = true/include-system-site-packages = false/" .venv/pyvenv.cfg \
  && site=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_path(\"purelib\"))") \
  && for m in /usr/lib/python3/dist-packages/{sense_hat,RTIMU.*.so,smbus.*.so}; do \
       if [[ -e $m ]]; then ln -sfn "$m" "$site/"; fi; done \
  && .venv/bin/pip install -q pillow'

echo "== benchmark (DEPLOYMENT.md step 6)"
ssh "$TARGET" "cd quielq-vcm && .venv/bin/python scripts/benchmark_pi.py && free -m"

echo "== audio device rules (WirePlumber: pin the USB mic and the soundbar, no saved stream volumes)"
ssh "$TARGET" 'mkdir -p ~/.config/wireplumber/main.lua.d \
  && cp quielq-vcm/deploy/wireplumber/51-kiwi-audio.lua ~/.config/wireplumber/main.lua.d/ \
  && systemctl --user restart wireplumber && sleep 2 \
  && echo "  input:  $(pactl get-default-source)" && echo "  output: $(pactl get-default-sink)"'

if [[ $SERVICES == 1 ]]; then
  echo "== start on boot, restart on failure (DEPLOYMENT.md step 9)"
  ssh -t "$TARGET" 'mkdir -p ~/.config/systemd/user \
    && cp quielq-vcm/deploy/systemd/vcm-home.service quielq-vcm/deploy/systemd/vcm.service ~/.config/systemd/user/ \
    && systemctl --user daemon-reload && systemctl --user enable vcm-home vcm && systemctl --user restart vcm-home vcm \
    && sudo loginctl enable-linger "$USER" \
    && systemctl --user --no-pager status vcm-home vcm | grep -E "Active|●"'
fi

HOST="${TARGET#*@}"
echo
echo "Done. Next: DEPLOYMENT.md step 5 (microphone), then step 7:"
echo "  ssh $TARGET"
echo "  cd quielq-vcm && .venv/bin/python scripts/vcm_listen.py --server http://127.0.0.1:8000"
echo "Dashboard: http://$HOST:8000 (once vcm.home.server is running)"
