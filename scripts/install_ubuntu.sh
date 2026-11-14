#!/usr/bin/env bash
# Install the character lamp on a clean Ubuntu 24.04 LTS laptop
# (4 cores, 8 GB RAM, no GPU, built-in or USB camera, PipeWire audio).
#
#   ./scripts/install_ubuntu.sh            # system packages + venv + models
#   ./scripts/install_ubuntu.sh --services # also install systemd user services
#
# Then put your key in .env and run ./run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 system packages"
sudo apt-get update -q
sudo apt-get install -y -q \
  python3 python3-venv python3-pip \
  libportaudio2 pipewire-alsa \
  libgl1 libegl1 libglfw3 libglib2.0-0 \
  v4l-utils espeak-ng
#  libportaudio2 + pipewire-alsa : microphone/speaker for sounddevice (PortAudio -> ALSA -> PipeWire)
#  libgl1 libegl1 libglfw3       : MuJoCo viewer (OpenGL on the integrated GPU) and headless EGL
#  libglib2.0-0                  : OpenCV runtime
#  v4l-utils                     : `v4l2-ctl --list-devices` to find the camera
#  espeak-ng                     : fallback voice if the Piper model can't be downloaded

echo "== 2/5 camera and audio permissions"
for g in video audio; do
  if ! id -nG "$USER" | grep -qw "$g"; then
    sudo usermod -aG "$g" "$USER"
    echo "   added $USER to group '$g' (log out and back in for it to apply)"
  fi
done
ls /dev/video* >/dev/null 2>&1 && v4l2-ctl --list-devices || echo "   WARNING: no /dev/video* camera found"

echo "== 3/5 Python environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== 4/5 API key"
if [ ! -f .env ]; then
  read -r -p "   Anthropic API key (sk-ant-...): " KEY
  printf 'ANTHROPIC_API_KEY=%s\n' "$KEY" > .env
  chmod 600 .env
fi

echo "== 5/5 models + self-test (downloads Whisper base.en ~140 MB and a Piper voice ~60 MB)"
python scripts/check_setup.py || echo "   some checks failed; see above"

if [[ "${1:-}" == "--services" ]]; then
  mkdir -p ~/.config/systemd/user
  for unit in deploy/lamp-body.service deploy/lamp-brain.service; do
    sed "s|@ROOT@|$PWD|g" "$unit" > ~/.config/systemd/user/"$(basename "$unit")"
  done
  systemctl --user daemon-reload
  systemctl --user enable lamp-body.service lamp-brain.service
  loginctl enable-linger "$USER"
  echo "   services enabled: they start at login. Logs: journalctl --user -u lamp-brain -f"
fi

echo
echo "Done. Start the lamp with:  ./run.sh"
