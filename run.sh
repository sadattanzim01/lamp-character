#!/usr/bin/env bash
# Start the lamp: body (simulator + protocol server) then brain. Ctrl+C stops both.
#
#   ./run.sh                 # 3D window + camera debug window
#   ./run.sh --headless      # no windows (SSH, kiosk, CI)
#   ./run.sh --no-voice      # extra args go to the brain (see python -m brain.main -h)
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p logs

HEADLESS=0
BRAIN_ARGS=()
for a in "$@"; do
  if [[ "$a" == "--headless" ]]; then HEADLESS=1; else BRAIN_ARGS+=("$a"); fi
done

if [[ "$(uname)" == "Darwin" ]]; then
  BODY=(mjpython -m body.server)                    # macOS: viewer needs mjpython
else
  BODY=(python -m body.server)
fi
if [[ $HEADLESS == 1 || -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" && "$(uname)" != "Darwin" ]]; then
  BODY+=(--headless)
  export MUJOCO_GL=egl
else
  BRAIN_ARGS+=(--show)
fi

"${BODY[@]}" > logs/body.log 2>&1 &
BODY_PID=$!
trap 'kill $BODY_PID 2>/dev/null; wait $BODY_PID 2>/dev/null' EXIT

# Wait for the body's WebSocket port instead of sleeping a fixed time.
for _ in $(seq 1 50); do
  if python - <<'EOF' 2>/dev/null
import socket; socket.create_connection(("127.0.0.1", 8765), timeout=0.2).close()
EOF
  then break; fi
  kill -0 $BODY_PID 2>/dev/null || { echo "body failed to start:"; cat logs/body.log; exit 1; }
  sleep 0.2
done

python -m brain.main ${BRAIN_ARGS[@]+"${BRAIN_ARGS[@]}"}
