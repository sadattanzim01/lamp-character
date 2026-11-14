# Brain ↔ Body protocol (v1)

**Transport:** WebSocket, one JSON object per message, default `ws://127.0.0.1:8765`.
The body is the server; the brain connects to it.

**Why this split:** the body is the only process that touches the simulator (or, on
a real lamp, the servos). It owns safety: limits, self-collision, velocity and
acceleration. The brain owns *intent*. It never sends raw motor torques and never
needs to know about joint limits or the collision geometry. Replacing MuJoCo with
real hardware means re-implementing only `body/lamp_body.py`.

**Why WebSocket + JSON:** it works from any language, is easy to inspect, and
messages are tiny (< 1 kB), so serialization cost is negligible at these rates.
It also works over a network, so the brain could run on a laptop while the body
runs on the lamp's embedded board. A shared-memory or binary (protobuf/ZeroMQ)
transport would only be worth the complexity at kHz control rates. Here
low-level control stays inside the body at 100 Hz.

## Brain → Body (commands)

Every command carries a client-chosen `id`. Angles are in radians; `speed` is a
fraction (0–1) of each joint's URDF velocity limit.

| `type` | Fields | Channel |
|---|---|---|
| `pose` | `name` (from `hello.capabilities.poses`), `speed` | motion |
| `gaze` | `azimuth`, `elevation` (lamp frame, see below), `speed` | motion |
| `gesture` | `name` (`nod`, `shake`, `wiggle`, `perk`, `droop`, `tilt`), `intensity` | motion |
| `joints` | `q`: `{joint_name: angle}` (partial OK), `speed` | motion |
| `light` | `rgb` [0–1]×3, `intensity` 0–1, `fade` seconds | light |
| `stop` | none (hold the current position) | motion |

**Gaze frame:** azimuth 0 is the lamp's forward direction (world −x), positive turns
to the lamp's left. Elevation 0 is horizontal, positive is up. The reachable
elevation is advertised in `hello` (currently −1.05 to +0.22 rad).

**Channels:** a new motion command preempts the running one. Light commands run
independently, so the lamp can change colour mid-gesture.

## Body → Brain

| `type` | When | Fields |
|---|---|---|
| `hello` | on connect | `v`, `capabilities` (joints, soft/velocity limits, poses, gestures, gaze range) |
| `ack` | once per command, immediately | `id`, `ok`, `adjusted` (list of changes the body made), `error`, `body_ms` |
| `done` | when a motion command ends | `id`, `status`: `completed` / `preempted` / `rejected` |
| `state` | 20 Hz broadcast | `t`, `q`, `qd`, `torque_frac` (of URDF effort), `at_goal`, `light` |

## Safety the body applies to every motion goal

1. Clamp to URDF **soft** limits.
2. Check the goal pose for self-collision or table contact; if it collides, raise
   the head in 0.05 rad steps until it is clear, or reject the goal.
3. Approach the goal with a per-joint velocity limit (URDF `velocity` × `speed`)
   and acceleration limit (3–12 rad/s², lower for joints that swing the whole arm).
4. Servo torque is capped at the URDF `effort`, including gravity compensation.

Every adjustment is reported in the `ack`, so the brain (and the logs) know what
the body actually did.

## Example

```json
→ {"id": "c7", "type": "gaze", "azimuth": 0.6, "elevation": 0.8, "speed": 0.7}
← {"type": "ack", "id": "c7", "ok": true, "adjusted": ["elevation clipped to (-1.05, 0.22)"], "body_ms": 1.3}
← {"type": "done", "id": "c7", "status": "completed"}
```

## Measured (Linux container, headless, 18 commands)

Command → ack round trip: median 1.1 ms, p95 3.3 ms.
Gaze IK: median error 0.1°, 95th percentile 2.2°, 0 self-collisions over 221 targets, about 1 ms per solve.
