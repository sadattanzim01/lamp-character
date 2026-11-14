"""Goal-directed action: a spoken goal about the live scene becomes lamp actions,
then the lamp looks again to check before it says it is done.

    "shine your light on the book"
      1. PLAN     fresh camera frame + goal -> Claude (plan_goal tool):
                  target object located in the image + a short sequence of ops
      2. ACT      ops executed in order (aim / light / gesture / look_at_person)
      3. VERIFY   new frame -> Claude (locate tool): where is the target *now*?
                  compare with the lamp's *measured* gaze (from joint state)
      4. CORRECT  if the error > 5 deg (object moved, estimate was off), re-aim
                  and verify again, up to 3 times; then report the outcome

Boundary between model and body (the "action representation"):
  * The model decides WHAT: which object (as a point in the image) and which
    ops, from a fixed vocabulary. It never outputs joint angles.
  * The brain turns image points into gaze angles (camera geometry).
  * The body turns gaze angles into joint goals (IK) and enforces limits,
    collisions, speed and torque. It can adjust or reject, and says so.
  * Success is judged by fresh perception + measured body state, not by the
    model's own claim.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from brain.perception.attention import camera_angles_to_image
import cv2
import numpy as np

from brain.perception.scene import image_block

log = logging.getLogger("goals")

AIM_TOLERANCE_DEG = 5.0     # used when the object's box is tiny or missing
BOX_SHRINK = 0.8            # the aim point must land in the central 80% of the object's box
MAX_CORRECTIONS = 3
OPS = ["aim", "light", "gesture", "look_at_person"]

PLAN_TOOL = {
    "name": "plan_goal",
    "description": "Plan lamp actions for a goal about objects in the camera image.",
    "input_schema": {
        "type": "object",
        "properties": {
            "target": {
                "type": "object",
                "description": "The object the goal is about, located in THIS image.",
                "properties": {
                    "name": {"type": "string"},
                    "visible": {"type": "boolean"},
                    "x": {"type": "number", "description": "centre, 0=left edge 1=right edge"},
                    "y": {"type": "number", "description": "centre, 0=top edge 1=bottom edge"},
                    "w": {"type": "number", "description": "bounding-box width, fraction of image width"},
                    "h": {"type": "number", "description": "bounding-box height, fraction of image height"},
                },
                "required": ["name", "visible", "x", "y", "w", "h"],
            },
            "steps": {
                "type": "array",
                "description": "Ordered ops. aim = point the lamp head and light at the "
                               "target; light = set colour/brightness; gesture = body "
                               "gesture; look_at_person = turn back to the person.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": OPS},
                        "rgb": {"type": "array", "items": {"type": "number"}},
                        "brightness": {"type": "number"},
                        "gesture": {"type": "string",
                                    "enum": ["nod", "shake", "wiggle", "perk", "tilt"]},
                    },
                    "required": ["op"],
                },
            },
            "done_message": {"type": "string", "description": "short spoken line when done"},
            "cannot_message": {"type": "string",
                               "description": "short spoken line if the target is not visible"},
        },
        "required": ["target", "steps", "done_message", "cannot_message"],
    },
}

LOCATE_TOOL = {
    "name": "locate",
    "description": "Locate one object in the camera image.",
    "input_schema": {
        "type": "object",
        "properties": {
            "visible": {"type": "boolean"},
            "x": {"type": "number", "description": "centre, 0=left 1=right"},
            "y": {"type": "number", "description": "centre, 0=top 1=bottom"},
            "w": {"type": "number", "description": "bounding-box width, fraction of image width"},
            "h": {"type": "number", "description": "bounding-box height, fraction of image height"},
        },
        "required": ["visible", "x", "y", "w", "h"],
    },
}

PLAN_PROMPT = """You control a desk lamp robot. The image is from a webcam that sits at the
lamp and faces the same way, so pointing the lamp at an image location points it at
that object. Goal from the person: "{goal}"

Locate the object the goal refers to in this image (centre point, normalised) and plan
a short sequence of ops to achieve the goal. Typical: [aim] then [light] if a colour or
brightness is asked for, optionally a gesture. Use look_at_person only if asked.
{memory}Do not describe people."""


GRID_NOTE = ("The image has a coordinate grid: lines every 0.1, labelled along the top "
             "(x, 0=left to 1=right) and left edge (y, 0=top to 1=bottom). Read the object's "
             "box edges off the grid lines; be precise to about 0.02.")


def with_grid(frame: np.ndarray) -> np.ndarray:
    """Overlay a labelled 10x10 grid. Vision models read positions off reference lines
    far more precisely than they estimate raw coordinates (first real runs showed
    plan vs re-locate disagreeing by up to 0.34 of the image width)."""
    out = frame.copy()
    h, w = out.shape[:2]
    for i in range(1, 10):
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
        for pos, text in (((x + 3, 16), f".{i}"), ((3, y - 4), f".{i}")):
            cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


class GoalRunner:
    def __init__(self, client, model: str, character, get_frame, memory=None,
                 log_dir: Path | None = None, fallback_model: str | None = None):
        self.client, self.model = client, model
        self.fallback_model = fallback_model or model
        self._forced: dict[str, bool] = {}       # model -> supports forced tool_choice
        self.character, self.get_frame, self.memory = character, get_frame, memory
        self.log = (log_dir / "goals.jsonl").open("a") if log_dir else None
        self.overlay: dict = {}            # for the debug window: target + aim point

    # --------------------------------------------------------------- run
    def run(self, goal: str, say) -> dict:
        """Blocking; called on the conversation worker thread. `say(text, mood)` speaks."""
        rec: dict = {"t": round(time.time(), 3), "goal": goal, "attempts": []}
        t0 = time.perf_counter()
        frame = self.get_frame()
        if frame is None:
            say("I can't see anything right now.", "sad")
            return rec

        # 1. PLAN
        mem = ""
        if self.memory is not None and self.memory.items:
            mem = ("If the target is not in view, it may be something you saw earlier:\n"
                   + self.memory.describe() + "\n")
        plan = self._tool(PLAN_TOOL, frame, PLAN_PROMPT.format(goal=goal, memory=mem))
        rec["plan"] = plan
        rec["model"] = self.model
        rec["plan_s"] = round(time.perf_counter() - t0, 2)
        target = plan.get("target", {})
        name = str(target.get("name", "it"))
        log.info("plan (%.1fs): target=%s steps=%s", rec["plan_s"], target,
                 [s.get("op") for s in plan.get("steps", [])])
        if not target.get("visible"):
            say(plan.get("cannot_message") or f"I can't see the {name} right now.", "sad")
            rec["result"] = "not_visible"
            return self._finish(rec)

        x, y = _clip01(target.get("x")), _clip01(target.get("y"))
        box = (_clip01(target.get("w"), 0.0), _clip01(target.get("h"), 0.0))
        self.overlay = {"name": name, "target": (x, y), "box": box}

        # 2. ACT
        for step in plan.get("steps", [])[:6]:
            self._do(step, x, y)
        if not any(s.get("op") == "aim" for s in plan.get("steps", [])):
            self._do({"op": "aim"}, x, y)          # every scene goal needs the lamp on target

        # 3-4. VERIFY (+ CORRECT)
        ok = False
        for attempt in range(1, MAX_CORRECTIONS + 1):
            time.sleep(0.3)                          # let the arm settle
            frame = self.get_frame()
            loc = self._tool(LOCATE_TOOL, frame,
                             f'Where is the "{name}" in this image? Do not describe people.')
            gaze = self.character.body.state.get("gaze", {})
            entry = {"attempt": attempt, "seen": bool(loc.get("visible"))}
            if not loc.get("visible"):
                rec["attempts"].append(entry)
                say(f"Hmm, I lost sight of the {name}.", "sad")
                rec["result"] = "lost"
                return self._finish(rec)
            x, y = _clip01(loc.get("x")), _clip01(loc.get("y"))
            box = (_clip01(loc.get("w"), 0.0), _clip01(loc.get("h"), 0.0))
            want_az, want_el = self.character.gaze_for_image(x, y)
            g_az, g_el = gaze.get("azimuth", 0.0), gaze.get("elevation", 0.0)
            err = math.degrees(math.hypot(want_az - g_az, want_el - g_el))
            # Where the lamp actually points, in image coordinates (from measured joints).
            from brain.character import ATTENTIVE_EL
            ax, ay = camera_angles_to_image(g_az, g_el - ATTENTIVE_EL)
            inside = (box[0] > 0.02 and box[1] > 0.02
                      and abs(ax - x) <= BOX_SHRINK * box[0] / 2
                      and abs(ay - y) <= BOX_SHRINK * box[1] / 2)
            entry.update(target=[round(x, 3), round(y, 3)], box=[round(box[0], 3), round(box[1], 3)],
                         aim=[round(ax, 3), round(ay, 3)], error_deg=round(err, 2), on_object=inside)
            rec["attempts"].append(entry)
            self.overlay = {"name": name, "target": (x, y), "box": box}
            log.info("verify %d: %s at (%.2f, %.2f) size %.2fx%.2f, aim error %.1f deg, on object: %s",
                     attempt, name, x, y, box[0], box[1], err, inside)
            if inside or err <= AIM_TOLERANCE_DEG:
                ok = True
                break
            self._do({"op": "aim"}, x, y)            # correction

        rec["result"] = "done" if ok else "inaccurate"
        rec["total_s"] = round(time.perf_counter() - t0, 2)
        if ok:
            say(plan.get("done_message") or f"There, the {name} is lit up.", "happy")
        else:
            say(f"I tried, but I couldn't line up exactly on the {name}.", "sad")
        return self._finish(rec)

    # --------------------------------------------------------------- ops
    def _do(self, step: dict, x: float, y: float) -> None:
        ch = self.character
        op = step.get("op")
        if op == "aim":
            az, el = ch.gaze_for_image(x, y)
            # Bright neutral spot so the "light on it" is unmistakable, unless a
            # colour step follows and overrides it.
            ch.on_main(lambda: ch.body.light(*(ch.user_light or ((1.0, 0.95, 0.85), 1.0)), fade=0.3))
            done = ch.on_main(lambda: ch.aim(az, el))
        elif op == "light":
            rgb = step.get("rgb") or [1.0, 0.95, 0.85]
            if len(rgb) != 3:
                return
            done = None
            ch.on_main(lambda: ch.set_user_light(rgb, step.get("brightness", 1.0)))
        elif op == "gesture":
            done = ch.on_main(lambda: ch.body.gesture(step.get("gesture", "nod"))[1])
        elif op == "look_at_person":
            ch.on_main(ch.release_hold)
            done = None
        else:
            return
        if done is not None:
            try:
                done.result(timeout=8)
            except Exception:                         # noqa: BLE001
                pass

    def _tool(self, tool: dict, frame, prompt: str) -> dict:
        content = [image_block(with_grid(frame), width=768),
                   {"type": "text", "text": prompt + "\n" + GRID_NOTE}]
        out = self._call(self.model, tool, content)
        if out is None and self.model != self.fallback_model:
            log.warning("vision model %s gave no %s call; using %s for this step",
                        self.model, tool["name"], self.fallback_model)
            out = self._call(self.fallback_model, tool, content)
        return out or {}

    def _call(self, model: str, tool: dict, content: list) -> dict | None:
        """One tool call. Some models (e.g. claude-sonnet-5-5) reject a *forced*
        tool_choice; for those, use auto with an explicit instruction instead."""
        forced = self._forced.get(model, True)
        kwargs = dict(model=model, tools=[tool], messages=[{"role": "user", "content": content}])
        if forced:
            kwargs.update(max_tokens=500, tool_choice={"type": "tool", "name": tool["name"]})
        else:
            kwargs.update(max_tokens=4000, tool_choice={"type": "auto"},
                          system=f"Answer only by calling the {tool['name']} tool, exactly once.")
        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as exc:                   # noqa: BLE001
            if forced and "tool_choice" in str(exc):
                log.info("%s does not support forced tool_choice; switching to auto", model)
                self._forced[model] = False
                return self._call(model, tool, content)
            if model == self.fallback_model:
                raise
            log.warning("vision model %s failed (%s)", model, exc)
            return None
        return next((b.input for b in resp.content if b.type == "tool_use"), None)

    def _finish(self, rec: dict) -> dict:
        if self.log:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        log.info("goal %r -> %s", rec["goal"], rec.get("result"))
        return rec


def _clip01(v, default: float = 0.5) -> float:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return default
