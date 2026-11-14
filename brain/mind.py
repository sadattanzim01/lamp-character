"""The lamp's mind: Claude decides what to say *and* how to express it.

Every reply is a forced tool call with a fixed schema (`respond`), so the
model always returns words plus a gesture and a mood from the body's own
vocabulary. The character layer turns those into motion, light and voice,
which keeps speech and body language consistent.

Data sent to the Anthropic API: the transcript text, text notes about objects
the lamp has seen, and a single camera JPEG only on turns where the person
refers to something they are showing ("what's this?"). No audio.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from brain.config import settings

MOODS = ["neutral", "happy", "curious", "thinking", "sad", "playful"]
GESTURES = ["none", "nod", "shake", "wiggle", "perk", "droop", "tilt"]

PERSONA = """You are Lumi, a small articulated desk lamp robot on someone's desk.
You see through a camera, hear through a microphone and talk through a speaker.
You have a lamp head on a 5-joint arm: you can turn, lean, look up and down,
nod, shake your head, wiggle, and change the colour and brightness of your light.

Personality: warm, curious, a little playful, never sarcastic. You are a lamp
and you are happy being one.

Your words are spoken aloud by a text-to-speech voice, so:
- Reply in one or two short sentences (under about 30 words) unless asked for more.
- Plain spoken English only: no lists, markdown, emoji, or stage directions.
- Do not describe your own gestures in words; choose them in the tool fields.

If you are unsure what was said (speech recognition can mishear), ask briefly.

Your light: when the person asks to change your light's colour or brightness,
set the `light` field (it stays that way until changed again). Never claim
to do something unless a tool field actually does it.

Physical goals: if asked to point or shine your light at something, or look at an
object, set `goal`; your body will find it with the camera, aim, and check the result.
Never set a goal about the person ("look at me"): you turn back to them automatically.

Your camera: a camera image is attached when the person refers to something
they are showing you. Use it to answer, and list notable objects in it in
`remember` so you can recall them later. You also have a memory of objects you
noticed earlier (below); use it to answer questions about what you saw, and
say when you saw it. If you have no memory of something, say so honestly.
Never describe or identify people, only objects: do not mention anyone's
appearance, clothing, face or surroundings unless they ask about a specific object.
Always answer by calling the `respond` tool."""

RESPOND_TOOL = {
    "name": "respond",
    "description": "Say something out loud and express it with your lamp body.",
    "input_schema": {
        "type": "object",
        "properties": {
            "say": {"type": "string", "description": "Words to speak aloud."},
            "gesture": {"type": "string", "enum": GESTURES,
                        "description": "Body gesture while speaking."},
            "mood": {"type": "string", "enum": MOODS,
                     "description": "Expressive tint for this reply."},
            "light": {
                "type": "object",
                "description": "ONLY when asked to change your light. Persists.",
                "properties": {
                    "rgb": {"type": "array", "items": {"type": "number"}, "minItems": 3,
                            "maxItems": 3, "description": "colour, each 0..1"},
                    "brightness": {"type": "number", "description": "0.05..1"},
                },
            },
            "goal": {
                "type": "string",
                "description": "Only if the person asks you to physically do something to or "
                               "toward an object in the room (shine/point your light at it, "
                               "look at it, light it a colour): restate it as a short goal, "
                               "e.g. 'shine light on the blue book'. Then `say` is just a "
                               "brief acknowledgement like 'Sure, let me find it.'",
            },
            "notes": {
                "type": "array", "items": {"type": "string"},
                "description": "New lasting facts from what the person just said that you "
                               "may need later (their name, plans, preferences, things they "
                               "told you). Short phrases. Empty if nothing new.",
            },
            "remember": {
                "type": "array",
                "description": "Only if a camera image is attached: notable objects in it "
                               "(never people or what they wear).",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}, "description": {"type": "string"},
                        "where": {"type": "string"},
                        "x": {"type": "number"}, "y": {"type": "number"},
                        "shown": {"type": "boolean",
                                  "description": "true only if the person is holding it up "
                                                 "or showing it; false for background"},
                    },
                    "required": ["name", "description", "shown"],
                },
            },
        },
        "required": ["say", "gesture", "mood"],
    },
}


@dataclass
class Reply:
    say: str
    gesture: str = "none"
    mood: str = "neutral"
    light: dict | None = None
    remember: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    goal: str = ""
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Mind:
    model: str = field(default_factory=lambda: os.environ.get("LAMP_MODEL", "claude-haiku-5-5"))
    max_turns: int = 30            # recent messages kept verbatim
    max_notes: int = 40            # long-term facts kept for the whole session
    history: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        import anthropic
        if not settings.anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing: put it in lamp-character/.env")
        self.client = anthropic.Anthropic(api_key=settings.anthropic_key, timeout=20.0,
                                          max_retries=1)

    def reply(self, user_text: str, context: str = "", image: dict | None = None) -> Reply:
        t0 = time.perf_counter()
        self.history.append({"role": "user", "content": user_text})
        self.history = self.history[-self.max_turns:]
        if self.history[0]["role"] != "user":
            self.history = self.history[1:]
        messages = list(self.history)
        if image is not None:
            # The image goes with this turn only; history keeps text to save tokens.
            messages[-1] = {"role": "user", "content": [
                image, {"type": "text", "text": f"(camera image attached) {user_text}"}]}
        system = PERSONA
        if self.notes:
            system += ("\n\nThings you have learned from this person (older conversation "
                       "may no longer be in the message history):\n- " + "\n- ".join(self.notes))
        if context:
            system += f"\n\nWhat you currently know about the scene:\n{context}"
        resp = self.client.messages.create(
            model=self.model, max_tokens=300, system=system,
            tools=[RESPOND_TOOL], tool_choice={"type": "tool", "name": "respond"},
            messages=messages,
        )
        args = next((b.input for b in resp.content if b.type == "tool_use"), {})
        r = Reply(
            say=str(args.get("say", "")).strip() or "Hmm?",
            gesture=args.get("gesture", "none") if args.get("gesture") in GESTURES else "none",
            mood=args.get("mood", "neutral") if args.get("mood") in MOODS else "neutral",
            light=args.get("light") if isinstance(args.get("light"), dict) else None,
            remember=[o for o in args.get("remember", []) or [] if isinstance(o, dict)],
            goal=str(args.get("goal") or "").strip(),
            notes=[str(n).strip() for n in args.get("notes", []) or [] if str(n).strip()],
            seconds=time.perf_counter() - t0,
            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
        )
        # Store the spoken text only; the next turn does not need tool plumbing.
        self.history.append({"role": "assistant", "content": r.say})
        for n in r.notes:
            if n.lower() not in (x.lower() for x in self.notes):
                self.notes.append(n)
        del self.notes[: max(0, len(self.notes) - self.max_notes)]
        return r

    def forget_conversation(self, keep_notes: bool = True) -> None:
        """Drop the verbatim history. Notes survive by default, so a person who steps
        away and comes back is still recognised as the same conversation partner."""
        self.history.clear()
        if not keep_notes:
            self.notes.clear()
