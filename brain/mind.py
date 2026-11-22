"""Lumi's mind: Claude decides what to say AND how to act it out.

What this file does, in plain words:
    You say something; this file sends it to Claude (with Lumi's personality,
    what it remembers, and sometimes a camera photo) and gets back a reply.

    The important trick: Claude doesn't reply with plain text. It must fill in
    a form, the `respond` tool, with fixed fields:
        say       the words to speak
        gesture   a body movement from the lamp's list (nod, wiggle, tilt...)
        mood      a feeling, which sets the light colour (happy, curious...)
        light     only if you asked to change the light (stays changed)
        goal      only if you asked it to DO something physical ("shine on the book")
        notes     facts worth remembering about you (your name, your plans)
        remember  objects in the photo, if a photo was attached
    Because the words and the body language come out of the SAME answer, they
    always match. And because gestures/moods must come from fixed lists, Claude
    can't invent a movement the lamp can't do.

What is sent to the cloud (Anthropic API): the text of what you said, Lumi's
memory notes, and a camera photo ONLY when you refer to something you're
showing ("what's this?"). Never any audio.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from brain.config import settings

# The only moods and gestures Claude may pick. They match what character.py
# (moods -> light colours) and the body (body/executor.py gestures) can actually do.
MOODS = ["neutral", "happy", "curious", "thinking", "sad", "playful"]
GESTURES = ["none", "nod", "shake", "wiggle", "perk", "droop", "tilt"]

# Lumi's personality and rules: the "system prompt" Claude reads before every reply.
# Key rules in it: keep answers short (they're spoken aloud), never pretend to do
# something no field actually does, and never describe people.

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

# The "form" Claude must fill in for every reply. "enum" limits a field to a fixed
# list; "required" fields must always be filled. Each "description" tells Claude
# when and how to use that field.
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
    """Claude's answer, unpacked from the form (plus timing and token counts for the logs)."""
    say: str
    gesture: str = "none"
    mood: str = "neutral"
    light: dict | None = None
    remember: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    goal: str = ""
    seconds: float = 0.0           # how long Claude took
    input_tokens: int = 0          # how much text was sent (affects cost)
    output_tokens: int = 0         # how much text came back


@dataclass
class Mind:
    """Talks to Claude and keeps track of the conversation.

    It remembers in two ways:
      * history: the last 30 messages, word for word
      * notes:   up to 40 short facts (e.g. "Their name is Sadat") that last the
                 whole session, even after old messages drop out of the history.
                 This is why Lumi still knows your name after a long chat.
    """
    model: str = field(default_factory=lambda: os.environ.get("LAMP_MODEL", "claude-haiku-5-5"))
    max_turns: int = 30            # recent messages kept verbatim
    max_notes: int = 40            # long-term facts kept for the whole session
    history: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Runs right after the Mind is created: set up the Claude API client."""
        import anthropic
        if not settings.anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing: put it in lamp-character/.env")
        # Give up after 20 s, and retry a failed request once.
        self.client = anthropic.Anthropic(api_key=settings.anthropic_key, timeout=20.0,
                                          max_retries=1)

    def reply(self, user_text: str, context: str = "", image: dict | None = None) -> Reply:
        """Get Lumi's reply to what the person said.

        user_text  what speech recognition heard
        context    Lumi's memory of objects (from memory.py), as text
        image      a camera photo, only for "what's this?"-type questions
        """
        t0 = time.perf_counter()
        # Add the new message and keep only the last 30. The history must start
        # with a user message (an API rule), so drop a leading Lumi message if trimming left one.
        self.history.append({"role": "user", "content": user_text})
        self.history = self.history[-self.max_turns:]
        if self.history[0]["role"] != "user":
            self.history = self.history[1:]
        messages = list(self.history)
        if image is not None:
            # The image goes with this turn only; history keeps text to save tokens.
            messages[-1] = {"role": "user", "content": [
                image, {"type": "text", "text": f"(camera image attached) {user_text}"}]}
        # Build the instructions: personality + facts learned + scene memory.
        system = PERSONA
        if self.notes:
            system += ("\n\nThings you have learned from this person (older conversation "
                       "may no longer be in the message history):\n- " + "\n- ".join(self.notes))
        if context:
            system += f"\n\nWhat you currently know about the scene:\n{context}"
        # Ask Claude. tool_choice forces it to answer by filling in the `respond` form.
        resp = self.client.messages.create(
            model=self.model, max_tokens=300, system=system,
            tools=[RESPOND_TOOL], tool_choice={"type": "tool", "name": "respond"},
            messages=messages,
        )
        # Unpack the form. Anything missing or invalid gets a safe default
        # (e.g. an unknown gesture becomes "none"), so a bad answer can't break the lamp.
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
        # Save new facts (skipping duplicates), keeping only the newest 40.
        for n in r.notes:
            if n.lower() not in (x.lower() for x in self.notes):
                self.notes.append(n)
        del self.notes[: max(0, len(self.notes) - self.max_notes)]
        return r

    def forget_conversation(self, keep_notes: bool = True) -> None:
        """Clear the word-for-word history (called when the person leaves).
        Notes are kept by default, so if you step away and come back, Lumi
        still remembers who you are and what you talked about."""
        self.history.clear()
        if not keep_notes:
            self.notes.clear()
