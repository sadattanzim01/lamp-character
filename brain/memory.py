"""The lamp's memory of objects it has seen.

What this file does, in plain words:
    Every time the lamp looks at the scene, Claude returns a list of objects.
    This file keeps those as short text notes, one per object:
        name, description (colour, material, any text on it), where it was,
        its position in the picture, when it was first and last seen,
        how many times it's been seen, and whether the person SHOWED it
        (held it up) or it was just in the background.

    When you later ask "what did I show you earlier?", these notes are given
    to Claude as background info so it can answer, including WHEN.

Key decisions:
    * Text only, no photos are ever stored. The notes are also saved to
      logs/memory.json so memory survives a restart.
    * Same object, one memory: if you hold up the same mug twice (and Claude
      calls it "red mug" once and "coffee mug" the next time), it's matched as
      the same object and counted as two sightings, not two different mugs.
    * Shown objects are protected: background clutter (curtains, monitors) has
      its own smaller limit, so it can never push out the things you showed.
      (Before this rule, a long session forgot the objects you showed it.)
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Filler words ignored when comparing names ("a red mug" vs "the mug").
_STOP = {"a", "an", "the", "of", "with", "and", "on", "in", "small", "large", "some"}


def _words(s: str) -> set[str]:
    """The meaningful lowercase words in a phrase: "A red Coffee Mug" -> {"red", "coffee", "mug"}."""
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in _STOP}


def _head(name: str) -> str:
    """The main noun of a name (its last meaningful word, singular).
    "red coffee mug" -> "mug", "yellow sticky notes" -> "note"."""
    ws = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in _STOP]
    return ws[-1].rstrip("s") if ws else name.lower()


@dataclass
class SeenObject:
    """One remembered object."""
    name: str
    description: str = ""
    where: str = ""                                    # e.g. "held up by the person", "on the desk, left"
    x: float = 0.5          # position in the picture: 0 = left edge, 1 = right edge
    y: float = 0.5          # 0 = top, 1 = bottom
    first_seen: float = field(default_factory=time.time)   # time stamps (seconds since 1970)
    last_seen: float = field(default_factory=time.time)
    sightings: int = 1                                  # how many times it's been seen
    shown: bool = False     # True = the person held it up / showed it; False = just in the background


class SceneMemory:
    """All remembered objects, kept newest first."""

    def __init__(self, path: Path | None = None, max_shown: int = 40, max_background: int = 15):
        # Keep up to 40 shown objects and 15 background ones.
        self.path, self.max_shown, self.max_background = path, max_shown, max_background
        self.items: list[SeenObject] = []
        self._lock = threading.Lock()   # several threads (scene watcher, conversation) use memory at once
        # Load what was remembered last time, if the file exists and is readable.
        if path and path.exists():
            try:
                self.items = [SeenObject(**d) for d in json.loads(path.read_text())]
            except (ValueError, TypeError):
                self.items = []         # damaged file: start fresh rather than crash

    def update(self, objects: list[dict], now: float | None = None) -> list[SeenObject]:
        """Add a batch of sightings from Claude.

        Each object either updates a matching memory (new time, +1 sighting,
        better description if the new one is longer) or becomes a new memory.
        Returns only the objects that are NEW, so the lamp can react to them.
        """
        now = now or time.time()
        new: list[SeenObject] = []
        with self._lock:
            for o in objects:
                name = str(o.get("name", "")).strip()
                if not name:
                    continue
                desc = str(o.get("description", "")).strip()
                shown = bool(o.get("shown", False))
                match = self._find(name, desc)
                if match:
                    # Seen before: refresh it. Once something has been shown, it stays "shown".
                    match.shown = match.shown or shown
                    match.last_seen, match.sightings = now, match.sightings + 1
                    match.where = o.get("where", match.where) or match.where
                    match.x = float(o.get("x", match.x))
                    match.y = float(o.get("y", match.y))
                    if len(desc) > len(match.description):
                        match.description = desc        # keep the most detailed description
                else:
                    item = SeenObject(name, desc, str(o.get("where", "")),
                                      float(o.get("x", 0.5)), float(o.get("y", 0.5)), now, now,
                                      shown=shown)
                    self.items.append(item)
                    new.append(item)
            # Trim memory, keeping shown and background objects in separate
            # "buckets" so background clutter can never push out shown objects.
            # Within each bucket, the most recently seen are kept.
            self.items.sort(key=lambda i: i.last_seen, reverse=True)
            shown_items = [i for i in self.items if i.shown][: self.max_shown]
            background = [i for i in self.items if not i.shown][: self.max_background]
            self.items = sorted(shown_items + background, key=lambda i: i.last_seen, reverse=True)
            self._save()
        return new

    def _find(self, name: str, desc: str) -> SeenObject | None:
        """Is this the same object as one we already remember?

        Two rules: the main noun must match ("mug" = "mug"), and the words
        must overlap enough. Overlap = shared words / all words (a "Jaccard"
        similarity); at least 0.2 counts as the same object.
        """
        head, words = _head(name), _words(name + " " + desc)
        best, score = None, 0.0
        for it in self.items:
            if _head(it.name) != head:
                continue                                 # different kind of thing
            other = _words(it.name + " " + it.description)
            s = len(words & other) / max(1, len(words | other))
            if s > score:
                best, score = it, s
        return best if best is not None and score >= 0.2 else None

    def find(self, query: str) -> SeenObject | None:
        """Find the best match for something the person said, like "the mug".
        Picks the object sharing the most words with the query (most recent wins ties)."""
        q = _words(query)
        with self._lock:
            ranked = sorted(self.items, key=lambda i: (len(q & _words(i.name + " " + i.description)),
                                                        i.last_seen), reverse=True)
        return ranked[0] if ranked and q & _words(ranked[0].name + " " + ranked[0].description) else None

    def describe(self, now: float | None = None) -> str:
        """Write memory out as plain text for Claude to read before it answers.

        Shown objects get a full line each, with when they were seen; background
        items get one short summary line. Example:
            Objects the person showed you (most recent first):
            - red mug: red ceramic mug. Where: held up. Last seen 2 minutes ago (14:31), seen 1 time.
            Also visible in the background of the room: monitor, curtain.
            Current time: 14:33.
        """
        now = now or time.time()
        with self._lock:
            shown = [i for i in self.items if i.shown]
            background = [i for i in self.items if not i.shown]
        if not self.items:
            return "You have not noticed any objects yet."
        lines = ["Objects the person showed you (most recent first):" if shown
                 else "The person has not shown you any objects yet."]
        for it in shown:
            ago = _ago(now - it.last_seen)
            stamp = time.strftime("%H:%M", time.localtime(it.last_seen))
            first = (f"; first seen {_ago(now - it.first_seen)}"
                     if it.sightings > 1 else "")
            lines.append(f"- {it.name}: {it.description or 'no details'}. "
                         f"Where: {it.where or 'unknown'}. Last seen {ago} ({stamp}){first}, "
                         f"seen {it.sightings} time{'s' * (it.sightings > 1)}.")
        if background:
            lines.append("Also visible in the background of the room: "
                         + ", ".join(i.name for i in background[:12]) + ".")
        # Including the current time lets Claude say things like "about 5 minutes ago".
        lines.append(f"Current time: {time.strftime('%H:%M', time.localtime(now))}.")
        return "\n".join(lines)

    def clear(self) -> None:
        """Forget everything (used by `--forget`)."""
        with self._lock:
            self.items.clear()
            self._save()

    def _save(self) -> None:
        """Write memory to logs/memory.json."""
        if self.path:
            self.path.write_text(json.dumps([asdict(i) for i in self.items], indent=1))


def _ago(s: float) -> str:
    """Seconds -> friendly text: "just now", "45 seconds ago", "3 minutes ago", "1.5 hours ago"."""
    if s < 20:
        return "just now"
    if s < 90:
        return f"{int(s)} seconds ago"
    if s < 3600:
        return f"{round(s / 60)} minutes ago"
    return f"{s / 3600:.1f} hours ago"
