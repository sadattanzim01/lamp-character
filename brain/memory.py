"""Scene memory: what the lamp has seen, as compact text facts.

Stored per object: name, description (colour, material, any visible text),
where it was, its rough image position, and when it was first / last seen.
No images are kept, only these notes, which is what the mind gets as context
and what is written to logs/memory.json (so memory survives a restart).

Merging: a new sighting updates an existing entry when the head noun matches
("red mug" vs "coffee mug") and the descriptions overlap, so holding the same
mug up twice gives one memory with two sightings, not two mugs.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_STOP = {"a", "an", "the", "of", "with", "and", "on", "in", "small", "large", "some"}


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in _STOP}


def _head(name: str) -> str:
    ws = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in _STOP]
    return ws[-1].rstrip("s") if ws else name.lower()


@dataclass
class SeenObject:
    name: str
    description: str = ""
    where: str = ""
    x: float = 0.5          # image position, 0..1 (left..right)
    y: float = 0.5          # 0..1 (top..bottom)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    sightings: int = 1
    shown: bool = False     # held up / presented by the person (vs background)


class SceneMemory:
    def __init__(self, path: Path | None = None, max_shown: int = 40, max_background: int = 15):
        self.path, self.max_shown, self.max_background = path, max_shown, max_background
        self.items: list[SeenObject] = []
        self._lock = threading.Lock()
        if path and path.exists():
            try:
                self.items = [SeenObject(**d) for d in json.loads(path.read_text())]
            except (ValueError, TypeError):
                self.items = []

    def update(self, objects: list[dict], now: float | None = None) -> list[SeenObject]:
        """Merge sightings. Returns the objects that are new to memory."""
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
                    match.shown = match.shown or shown
                    match.last_seen, match.sightings = now, match.sightings + 1
                    match.where = o.get("where", match.where) or match.where
                    match.x = float(o.get("x", match.x))
                    match.y = float(o.get("y", match.y))
                    if len(desc) > len(match.description):
                        match.description = desc
                else:
                    item = SeenObject(name, desc, str(o.get("where", "")),
                                      float(o.get("x", 0.5)), float(o.get("y", 0.5)), now, now,
                                      shown=shown)
                    self.items.append(item)
                    new.append(item)
            # Evict separately so background clutter can never push out shown objects.
            self.items.sort(key=lambda i: i.last_seen, reverse=True)
            shown_items = [i for i in self.items if i.shown][: self.max_shown]
            background = [i for i in self.items if not i.shown][: self.max_background]
            self.items = sorted(shown_items + background, key=lambda i: i.last_seen, reverse=True)
            self._save()
        return new

    def _find(self, name: str, desc: str) -> SeenObject | None:
        head, words = _head(name), _words(name + " " + desc)
        best, score = None, 0.0
        for it in self.items:
            if _head(it.name) != head:
                continue
            other = _words(it.name + " " + it.description)
            s = len(words & other) / max(1, len(words | other))
            if s > score:
                best, score = it, s
        return best if best is not None and score >= 0.2 else None

    def find(self, query: str) -> SeenObject | None:
        """Best match for a spoken reference like 'the mug' (used by goals)."""
        q = _words(query)
        with self._lock:
            ranked = sorted(self.items, key=lambda i: (len(q & _words(i.name + " " + i.description)),
                                                        i.last_seen), reverse=True)
        return ranked[0] if ranked and q & _words(ranked[0].name + " " + ranked[0].description) else None

    def describe(self, now: float | None = None) -> str:
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
        lines.append(f"Current time: {time.strftime('%H:%M', time.localtime(now))}.")
        return "\n".join(lines)

    def clear(self) -> None:
        with self._lock:
            self.items.clear()
            self._save()

    def _save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps([asdict(i) for i in self.items], indent=1))


def _ago(s: float) -> str:
    if s < 20:
        return "just now"
    if s < 90:
        return f"{int(s)} seconds ago"
    if s < 3600:
        return f"{round(s / 60)} minutes ago"
    return f"{s / 3600:.1f} hours ago"
