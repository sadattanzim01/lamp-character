"""Audio: one mixer for music, sound effects and voice.

Everything is synthesised in numpy (no audio assets to license or ship).
One output stream mixes three layers:

  music  - looping bed with a target volume and smooth fades
  sfx    - one-shots, overlapping allowed
  voice  - speech; while it plays the music ducks so words stay clear

A single mixer (rather than each feature opening the speaker itself) is what
lets these layers react to each other, and on Linux it is one PortAudio
client on PipeWire/PulseAudio instead of several competing ones.
"""
from __future__ import annotations

import threading
import time

import numpy as np

SR = 22050          # matches Piper TTS voices, so speech needs no resampling
BLOCK = 512         # ~23 ms per audio callback


# ------------------------------------------------------------------ synthesis
def _env(n: int, attack: float, release: float) -> np.ndarray:
    e = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def _glide(f0: float, f1: float, dur: float, harm=(1.0, 0.3, 0.1)) -> np.ndarray:
    n = int(dur * SR)
    f = np.geomspace(f0, f1, n)
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = sum(a * np.sin((k + 1) * ph) for k, a in enumerate(harm))
    return tone * _env(n, 0.01, dur * 0.4)


def _bell(freq: float, dur: float, decay: float = 2.5) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    partials = [(1.0, 1.0), (2.0, 0.35), (3.01, 0.12), (4.2, 0.06)]
    s = sum(a * np.sin(2 * np.pi * freq * m * t) * np.exp(-decay * m * t) for m, a in partials)
    return s * _env(n, 0.004, 0.05)


def _silence(d: float) -> np.ndarray:
    return np.zeros(int(d * SR))


def _seq(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)


def _norm(x: np.ndarray, peak: float = 0.8) -> np.ndarray:
    m = np.max(np.abs(x)) or 1.0
    return (x / m * peak).astype(np.float32)


def make_sfx() -> dict[str, np.ndarray]:
    return {
        # curious upward "hm?" when the lamp first notices someone
        "notice": _norm(_glide(500, 820, 0.18), 0.5),
        # bright two-step chirp: hello!
        "hello": _norm(_seq(_glide(600, 900, 0.09), _silence(0.04), _glide(750, 1300, 0.14))),
        # sighing descending glide: bye
        "bye": _norm(_glide(900, 380, 0.45, harm=(1.0, 0.2)), 0.6),
        # soft blip: "I'm listening"
        "listen": _norm(_glide(880, 1100, 0.07), 0.45),
        # bubbly triple blip: thinking
        "think": _norm(_seq(*[_seq(_glide(f, f * 1.1, 0.05), _silence(0.05))
                              for f in (700, 900, 800)]), 0.4),
        # rising arpeggio: success
        "success": _norm(_seq(*[_bell(f, 0.16, 4) for f in (523, 659, 784)], _bell(1047, 0.5, 3))),
        # low wobble: can't do that
        "error": _norm(_seq(_glide(300, 240, 0.15), _glide(260, 200, 0.25)), 0.6),
    }


def make_music() -> dict[str, np.ndarray]:
    """'idle': a slow music-box lullaby in C major pentatonic (~16 s loop)."""
    beat = 60 / 72
    notes = [  # (midi, beats)
        (72, 1), (76, 1), (79, 2), (76, 1), (74, 1), (72, 2),
        (69, 1), (72, 1), (74, 2), (76, 1), (74, 1), (72, 2),
        (72, 1), (76, 1), (79, 1), (81, 1), (79, 2), (76, 2),
        (74, 1), (76, 1), (72, 2), (67, 2), (72, 2),
    ]
    total = sum(b for _, b in notes) * beat
    n = int(total * SR)
    melody = np.zeros(n + SR * 3)
    pos = 0.0
    for midi, beats in notes:
        f = 440 * 2 ** ((midi - 69) / 12)
        tone = _bell(f, 2.5, decay=1.6)
        i = int(pos * SR)
        melody[i : i + len(tone)] += tone
        pos += beats * beat
    # Wrap the tail into the start so the loop is seamless.
    tail = melody[n:]
    melody = melody[:n]
    melody[: len(tail)] += tail
    # Soft drone pad (C + G) for warmth.
    t = np.arange(n) / SR
    pad = 0.12 * (np.sin(2 * np.pi * 130.8 * t) + 0.7 * np.sin(2 * np.pi * 196.0 * t))
    pad *= 0.6 + 0.4 * np.sin(2 * np.pi * t / total)
    return {"idle": _norm(melody + pad, 0.6)}


# ------------------------------------------------------------------ mixer
class Mixer:
    def __init__(self, output: bool = True, device=None):
        self.sfx_bank = make_sfx()
        self.music_bank = make_music()
        self._lock = threading.Lock()
        self._sfx: list[list] = []                 # [buffer, pos, gain]
        self._voice: list = []                     # [buffer, pos, done_event]
        self._music: np.ndarray | None = None
        self._music_pos = 0
        self._music_vol = 0.0
        self._music_target = 0.0
        self._music_rate = 1.0                     # volume units per second
        self.master = 0.9
        self.duck = 0.25                            # music level under voice
        self._stream = None
        if output:
            import sounddevice as sd
            self._stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32",
                                           blocksize=BLOCK, device=device,
                                           callback=self._callback)
            self._stream.start()

    # ---------------------------------------------------------- API
    def sfx(self, name: str, gain: float = 1.0) -> float:
        """Play a one-shot. Returns its duration in seconds."""
        buf = self.sfx_bank[name]
        with self._lock:
            self._sfx.append([buf, 0, gain])
        return len(buf) / SR

    def music(self, name: str | None, volume: float = 0.35, fade: float = 1.5) -> None:
        """Start / change / fade out (name=None) the music bed."""
        with self._lock:
            if name is not None and self._music is not self.music_bank[name]:
                self._music, self._music_pos = self.music_bank[name], 0
                self._music_vol = 0.0
            self._music_target = 0.0 if name is None else volume
            self._music_rate = 1.0 / max(fade, 0.05)

    def voice(self, pcm: np.ndarray, sr: int = SR) -> threading.Event:
        """Queue speech audio. Returns an Event set when it finishes playing."""
        pcm = np.asarray(pcm, dtype=np.float32)
        if sr != SR:
            x = np.linspace(0, len(pcm), int(len(pcm) * SR / sr), endpoint=False)
            pcm = np.interp(x, np.arange(len(pcm)), pcm).astype(np.float32)
        done = threading.Event()
        with self._lock:
            self._voice.append([pcm, 0, done])
        return done

    def stop_voice(self) -> None:
        with self._lock:
            for v in self._voice:
                v[2].set()
            self._voice.clear()

    @property
    def speaking(self) -> bool:
        return bool(self._voice)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    # ---------------------------------------------------------- mixing
    def _mix(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            # voice: play queued utterances back to back
            if self._voice:
                buf, pos, done = self._voice[0]
                k = min(n, len(buf) - pos)
                out[:k] += buf[pos : pos + k]
                self._voice[0][1] += k
                if self._voice[0][1] >= len(buf):
                    done.set()
                    self._voice.pop(0)
            talking = bool(self._voice) or np.any(out)
            # music: fade toward target, duck under voice
            if self._music is not None:
                step = self._music_rate * n / SR
                self._music_vol += np.clip(self._music_target - self._music_vol, -step, step)
                if self._music_vol > 1e-4:
                    m = self._music
                    idx = (self._music_pos + np.arange(n)) % len(m)
                    level = self._music_vol * (self.duck if talking else 1.0)
                    out += m[idx] * level
                self._music_pos = (self._music_pos + n) % len(self._music)
            # sfx
            alive = []
            for s in self._sfx:
                buf, pos, gain = s
                k = min(n, len(buf) - pos)
                out[:k] += buf[pos : pos + k] * gain
                s[1] += k
                if s[1] < len(buf):
                    alive.append(s)
            self._sfx = alive
        return np.tanh(out * self.master)          # soft limiter, no clipping

    def _callback(self, outdata, frames, _time, _status) -> None:
        outdata[:, 0] = self._mix(frames)

    def render(self, seconds: float) -> np.ndarray:
        """Offline rendering (tests, headless machines)."""
        blocks = [self._mix(BLOCK) for _ in range(int(seconds * SR / BLOCK))]
        return np.concatenate(blocks)


def save_wav(path: str, pcm: np.ndarray, sr: int = SR) -> None:
    import wave
    data = (np.clip(pcm, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


if __name__ == "__main__":       # python -m brain.audio  -> hear every sound
    mx = Mixer()
    mx.music("idle", 0.4, fade=1.0)
    time.sleep(4)
    for name in mx.sfx_bank:
        print("sfx:", name)
        time.sleep(mx.sfx(name) + 0.6)
    mx.music(None, fade=2.0)
    time.sleep(2.5)
    mx.close()
