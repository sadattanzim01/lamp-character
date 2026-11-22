"""All of Lumi's sound: sound effects, the lullaby music, and its voice.

What this file does, in plain words:
    1. It CREATES every sound effect and the music from scratch with math
       (sine waves), so there are no audio files to download or license.
    2. It MIXES three layers into one stream going to the speaker:
         music  a looping background tune that can fade in and out smoothly
         sfx    short sound effects (chirps, blips); several can overlap
         voice  Lumi's speech; while it plays, the music automatically gets
                quieter ("ducking") so you can hear the words

Why one mixer instead of each part playing its own sounds:
    Because everything goes through one place, the layers can react to each
    other (the music ducks under the voice). And on Linux it's a single audio
    connection instead of several fighting over the speaker.

Quick listen: `python -m brain.audio` plays the music and every sound effect.
"""
from __future__ import annotations

import threading
import time

import numpy as np

SR = 22050          # sample rate: 22,050 sound samples per second (same as the Piper voice, so no conversion needed)
BLOCK = 512         # the speaker asks for 512 samples at a time (~23 ms of sound)


# ------------------------------------------------------------------ making sounds
# A sound is just a long list of numbers between -1 and 1 (the speaker's position
# over time). These helpers build those lists.

def _env(n: int, attack: float, release: float) -> np.ndarray:
    """A volume shape ("envelope"): fade in over `attack` seconds, fade out over
    `release` seconds. Without it, sounds would start/stop with a harsh click."""
    e = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def _glide(f0: float, f1: float, dur: float, harm=(1.0, 0.3, 0.1)) -> np.ndarray:
    """A tone that slides in pitch from f0 to f1 Hz over `dur` seconds, like "wheee".

    `harm` adds quieter copies at 2x and 3x the pitch ("harmonics"), which makes
    it sound richer and more like a cute robot than a pure beep.
    """
    n = int(dur * SR)
    f = np.geomspace(f0, f1, n)                 # pitch at every sample (even steps on a musical scale)
    ph = 2 * np.pi * np.cumsum(f) / SR           # running "phase" so the slide is smooth
    tone = sum(a * np.sin((k + 1) * ph) for k, a in enumerate(harm))
    return tone * _env(n, 0.01, dur * 0.4)


def _bell(freq: float, dur: float, decay: float = 2.5) -> np.ndarray:
    """A bell / music-box note: several tones that ring and then die away.
    Higher tones die faster, like a real bell."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    partials = [(1.0, 1.0), (2.0, 0.35), (3.01, 0.12), (4.2, 0.06)]   # (pitch multiple, loudness)
    s = sum(a * np.sin(2 * np.pi * freq * m * t) * np.exp(-decay * m * t) for m, a in partials)
    return s * _env(n, 0.004, 0.05)


def _silence(d: float) -> np.ndarray:
    """`d` seconds of quiet."""
    return np.zeros(int(d * SR))


def _seq(*parts: np.ndarray) -> np.ndarray:
    """Play sounds one after another."""
    return np.concatenate(parts)


def _norm(x: np.ndarray, peak: float = 0.8) -> np.ndarray:
    """Scale a sound so its loudest point is `peak` (keeps volumes consistent)."""
    m = np.max(np.abs(x)) or 1.0
    return (x / m * peak).astype(np.float32)


def make_sfx() -> dict[str, np.ndarray]:
    """Build every sound effect. Each one is designed to "mean" something."""
    return {
        # a curious upward "hm?" when the lamp first notices someone
        "notice": _norm(_glide(500, 820, 0.18), 0.5),
        # a bright two-step chirp: "hello!"
        "hello": _norm(_seq(_glide(600, 900, 0.09), _silence(0.04), _glide(750, 1300, 0.14))),
        # a sighing slide downward: "bye"
        "bye": _norm(_glide(900, 380, 0.45, harm=(1.0, 0.2)), 0.6),
        # a soft blip: "I'm listening"
        "listen": _norm(_glide(880, 1100, 0.07), 0.45),
        # three bubbly blips: "thinking..."
        "think": _norm(_seq(*[_seq(_glide(f, f * 1.1, 0.05), _silence(0.05))
                              for f in (700, 900, 800)]), 0.4),
        # a rising arpeggio (C-E-G-C): "success!"
        "success": _norm(_seq(*[_bell(f, 0.16, 4) for f in (523, 659, 784)], _bell(1047, 0.5, 3))),
        # a low wobble: "can't do that"
        "error": _norm(_seq(_glide(300, 240, 0.15), _glide(260, 200, 0.25)), 0.6),
    }


def make_music() -> dict[str, np.ndarray]:
    """Build the music. 'idle' is a slow music-box lullaby (~16 s, loops forever),
    played while the lamp is asleep."""
    beat = 60 / 72                     # 72 beats per minute
    # The melody as (note number, length in beats). Note numbers follow the MIDI
    # standard: 60 = middle C, each +1 = one semitone up. Only the notes C, D, E,
    # G, A are used (a "pentatonic" scale), which always sounds gentle.
    notes = [
        (72, 1), (76, 1), (79, 2), (76, 1), (74, 1), (72, 2),
        (69, 1), (72, 1), (74, 2), (76, 1), (74, 1), (72, 2),
        (72, 1), (76, 1), (79, 1), (81, 1), (79, 2), (76, 2),
        (74, 1), (76, 1), (72, 2), (67, 2), (72, 2),
    ]
    total = sum(b for _, b in notes) * beat     # length of the tune in seconds
    n = int(total * SR)
    melody = np.zeros(n + SR * 3)               # 3 s extra so the last notes can ring out
    pos = 0.0
    for midi, beats in notes:
        f = 440 * 2 ** ((midi - 69) / 12)        # note number -> pitch in Hz (69 = A = 440 Hz)
        tone = _bell(f, 2.5, decay=1.6)
        i = int(pos * SR)
        melody[i : i + len(tone)] += tone
        pos += beats * beat
    # The ringing that spills past the end is added back onto the start, so when
    # the loop repeats there's no gap or click.
    tail = melody[n:]
    melody = melody[:n]
    melody[: len(tail)] += tail
    # A soft, slowly swelling background hum (the notes C and G) for warmth.
    t = np.arange(n) / SR
    pad = 0.12 * (np.sin(2 * np.pi * 130.8 * t) + 0.7 * np.sin(2 * np.pi * 196.0 * t))
    pad *= 0.6 + 0.4 * np.sin(2 * np.pi * t / total)
    return {"idle": _norm(melody + pad, 0.6)}


# ------------------------------------------------------------------ the mixer
class Mixer:
    """Plays music, sound effects and voice together through the speaker.

    How playback works: the speaker library (sounddevice) calls our _callback
    about 43 times a second asking for the next 512 samples. Each time, we add
    up whatever is playing (voice + music + effects) and hand it over.
    """

    def __init__(self, output: bool = True, device=None):
        self.sfx_bank = make_sfx()             # all sound effects, built once at startup
        self.music_bank = make_music()
        self._lock = threading.Lock()          # the speaker thread and the brain share these lists
        self._sfx: list[list] = []             # effects playing now: [sound, position, volume]
        self._voice: list = []                 # speech waiting/playing: [sound, position, finished-signal]
        self._music: np.ndarray | None = None  # current music track
        self._music_pos = 0                    # where we are in the music loop
        self._music_vol = 0.0                  # current music volume...
        self._music_target = 0.0               # ...and the volume it's fading toward
        self._music_rate = 1.0                 # how fast it fades (volume change per second)
        self.master = 0.9                      # overall volume
        self.duck = 0.25                       # music plays at 25% volume while Lumi talks
        self._stream = None
        if output:
            # output=False is for tests and machines without a speaker.
            import sounddevice as sd
            self._stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32",
                                           blocksize=BLOCK, device=device,
                                           callback=self._callback)
            self._stream.start()

    # ---------------------------------------------------------- what the brain calls
    def sfx(self, name: str, gain: float = 1.0) -> float:
        """Play a sound effect (e.g. "hello"). Returns how long it lasts, in seconds."""
        buf = self.sfx_bank[name]
        with self._lock:
            self._sfx.append([buf, 0, gain])
        return len(buf) / SR

    def music(self, name: str | None, volume: float = 0.35, fade: float = 1.5) -> None:
        """Start a music track, change its volume, or fade it out (name=None),
        smoothly over `fade` seconds."""
        with self._lock:
            if name is not None and self._music is not self.music_bank[name]:
                # A different track: start it from the beginning, silent, then fade up.
                self._music, self._music_pos = self.music_bank[name], 0
                self._music_vol = 0.0
            self._music_target = 0.0 if name is None else volume
            self._music_rate = 1.0 / max(fade, 0.05)

    def voice(self, pcm: np.ndarray, sr: int = SR) -> threading.Event:
        """Queue Lumi's speech to play.

        Returns an Event (a flag) that becomes "set" when the speech has
        finished playing, so the conversation code can wait for it.
        """
        pcm = np.asarray(pcm, dtype=np.float32)
        if sr != SR:
            # Different sample rate (e.g. the macOS fallback voice): stretch it to match ours.
            x = np.linspace(0, len(pcm), int(len(pcm) * SR / sr), endpoint=False)
            pcm = np.interp(x, np.arange(len(pcm)), pcm).astype(np.float32)
        done = threading.Event()
        with self._lock:
            self._voice.append([pcm, 0, done])
        return done

    def stop_voice(self) -> None:
        """Cut off any speech right away (and tell anyone waiting that it's done)."""
        with self._lock:
            for v in self._voice:
                v[2].set()
            self._voice.clear()

    @property
    def speaking(self) -> bool:
        """True while speech is playing or queued."""
        return bool(self._voice)

    def close(self) -> None:
        """Stop the audio stream (on shutdown)."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    # ---------------------------------------------------------- mixing
    def _mix(self, n: int) -> np.ndarray:
        """Produce the next `n` samples: voice + (fading, ducked) music + effects."""
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            # 1. Voice: play queued sentences one after another.
            if self._voice:
                buf, pos, done = self._voice[0]
                k = min(n, len(buf) - pos)
                out[:k] += buf[pos : pos + k]
                self._voice[0][1] += k
                if self._voice[0][1] >= len(buf):
                    done.set()                   # this sentence finished
                    self._voice.pop(0)
            talking = bool(self._voice) or np.any(out)
            # 2. Music: move the volume a small step toward its target (smooth
            #    fades), drop it to the "duck" level while Lumi talks, and loop.
            if self._music is not None:
                step = self._music_rate * n / SR
                self._music_vol += np.clip(self._music_target - self._music_vol, -step, step)
                if self._music_vol > 1e-4:
                    m = self._music
                    idx = (self._music_pos + np.arange(n)) % len(m)   # % wraps around = looping
                    level = self._music_vol * (self.duck if talking else 1.0)
                    out += m[idx] * level
                self._music_pos = (self._music_pos + n) % len(self._music)
            # 3. Sound effects: add each one, and drop those that have finished.
            alive = []
            for s in self._sfx:
                buf, pos, gain = s
                k = min(n, len(buf) - pos)
                out[:k] += buf[pos : pos + k] * gain
                s[1] += k
                if s[1] < len(buf):
                    alive.append(s)
            self._sfx = alive
        # tanh gently squashes anything louder than the speaker can handle, so
        # overlapping sounds never distort with harsh clipping.
        return np.tanh(out * self.master)

    def _callback(self, outdata, frames, _time, _status) -> None:
        """Called by the speaker library whenever it needs more sound."""
        outdata[:, 0] = self._mix(frames)

    def render(self, seconds: float) -> np.ndarray:
        """Mix sound without a speaker (for tests, or to save a preview file)."""
        blocks = [self._mix(BLOCK) for _ in range(int(seconds * SR / BLOCK))]
        return np.concatenate(blocks)


def save_wav(path: str, pcm: np.ndarray, sr: int = SR) -> None:
    """Save sound to a .wav file (16-bit, mono)."""
    import wave
    data = (np.clip(pcm, -1, 1) * 32767).astype("<i2")   # -1..1 floats -> 16-bit whole numbers
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


# Run `python -m brain.audio` to hear the lullaby, then every sound effect in turn.
if __name__ == "__main__":
    mx = Mixer()
    mx.music("idle", 0.4, fade=1.0)
    time.sleep(4)
    for name in mx.sfx_bank:
        print("sfx:", name)
        time.sleep(mx.sfx(name) + 0.6)
    mx.music(None, fade=2.0)
    time.sleep(2.5)
    mx.close()
