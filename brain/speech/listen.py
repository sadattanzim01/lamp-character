"""Listens to the microphone and figures out when you start and stop talking.

What this file does, in plain words:
    It records the microphone in tiny 30 ms pieces. When the sound gets clearly
    louder than the room's background noise, it decides "someone started
    talking" and starts collecting. When it's been quiet for 0.8 s, it decides
    "they finished" and hands the whole recording (one "utterance") to the
    speech recogniser. Working out where speech starts and ends like this is
    called "endpointing".

Three design choices:
    * No wake word ("Hey Lumi"). The lamp only listens while you're engaged
      with it, so LOOKING at the lamp is what turns the microphone on.
    * It never hears itself. While Lumi is talking the listener is paused
      (called "half-duplex"), so it can't transcribe its own voice coming out
      of the laptop speaker.
    * It adapts to the room. It keeps measuring the background noise (the
      "noise floor"), so a noisy fan or quiet room both work.

Privacy: the audio stays in memory and only goes to the speech recogniser,
which runs on the laptop. No audio is ever sent to the cloud.
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

SR = 16000            # 16,000 samples per second: what the Whisper recogniser expects
BLOCK = 480           # 480 samples = 30 ms per piece
START_BLOCKS = 4      # 4 loud pieces in a row (120 ms) = speech has started
END_SILENCE = 0.8     # 0.8 s of quiet = you've finished talking
PRE_ROLL = 0.35       # also keep 0.35 s from BEFORE speech was detected, so the first syllable isn't cut off
MAX_SPEECH = 12.0     # stop recording after 12 s no matter what
MIN_SPEECH = 0.35     # anything shorter is probably a click or cough: ignore it


@dataclass
class Utterance:
    """One finished stretch of speech."""
    audio: np.ndarray        # the recorded sound (16 kHz, mono)
    t_start: float           # when speech started (wall-clock seconds)
    t_end: float             # when we decided it ended

    @property
    def seconds(self) -> float:
        """How long the recording is."""
        return len(self.audio) / SR


class Endpointer:
    """The start/stop-of-speech logic, with no microphone attached.

    Kept separate from the microphone so it can be tested by feeding it
    made-up audio.
    """

    def __init__(self) -> None:
        self.floor = 0.005                     # current estimate of the background noise level
        # A rolling buffer of the most recent pieces, so we can include the
        # moment just before speech was detected.
        self.pre = collections.deque(maxlen=int(PRE_ROLL * SR / BLOCK))
        self.buf: list[np.ndarray] = []         # pieces of the current utterance
        self.loud = 0                           # loud pieces in a row (to detect a start)
        self.quiet = 0                          # quiet pieces in a row (to detect an end)
        self.active = False                     # are we in the middle of someone talking?
        self.t_start = 0.0

    def reset(self) -> None:
        """Forget any half-collected speech and start fresh."""
        self.pre.clear()
        self.buf, self.loud, self.quiet, self.active = [], 0, 0, False

    def push(self, block: np.ndarray, now: float) -> Utterance | None:
        """Feed in one 30 ms piece. Returns an Utterance when one just finished, else None."""
        # Loudness of this piece (RMS = average energy). The tiny 1e-9 avoids zero.
        rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
        # Thresholds relative to the room's noise: "loud" = 3.5x the background,
        # "quiet" = under 2x. Having two different thresholds stops it flickering
        # on and off around a single value. The max() sets a minimum for very quiet rooms.
        on = max(0.012, self.floor * 3.5)
        off = max(0.008, self.floor * 2.0)
        if not self.active:
            # Nobody talking yet. Slowly update our idea of the background noise
            # (quickly for quiet pieces, very slowly for loud ones, so speech
            # doesn't get mistaken for background).
            self.floor += (0.02 if rms < self.floor * 3 else 0.002) * (rms - self.floor)
            self.pre.append(block)
            self.loud = self.loud + 1 if rms > on else 0
            if self.loud >= START_BLOCKS:
                # Speech started. Begin the recording with the pre-roll pieces.
                self.active, self.quiet = True, 0
                self.t_start = now - len(self.pre) * BLOCK / SR
                self.buf = list(self.pre)
            return None
        # Someone is talking: keep collecting, and count quiet pieces.
        self.buf.append(block)
        self.quiet = self.quiet + 1 if rms < off else 0
        dur = len(self.buf) * BLOCK / SR
        if self.quiet * BLOCK / SR >= END_SILENCE or dur >= MAX_SPEECH:
            # They've stopped (or it's too long). Package the recording.
            audio = np.concatenate(self.buf)
            # Cut most of the silence at the end (keep ~0.18 s so words aren't clipped).
            trim = max(0, self.quiet - 6) * BLOCK
            audio = audio[: len(audio) - trim] if trim else audio
            self.reset()
            if len(audio) / SR - 0.18 < MIN_SPEECH:
                return None                    # too short to be real speech
            return Utterance(audio.astype(np.float32), self.t_start, now)
        return None


class Listener:
    """The live microphone. Finished utterances appear in `utterances` (a queue)."""

    def __init__(self, device=None):
        import sounddevice as sd
        self._q: queue.Queue[np.ndarray] = queue.Queue()            # raw 30 ms pieces from the mic
        self.utterances: queue.Queue[Utterance] = queue.Queue()     # finished speech, ready to transcribe
        self.enabled = threading.Event()        # the on/off switch: only listens while set
        self.ep = Endpointer()
        self.level = 0.0                        # current mic loudness (for debugging)
        # The audio library calls our little callback with each 30 ms piece; we
        # just drop it in the queue and return quickly (audio callbacks must be fast).
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                      blocksize=BLOCK, device=device,
                                      callback=lambda d, *_: self._q.put(d[:, 0].copy()))
        self._stream.start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        """Background thread: run each piece through the endpointer."""
        while True:
            block = self._q.get()
            self.level = float(np.sqrt(np.mean(block ** 2)))
            if not self.enabled.is_set():
                # Switched off: throw away anything half-collected.
                if self.ep.active:
                    self.ep.reset()
                continue
            utt = self.ep.push(block, time.time())
            if utt is not None:
                self.utterances.put(utt)

    def pause(self) -> None:
        """Stop listening (e.g. while Lumi speaks), and throw away audio already captured."""
        self.enabled.clear()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def resume(self) -> None:
        """Start listening again with a clean slate."""
        self.ep.reset()
        self.enabled.set()

    @property
    def hearing_speech(self) -> bool:
        """True while someone is talking right now (shown as "hearing you" in the camera window)."""
        return self.ep.active

    def close(self) -> None:
        """Turn off the microphone."""
        self._stream.stop()
        self._stream.close()
