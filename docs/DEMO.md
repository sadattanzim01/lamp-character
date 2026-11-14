# Demo: one continuous take (about 3 minutes)

Covers all five required moments and all five channels (motion, light, voice, SFX, music)
in one interaction. The numbers in brackets are the challenge's moments.

## Setup (before recording)

1. Put a **book** (or mug) on the desk within camera view, plus one more object to hold up.
2. `./run.sh --forget` (macOS) so memory starts empty. Arrange the windows: **MuJoCo on the left,
   "lamp eyes" camera window on the right**.
3. Speaker volume at about 60 % so the mic picks up the lamp's voice, music and SFX in the room.
4. Record with **QuickTime → New Screen Recording → Options → Microphone: MacBook Microphone**
   (QuickTime does not capture system audio directly; the built-in mic hears the speakers).
   OBS with desktop audio also works.
5. Look **away** from the screen before you press record.

## Script

| # | You do / say | Lamp should | Shows |
|---|---|---|---|
| 1 | Stay looking away for 5 s | asleep, slumped, slow warm light "breathing", lullaby | music · light |
| 2 | Turn and look at the screen | "hm?" blip, lifts head → chirp, orange flash, perk + wiggle, music fades, says "Oh! Hi there." | **[1] engagement, [2] character response**: motion · light · SFX · voice |
| 3 | "Hi Lumi, my name is Sadat. What do you do?" | answers out loud with a gesture; light pulses with its voice | **[3] spoken interaction** |
| 4 | Hold up the second object, keep it still for 1 s | glances at it, blip, "Ooh, a …" | **[4] scene memory**: observe |
| 5 | "What am I holding?" | describes it (colour, text on it) | [4] retains details |
| 6 | Put it away. "Can you make your light a bit warmer?" | the light really changes and stays | light control |
| 7 | "Shine your light on the book." | "Sure…" → turns to the book (red box + yellow circle meet in the camera window), re-checks, "There, the book is lit up." | **[5] goal-directed action** |
| 7b | *(optional)* While it aims, slide the book a little | the re-check sees the miss and re-aims | re-observe + correct |
| 8 | "Look at me again." Then: "What did I show you earlier?" | names the object and roughly when | **[4] later recall** |
| 9 | "Thanks Lumi, see you." Then look away for about 3 s | blue light + head tilt → "bye" sigh, droop, dims, lullaby returns | **[1] disengage**, music |

Stop recording after the lullaby returns.

## Tips

- Speak, then pause: the lamp waits 0.8 s of silence before answering.
- If it mishears, just repeat; the camera window shows `hearing you` while it listens.
- If a take goes wrong, `Ctrl+C` and start again with `./run.sh --forget`.
- After recording: `python scripts/report.py` refreshes `docs/MEASUREMENTS.md` with this session's numbers.
