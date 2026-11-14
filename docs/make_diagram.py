"""Generate docs/architecture.svg (+ .png): architecture and data-flow diagram.

    python docs/make_diagram.py
Hand-placed layout so the note's one diagram reads left-to-right:
sensors -> brain -> body, with the cloud boundary on top.
"""
from __future__ import annotations

from pathlib import Path

W, H = 1600, 930
OUT = Path(__file__).resolve().parent

C = {  # fill, stroke, text
    "hw": ("#eef1f8", "#4a5d8a", "#1d2747"),
    "local": ("#eef6ee", "#3a7d44", "#16331b"),
    "cloud": ("#fdf0e6", "#c46a1b", "#5a2e08"),
    "body": ("#f3eefa", "#6b4ea1", "#2c1e47"),
    "store": ("#fffbe8", "#a68a1f", "#4a3d08"),
}
FONT = "Helvetica, Arial, sans-serif"
parts: list[str] = []
boxes: dict[str, tuple[float, float, float, float]] = {}


def box(key, x, y, w, h, title, sub="", kind="local", bold=True):
    fill, stroke, text = C[kind]
    boxes[key] = (x, y, w, h)
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="1.6"/>')
    lines = sub.split("|") if sub else []
    total = 1 + len(lines)
    y0 = y + h / 2 - (total - 1) * 8.5 + 5
    parts.append(f'<text x="{x + w / 2}" y="{y0}" text-anchor="middle" font-family="{FONT}" '
                 f'font-size="15" font-weight="{700 if bold else 400}" fill="{text}">{title}</text>')
    for i, ln in enumerate(lines):
        parts.append(f'<text x="{x + w / 2}" y="{y0 + 17 * (i + 1)}" text-anchor="middle" '
                     f'font-family="{FONT}" font-size="12.5" fill="{text}">{ln}</text>')


def group(x, y, w, h, title, kind):
    fill, stroke, text = C[kind]
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" '
                 f'stroke="{stroke}" stroke-width="2" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{x + 14}" y="{y + h - 14}" font-family="{FONT}" font-size="16" '
                 f'font-weight="700" fill="{stroke}">{title}</text>')


def side(key, s):
    x, y, w, h = boxes[key]
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y), "b": (x + w / 2, y + h)}[s]


def arrow(a, b, label="", color="#555", width=1.6, both=False, via=None, lx=0, ly=-6, dash=""):
    pts = [a] + (via or []) + [b]
    d = "M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts)
    marker = 'marker-end="url(#ah)"' + (' marker-start="url(#ah)"' if both else "")
    da = f'stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" {da} {marker}/>')
    if label:
        mx = (pts[len(pts) // 2 - 1][0] + pts[len(pts) // 2][0]) / 2 + lx
        my = (pts[len(pts) // 2 - 1][1] + pts[len(pts) // 2][1]) / 2 + ly
        for i, ln in enumerate(label.split("|")):
            parts.append(f'<text x="{mx}" y="{my + 14 * i}" text-anchor="middle" font-family="{FONT}" '
                         f'font-size="12" fill="#333" stroke="white" stroke-width="4" '
                         f'paint-order="stroke">{ln}</text>')


# ------------------------------------------------------------------ layout
group(150, 120, 1030, 790, "Brain process · brain.main · 15 Hz main loop", "local")
group(1215, 120, 370, 790, "Body process · body.server · 100 Hz", "body")

box("api", 470, 18, 520, 62, "Anthropic API (Claude)",
    "text turns + memory notes · one JPEG only on scene change, 'what's this?', goals", "cloud")

box("cam", 15, 200, 115, 50, "Camera", kind="hw")
box("mic", 15, 520, 115, 50, "Microphone", kind="hw")
box("spk", 15, 790, 115, 50, "Speaker", kind="hw")

# perception row
box("att", 185, 170, 190, 64, "Attention", "Haar face + eyes · ~7 ms")
box("eng", 425, 165, 215, 74, "Engagement FSM", "idle → noticing → engaged|→ leaving (hysteresis)")
box("scn", 185, 290, 190, 64, "Scene watcher", "local change detect, face masked")
box("mem", 425, 290, 215, 64, "Memory", "shown objects · background ·|conversation notes", "store")
# speech row
box("ep", 185, 505, 190, 64, "Endpointer", "mic open only while engaged")
box("stt", 185, 610, 190, 64, "Speech-to-text", "Whisper base.en · local · 0.7 s")
box("mind", 425, 480, 215, 118, "Mind", "respond tool → say · gesture ·|mood · light · notes ·|remember · goal")
box("tts", 425, 650, 215, 64, "Text-to-speech", "Piper · local · 0.3 s")
box("goal", 730, 400, 200, 80, "Goal runner", "plan → act → verify|→ correct (≤3)")
# expression column
box("chr", 730, 520, 200, 120, "Character", "single owner of expression:|motion · light · SFX ·|music · voice")
box("mix", 730, 760, 200, 72, "Mixer", "music bed · SFX · voice|(music ducks under speech)")
box("bc", 985, 545, 175, 64, "Body client", "commands · state")

# body column
box("srv", 1240, 170, 320, 64, "Protocol server", "main thread owns physics state", "body")
box("exe", 1240, 290, 320, 80, "Executor", "pose · gaze · gesture · joints · light · stop|ids · preemption · done events", "body")
box("ik", 1240, 425, 320, 64, "Gaze IK", "damped least squares on MuJoCo kinematics", "body")
box("lb", 1240, 545, 320, 96, "LampBody (safety)", "URDF soft limits · self-collision check|velocity + acceleration profiles|torque ≤ URDF effort", "body")
box("sim", 1240, 700, 320, 80, "MuJoCo", "supplied URDF untouched · spotlight|+ emissive shade = light", "body")

# ------------------------------------------------------------------ edges
arrow(side("cam", "r"), side("att", "l"), via=[(150, 225), (160, 202)])
arrow(side("cam", "r"), side("scn", "l"), via=[(150, 225), (160, 322)])
arrow(side("att", "r"), side("eng", "l"))
arrow(side("eng", "r"), (730, 560), "state", via=[(700, 202), (700, 560)], lx=14, ly=-60)
arrow(side("scn", "r"), side("mem", "l"))
arrow(side("mic", "r"), side("ep", "l"))
arrow(side("ep", "b"), side("stt", "t"))
arrow(side("stt", "r"), (425, 570), "text", via=[(400, 642), (400, 570)], lx=-14, ly=-30)
arrow(side("mem", "b"), side("mind", "t"))
arrow(side("mind", "b"), side("tts", "t"))
arrow(side("tts", "r"), (730, 615), "audio", via=[(712, 682), (712, 615)], lx=-24, ly=40)
arrow((640, 500), side("goal", "l"), "goal", via=[(680, 500), (680, 440)], lx=-16, ly=-6)
arrow(side("goal", "b"), side("chr", "t"))
arrow((640, 540), (730, 540), "")
arrow(side("chr", "b"), side("mix", "t"))
arrow(side("mix", "l"), side("spk", "r"), via=[(150, 796), (140, 815)])
arrow(side("chr", "r"), side("bc", "l"))
# cloud
cl = "#c46a1b"
arrow((340, 290), (470, 62), "JPEG on change", color=cl, via=[(340, 262), (400, 262), (400, 62)], dash="5 4", lx=-8, ly=-6)
arrow((600, 480), (655, 80), "transcript + notes", color=cl, both=True, via=[(600, 460), (655, 460)], dash="5 4", lx=-20, ly=-196)
arrow(side("goal", "t"), (830, 80), "frame + goal,|locate", color=cl, both=True, dash="5 4", lx=52, ly=-40)
# protocol
arrow(side("bc", "r"), (1240, 202), "", color="#6b4ea1", width=2.6, both=True,
      via=[(1200, 577), (1200, 202)])
parts.append(f'<g font-family="{FONT}" font-size="12.5" fill="#2c1e47">'
             f'<rect x="990" y="630" width="170" height="132" rx="6" fill="white" stroke="#6b4ea1"/>'
             f'<text x="1000" y="649" font-weight="700">WebSocket + JSON</text>'
             f'<text x="1000" y="668">→ pose · gaze · gesture</text>'
             f'<text x="1000" y="685">   joints · light · stop</text>'
             f'<text x="1000" y="704">← ack (+adjusted)</text>'
             f'<text x="1000" y="721">← done (completed /</text>'
             f'<text x="1000" y="738">   preempted)</text>'
             f'<text x="1000" y="755">← state 20 Hz</text></g>')
for a, b in (("srv", "exe"), ("exe", "ik"), ("ik", "lb"), ("lb", "sim")):
    arrow(side(a, "b"), side(b, "t"))

# legend
lx, ly = 1240, 800
for i, (k, t) in enumerate((("local", "runs on the laptop"), ("cloud", "cloud (only text + occasional JPEG)"),
                            ("body", "body: owns safety"), ("store", "memory (text only)"))):
    f, s, _ = C[k]
    parts.append(f'<rect x="{lx}" y="{ly + 22 * i}" width="16" height="14" rx="3" fill="{f}" stroke="{s}"/>'
                 f'<text x="{lx + 24}" y="{ly + 12 + 22 * i}" font-family="{FONT}" font-size="12.5" '
                 f'fill="#333">{t}</text>')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
       'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="white"/>' + "".join(parts) + "</svg>")
(OUT / "architecture.svg").write_text(svg)
print("wrote", OUT / "architecture.svg")
