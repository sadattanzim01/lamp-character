"""Builds the simulated lamp that MuJoCo runs.

What this file does, in plain words:
    The challenge gave us a URDF file (a text description of the lamp: its
    parts, joints, sizes and limits). MuJoCo can read URDF, but not exactly
    the way we need. So this file loads the URDF *without changing it* and
    adds what the simulation is missing.

The steps:
    1. The lampshade mesh came as a text ("ASCII") STL file, but MuJoCo only
       reads the compact binary STL format, so we convert it once.
    2. We tell MuJoCo to keep the parts it would normally throw away: the
       visible shapes, and the empty marker frames for the light, camera and
       speaker.
    3. We add what the URDF leaves out on purpose: a table to stand on, room
       lighting, a spotlight inside the shade (the lamp's light), and a motor
       ("actuator") for every joint, each limited to the torque the URDF allows.

Why it matters: everything physical about the lamp (sizes, weights, joint
limits, motor strength) comes from the supplied file, so the simulation
respects the real robot's constraints.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import mujoco
import numpy as np

# Folder paths. ROOT is the top of the project (one level above body/).
ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = ROOT / "robot" / "dummy_lamp_5dof.urdf"   # the supplied robot file
GENERATED = ROOT / "generated"                         # files we create (git-ignored)

# The lamp's 5 joints (its 5 degrees of freedom), from the base up to the head.
# This order is used everywhere else in the project.
JOINTS = [
    "base_yaw_joint",        # spins the whole lamp left/right
    "shoulder_pitch_joint",  # tilts the lower arm forward/back
    "elbow_pitch_joint",     # bends the upper arm
    "neck_yaw_joint",        # turns the head left/right
    "head_pitch_joint",      # nods the head up/down
]

# How stiff each joint's motor is: (kp, kv).
#   kp = how hard the motor pushes toward its target angle (spring strength)
#   kv = how much it resists moving fast (damping, stops wobbling)
# Joints near the base carry more of the arm's weight, so they get stronger
# values. Gravity is handled separately (see "gravcomp" below), so these
# numbers only decide how crisply each joint follows its target.
SERVO_GAINS = {
    "base_yaw_joint": (40.0, 4.0),
    "shoulder_pitch_joint": (60.0, 6.0),
    "elbow_pitch_joint": (45.0, 4.5),
    "neck_yaw_joint": (15.0, 1.2),
    "head_pitch_joint": (15.0, 1.2),
}


def _ascii_stl_to_binary(src: Path, dst: Path) -> None:
    """Convert the text (ASCII) lampshade mesh into binary STL for MuJoCo.

    An STL mesh is a list of triangles. Each triangle has a "normal" (the
    direction it faces) and 3 corner points. We read those numbers out of the
    text file and write them in the binary layout: an 80-byte header, the
    triangle count, then 50 bytes per triangle.
    """
    text = src.read_text()
    normals = re.findall(r"facet normal\s+(\S+)\s+(\S+)\s+(\S+)", text)
    verts = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    # Every triangle must have exactly 3 corners, otherwise the file is broken.
    if len(verts) != 3 * len(normals):
        raise ValueError(f"Malformed STL {src}: {len(normals)} facets, {len(verts)} vertices")
    with dst.open("wb") as f:
        f.write(b"binary STL converted from ASCII".ljust(80, b"\0"))  # 80-byte header
        f.write(struct.pack("<I", len(normals)))                     # number of triangles
        for i, n in enumerate(normals):
            f.write(struct.pack("<3f", *map(float, n)))               # facing direction
            for v in verts[3 * i : 3 * i + 3]:
                f.write(struct.pack("<3f", *map(float, v)))           # the 3 corners
            f.write(b"\0\0")                                          # unused 2 bytes


def _prepared_urdf() -> str:
    """Return the URDF text, ready for MuJoCo, without editing the original file.

    Two small changes are made to an in-memory copy:
      * point the lampshade at the converted binary mesh, and
      * insert a <mujoco> settings block that tells MuJoCo to keep visual
        shapes and the marker frames (light, camera, speaker) instead of
        merging or dropping them.
    """
    GENERATED.mkdir(exist_ok=True)
    mesh_src = ROOT / "robot" / "assets" / "lamp_shade.stl"
    mesh_dst = GENERATED / "lamp_shade.stl"
    # Only convert again if the binary copy is missing or older than the original.
    if not mesh_dst.exists() or mesh_dst.stat().st_mtime < mesh_src.stat().st_mtime:
        _ascii_stl_to_binary(mesh_src, mesh_dst)

    urdf = URDF_PATH.read_text()
    urdf = urdf.replace('filename="assets/lamp_shade.stl"', 'filename="lamp_shade.stl"')
    # MuJoCo-specific import settings:
    #   discardvisual="false" -> keep the shapes you can see (not just collision shapes)
    #   fusestatic="false"    -> keep fixed child frames (light/camera/speaker) as their own bodies
    #   balanceinertia="true" -> quietly fix slightly inconsistent inertia numbers
    ext = (
        "<mujoco><compiler "
        f'meshdir="{GENERATED.as_posix()}" strippath="false" '
        'discardvisual="false" fusestatic="false" balanceinertia="true"/>'
        "</mujoco>"
    )
    # Put the settings block right after the opening <robot ...> tag.
    return re.sub(r"(<robot[^>]*>)", r"\1" + ext, urdf, count=1)


def build_spec() -> mujoco.MjSpec:
    """Load the lamp and add the table, lights, glowing shade and joint motors.

    Returns an editable MuJoCo "spec" (a model description that hasn't been
    compiled yet). Call build_model() if you want the finished model.
    """
    spec = mujoco.MjSpec.from_string(_prepared_urdf())
    spec.modelname = "character_lamp"
    spec.option.timestep = 0.002                 # physics updates 500 times per second
    spec.visual.headlight.ambient = [0.25, 0.25, 0.28]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.38]
    spec.visual.quality.shadowsize = 4096        # sharper shadows

    # The URDF describes each part twice: a "visual" shape (what you see) and a
    # "collision" shape (a simpler shape used to detect bumping into things).
    # Visual shapes go in display group 2 and never collide; collision shapes go
    # in group 3, which is hidden in the viewer but still used for contact.
    for g in spec.geoms:
        is_visual = g.contype == 0 and g.conaffinity == 0
        if not is_visual:
            g.group = 3
        else:
            g.group = 2

    world = spec.worldbody
    # A tabletop. The URDF has no joint fixing the lamp to anything, so we treat
    # the heavy 3 kg base as sitting on a desk: the base is attached to the world
    # and can't slide or tip.
    world.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05],
        rgba=[0.32, 0.27, 0.23, 1], group=2,
    )
    # A soft room light so the scene isn't pitch black.
    world.add_light(name="key", pos=[1.2, -1.0, 2.0], dir=[-0.5, 0.4, -0.8],
                    diffuse=[0.35, 0.35, 0.38], castshadow=False)

    # The lamp's own light: a spotlight placed at the URDF's light_emitter frame,
    # shining straight out of the shade. Its colour and brightness are changed
    # while the program runs (see lamp_body.py).
    emitter = spec.body("light_emitter_link")
    emitter.add_light(
        name="lamp_spot", type=mujoco.mjtLightType.mjLIGHT_SPOT,
        pos=[0, 0, 0], dir=[1, 0, 0], cutoff=40, exponent=8,   # 40 degree cone
        diffuse=[1.0, 0.85, 0.6], specular=[0.2, 0.2, 0.2], castshadow=True,
        attenuation=[0.6, 0.0, 0.6],                            # fades with distance
    )

    # Make the bulb and the disk inside the shade glow. They share one
    # "emissive" material, so changing that material's colour at runtime
    # recolours the whole inside of the shade at once.
    spec.add_material(name="bulb", rgba=[1.0, 0.85, 0.6, 1.0], emission=0.8)
    for g in spec.geoms:
        # Find the bulb/disk by the light-yellow colour the URDF gave them.
        if g.group == 2 and np.allclose(g.rgba, [1.0, 0.95, 0.76, 0.95], atol=1e-3):
            g.material = "bulb"
            # Reset the shape's own colour to MuJoCo's default; otherwise it
            # would override the material colour and the glow wouldn't show.
            g.rgba = [0.5, 0.5, 0.5, 1]

    # Gravity compensation: the motors automatically add the torque needed to
    # hold the arm up against gravity. "actgravcomp" makes that holding torque
    # come out of each motor's own torque limit, like a real servo that has to
    # spend part of its strength just holding the arm still.
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0

    # One position motor per joint: you give it a target angle and it drives
    # the joint there.
    for name in JOINTS:
        j = spec.joint(name)
        j.actgravcomp = True
        kp, kv = SERVO_GAINS[name]
        act = spec.add_actuator(name=name.replace("_joint", ""), target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=kp, kv=kv)
        # Targets can never be outside the joint's range.
        act.ctrllimited = True
        act.ctrlrange = list(j.range)
        # The URDF's "effort" (max torque) becomes the motor's strength cap.
        effort = _urdf_effort(name)
        act.forcelimited = True
        act.forcerange = [-effort, effort]
        j.actfrclimited = True
        j.actfrcrange = [-effort, effort]

    return spec


# Filled the first time _urdf_effort() runs, so the URDF is only read once.
_EFFORT_CACHE: dict[str, float] = {}


def _urdf_effort(joint: str) -> float:
    """Max torque (N*m) the URDF allows for this joint."""
    if not _EFFORT_CACHE:
        text = URDF_PATH.read_text()
        for m in re.finditer(r'<joint name="([^"]+)".*?<limit[^>]*effort="([^"]+)"', text, re.S):
            _EFFORT_CACHE[m.group(1)] = float(m.group(2))
    return _EFFORT_CACHE[joint]


def urdf_limits() -> dict[str, dict[str, float]]:
    """Read every joint's limits from the URDF.

    For each joint you get:
      lower / upper            hard limits: the joint physically can't go past these
      soft_lower / soft_upper  safe limits, a little inside the hard ones; our
                               motion code stays within these
      velocity                 max speed (rad/s)
      effort                   max torque (N*m)
    """
    text = URDF_PATH.read_text()
    out: dict[str, dict[str, float]] = {}
    for block in re.findall(r"<joint .*?</joint>", text, re.S):
        name = re.search(r'name="([^"]+)"', block).group(1)
        lim = re.search(r"<limit ([^/]*)/>", block)
        soft = re.search(r"<safety_controller ([^/]*)/>", block)
        if not lim:
            continue          # fixed joints (e.g. the light frame) have no limits
        attrs = dict(re.findall(r'(\w+)="([^"]+)"', lim.group(1)))
        sattrs = dict(re.findall(r'(\w+)="([^"]+)"', soft.group(1))) if soft else {}
        out[name] = {
            "lower": float(attrs["lower"]),
            "upper": float(attrs["upper"]),
            "velocity": float(attrs["velocity"]),
            "effort": float(attrs["effort"]),
            # If a joint had no safety limits, fall back to the hard limits.
            "soft_lower": float(sattrs.get("soft_lower_limit", attrs["lower"])),
            "soft_upper": float(sattrs.get("soft_upper_limit", attrs["upper"])),
        }
    return out


def build_model() -> mujoco.MjModel:
    """The finished, ready-to-simulate lamp model."""
    return build_spec().compile()


# Run `python -m body.model` to build the model and save it as a MuJoCo XML
# file you can inspect (generated/lamp_scene.xml).
if __name__ == "__main__":
    spec = build_spec()
    model = spec.compile()
    out = GENERATED / "lamp_scene.xml"
    out.write_text(spec.to_xml())
    print(f"bodies={model.nbody} joints={model.njnt} actuators={model.nu} "
          f"geoms={model.ngeom} lights={model.nlight}")
    print(f"wrote {out}")
