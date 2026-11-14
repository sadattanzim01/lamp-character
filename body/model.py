"""Build the MuJoCo model for the lamp from the supplied URDF.

The supplied URDF is never edited. Instead this module:
  1. converts the ASCII STL shade to binary (MuJoCo only reads binary STL),
  2. injects MuJoCo compiler options so visuals and the semantic frames
     (light_emitter, camera, speaker) survive the import,
  3. adds what the URDF deliberately leaves out: a table, scene lighting,
     a steerable spotlight in the shade, and one position actuator per joint
     whose force limit equals the URDF effort limit.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = ROOT / "robot" / "dummy_lamp_5dof.urdf"
GENERATED = ROOT / "generated"

JOINTS = [
    "base_yaw_joint",
    "shoulder_pitch_joint",
    "elbow_pitch_joint",
    "neck_yaw_joint",
    "head_pitch_joint",
]

# Position-servo gains per joint (N*m/rad and N*m*s/rad). Bigger joints carry
# more of the arm, so they get stiffer servos. Gravity is compensated inside
# the actuator budget (actuatorgravcomp), so these gains only shape tracking.
SERVO_GAINS = {
    "base_yaw_joint": (40.0, 4.0),
    "shoulder_pitch_joint": (60.0, 6.0),
    "elbow_pitch_joint": (45.0, 4.5),
    "neck_yaw_joint": (15.0, 1.2),
    "head_pitch_joint": (15.0, 1.2),
}


def _ascii_stl_to_binary(src: Path, dst: Path) -> None:
    """Convert an ASCII STL to binary STL using only the standard library."""
    text = src.read_text()
    normals = re.findall(r"facet normal\s+(\S+)\s+(\S+)\s+(\S+)", text)
    verts = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if len(verts) != 3 * len(normals):
        raise ValueError(f"Malformed STL {src}: {len(normals)} facets, {len(verts)} vertices")
    with dst.open("wb") as f:
        f.write(b"binary STL converted from ASCII".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(normals)))
        for i, n in enumerate(normals):
            f.write(struct.pack("<3f", *map(float, n)))
            for v in verts[3 * i : 3 * i + 3]:
                f.write(struct.pack("<3f", *map(float, v)))
            f.write(b"\0\0")


def _prepared_urdf() -> str:
    """Return URDF text that points at the binary mesh and keeps visuals."""
    GENERATED.mkdir(exist_ok=True)
    mesh_src = ROOT / "robot" / "assets" / "lamp_shade.stl"
    mesh_dst = GENERATED / "lamp_shade.stl"
    if not mesh_dst.exists() or mesh_dst.stat().st_mtime < mesh_src.stat().st_mtime:
        _ascii_stl_to_binary(mesh_src, mesh_dst)

    urdf = URDF_PATH.read_text()
    urdf = urdf.replace('filename="assets/lamp_shade.stl"', 'filename="lamp_shade.stl"')
    ext = (
        "<mujoco><compiler "
        f'meshdir="{GENERATED.as_posix()}" strippath="false" '
        'discardvisual="false" fusestatic="false" balanceinertia="true"/>'
        "</mujoco>"
    )
    return re.sub(r"(<robot[^>]*>)", r"\1" + ext, urdf, count=1)


def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_string(_prepared_urdf())
    spec.modelname = "character_lamp"
    spec.option.timestep = 0.002
    spec.visual.headlight.ambient = [0.25, 0.25, 0.28]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.38]
    spec.visual.quality.shadowsize = 4096

    # URDF visuals and collisions both import as geoms. Keep visuals for
    # rendering only (group 2, no contacts); collisions go to hidden group 3.
    for g in spec.geoms:
        is_visual = g.contype == 0 and g.conaffinity == 0
        if not is_visual:
            g.group = 3
        else:
            g.group = 2

    world = spec.worldbody
    # Tabletop. The URDF has no world joint; we treat the 3 kg base as resting
    # on a table, so base_link is welded to the world (see README: physical notes).
    world.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05],
        rgba=[0.32, 0.27, 0.23, 1], group=2,
    )
    world.add_light(name="key", pos=[1.2, -1.0, 2.0], dir=[-0.5, 0.4, -0.8],
                    diffuse=[0.35, 0.35, 0.38], castshadow=False)

    # The lamp's own light: a spotlight at light_emitter_link shining along +x
    # of the head. Colour and intensity are changed at runtime.
    emitter = spec.body("light_emitter_link")
    emitter.add_light(
        name="lamp_spot", type=mujoco.mjtLightType.mjLIGHT_SPOT,
        pos=[0, 0, 0], dir=[1, 0, 0], cutoff=40, exponent=8,
        diffuse=[1.0, 0.85, 0.6], specular=[0.2, 0.2, 0.2], castshadow=True,
        attenuation=[0.6, 0.0, 0.6],
    )

    # The bulb and the diffuser disk share one emissive material so the shade
    # visibly glows; its colour and emission are driven at runtime.
    spec.add_material(name="bulb", rgba=[1.0, 0.85, 0.6, 1.0], emission=0.8)
    for g in spec.geoms:
        if g.group == 2 and np.allclose(g.rgba, [1.0, 0.95, 0.76, 0.95], atol=1e-3):
            g.material = "bulb"
            g.rgba = [0.5, 0.5, 0.5, 1]  # MuJoCo default -> material colour wins

    # Gravity compensation is applied through the actuators so that holding
    # torque counts against each joint's URDF effort limit, like a real servo.
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0

    for name in JOINTS:
        j = spec.joint(name)
        j.actgravcomp = True
        kp, kv = SERVO_GAINS[name]
        act = spec.add_actuator(name=name.replace("_joint", ""), target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=kp, kv=kv)
        act.ctrllimited = True
        act.ctrlrange = list(j.range)
        # URDF effort limit becomes the actuator torque budget.
        effort = _urdf_effort(name)
        act.forcelimited = True
        act.forcerange = [-effort, effort]
        j.actfrclimited = True
        j.actfrcrange = [-effort, effort]

    return spec


_EFFORT_CACHE: dict[str, float] = {}


def _urdf_effort(joint: str) -> float:
    if not _EFFORT_CACHE:
        text = URDF_PATH.read_text()
        for m in re.finditer(r'<joint name="([^"]+)".*?<limit[^>]*effort="([^"]+)"', text, re.S):
            _EFFORT_CACHE[m.group(1)] = float(m.group(2))
    return _EFFORT_CACHE[joint]


def urdf_limits() -> dict[str, dict[str, float]]:
    """Hard limits, soft limits and velocity limits for each joint, read from the URDF."""
    text = URDF_PATH.read_text()
    out: dict[str, dict[str, float]] = {}
    for block in re.findall(r"<joint .*?</joint>", text, re.S):
        name = re.search(r'name="([^"]+)"', block).group(1)
        lim = re.search(r"<limit ([^/]*)/>", block)
        soft = re.search(r"<safety_controller ([^/]*)/>", block)
        if not lim:
            continue
        attrs = dict(re.findall(r'(\w+)="([^"]+)"', lim.group(1)))
        sattrs = dict(re.findall(r'(\w+)="([^"]+)"', soft.group(1))) if soft else {}
        out[name] = {
            "lower": float(attrs["lower"]),
            "upper": float(attrs["upper"]),
            "velocity": float(attrs["velocity"]),
            "effort": float(attrs["effort"]),
            "soft_lower": float(sattrs.get("soft_lower_limit", attrs["lower"])),
            "soft_upper": float(sattrs.get("soft_upper_limit", attrs["upper"])),
        }
    return out


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


if __name__ == "__main__":
    spec = build_spec()
    model = spec.compile()
    out = GENERATED / "lamp_scene.xml"
    out.write_text(spec.to_xml())
    print(f"bodies={model.nbody} joints={model.njnt} actuators={model.nu} "
          f"geoms={model.ngeom} lights={model.nlight}")
    print(f"wrote {out}")
