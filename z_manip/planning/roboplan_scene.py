"""Build a roboplan ``Scene`` from the enforced capsule collision model.

Why this file does not just hand roboplan the URDF
==================================================
``roboplan.core.Scene`` loads the URDF's own ``<collision>`` geometry and makes
every bit of it live.  On this robot that is wrong twice over:

1.  **Safety.**  ``configs/piper_collision_capsules.json`` is the only enforced
    model of the chassis, NUC and Mid-360, and it opens by stating that the
    matching URDF ``<collision>`` tags are INERT --
    ``PinocchioSelfCollisionChecker`` drops every base-fixed geometry before a
    pair is built.  ``tests/test_lidar_keepout.py`` exists because that drift
    once let the gripper strike the Mid-360.  Loading the URDF geometry would
    silently promote eight inert, unaudited envelopes to enforced ones.

2.  **Speed.**  The PiPER ships visual-quality STLs as collision geometry
    (``link2.STL`` is 3.5 MB).  Measured on this model: 2770 us per collision
    check with the STLs, 15 us with primitives -- 185x, and the 2.8 ms figure
    is *worse* than the capsule/KD-tree path we already run at 1.76 ms.

So we strip every URDF ``<collision>`` and re-emit the 22 capsules as primitive
geometry on dedicated fixed-joint links.  One source of truth, no semantic
change, and the ACM falls out of ``self_collision.pairs`` at link granularity.

A capsule spans two link frames and is not in general rigid in either
(``wrist`` runs piper_link4 -> piper_link6 across two joints).  We attach each
one to whichever frame it moves least in and inflate the radius by the measured
residual, so the emitted body always *contains* the true swept capsule.  For 21
of 22 the residual is zero; ``wrist`` costs 0.18 mm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from z_manip.collision.pointcloud import CapsuleSpec, RobotCollisionModel
from z_manip.kinematics.chain import KinematicChain

__all__ = [
    "CapsuleBody",
    "capsule_bodies",
    "write_planning_model",
    "build_scene",
]

# Configurations sampled when measuring how rigid a capsule is in a candidate
# frame.  The residual feeds a radius inflation, so under-sampling makes the
# model optimistic -- do not lower this without re-reading the class docstring.
_RIGIDITY_SAMPLES = 512
_RIGIDITY_SEED = 20260803


@dataclass(frozen=True)
class CapsuleBody:
    """One capsule, resolved to a rigid body on a single link frame."""

    name: str
    parent_frame: str
    #: Capsule axis midpoint, in ``parent_frame``.
    origin: np.ndarray
    #: Unit axis direction, in ``parent_frame``.
    axis: np.ndarray
    length: float
    radius: float
    #: Radius added to absorb residual motion of the far endpoint, in metres.
    inflation: float

    @property
    def link_name(self) -> str:
        return f"cap_{self.name}"


def _endpoints(chain: KinematicChain, capsule: CapsuleSpec, q: np.ndarray):
    transforms = chain.link_transforms(q)
    start_tf = transforms[capsule.start_frame]
    end_tf = transforms[capsule.end_frame]
    start = start_tf[:3, :3] @ np.asarray(capsule.start_offset, float) + start_tf[:3, 3]
    end = end_tf[:3, :3] @ np.asarray(capsule.end_offset, float) + end_tf[:3, 3]
    return start, end, start_tf, end_tf


def _resolve(chain: KinematicChain, capsule: CapsuleSpec, samples: np.ndarray) -> CapsuleBody:
    """Pick the frame the capsule is most rigid in and measure the residual."""

    local: dict[str, list[np.ndarray]] = {capsule.start_frame: [], capsule.end_frame: []}
    for q in samples:
        start, end, start_tf, end_tf = _endpoints(chain, capsule, q)
        local[capsule.start_frame].append(
            np.concatenate(
                (
                    start_tf[:3, :3].T @ (start - start_tf[:3, 3]),
                    start_tf[:3, :3].T @ (end - start_tf[:3, 3]),
                )
            )
        )
        local[capsule.end_frame].append(
            np.concatenate(
                (
                    end_tf[:3, :3].T @ (start - end_tf[:3, 3]),
                    end_tf[:3, :3].T @ (end - end_tf[:3, 3]),
                )
            )
        )

    best_frame, best_drift, best_local = None, np.inf, None
    for frame, points in local.items():
        stacked = np.asarray(points)
        # Worst distance of either endpoint from its own mean placement: the
        # radius must grow by this much for the rigid body to contain the truth.
        centre = stacked.mean(axis=0)
        drift = float(
            np.linalg.norm((stacked - centre).reshape(len(stacked), 2, 3), axis=2).max()
        )
        if drift < best_drift:
            best_frame, best_drift, best_local = frame, drift, centre

    start_local, end_local = best_local[:3], best_local[3:]
    axis_vec = end_local - start_local
    length = float(np.linalg.norm(axis_vec))
    # A degenerate capsule is a sphere; any unit axis will do.
    axis = axis_vec / length if length > 1e-12 else np.array([0.0, 0.0, 1.0])
    return CapsuleBody(
        name=capsule.name,
        parent_frame=best_frame,
        origin=0.5 * (start_local + end_local),
        axis=axis,
        length=length,
        radius=capsule.radius + best_drift,
        inflation=best_drift,
    )


def capsule_bodies(
    chain: KinematicChain,
    model: RobotCollisionModel,
    *,
    samples: int = _RIGIDITY_SAMPLES,
) -> tuple[CapsuleBody, ...]:
    """Resolve every capsule to a rigid, conservatively inflated body."""

    rng = np.random.default_rng(_RIGIDITY_SEED)
    grid = rng.uniform(chain.lower_limits, chain.upper_limits, size=(samples, chain.dof))
    # Joint limits are where the residual is largest, so pin both corners.
    grid = np.vstack((grid, chain.lower_limits, chain.upper_limits, np.zeros(chain.dof)))
    return tuple(_resolve(chain, capsule, grid) for capsule in model.capsules)


def _rotation_to(axis: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw taking +Z onto ``axis`` (URDF cylinders extend along Z)."""

    x, y, z = axis
    return np.array([0.0, float(np.arctan2(np.hypot(x, y), z)), float(np.arctan2(y, x))])


def _sub(parent: ElementTree.Element, tag: str, **attrib: str) -> ElementTree.Element:
    return ElementTree.SubElement(parent, tag, {k: v for k, v in attrib.items()})


def _add_capsule_link(robot: ElementTree.Element, body: CapsuleBody) -> None:
    """Emit a fixed-joint link carrying cylinder + two end spheres."""

    link = _sub(robot, "link", name=body.link_name)
    rpy = _rotation_to(body.axis)
    radius = f"{body.radius:.9g}"
    if body.length > 1e-9:
        collision = _sub(link, "collision", name=f"{body.link_name}_shaft")
        _sub(collision, "origin", xyz="0 0 0", rpy=" ".join(f"{v:.9g}" for v in rpy))
        _sub(_sub(collision, "geometry"), "cylinder", radius=radius, length=f"{body.length:.9g}")
    for sign, end in ((-0.5, "start"), (0.5, "end")):
        offset = sign * body.length * body.axis
        collision = _sub(link, "collision", name=f"{body.link_name}_{end}")
        _sub(collision, "origin", xyz=" ".join(f"{v:.9g}" for v in offset), rpy="0 0 0")
        _sub(_sub(collision, "geometry"), "sphere", radius=radius)

    joint = _sub(robot, "joint", name=f"{body.link_name}_joint", type="fixed")
    _sub(joint, "parent", link=body.parent_frame)
    _sub(joint, "child", link=body.link_name)
    _sub(joint, "origin", xyz=" ".join(f"{v:.9g}" for v in body.origin), rpy="0 0 0")


def write_planning_model(
    urdf_path: str | Path,
    srdf_path: str | Path,
    model: RobotCollisionModel,
    bodies: tuple[CapsuleBody, ...],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the derived URDF/SRDF pair roboplan's ``Scene`` consumes.

    The URDF keeps the source kinematics verbatim and replaces all collision
    geometry with the capsule bodies.  The SRDF keeps the source groups and
    appends an ACM that leaves exactly ``self_collision.pairs`` enabled.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = ElementTree.parse(urdf_path).getroot()
    for link in robot.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)
    for body in bodies:
        _add_capsule_link(robot, body)
    out_urdf = out_dir / "planning.urdf"
    ElementTree.ElementTree(robot).write(out_urdf, encoding="unicode", xml_declaration=True)

    srdf = ElementTree.parse(srdf_path).getroot()
    enabled = {
        tuple(sorted(pair))
        for pair in (model.self_collision.pairs or ())
        if tuple(sorted(pair)) not in {tuple(sorted(p)) for p in model.self_collision.ignore_pairs}
    }
    names = [body.name for body in bodies]
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if tuple(sorted((first, second))) in enabled:
                continue
            _sub(
                srdf,
                "disable_collisions",
                link1=f"cap_{first}",
                link2=f"cap_{second}",
                reason="not in capsule self_collision.pairs",
            )
    out_srdf = out_dir / "planning.srdf"
    ElementTree.ElementTree(srdf).write(out_srdf, encoding="unicode", xml_declaration=True)
    return out_urdf, out_srdf


def build_scene(
    urdf_path: str | Path,
    srdf_path: str | Path,
    collision_model_path: str | Path,
    *,
    base_link: str,
    tip_link: str,
    out_dir: str | Path,
    package_paths: tuple[str, ...] = (),
    name: str = "z_manip",
):
    """Build a roboplan ``Scene`` carrying the enforced capsule model.

    Imported lazily: roboplan lives on the 4090 only, never on the NUC thin
    deploy (see the NUC thin-deploy contract in README.md).
    """

    import roboplan.core as rc  # noqa: PLC0415 -- 4090-only dependency

    model = RobotCollisionModel.from_mapping(json.loads(Path(collision_model_path).read_text()))
    chain = KinematicChain.from_urdf(urdf_path, base_link, tip_link)
    bodies = capsule_bodies(chain, model)
    out_urdf, out_srdf = write_planning_model(urdf_path, srdf_path, model, bodies, out_dir)
    urdf_dir = str(Path(urdf_path).resolve().parent)
    return rc.Scene(name, str(out_urdf), str(out_srdf), [urdf_dir, *package_paths])
