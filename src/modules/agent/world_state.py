"""Avatar world-state tracker for the closed-loop agent runner.

Holds the minimum kinematic state the conditions module needs to evaluate
termination predicates against:

  - root position in world space (x, y, z)
  - root heading (yaw in radians, world frame)
  - linear velocity (smoothed over the last few frames)
  - reference to known scene objects (positions provided by the M2 planner)
  - elapsed frames in the current action

State is updated by the runner once per emitted motion chunk. Conditions
read it as-is; the world_state is the only mutable shared surface between
the runner and the conditions layer, so the predicates stay stateless.

Frames are SMPL-X 168-d feature vectors. The first 3 channels are root
orientation (axis-angle), channels 3..6 are root translation (x, y, z).
See ``src/shared/constants.py`` for the exact slice indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.shared.constants import CONSTS, SMPLX


@dataclass
class SceneObject:
    """Static scene element the planner can target (tree, ball, chair...).

    Position is fixed at scene-init time and never updated by the runner.
    The M2 scene planner populates these from the parsed instruction.
    """

    name: str
    position: np.ndarray  # (3,) world-space xyz

    def __post_init__(self) -> None:
        if self.position.shape != (3,):
            raise ValueError(f"position must be (3,); got {self.position.shape}")


@dataclass
class WorldState:
    """Mutable avatar + scene state shared between runner and conditions.

    Use :meth:`reset_for_new_action` when the runner pulls a new action off
    the queue -- that resets the per-action counters (frames_in_action,
    initial_yaw_for_rotation) without clearing the global avatar pose.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading_yaw: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scene_objects: dict[str, SceneObject] = field(default_factory=dict)
    frames_in_action: int = 0
    yaw_at_action_start: float = 0.0

    def update_from_frame(self, frame: np.ndarray, prev_position: np.ndarray | None = None) -> None:
        """Pull root pose + translation from a single 168-d motion frame.

        Velocity is an instantaneous finite difference; the runner can pass
        the position from the previous frame to get a clean delta. When
        ``prev_position`` is None, velocity is left unchanged (e.g. on the
        first frame of an action when prev is undefined).
        """
        if frame.shape != (168,) and not (frame.ndim == 1 and frame.shape[0] >= 6):
            raise ValueError(f"frame must be 1-D length-168 (SMPL-X); got {frame.shape}")
        root_orient = frame[SMPLX.root_orient_slice]
        new_position = frame[SMPLX.transl_slice].copy()
        # axis-angle root_orient -> yaw is the y-axis rotation magnitude when the
        # body-up axis is the global y-axis (HumanML3D convention). For our
        # condition predicates the precise yaw extraction only needs to be
        # consistent, not anatomically exact, so we use the y-component of the
        # axis-angle vector as a yaw proxy. (Correct retargeting happens in M4.)
        self.heading_yaw = float(root_orient[1])

        if prev_position is not None:
            self.velocity = new_position - prev_position
        self.position = new_position
        self.frames_in_action += 1

    def reset_for_new_action(self) -> None:
        """Called by the runner when the action queue advances.

        Captures the current yaw as the reference for rotation-based termination
        predicates ("until rotated 90deg") and zeroes the per-action counter.
        """
        self.yaw_at_action_start = self.heading_yaw
        self.frames_in_action = 0

    def add_scene_object(self, name: str, position: np.ndarray) -> None:
        self.scene_objects[name] = SceneObject(name=name, position=position)

    def distance_to(self, object_name: str) -> float:
        """Euclidean distance from the avatar root to a named scene object.

        Raises if the object isn't in the scene -- the conditions module
        should validate target names against ``scene_objects.keys()`` at
        parse time, not at evaluation time.
        """
        if object_name not in self.scene_objects:
            raise KeyError(
                f"unknown scene object {object_name!r}; known: {sorted(self.scene_objects.keys())}"
            )

        return float(np.linalg.norm(self.position - self.scene_objects[object_name].position))

    def rotated_since_start_deg(self) -> float:
        """Absolute yaw change since the current action started, in degrees.

        Used by rotation-based termination predicates ("rotated 90 deg").
        Always non-negative; the sign of the turn doesn't matter for these
        predicates.
        """
        return float(np.degrees(abs(self.heading_yaw - self.yaw_at_action_start)))
