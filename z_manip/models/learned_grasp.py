"""Adapter from a remote learned 6-DoF model to the GraspSource contract."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from z_manip.inference.grasp_client import GraspInferenceResult

from .grasp_source import GraspCandidates, GraspContext, GraspGenerationError


class LearnedClient(Protocol):
    def infer(
        self,
        *,
        object_points: object,
        colors: object | None,
        scene_bounds: object,
        frame: str,
    ) -> GraspInferenceResult:
        ...


class LearnedGraspSource:
    """Request observation-frame candidates from an isolated inference server."""

    def __init__(self, client: LearnedClient, *, bounds_margin_m: float = 0.03):
        if bounds_margin_m <= 0.0:
            raise ValueError("scene-bounds margin must be positive")
        self.client = client
        self.bounds_margin_m = float(bounds_margin_m)

    def generate(self, context: GraspContext) -> GraspCandidates:
        if context.object_points is None:
            raise GraspGenerationError("learned grasp inference requires an object point cloud")
        points = np.asarray(context.object_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 1:
            raise GraspGenerationError("object point cloud must have shape (N, 3), N >= 1")
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < 1:
            raise GraspGenerationError("object point cloud has no finite points")
        scene = points
        if context.scene_points is not None:
            candidate_scene = np.asarray(context.scene_points, dtype=np.float32)
            if candidate_scene.ndim != 2 or candidate_scene.shape[1:] != (3,):
                raise GraspGenerationError("scene point cloud must have shape (N, 3)")
            candidate_scene = candidate_scene[np.all(np.isfinite(candidate_scene), axis=1)]
            if len(candidate_scene):
                scene = candidate_scene
        lower = np.min(scene, axis=0) - self.bounds_margin_m
        upper = np.max(scene, axis=0) + self.bounds_margin_m
        context.progress_cb("learned_grasp_request", 0.15)
        result = self.client.infer(
            object_points=points,
            colors=None,
            scene_bounds=np.stack((lower, upper)),
            frame=context.source_frame,
        )
        if result.frame != context.source_frame:
            raise GraspGenerationError(
                f"learned grasp frame {result.frame!r} differs from observation frame "
                f"{context.source_frame!r}",
            )
        context.progress_cb("learned_grasp_candidates", 1.0)
        return GraspCandidates(
            grasps=result.grasps,
            scores=result.scores,
            centroid=np.median(points, axis=0),
            frame=result.frame,
            num_raw=len(result.grasps),
            widths=result.widths,
        )
