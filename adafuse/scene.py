"""Encode agent poses and a textual scene summary for the fusion selector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AgentPose:
    """2D pose of one agent in a common world frame (e.g. CARLA map)."""

    agent_id: str
    x: float
    y: float
    yaw_rad: float = 0.0

    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class SceneContext:
    """Everything the LLM (or heuristic) sees for one decision step."""

    agent_poses: List[AgentPose]
    scene_summary: str
    num_target_vehicles: int = 50
    extra_tags: List[str] = field(default_factory=list)

    def positions_jsonable(self) -> List[dict]:
        return [
            {
                "id": p.agent_id,
                "x": round(p.x, 3),
                "y": round(p.y, 3),
                "yaw_deg": round(float(np.degrees(p.yaw_rad)), 2),
            }
            for p in self.agent_poses
        ]


def _spread_metric(poses: Sequence[AgentPose]) -> float:
    if len(poses) < 2:
        return 0.0
    xy = np.array([[p.x, p.y] for p in poses], dtype=np.float64)
    return float(np.max(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))


def build_scene_summary(
    poses: Sequence[AgentPose],
    num_target_vehicles: int = 50,
    extra_lines: Optional[Sequence[str]] = None,
) -> str:
    """
    Build a short French/English-neutral textual description from geometry.

    Parameters
    ----------
    poses
        All agents participating in cooperation.
    num_target_vehicles
        Number of dynamic objects to detect (CARLA scenario constant).
    extra_lines
        Optional lines (weather, occlusion hints, etc.).
    """
    n = len(poses)
    spread = _spread_metric(poses)
    lines = [
        f"Multi-agent cooperative perception: {n} LiDAR-equipped agents.",
        f"Approximately {num_target_vehicles} other vehicles to detect in the scene.",
        f"Fleet spatial spread (max distance from centroid): {spread:.1f} m.",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    return " ".join(lines)


def random_poses_circle(
    n_agents: int,
    radius_m: float = 80.0,
    prefix: str = "Tesla",
    seed: Optional[int] = None,
) -> List[AgentPose]:
    """Synthetic poses on a circle (demo / simulation)."""
    rng = np.random.default_rng(seed)
    poses: List[AgentPose] = []
    for i in range(n_agents):
        t = 2 * np.pi * (i / max(n_agents, 1)) + rng.normal(0, 0.05)
        x = radius_m * np.cos(t) + rng.normal(0, 3.0)
        y = radius_m * np.sin(t) + rng.normal(0, 3.0)
        yaw = t + np.pi / 2 + rng.normal(0, 0.1)
        poses.append(
            AgentPose(agent_id=f"{prefix}_{i+1}", x=float(x), y=float(y), yaw_rad=float(yaw))
        )
    return poses
