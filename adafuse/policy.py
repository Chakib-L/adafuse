"""Structured fusion decisions for OpenCOOD-style intermediate vs late fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set


class FusionMode(str, Enum):
    """High-level fusion strategy."""

    INTERMEDIATE = "intermediate"
    LATE = "late"
    HYBRID = "hybrid"


@dataclass
class FusionPlan:
    """
    Fusion plan for a multi-agent frame.

    - **intermediate_clusters**: each inner list is a set of agent ids (strings)
      that perform **intermediate fusion** together (feature-level exchange).
    - **late_fusion_agents**: agents that participate in **late fusion** (detection-level
      merge). In hybrid mode, typically includes all agents; clusters exchange features
      internally, then boxes are merged across the fleet.
    - **rationale**: short natural-language justification (e.g. from an LLM).
    """

    mode: FusionMode
    intermediate_clusters: List[List[str]] = field(default_factory=list)
    late_fusion_agents: Optional[List[str]] = None
    rationale: str = ""

    def __post_init__(self):
        if self.late_fusion_agents is None:
            self.late_fusion_agents = []

    def all_agents_in_clusters(self) -> Set[str]:
        s: Set[str] = set()
        for c in self.intermediate_clusters:
            s.update(c)
        return s

    def to_json_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "intermediate_clusters": self.intermediate_clusters,
            "late_fusion_agents": list(self.late_fusion_agents or []),
            "rationale": self.rationale,
        }

    @staticmethod
    def from_json_dict(d: dict) -> "FusionPlan":
        mode = FusionMode(d.get("mode", "late"))
        clusters = d.get("intermediate_clusters") or []
        late = d.get("late_fusion_agents") or []
        rationale = d.get("rationale") or ""
        return FusionPlan(
            mode=mode,
            intermediate_clusters=[list(c) for c in clusters],
            late_fusion_agents=list(late),
            rationale=rationale,
        )

