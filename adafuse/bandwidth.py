"""Bandwidth constraints and load estimates for intermediate vs late fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

import numpy as np

from adafuse.policy import FusionMode, FusionPlan


@dataclass
class BandwidthConfig:
    """
    Order-of-magnitude bitrates pour la fusion (calibrable par déploiement).

    **Topologie en étoile (hub = ``fusion_hub_index``)** — plus réaliste qu’un maillage complet :
    - *Intermediate* : chaque agent ≠ hub envoie des features vers le hub sur **un** lien à
      ``intermediate_bps`` (pas une exigence par paire du cluster).
    - *Late* : idem avec ``late_bps`` (détections compactes vers le hub).
    - *Hybrid* : une étoile **par cluster** (hub = plus petit indice dans le cluster) pour
      l’intermediate ; le **late inter-clusters** reste modélisé par paires d’agents de clusters
      distincts (fusion tardive de boîtes entre groupes).
    """

    intermediate_bps: float = 2.5e6
    late_bps: float = 2e5
    fusion_hub_index: int = 0
    # Capacity model: C_ij = clip( C_max / (1 + ((d_ij * γ)/d0)^eta), floor, max ) [bits/s]
    # γ = distance_coupling : >1 renforce « proches = meilleurs liens », <1 atténue l'effet distance.
    # Plancher ≥ débit intermediate par lien (étoile) pour éviter l’infaisabilité systématique sur les liens lointains.
    capacity_max_bps: float = 12e6
    capacity_floor_bps: float = 2.5e6
    reference_distance_m: float = 50.0
    path_loss_exponent: float = 2.2
    distance_coupling: float = 1.0


@dataclass
class FeasibilityReport:
    """Feasibility of a fusion plan under pairwise capacities."""

    plan: FusionPlan
    required_bps_matrix: np.ndarray
    capacity_bps_matrix: np.ndarray
    overload_ratio: float
    feasible: bool
    notes: str = ""

    def total_estimated_load_bps(self) -> float:
        """Scalar aggregate load proxy (sum of required over congested links)."""
        req = np.maximum(self.required_bps_matrix, 0)
        cap = np.maximum(self.capacity_bps_matrix, 1e-9)
        return float(np.sum(np.maximum(req - cap, 0.0)))


class BandwidthModel:
    """
    Pairwise capacities from distances and estimated traffic for a ``FusionPlan``.
    """

    def __init__(self, agent_ids: Sequence[str], config: Optional[BandwidthConfig] = None):
        self.agent_ids = list(agent_ids)
        self.id_to_idx = {a: i for i, a in enumerate(self.agent_ids)}
        self.n = len(self.agent_ids)
        self.config = config or BandwidthConfig()

    def pairwise_distances_m(self, xy: np.ndarray) -> np.ndarray:
        """xy: (n, 2) array of positions."""
        d = xy[:, None, :] - xy[None, :, :]
        return np.sqrt(np.sum(d * d, axis=-1))

    def capacity_matrix_bps(self, distances_m: np.ndarray) -> np.ndarray:
        """Symmetric capacity matrix [bits/s] from pairwise distances."""
        c = self.config
        d = np.maximum(distances_m, 1e-3)
        gamma = max(float(c.distance_coupling), 1e-6)
        cap = c.capacity_max_bps / (
            1.0 + np.power((d * gamma) / c.reference_distance_m, c.path_loss_exponent)
        )
        np.fill_diagonal(cap, c.capacity_max_bps)
        cap = np.clip(cap, c.capacity_floor_bps, c.capacity_max_bps)
        return cap

    def required_rate_matrix(
        self,
        plan: FusionPlan,
    ) -> np.ndarray:
        """
        Débit requis [bit/s] sur chaque lien ``(i,j)`` (matrice symétrique : charge agrégée sur l’arête).

        Modèle **étoile** pour late / intermediate global ; **étoile par cluster** en hybrid ;
        **paires** uniquement pour le late **inter-clusters**.
        Les flux qui partagent le même lien **s’additionnent**.
        """
        req = np.zeros((self.n, self.n), dtype=np.float64)
        c = self.config
        gh = int(np.clip(c.fusion_hub_index, 0, max(0, self.n - 1)))

        def add_undirected(i: int, j: int, rate: float) -> None:
            if i == j or rate <= 0.0:
                return
            req[i, j] += rate
            req[j, i] = req[i, j]

        all_idx = set(range(self.n))

        if plan.mode == FusionMode.LATE:
            for i in range(self.n):
                if i != gh:
                    add_undirected(gh, i, c.late_bps)
            return req

        if plan.mode == FusionMode.INTERMEDIATE:
            for i in range(self.n):
                if i != gh:
                    add_undirected(gh, i, c.intermediate_bps)
            return req

        # HYBRID
        cluster_sets: List[Set[int]] = []
        for cluster in plan.intermediate_clusters:
            idxs = {self.id_to_idx[a] for a in cluster if a in self.id_to_idx}
            if idxs:
                cluster_sets.append(idxs)

        if not cluster_sets:
            for i in range(self.n):
                if i != gh:
                    add_undirected(gh, i, c.late_bps)
            return req

        # Intra-cluster : étoile vers hub local (plus petit indice dans le cluster)
        for s in cluster_sets:
            if len(s) < 2:
                continue
            hub_local = min(s)
            for i in s:
                if i != hub_local:
                    add_undirected(hub_local, i, c.intermediate_bps)

        late_agents = [self.id_to_idx[a] for a in (plan.late_fusion_agents or []) if a in self.id_to_idx]
        late_set = set(late_agents) if late_agents else all_idx

        # Late inter-clusters : paire (i,j) dans des clusters différents (ou hors cluster)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                same = any(i in s and j in s for s in cluster_sets)
                if same:
                    continue
                if i in late_set and j in late_set:
                    add_undirected(i, j, c.late_bps)

        return req

    def feasibility(
        self,
        plan: FusionPlan,
        distances_m: np.ndarray,
        capacity_scale: float = 1.0,
    ) -> FeasibilityReport:
        cap = self.capacity_matrix_bps(distances_m) * float(capacity_scale)
        req = self.required_rate_matrix(plan)
        diff = req - cap
        raw_over = float(np.max(diff)) if self.n else 0.0
        overload = max(0.0, raw_over)
        feasible = raw_over <= 0.0
        notes = "ok" if feasible else "at least one link requires more than capacity"
        return FeasibilityReport(
            plan=plan,
            required_bps_matrix=req,
            capacity_bps_matrix=cap,
            overload_ratio=overload,
            feasible=feasible,
            notes=notes,
        )


def positions_to_xy(poses: Sequence) -> np.ndarray:
    """Accept AgentPose-like objects with .x and .y."""
    return np.array([[p.x, p.y] for p in poses], dtype=np.float64)


def estimate_objective_hint(
    feasibility: FeasibilityReport,
    ap_proxy: float = 0.5,
    lambda_bw: float = 1e-7,
) -> float:
    """
    Scalar hint: higher is better — AP proxy minus penalty on bandwidth stress.

    ``ap_proxy`` stands in for real OpenCOOD AP when not available.
    ``lambda_bw`` scales FeasibilityReport total overload (bits/s gap).
    """

    stress = feasibility.total_estimated_load_bps()
    return ap_proxy - lambda_bw * stress
