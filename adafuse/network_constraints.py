"""
Modélisation d'infrastructure réseau pour la coopération multi-agents (AdaFuse).

Vue d'ensemble
--------------

1. **Capacité par paire (i, j)** — Matrice symétrique ``C_ij`` [bit/s], dérivée de la distance
   (affaiblissement de trajet) ou fournie par mesures / opérateur. C'est le **plafond durable**
   sur le lien ; tout débit requis au-delà de ``C_ij`` est considéré comme **surcharge**.

2. **Charge utile par mode de fusion** — On ne confond plus « débit d'un mode » et « taille de
   message » :
   - *Intermediate* : chaque message porte une représentation intermédiaire (carte BEV / features)
     de taille ``feature_payload_bits`` (après quantification éventuelle).
   - *Late* : échange de détections compactes ``detection_payload_bits`` (boîtes + scores).

3. **Temps de communication (latence de transfert)** — Pour l'intermediate fusion, le **temps
   minimal** avant que le nœud de fusion dispose de tout dépend du **schéma de communication** :
   - **Étoile (fusion vers un ego k)** : chaque agent i envoie ``S`` bits vers k en
     ``T_i = S / C_{i,k}`` ; en parallèle sur les liens disjoints, un proxy courant est
     ``T_star = max_i S / C_{i,k}`` (temps du maillon le plus lent vers le centre).
   - **Tous contre tous / agrégation itérative** : borne supérieure simple ``O(n^2)`` messages
     ou modélisation par **goulot** sur la coupe minimale ; on expose une borne **prudente**
     (somme séquentielle ou max selon le paramètre ``schedule``).

4. **Faisabilité temporelle** — Si la période entre deux acquisitions LiDAR est ``frame_period_s``
   (ex. 0,1 s) ou un **budget latence** ``max_comm_latency_s``, on exige
   ``T_comm + T_compute* <= budget`` pour valider un plan *intermediate* (sinon repli late ou
   clusters plus petits).

Cette couche complète la vérification **instantanée** (débit requis vs capacité) déjà présente
dans ``BandwidthModel`` avec une vision **volume + temps** compatible avec les décisions LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Set

import numpy as np

from adafuse.policy import FusionMode, FusionPlan


@dataclass
class InfrastructureConfig:
    """
    Paramètres physiques et tailles de charge utile (ordres de grandeur calibrables sur le déploiement).

    Les capacités ``C_ij`` suivent le même modèle d'affaiblissement que ``BandwidthConfig``.
    """

    capacity_max_bps: float = 12e6
    capacity_floor_bps: float = 2.5e6
    reference_distance_m: float = 50.0
    path_loss_exponent: float = 2.2
    distance_coupling: float = 1.0
    # Taille d'une représentation intermédiaire (une fois par message utile vers le partenaire)
    feature_payload_bits: float = 2e7
    # Taille d'un paquet de détections (late fusion)
    detection_payload_bits: float = 5e4
    # Temps de calcul fusion + tête de détection côté ego après réception (borne simple)
    compute_latency_s: float = 0.02
    # Période entre frames LiDAR (borne haute pour latence bout-en-bout)
    frame_period_s: float = 0.1


class IntermediateSchedule(str, Enum):
    """Schéma de collecte des features vers le nœud de fusion."""

    STAR_TO_EGO = "star_to_ego"
    """Tous les agents envoient vers l'ego (index 0 par convention si non précisé)."""
    SEQUENTIAL_PAIRWISE = "sequential_pairwise"
    """Borne pessimiste : transferts séquentiels sur les arêtes nécessaires (proxy)."""


@dataclass
class TemporalFeasibilityReport:
    """Résultat d'une analyse débit + temps pour un plan donné."""

    plan: FusionPlan
    communication_time_s: float
    end_to_end_time_s: float
    feasible_under_deadline: bool
    deadline_s: float
    total_bits_moved_upper_bound: float
    notes: str = ""


@dataclass
class NetworkInfrastructureModel:
    """
    Capacités par paire, volumes en bits, et bornes de **latence de communication**.

    Complète ``BandwidthModel`` (taux instantanés) par une estimation **temporelle** du
    pipeline intermediate (transfert puis prédiction).
    """

    agent_ids: Sequence[str]
    config: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    fusion_head_index: int = 0

    def __post_init__(self):
        self.agent_ids = list(self.agent_ids)
        self.id_to_idx = {a: i for i, a in enumerate(self.agent_ids)}
        self.n = len(self.agent_ids)

    def pairwise_distances_m(self, xy: np.ndarray) -> np.ndarray:
        d = xy[:, None, :] - xy[None, :, :]
        return np.sqrt(np.sum(d * d, axis=-1))

    def capacity_matrix_bps(self, distances_m: np.ndarray) -> np.ndarray:
        c = self.config
        d = np.maximum(distances_m, 1e-3)
        gamma = max(float(c.distance_coupling), 1e-6)
        cap = c.capacity_max_bps / (
            1.0 + np.power((d * gamma) / c.reference_distance_m, c.path_loss_exponent)
        )
        np.fill_diagonal(cap, c.capacity_max_bps)
        return np.clip(cap, c.capacity_floor_bps, c.capacity_max_bps)

    def _cluster_indices(self, plan: FusionPlan) -> List[Set[int]]:
        out: List[Set[int]] = []
        for cluster in plan.intermediate_clusters:
            s = {self.id_to_idx[a] for a in cluster if a in self.id_to_idx}
            if s:
                out.append(s)
        return out

    def estimate_intermediate_star_time_s(
        self,
        cluster_agent_indices: Set[int],
        cap_bps: np.ndarray,
        hub_idx: int,
    ) -> float:
        """
        Temps pour que tous les membres du cluster envoient ``feature_payload_bits`` vers ``hub_idx``
        (parallèle sur liens entrants distincts → max des durées par source).
        """
        if not cluster_agent_indices:
            return 0.0
        c = self.config
        S = c.feature_payload_bits
        times = []
        for i in cluster_agent_indices:
            if i == hub_idx:
                continue
            rate = float(cap_bps[i, hub_idx])
            if rate < 1e-9:
                times.append(float("inf"))
            else:
                times.append(S / rate)
        return max(times) if times else 0.0

    def estimate_late_broadcast_time_s(
        self,
        agent_indices: Set[int],
        cap_bps: np.ndarray,
        hub_idx: int,
    ) -> float:
        """Chaque agent envoie ``detection_payload_bits`` vers le hub (fusion tardive)."""
        c = self.config
        S = c.detection_payload_bits
        times = []
        for i in agent_indices:
            if i == hub_idx:
                continue
            rate = float(cap_bps[i, hub_idx])
            if rate < 1e-9:
                times.append(float("inf"))
            else:
                times.append(S / rate)
        return max(times) if times else 0.0

    def upper_bound_bits_intermediate_frame(
        self,
        plan: FusionPlan,
    ) -> float:
        """
        Borne supérieure simple sur le volume **bit** échangé par frame (pour métriques d'efficacité).

        - Late : chaque paire impliquée échange un message de détection (symétrique simplifié).
        - Intermediate global : chaque paire reçoit potentiellement des features (sur-estimation).
        - Hybrid : intra-cluster feature volume + inter late.
        """
        c = self.config
        n = self.n
        Sf, Sd = c.feature_payload_bits, c.detection_payload_bits

        if plan.mode == FusionMode.LATE:
            # n*(n-1)/2 paires, 2 directions comptées une fois chaque message : approximation  n*(n-1)*Sd
            return float(max(0, n * (n - 1)) * Sd)

        if plan.mode == FusionMode.INTERMEDIATE:
            return float(max(0, n * (n - 1)) * Sf)

        bits = 0.0
        cluster_sets = self._cluster_indices(plan)
        seen_pairs = set()
        for s in cluster_sets:
            k = len(s)
            bits += float(k * (k - 1) * Sf)
        all_idx = set(range(n))
        late_agents = {self.id_to_idx[a] for a in (plan.late_fusion_agents or []) if a in self.id_to_idx}
        late_set = late_agents if late_agents else all_idx
        for i in range(n):
            for j in range(i + 1, n):
                same = any(i in s and j in s for s in cluster_sets)
                if same:
                    continue
                if i in late_set and j in late_set:
                    pair = (i, j)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        bits += 2 * Sd
        return bits

    def temporal_feasibility(
        self,
        plan: FusionPlan,
        distances_m: np.ndarray,
        max_latency_s: Optional[float] = None,
        schedule: IntermediateSchedule = IntermediateSchedule.STAR_TO_EGO,
    ) -> TemporalFeasibilityReport:
        """
        Vérifie si le **temps de communication** (plus calcul) tient dans ``max_latency_s``
        (défaut : ``frame_period_s`` de la config).
        """
        c = self.config
        deadline = max_latency_s if max_latency_s is not None else c.frame_period_s
        cap = self.capacity_matrix_bps(distances_m)
        hub = self.fusion_head_index

        T_comm = 0.0
        bits_ub = self.upper_bound_bits_intermediate_frame(plan)

        if plan.mode == FusionMode.LATE:
            T_comm = self.estimate_late_broadcast_time_s(set(range(self.n)), cap, hub)
        elif plan.mode == FusionMode.INTERMEDIATE:
            if schedule == IntermediateSchedule.STAR_TO_EGO:
                T_comm = self.estimate_intermediate_star_time_s(set(range(self.n)), cap, hub)
            else:
                # Pessimiste : somme des temps sur un ordre fixe (très conservateur)
                S = c.feature_payload_bits
                order = list(range(self.n))
                acc = 0.0
                for idx in range(1, len(order)):
                    i, j = order[idx], hub
                    r = float(cap[i, j])
                    acc += S / r if r > 1e-9 else float("inf")
                T_comm = acc
        else:
            # HYBRID : max des temps par cluster (parallèle entre clusters) + late cross
            cluster_sets = self._cluster_indices(plan)
            times = []
            for s in cluster_sets:
                times.append(self.estimate_intermediate_star_time_s(s, cap, hub))
            T_cross = 0.0
            late_agents = {self.id_to_idx[a] for a in (plan.late_fusion_agents or []) if a in self.id_to_idx}
            late_set = late_agents if late_agents else set(range(self.n))
            for i in late_set:
                for j in late_set:
                    if i >= j:
                        continue
                    same = any(i in s and j in s for s in cluster_sets)
                    if not same:
                        r = float(cap[i, j])
                        if r > 1e-9:
                            T_cross = max(T_cross, c.detection_payload_bits / r)
            T_comm = (max(times) if times else 0.0) + T_cross

        T_total = T_comm + c.compute_latency_s
        feasible = T_total <= deadline and T_comm < float("inf")
        notes = (
            f"T_comm≈{T_comm:.4f}s, T_compute={c.compute_latency_s}s, deadline={deadline}s"
        )
        return TemporalFeasibilityReport(
            plan=plan,
            communication_time_s=float(T_comm),
            end_to_end_time_s=float(T_total),
            feasible_under_deadline=feasible,
            deadline_s=deadline,
            total_bits_moved_upper_bound=bits_ub,
            notes=notes,
        )
