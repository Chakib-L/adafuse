"""
Évaluation comparative de plans de fusion : métriques réseau + nMAP / mIoU (GT CARLA) lorsque disponible.

Sans fichier GT, les champs nMAP / mIoU restent vides ; les métriques infrastructure (débit, bits, temps)
sont toujours calculées.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adafuse.bandwidth import BandwidthConfig, BandwidthModel
from adafuse.gt_map_eval import GtBoxBev, compute_nmap_miou_from_gt, parse_carla_gt_bev
from adafuse.network_constraints import (
    InfrastructureConfig,
    IntermediateSchedule,
    NetworkInfrastructureModel,
)
from adafuse.policy import FusionMode, FusionPlan


def infrastructure_config_from_bandwidth_config(bw: BandwidthConfig) -> InfrastructureConfig:
    return InfrastructureConfig(
        capacity_max_bps=bw.capacity_max_bps,
        capacity_floor_bps=bw.capacity_floor_bps,
        reference_distance_m=bw.reference_distance_m,
        path_loss_exponent=bw.path_loss_exponent,
        distance_coupling=bw.distance_coupling,
    )


def _ap_proxy_base(mode: FusionMode) -> float:
    if mode == FusionMode.LATE:
        return 0.63
    if mode == FusionMode.INTERMEDIATE:
        return 0.78
    return 0.70


def _ap_proxy_from_feasibility(feas: Any, mode: FusionMode) -> float:
    base = _ap_proxy_base(mode)
    stress = feas.total_estimated_load_bps()
    pen_stress = 1.2e-8 * stress
    pen_inf = 0.04 if not feas.feasible else 0.0
    pen_ol = max(0.0, float(feas.overload_ratio)) * 2e-9
    return float(np.clip(base - pen_stress - pen_inf - pen_ol, 0.0, 0.99))


@dataclass
class PlanEvalResult:
    label: str
    plan: FusionPlan
    ap_proxy: float
    feasible_bandwidth: bool
    overload_max_bps: float
    stress_bps: float
    bits_upper_bound: float
    temporal_feasible: bool
    comm_time_s: float
    end_to_end_time_s: float
    per_cluster: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    nmap: Optional[float] = None
    miou: Optional[float] = None
    gt_eval_detail: Optional[Dict[str, Any]] = None

    def to_display_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "label": self.label,
            "mode": self.plan.mode.value,
            "ap_proxy": round(self.ap_proxy, 4),
            "nmap": None if self.nmap is None else round(self.nmap, 4),
            "miou": None if self.miou is None else round(self.miou, 4),
            "faisable_debit": self.feasible_bandwidth,
            "surcharge_max_bps": round(self.overload_max_bps, 2),
            "stress_bps": round(self.stress_bps, 2),
            "bits_frame_ub": round(self.bits_upper_bound, 0),
            "faisable_temps": self.temporal_feasible,
            "T_comm_s": round(self.comm_time_s, 5),
            "T_e2e_s": round(self.end_to_end_time_s, 5),
            "clusters": self.per_cluster,
            "rationale": self.plan.rationale,
        }
        if self.gt_eval_detail:
            d["gt_detail"] = self.gt_eval_detail
        return d


def _evaluate_subplan(
    label: str,
    plan: FusionPlan,
    model: BandwidthModel,
    dist: np.ndarray,
    cap_scale: float,
    infra: NetworkInfrastructureModel,
    agent_xy: Optional[np.ndarray],
    gt_boxes: Optional[List[GtBoxBev]],
    eval_seed: int,
) -> PlanEvalResult:
    rep = model.feasibility(plan, dist, capacity_scale=cap_scale)
    ap = _ap_proxy_from_feasibility(rep, plan.mode)
    temp = infra.temporal_feasibility(plan, dist, schedule=IntermediateSchedule.STAR_TO_EGO)

    nmap: Optional[float] = None
    miou: Optional[float] = None
    gt_detail: Optional[Dict[str, Any]] = None
    if gt_boxes is not None and agent_xy is not None and len(gt_boxes) > 0:
        rng = np.random.default_rng(eval_seed)
        nmap, miou, gt_detail = compute_nmap_miou_from_gt(
            gt_boxes, agent_xy, plan, rep, iou_thresh=0.5, rng=rng
        )

    per_cluster: List[Dict[str, Any]] = []
    if plan.mode == FusionMode.HYBRID and plan.intermediate_clusters:
        for ci, cl in enumerate(plan.intermediate_clusters):
            if len(cl) < 2:
                continue
            sub = FusionPlan(
                mode=FusionMode.INTERMEDIATE,
                intermediate_clusters=[list(cl)],
                late_fusion_agents=list(cl),
                rationale=f"Cluster intermédiaire {ci}",
            )
            r_sub = model.feasibility(sub, dist, capacity_scale=cap_scale)
            per_cluster.append(
                {
                    "agents": list(cl),
                    "ap_proxy_cluster": round(_ap_proxy_from_feasibility(r_sub, FusionMode.INTERMEDIATE), 4),
                    "faisable_debit": r_sub.feasible,
                }
            )
    elif plan.mode == FusionMode.INTERMEDIATE:
        per_cluster.append(
            {
                "agents": list(model.agent_ids),
                "ap_proxy_cluster": round(ap, 4),
                "faisable_debit": rep.feasible,
            }
        )

    return PlanEvalResult(
        label=label,
        plan=plan,
        ap_proxy=ap,
        feasible_bandwidth=rep.feasible,
        overload_max_bps=rep.overload_ratio,
        stress_bps=rep.total_estimated_load_bps(),
        bits_upper_bound=temp.total_bits_moved_upper_bound,
        temporal_feasible=temp.feasible_under_deadline,
        comm_time_s=temp.communication_time_s,
        end_to_end_time_s=temp.end_to_end_time_s,
        per_cluster=per_cluster,
        notes=temp.notes,
        nmap=nmap,
        miou=miou,
        gt_eval_detail=gt_detail,
    )


def baseline_late_all_agents(agent_ids: List[str]) -> FusionPlan:
    return FusionPlan(
        mode=FusionMode.LATE,
        late_fusion_agents=list(agent_ids),
        rationale="Baseline late fusion (tous les agents, détections uniquement).",
    )


def baseline_intermediate_all_agents(agent_ids: List[str]) -> FusionPlan:
    return FusionPlan(
        mode=FusionMode.INTERMEDIATE,
        intermediate_clusters=[list(agent_ids)],
        late_fusion_agents=list(agent_ids),
        rationale="Baseline intermediate fusion (un cluster global).",
    )


def compare_fusion_strategies(
    agent_ids: List[str],
    model: BandwidthModel,
    dist: np.ndarray,
    cap_scale: float,
    infra_cfg: InfrastructureConfig,
    llm_position_only_plan: Optional[FusionPlan] = None,
    heuristic_plan: Optional[FusionPlan] = None,
    gt_boxes: Optional[List[GtBoxBev]] = None,
    agent_xy: Optional[np.ndarray] = None,
    eval_seed: int = 0,
) -> Dict[str, PlanEvalResult]:
    infra = NetworkInfrastructureModel(agent_ids, config=infra_cfg)
    out: Dict[str, PlanEvalResult] = {}

    out["late_fusion"] = _evaluate_subplan(
        "late_fusion (global)",
        baseline_late_all_agents(agent_ids),
        model,
        dist,
        cap_scale,
        infra,
        agent_xy,
        gt_boxes,
        eval_seed + 1,
    )
    out["intermediate_fusion"] = _evaluate_subplan(
        "intermediate_fusion (tous les robots)",
        baseline_intermediate_all_agents(agent_ids),
        model,
        dist,
        cap_scale,
        infra,
        agent_xy,
        gt_boxes,
        eval_seed + 2,
    )

    if heuristic_plan is not None:
        out["heuristique"] = _evaluate_subplan(
            "heuristique (positions + faisabilité)",
            heuristic_plan,
            model,
            dist,
            cap_scale,
            infra,
            agent_xy,
            gt_boxes,
            eval_seed + 3,
        )

    if llm_position_only_plan is not None:
        out["llm_clusters"] = _evaluate_subplan(
            "LLM (positions + capacités, clusters)",
            llm_position_only_plan,
            model,
            dist,
            cap_scale,
            infra,
            agent_xy,
            gt_boxes,
            eval_seed + 4,
        )

    return out


def load_gt_boxes_optional(path: Optional[str]) -> Optional[List[GtBoxBev]]:
    if not path or not str(path).strip():
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    return parse_carla_gt_bev(p)
