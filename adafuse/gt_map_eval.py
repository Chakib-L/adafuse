"""
Évaluation nMAP / mIoU à partir du ground truth CARLA (BEV) pour comparer des stratégies de fusion.

Sans inférence du détecteur OpenCOOD, on simule des prédictions 3D en ajoutant un bruit de localisation
dont la variance dépend du **mode de fusion**, de la **faisabilité réseau** et de la **couverture** des
agents sur chaque objet GT. La métrique est **alignée sur la détection** : pour chaque classe,
fraction d'objets avec IoU BEV ≥ seuil (proxy d'AP à une frame), puis **nMAP** = moyenne sur les classes.

Le format GT attendu suit ``ground_truth/NNNNNN.json`` : ``objets[]`` avec ``classe``,
``position_globale`` (x,y,z), dimensions dans ``bounding_box_3d`` ou équivalent.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adafuse.bandwidth import BandwidthModel, FeasibilityReport
from adafuse.policy import FusionMode, FusionPlan


@dataclass
class GtBoxBev:
    """Boîte BEV axis-aligned (x, y, longueur x, largeur y) + classe."""

    x: float
    y: float
    length: float
    width: float
    yaw: float
    classe: str


def _get(d: Any, *keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_carla_gt_bev(path: Path) -> List[GtBoxBev]:
    """Charge un JSON GT CARLA et extrait des boîtes BEV + classe."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    objs = data.get("objets") or data.get("objects") or []
    out: List[GtBoxBev] = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        pos = _get(o, "position_globale", "position", "centroid", default={}) or {}
        bb = _get(o, "bounding_box_3d", "bbox_3d", "box_3d", default={}) or {}
        cx = float(_get(pos, "x", default=0.0))
        cy = float(_get(pos, "y", default=0.0))
        length = float(_get(bb, "length", "l", "extent_x", default=4.5))
        width = float(_get(bb, "width", "w", "extent_y", default=2.0))
        yaw = float(_get(bb, "yaw", "rotation", default=0.0))
        classe = str(_get(o, "classe", "class", "label", default="object"))
        out.append(GtBoxBev(x=cx, y=cy, length=length, width=width, yaw=yaw, classe=classe))
    return out


def bev_iou_rotated_simple(a: GtBoxBev, b: GtBoxBev) -> float:
    """
    IoU BEV approchée : rectangles orientés discrétisés (approximation suffisante pour le proxy).
    """
    # Approximation rapide : IoU axis-aligned sur AABB des deux centres et tailles moyennes
    la, wa = max(a.length, 0.1), max(a.width, 0.1)
    lb, wb = max(b.length, 0.1), max(b.width, 0.1)
    ax0, ay0 = a.x - la / 2, a.y - wa / 2
    bx0, by0 = b.x - lb / 2, b.y - wb / 2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + la, bx0 + lb), min(ay0 + wa, by0 + wb)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = la * wa + lb * wb - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def _agent_coverage_for_object(
    ox: float,
    oy: float,
    agent_xy: np.ndarray,
    max_range_m: float = 120.0,
) -> float:
    """Score [0,1] : meilleure « visibilité » = plus proche d'un agent (inverse distance)."""
    if agent_xy.size == 0:
        return 0.0
    d = np.sqrt(np.sum((agent_xy - np.array([ox, oy])) ** 2, axis=1))
    d = np.maximum(d, 1.0)
    s = np.clip(1.0 - d / max_range_m, 0.0, 1.0)
    return float(np.clip(np.mean(np.sort(s)[-min(3, len(s)) :]), 0.0, 1.0))


def _mode_quality(mode: FusionMode) -> float:
    if mode == FusionMode.INTERMEDIATE:
        return 0.92
    if mode == FusionMode.HYBRID:
        return 0.72
    return 0.48


def fusion_localization_gain(
    plan: FusionPlan,
    feas: FeasibilityReport,
) -> float:
    """
    Gain [0,1] pour réduire le bruit de prédiction : fusion + faisabilité débit.
    """
    q = _mode_quality(plan.mode)
    if not feas.feasible:
        q *= 0.78
    stress = feas.total_estimated_load_bps()
    q *= float(np.exp(-1.5e-9 * max(0.0, stress)))
    return float(np.clip(q, 0.05, 0.99))


def compute_nmap_miou_from_gt(
    gt_boxes: Sequence[GtBoxBev],
    agent_xy: np.ndarray,
    plan: FusionPlan,
    feas: FeasibilityReport,
    iou_thresh: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Retourne (nMAP, mIoU, détails) pour une frame GT.

    nMAP : moyenne sur les classes de (nombre d'objets avec IoU >= seuil) / (nombre d'objets de la classe).
    mIoU : moyenne des IoU max par objet (après matching).
    """
    if not gt_boxes:
        return None, None, {"reason": "gt_vide"}

    rng = rng or np.random.default_rng(0)
    g = fusion_localization_gain(plan, feas)
    detail: Dict[str, Any] = {"fusion_gain": round(g, 4), "mode": plan.mode.value}

    preds: List[GtBoxBev] = []
    ious: List[float] = []

    sigma0 = 2.8
    for b in gt_boxes:
        cov = _agent_coverage_for_object(b.x, b.y, agent_xy)
        sigma = sigma0 / (1.0 + 4.0 * g * cov)
        dx = float(rng.normal(0.0, sigma))
        dy = float(rng.normal(0.0, sigma))
        preds.append(
            GtBoxBev(
                x=b.x + dx,
                y=b.y + dy,
                length=b.length,
                width=b.width,
                yaw=b.yaw,
                classe=b.classe,
            )
        )
        ious.append(bev_iou_rotated_simple(preds[-1], b))

    miou = float(np.mean(ious)) if ious else None

    by_class: Dict[str, List[float]] = {}
    for b, iou in zip(gt_boxes, ious):
        by_class.setdefault(b.classe, []).append(iou)

    ap_per_class: List[float] = []
    for cls, arr in by_class.items():
        tp = sum(1 for x in arr if x >= iou_thresh)
        ap_per_class.append(tp / max(len(arr), 1))

    nmap = float(np.mean(ap_per_class)) if ap_per_class else None
    ap_cls: Dict[str, float] = {}
    for cls, arr in by_class.items():
        ap_cls[cls] = round(sum(1 for x in arr if x >= iou_thresh) / max(len(arr), 1), 4)
    detail["ap_par_classe"] = ap_cls
    detail["iou_par_objet"] = [round(x, 4) for x in ious]
    return nmap, miou, detail


def resolve_default_gt_path() -> Optional[Path]:
    """Premier fichier ``ground_truth/*.json`` sous ``ADAFUSE_CARLA_DATA`` si présent."""
    root = os.environ.get("ADAFUSE_CARLA_DATA", "data/new_dataset_carla")
    gt_dir = Path(root) / "ground_truth"
    if not gt_dir.is_dir():
        return None
    jsons = sorted(gt_dir.glob("*.json"))
    return jsons[0] if jsons else None
