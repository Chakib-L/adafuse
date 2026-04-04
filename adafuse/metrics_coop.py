"""
Métriques de détection + coût réseau pour AdaFuse (ground truth CARLA / OpenCOOD).

L'**AP** (Average Precision) sur boîtes 3D se calcule en comparant prédictions et
``ground_truth/*.json`` après alignement de repère et filtrage des classes — le pipeline
d'évaluation complet est celui d'**OpenCOOD** (``eval_utils``). Ce module fixe les **définitions**
de métriques **complémentaires** liées à l'infrastructure :

- **Efficacité réseau** : bits échangés par frame (borne ``total_bits_moved_upper_bound``).
- **Latence de coopération** : ``communication_time_s``, ``end_to_end_time_s`` vs budget (frame).
- **Score composite** (recherche multi-objectif) : combiner AP (depuis eval) et pénalité sur
  bits ou latence pour comparer des stratégies de fusion.

Les chemins GT du jeu ``data/new_dataset_carla/ground_truth/NNNNNN.json`` suivent le schéma
``frame`` + ``objets[]`` avec ``bounding_box_3d``, ``position_globale``, ``classe``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CoopEfficiencyMetrics:
    """Agrégat pour un run ou une séquence (à remplir après évaluation détection)."""

    mean_ap_3d: Optional[float] = None
    """AP@IoU moyen (ou par classe) — issu d'OpenCOOD / script d'éval sur prédictions vs GT."""

    mean_bits_per_frame: Optional[float] = None
    """Borne ou mesure des bits échangés par frame (voir ``NetworkInfrastructureModel``)."""

    mean_comm_latency_s: Optional[float] = None
    """Temps de communication moyen sous le plan de fusion retenu."""

    deadline_violation_rate: Optional[float] = None
    """Fraction de frames où ``end_to_end_time > frame_period`` (si suivi frame par frame)."""

    notes: str = ""


def load_carla_ground_truth_json(path: Path) -> Dict[str, Any]:
    """Charge un fichier ``ground_truth/NNNNNN.json`` du jeu CARLA exporté."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_gt_object_classes(gt: Dict[str, Any]) -> List[str]:
    """Liste des libellés CARLA (``vehicle.*``) présents dans une frame GT."""
    objs = gt.get("objets") or []
    return [str(o.get("classe", "")) for o in objs]


def tradeoff_score(
    ap: float,
    bits_per_frame: float,
    lambda_bits: float = 1e-8,
    lambda_latency: float = 0.0,
    latency_s: float = 0.0,
) -> float:
    """
    Exemple de score scalaire pour optimisation bi-objectif (à calibrer) :

    ``ap - λ_bits * bits - λ_lat * latency``
    """
    return ap - lambda_bits * bits_per_frame - lambda_latency * latency_s
