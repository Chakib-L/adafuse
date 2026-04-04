"""LLM-based fusion selector with heuristic fallback (no API key required)."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adafuse.bandwidth import BandwidthModel, FeasibilityReport, positions_to_xy
from adafuse.policy import FusionMode, FusionPlan
from adafuse.scene import SceneContext

# Hugging Face Inference Providers (routeur OpenAI-compatible, remplace api-inference.huggingface.co)
DEFAULT_HF_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
_DOTENV_LOADED = False
_LAST_DOTENV_PATH: Optional[str] = None


SYSTEM_PROMPT = """You are a planning module for cooperative 3D object detection (OpenCOOD).
Choose a fusion strategy that MAXIMIZES detection quality (AP) while MINIMIZING wireless traffic.

Modes:
- "intermediate": all agents share intermediate feature representations (high bitrate).
- "late": agents only share final detections (low bitrate).
- "hybrid": some groups use intermediate fusion internally (clusters), then late fusion
  merges detections across groups.

Output ONLY valid JSON with keys:
- "mode": one of "intermediate", "late", "hybrid"
- "intermediate_clusters": array of arrays of agent id strings (empty if not hybrid/intermediate-split)
- "late_fusion_agents": array of all agent ids that participate in late fusion (usually everyone)
- "rationale": one short sentence

Respect bandwidth: if many links are capacity-limited, prefer "late" or small intermediate clusters.
"""

SYSTEM_PROMPT_POSITION_ONLY = """You are a planning module for cooperative 3D object detection (OpenCOOD).
You MUST choose a fusion strategy using ONLY:
- Agent positions (world frame)
- Pairwise link capacities (Mbit/s) — derived from distance; physically closer agents tend to have higher capacity

**Priority — intermediate clustering (preferred):**
Prefer mode **"hybrid"** with **several non-overlapping intermediate clusters** (2–4): strong **intermediate fusion inside each cluster** (star), then **late fusion** between clusters to merge detections. Group agents that are **close** and **well connected** in the same cluster.

Use **"intermediate"** (single global cluster, all agents) **only** when you want maximum feature sharing in one group **and** the capacity matrix clearly supports it for every hub link — otherwise **hybrid clustering is preferred**.

Use **"late"** only when neither hybrid nor global intermediate can respect link capacities.

Read **"## Bandwidth model (computed)"** in the user message for feasibility hints (full intermediate vs clustering).

Modes:
- "hybrid": **default choice** — multiple `intermediate_clusters` + `late_fusion_agents` (usually all).
- "intermediate": one cluster with all agents (optional, not default).
- "late": last resort.

Output ONLY valid JSON with keys:
- "mode": one of "intermediate", "late", "hybrid"
- "intermediate_clusters": for **hybrid**, non-empty disjoint clusters (each size ≥ 2 when possible); for **intermediate**, one cluster with all ids
- "late_fusion_agents": all agents that participate in late fusion (usually everyone)
- "rationale": **required** — one short sentence (French or English); never empty.
"""


def _parse_env_file(path: Path) -> None:
    """Load KEY=VALUE from ``.env`` into ``os.environ`` (values in file win over empty/missing)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        prev = os.environ.get(key)
        # HUGGINGFACEHUB_API_TOKEN : toujours depuis .env si la ligne existe (évite export vide / mauvaise valeur).
        if key == "HUGGINGFACEHUB_API_TOKEN":
            os.environ[key] = val
        elif prev is None or prev == "":
            os.environ[key] = val


def load_dotenv_from_project() -> Optional[str]:
    """
    Cherche un fichier ``.env`` (cwd → parents → racine du paquet) et le charge une fois.
    Retourne le chemin absolu du fichier chargé, ou ``None``.
    """
    global _DOTENV_LOADED, _LAST_DOTENV_PATH
    if _DOTENV_LOADED:
        return _LAST_DOTENV_PATH
    _DOTENV_LOADED = True
    here = Path.cwd()
    for _ in range(8):
        candidate = here / ".env"
        if candidate.is_file():
            _parse_env_file(candidate)
            _LAST_DOTENV_PATH = str(candidate.resolve())
            return _LAST_DOTENV_PATH
        if here.parent == here:
            break
        here = here.parent
    pkg_root = Path(__file__).resolve().parent.parent
    env_pkg = pkg_root / ".env"
    if env_pkg.is_file():
        _parse_env_file(env_pkg)
        _LAST_DOTENV_PATH = str(env_pkg.resolve())
        return _LAST_DOTENV_PATH
    return None


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in model output")
    return json.loads(m.group(0))


def _hf_router_chat_completions(
    user_content: str,
    model_id: str,
    token: str,
    timeout_s: float = 120.0,
    max_retries: int = 3,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
) -> str:
    """
    Appelle le routeur Hugging Face (format OpenAI ``/v1/chat/completions``).

    L'ancien ``api-inference.huggingface.co`` renvoie HTTP 410 ; il faut ce routeur.
    Auth : ``HUGGINGFACEHUB_API_TOKEN`` avec permission Inference Providers.
    """
    url = os.environ.get("ADAFUSE_HF_ROUTER_URL", DEFAULT_HF_ROUTER_CHAT_URL)
    sys_p = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    body = json.dumps(
        {
            "model": model_id,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as e:
            err_body = ""
            err_payload: Dict[str, Any] = {}
            try:
                err_body = e.read().decode("utf-8", errors="replace")
                err_payload = (
                    json.loads(err_body) if err_body.strip().startswith("{") else {}
                )
            except (json.JSONDecodeError, ValueError):
                err_payload = {}
            err_msg = err_payload.get("error") if isinstance(err_payload, dict) else None
            if err_msg is None and isinstance(err_payload.get("message"), str):
                err_msg = err_payload["message"]
            err_msg = err_msg or err_body or str(e)
            if e.code in (503, 429) and attempt + 1 < max_retries:
                wait = min(30.0, 2.0 ** (attempt + 1))
                if isinstance(err_payload, dict) and isinstance(
                    err_payload.get("estimated_time"), (int, float)
                ):
                    wait = max(wait, float(err_payload["estimated_time"]))
                time.sleep(wait)
                last_err = e
                continue
            raise RuntimeError(
                f"Hugging Face Router HTTP {e.code}: {err_msg}"
            ) from e

        text = _parse_openai_style_chat_response(payload)
        if text:
            return text.strip()
        last_err = RuntimeError("Réponse vide du routeur Hugging Face")
        time.sleep(2.0 * (attempt + 1))

    raise last_err or RuntimeError("Hugging Face Router: échec après reprises")


def _parse_openai_style_chat_response(payload: Any) -> str:
    """Extrait le texte assistant d'une réponse ``chat/completions`` (OpenAI / routeur HF)."""
    if not isinstance(payload, dict):
        return ""
    if "error" in payload:
        err = payload["error"]
        if isinstance(err, dict):
            raise RuntimeError(str(err.get("message", err)))
        raise RuntimeError(str(err))
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict) and "content" in msg:
            c = msg.get("content")
            return str(c) if c is not None else ""
    return ""


def _full_intermediate_plan(agent_ids: List[str]) -> FusionPlan:
    """Fusion intermediate globale (étoile vers le hub), un cluster = toute la flotte."""
    return FusionPlan(
        mode=FusionMode.INTERMEDIATE,
        intermediate_clusters=[list(agent_ids)],
        late_fusion_agents=list(agent_ids),
        rationale="Intermediate global (étoile).",
    )


def build_user_prompt_position_only(
    scene: SceneContext,
    bandwidth_model: BandwidthModel,
    distances_m: np.ndarray,
    capacity_scale: float = 1.0,
) -> str:
    """
    Positions + matrice C_ij + indices de faisabilité (intermediate global vs hybrid 2 clusters).
    """
    cap = bandwidth_model.capacity_matrix_bps(distances_m) * float(capacity_scale)
    cap_mbps = np.round(cap / 1e6, 4)
    mat = cap_mbps.tolist()
    agent_ids = list(bandwidth_model.agent_ids)
    full_inter = _full_intermediate_plan(agent_ids)
    rep_inter = bandwidth_model.feasibility(full_inter, distances_m, capacity_scale=capacity_scale)
    inter_ok = rep_inter.feasible
    hyb_x = _geographic_hybrid_two(scene, agent_ids, "x")
    rep_hx = bandwidth_model.feasibility(hyb_x, distances_m, capacity_scale=capacity_scale)
    hybrid2_ok = rep_hx.feasible
    parts: List[str] = [
        "## Agent positions (world frame)",
        json.dumps(scene.positions_jsonable(), indent=2),
        "",
        "## Pairwise link capacities (Mbit/s) — form intermediate clusters using strong links / proximity",
        json.dumps(
            {"agents": bandwidth_model.agent_ids, "capacity_mbps_symmetric": mat},
            indent=2,
        ),
        "",
        "## Bandwidth model (computed)",
        "Intermediate = étoile (débit `intermediate_bps` / lien hub↔agent). Hybrid = étoile **par cluster** + late **inter-clusters**.",
        f"- Intermediate **global** (un seul cluster, tous les agents): faisable = **{inter_ok}**",
        f"- Hybrid **2 clusters** (coupe géographique simple sur x, exemple): faisable = **{hybrid2_ok}**",
        "→ **Privilégier le mode hybrid avec clustering intermediate** lorsque des groupes faisables existent ; "
        "l’intermediate global n’est qu’une option si un seul grand cluster est pertinent.",
        "",
        "## Scene summary",
        scene.scene_summary,
        "",
        "## Task",
        f"There are {len(scene.agent_poses)} agents and about {scene.num_target_vehicles} vehicles to detect.",
    ]
    if scene.extra_tags:
        parts.extend(["", "## Extra", "\n".join(scene.extra_tags)])
    return "\n".join(parts)


def build_user_prompt(
    scene: SceneContext,
    feasibility_by_mode: Optional[Dict[str, FeasibilityReport]] = None,
) -> str:
    parts: List[str] = [
        "## Agent positions (world frame)",
        json.dumps(scene.positions_jsonable(), indent=2),
        "",
        "## Scene summary",
        scene.scene_summary,
        "",
        "## Task",
        f"There are {len(scene.agent_poses)} agents and about {scene.num_target_vehicles} vehicles to detect.",
    ]
    if scene.extra_tags:
        parts.extend(["", "## Extra", "\n".join(scene.extra_tags)])
    if feasibility_by_mode:
        parts.append("")
        parts.append("## Feasibility hints (required vs capacity per link, simplified)")
        for name, rep in feasibility_by_mode.items():
            parts.append(
                f"- {name}: feasible={rep.feasible}, max_overload_bps={rep.overload_ratio:.3g}, notes={rep.notes}"
            )
    return "\n".join(parts)


def heuristic_fusion_plan(
    scene: SceneContext,
    model: BandwidthModel,
    distances_m,
    capacity_scale: float = 1.0,
) -> FusionPlan:
    """
    Politique : **privilégier le hybrid** (2 clusters intermediate + late) lorsqu’il est faisable,
    puis intermediate global, puis late.
    """
    agent_ids = model.agent_ids
    n = len(agent_ids)
    if n == 0:
        return FusionPlan(mode=FusionMode.LATE, rationale="No agents.")

    # 1) Hybrid géographique (coupe x) — clustering intermediate privilégié
    xs = [(i, scene.agent_poses[i].x) for i in range(n)]
    xs.sort(key=lambda t: t[1])
    mid = max(1, n // 2)
    left = [agent_ids[xs[i][0]] for i in range(mid)]
    right = [agent_ids[xs[i][0]] for i in range(mid, n)]
    if left and right:
        hybrid = FusionPlan(
            mode=FusionMode.HYBRID,
            intermediate_clusters=[left, right],
            late_fusion_agents=list(agent_ids),
            rationale="Hybrid privilégié : deux clusters intermediate (coupe x) + late inter-clusters.",
        )
        rep_h = model.feasibility(hybrid, distances_m, capacity_scale=capacity_scale)
        if rep_h.feasible:
            return hybrid

    # 2) Même logique sur l’axe y
    # Second essai : découpe suivant y (véhicules alignés différemment)
    ys = [(i, scene.agent_poses[i].y) for i in range(n)]
    ys.sort(key=lambda t: t[1])
    mid_y = max(1, n // 2)
    low = [agent_ids[ys[i][0]] for i in range(mid_y)]
    high = [agent_ids[ys[i][0]] for i in range(mid_y, n)]
    if low and high and set(low) != set(left):
        hybrid_y = FusionPlan(
            mode=FusionMode.HYBRID,
            intermediate_clusters=[low, high],
            late_fusion_agents=list(agent_ids),
            rationale="Hybrid privilégié : deux clusters intermediate (coupe y) + late inter-clusters.",
        )
        if model.feasibility(hybrid_y, distances_m, capacity_scale=capacity_scale).feasible:
            return hybrid_y

    # 3) Intermediate global si faisable (un seul cluster)
    full_inter = _full_intermediate_plan(list(agent_ids))
    rep_i = model.feasibility(full_inter, distances_m, capacity_scale=capacity_scale)
    if rep_i.feasible:
        return FusionPlan(
            mode=FusionMode.INTERMEDIATE,
            intermediate_clusters=[list(agent_ids)],
            late_fusion_agents=list(agent_ids),
            rationale="Hybrid à 2 clusters infaisable ; intermediate global faisable.",
        )

    full_late = FusionPlan(mode=FusionMode.LATE, late_fusion_agents=list(agent_ids))
    rep_l = model.feasibility(full_late, distances_m, capacity_scale=capacity_scale)
    if rep_l.feasible:
        return FusionPlan(
            mode=FusionMode.LATE,
            late_fusion_agents=list(agent_ids),
            rationale="Intermediate / hybrid infeasible; late fusion fits capacity.",
        )

    return FusionPlan(
        mode=FusionMode.LATE,
        late_fusion_agents=list(agent_ids),
        rationale="Conservative late fusion (capacity stress on several links).",
    )


def _merge_singleton_chunks(chunks: List[List[str]]) -> List[List[str]]:
    """Fusionne les clusters d'un seul agent avec un voisin adjacent."""
    chunks = [list(c) for c in chunks if c]
    changed = True
    while changed and len(chunks) > 1:
        changed = False
        for i, c in enumerate(chunks):
            if len(c) != 1:
                continue
            if i + 1 < len(chunks):
                chunks[i + 1] = c + chunks[i + 1]
                chunks.pop(i)
            else:
                chunks[-2].extend(c)
                chunks.pop()
            changed = True
            break
    return chunks


def _angular_hybrid_k(
    scene: SceneContext,
    agent_ids: List[str],
    k: int,
) -> FusionPlan:
    """Découpe en k secteurs angulaires (tri par angle autour du barycentre) → clusters intermediate."""
    n = len(agent_ids)
    if n < 2 or k < 2:
        return FusionPlan(
            mode=FusionMode.LATE,
            late_fusion_agents=list(agent_ids),
            rationale="Découpe angulaire impossible (k ou n trop petit).",
        )
    id_to_pose = {p.agent_id: p for p in scene.agent_poses}
    xs = np.array([float(id_to_pose[a].x) for a in agent_ids])
    ys = np.array([float(id_to_pose[a].y) for a in agent_ids])
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    angles = [math.atan2(ys[i] - cy, xs[i] - cx) for i in range(n)]
    order = sorted(range(n), key=lambda i: angles[i])
    raw_chunks: List[List[str]] = []
    for seg in range(k):
        i0 = int(seg * n / k)
        i1 = n if seg == k - 1 else int((seg + 1) * n / k)
        raw_chunks.append([agent_ids[order[j]] for j in range(i0, i1)])
    clusters = _merge_singleton_chunks(raw_chunks)
    if not clusters:
        return FusionPlan(
            mode=FusionMode.LATE,
            late_fusion_agents=list(agent_ids),
            rationale="Clusters angulaires vides.",
        )
    return FusionPlan(
        mode=FusionMode.HYBRID,
        intermediate_clusters=clusters,
        late_fusion_agents=list(agent_ids),
        rationale=f"Réparation automatique : {k} secteurs angulaires (intermediate par secteur + late global).",
    )


def _iter_hybrid_repairs(
    scene: SceneContext,
    agent_ids: List[str],
) -> List[FusionPlan]:
    """Liste des plans hybrid candidates à tester (géographie + angles multi‑clusters)."""
    out: List[FusionPlan] = []
    for axis in ("x", "y"):
        out.append(_geographic_hybrid_two(scene, list(agent_ids), axis))
    n = len(agent_ids)
    k_hi = min(8, max(2, n - 1))
    for k in range(2, k_hi + 1):
        out.append(_angular_hybrid_k(scene, list(agent_ids), k))
    return out


def _geographic_hybrid_two(
    scene: SceneContext,
    agent_ids: List[str],
    axis: str,
) -> FusionPlan:
    n = len(agent_ids)
    if n < 2:
        return FusionPlan(
            mode=FusionMode.LATE,
            late_fusion_agents=list(agent_ids),
            rationale="Réparation géographique impossible (moins de 2 agents).",
        )
    if axis == "y":
        order = sorted(range(n), key=lambda i: scene.agent_poses[i].y)
    else:
        order = sorted(range(n), key=lambda i: scene.agent_poses[i].x)
    mid = max(1, n // 2)
    a = [agent_ids[order[i]] for i in range(mid)]
    b = [agent_ids[order[i]] for i in range(mid, n)]
    return FusionPlan(
        mode=FusionMode.HYBRID,
        intermediate_clusters=[a, b],
        late_fusion_agents=list(agent_ids),
        rationale=f"Réparation géographique (coupe sur {axis}) : deux clusters intermediate + late.",
    )


def fusion_plan_from_llm_json(data: Dict[str, Any]) -> FusionPlan:
    raw_mode = (data.get("mode") or "hybrid").strip().lower()
    try:
        mode = FusionMode(raw_mode)
    except ValueError:
        mode = FusionMode.HYBRID
    clusters = data.get("intermediate_clusters") or []
    late = data.get("late_fusion_agents") or []
    rationale = (data.get("rationale") or "").strip()
    if not rationale:
        rationale = (
            "Aucune justification fournie par le modèle (JSON incomplet ou réponse tronquée)."
        )
    return FusionPlan(
        mode=mode,
        intermediate_clusters=[list(c) for c in clusters],
        late_fusion_agents=list(late),
        rationale=rationale,
    )


class FusionSelector:
    """Select a ``FusionPlan`` via Hugging Face Inference (Llama 3) or heuristic."""

    def __init__(
        self,
        use_llm: Optional[bool] = None,
        model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        load_dotenv_from_project()
        self.hf_token = hf_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.model_id = model_id or os.environ.get(
            "ADAFUSE_HF_MODEL", DEFAULT_HF_MODEL_ID
        )
        if use_llm is None:
            use_llm = bool(self.hf_token)
        self.use_llm = use_llm
        self.last_decision_source: str = "heuristic"
        self.last_fallback_reason: Optional[str] = None
        self.last_error_detail: Optional[str] = None

    def _log_fallback(self, reason: str, detail: Optional[str] = None, exc: Optional[BaseException] = None) -> None:
        self.last_fallback_reason = reason
        if detail:
            self.last_error_detail = detail[:2000]
        elif exc is not None:
            self.last_error_detail = f"{type(exc).__name__}: {exc}"[:2000]
        msg = f"[adafuse] Fusion LLM → heuristique : {reason}"
        if self.last_error_detail:
            msg += f" | {self.last_error_detail}"
        print(msg, file=sys.stderr)
        if os.environ.get("ADAFUSE_VERBOSE", "").strip() in ("1", "true", "yes") and exc is not None:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

    def diagnostics_dict(self) -> Dict[str, Any]:
        return {
            "decision_source": self.last_decision_source,
            "fallback_reason": self.last_fallback_reason,
            "error_detail": self.last_error_detail,
            "hf_token_configured": bool(self.hf_token),
            "llm_requested": self.use_llm,
            "hf_model": self.model_id,
            "hf_router_url": os.environ.get(
                "ADAFUSE_HF_ROUTER_URL", DEFAULT_HF_ROUTER_CHAT_URL
            ),
            "dotenv_file": load_dotenv_from_project(),
        }

    def select(
        self,
        scene: SceneContext,
        bandwidth_model: BandwidthModel,
        capacity_scale: float = 1.0,
    ) -> FusionPlan:
        """Alias : même logique que ``select_llm_position_only`` (positions + capacités, pas d’oracle faisabilité)."""
        return self.select_llm_position_only(scene, bandwidth_model, capacity_scale=capacity_scale)

    def select_llm_position_only(
        self,
        scene: SceneContext,
        bandwidth_model: BandwidthModel,
        capacity_scale: float = 1.0,
    ) -> FusionPlan:
        """
        Variante LLM : positions + matrice de capacités uniquement (pas de résumés faisabilité late/inter).
        Retombe sur l'heuristique si pas de token ou erreur / plan infaisable.
        """
        self.last_decision_source = "heuristic"
        self.last_fallback_reason = None
        self.last_error_detail = None
        xy = positions_to_xy(scene.agent_poses)
        dist = bandwidth_model.pairwise_distances_m(xy)

        if not self.hf_token:
            self._log_fallback(
                "missing_hf_token_position_only",
                "Token HF requis pour la voie LLM positions+débit.",
            )
            return heuristic_fusion_plan(scene, bandwidth_model, dist, capacity_scale=capacity_scale)

        if not self.use_llm:
            self._log_fallback("llm_desactive_flag_no_llm")
            return heuristic_fusion_plan(scene, bandwidth_model, dist, capacity_scale=capacity_scale)

        user = build_user_prompt_position_only(scene, bandwidth_model, dist, capacity_scale=capacity_scale)
        try:
            raw = _hf_router_chat_completions(
                user,
                self.model_id,
                self.hf_token,
                system_prompt=SYSTEM_PROMPT_POSITION_ONLY,
                max_tokens=1024,
            )
            data = _extract_json_object(raw)
            plan = fusion_plan_from_llm_json(data)
            agent_ids = bandwidth_model.agent_ids
            if plan.mode == FusionMode.HYBRID and not any(plan.intermediate_clusters):
                plan = FusionPlan(
                    mode=FusionMode.LATE,
                    late_fusion_agents=list(agent_ids),
                    rationale=(plan.rationale or "") + " | hybrid sans clusters → late.",
                )
            rep = bandwidth_model.feasibility(plan, dist, capacity_scale=capacity_scale)
            if not rep.feasible:
                overload_bps = max(0.0, float(rep.overload_ratio))
                for repaired in _iter_hybrid_repairs(scene, list(agent_ids)):
                    if bandwidth_model.feasibility(repaired, dist, capacity_scale=capacity_scale).feasible:
                        self.last_decision_source = "llm_repaired_hybrid_auto"
                        repaired.rationale = (
                            f"{repaired.rationale} Plan LLM initial infaisable "
                            f"(surcharge max ≈ {overload_bps/1e6:.2f} Mbit/s). "
                            "Substitution par un hybrid géométrique/angulaire faisable."
                        )
                        return repaired
                late_only = FusionPlan(
                    mode=FusionMode.LATE,
                    late_fusion_agents=list(agent_ids),
                    rationale=(
                        f"Réparation auto : plan LLM infaisable (surcharge max ≈ {overload_bps/1e6:.2f} Mbit/s). "
                        "Aucun découpage en clusters hybrid faisable ; repli late fusion."
                    ),
                )
                if bandwidth_model.feasibility(late_only, dist, capacity_scale=capacity_scale).feasible:
                    self._log_fallback(
                        "llm_position_only_infeasible_fallback_late",
                        f"notes={rep.notes!r}, max_overload_bps={overload_bps!r}",
                    )
                    self.last_decision_source = "llm_repaired_late"
                    return late_only
                self._log_fallback(
                    "llm_position_only_infeasible",
                    f"notes={rep.notes!r}, max_overload_bps={overload_bps!r}",
                )
                return heuristic_fusion_plan(scene, bandwidth_model, dist, capacity_scale=capacity_scale)
            self.last_decision_source = "llm_position_only"
            return plan
        except Exception as exc:
            self._log_fallback("hf_api_ou_parse_json_position_only", exc=exc)
            return heuristic_fusion_plan(scene, bandwidth_model, dist, capacity_scale=capacity_scale)


def hf_router_chat_messages(
    messages: List[Dict[str, str]],
    model_id: Optional[str] = None,
    token: Optional[str] = None,
    temperature: float = 0.65,
    max_tokens: int = 1024,
    timeout_s: float = 120.0,
) -> str:
    """
    Chat multi-tours via le routeur Hugging Face (``/v1/chat/completions``).

    ``messages`` : liste OpenAI ``[{"role": "system"|"user"|"assistant", "content": "..."}, ...]``.
    """
    load_dotenv_from_project()
    tok = token or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not tok:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN manquant")
    mid = model_id or os.environ.get("ADAFUSE_HF_MODEL", DEFAULT_HF_MODEL_ID)
    url = os.environ.get("ADAFUSE_HF_ROUTER_URL", DEFAULT_HF_ROUTER_CHAT_URL)
    body = json.dumps(
        {
            "model": mid,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {tok}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.load(resp)
    return _parse_openai_style_chat_response(payload).strip()


def select_fusion_plan(
    scene: SceneContext,
    bandwidth_model: BandwidthModel,
    use_llm: Optional[bool] = None,
    capacity_scale: float = 1.0,
) -> FusionPlan:
    """Functional API."""
    return FusionSelector(use_llm=use_llm).select(scene, bandwidth_model, capacity_scale=capacity_scale)
