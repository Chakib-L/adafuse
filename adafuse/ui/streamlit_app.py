"""
Console AdaFuse — contraintes réseau, visualisation, décision fusion (Llama), chat contextuel.

Lancer en local :  streamlit run adafuse/ui/streamlit_app.py
Sur cluster SLURM : voir ``jobs/adafuse_ui.batch`` (tunnel SSH vers le port affiché).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# Racine du dépôt
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt
import streamlit as st

from adafuse.bandwidth import BandwidthConfig, BandwidthModel, estimate_objective_hint
from adafuse.carla_constants import NUM_LIDAR_AGENTS, NUM_TARGET_VEHICLES, list_agent_dirnames
from adafuse.fusion_eval import (
    compare_fusion_strategies,
    infrastructure_config_from_bandwidth_config,
    load_gt_boxes_optional,
)
from adafuse.gt_map_eval import resolve_default_gt_path
from adafuse.llm_selector import FusionSelector, heuristic_fusion_plan, hf_router_chat_messages, load_dotenv_from_project
from adafuse.scene import AgentPose, SceneContext, build_scene_summary, random_poses_circle

CHAT_SYSTEM = """Tu es l'assistant **AdaFuse** (perception coopérative véhicule, OpenCOOD).
Réponds en **français**, ton clair et concis. Tu t'appuies sur le **contexte de simulation** fourni
après ce bloc (positions, matrice de capacité, dernière décision de fusion).
Tu peux expliquer intermediate vs late fusion, débits V2V, et interpréter les décisions du modèle."""


def _inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=JetBrains+Mono:wght@400&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
h1 { font-weight: 700; letter-spacing: -0.03em; color: #e2e8f0 !important; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.stMetric { background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 12px; padding: 0.75rem; }
div[data-testid="stExpander"] { background: #0f172a; border: 1px solid #334155; border-radius: 12px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _fig_positions(xy: np.ndarray, labels: List[str], clusters: Optional[List[List[str]]] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    cmap = plt.get_cmap("tab10")
    id_to_i = {lab: i for i, lab in enumerate(labels)}

    if clusters:
        drawn = set()
        for ci, c in enumerate(clusters):
            if not c:
                continue
            col = cmap(ci % 10)
            idxs = [id_to_i[a] for a in c if a in id_to_i]
            for j in idxs:
                drawn.add(j)
            if not idxs:
                continue
            pts = xy[idxs, :]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=[col],
                s=140,
                edgecolors="#0f172a",
                linewidths=1.5,
                zorder=3,
            )
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            r = float(np.max(np.linalg.norm(pts - np.array([cx, cy]), axis=1))) + 4.0
            circ = plt.Circle((cx, cy), r, fill=False, edgecolor=col, linewidth=2.0, linestyle="--", alpha=0.9, zorder=2)
            ax.add_patch(circ)
        for i, name in enumerate(labels):
            if i not in drawn:
                ax.scatter([xy[i, 0]], [xy[i, 1]], c="#64748b", s=100, edgecolors="#334155", zorder=3)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c="#2dd4bf", s=120, edgecolors="#0d9488", linewidths=1.5, zorder=3)

    for i, name in enumerate(labels):
        ax.annotate(
            name.replace("Tesla_", ""),
            (xy[i, 0], xy[i, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            color="#94a3b8",
            fontsize=9,
        )
    ax.set_xlabel("x [m]", color="#94a3b8")
    ax.set_ylabel("y [m]", color="#94a3b8")
    ax.tick_params(colors="#64748b")
    ax.grid(True, alpha=0.2, color="#475569")
    ax.set_title("Flotte (plan XY)" + (" · clusters intermédiaires" if clusters else ""), color="#e2e8f0", fontsize=14)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def _fig_capacity(cap: np.ndarray, labels: List[str]) -> plt.Figure:
    n = cap.shape[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    im = ax.imshow(cap / 1e6, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Mbit/s", color="#94a3b8")
    cbar.ax.tick_params(colors="#64748b")
    short = [s.replace("Tesla_", "T") for s in labels]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8, color="#94a3b8")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=8, color="#94a3b8")
    ax.set_title("Capacité C_ij (symétrique)", color="#e2e8f0", fontsize=13)
    fig.tight_layout()
    return fig


def _build_simulation_context(
    plan_dict: Dict[str, Any],
    feasible: bool,
    overload: float,
    cap_scale: float,
    bw_cfg: BandwidthConfig,
) -> str:
    return (
        f"Échelle capacité appliquée : {cap_scale:.2f}\n"
        f"Config débit : intermediate={bw_cfg.intermediate_bps:.0f} b/s, late={bw_cfg.late_bps:.0f} b/s, "
        f"C_max={bw_cfg.capacity_max_bps:.0f} b/s, d_ref={bw_cfg.reference_distance_m} m, "
        f"γ distance={bw_cfg.distance_coupling:.2f}\n"
        f"Faisabilité (débit) : {'oui' if feasible else 'non'}, surcharge max (si >0) : {overload:.3g}\n"
        f"Décision fusion (JSON) :\n{json.dumps(plan_dict, indent=2, ensure_ascii=False)}"
    )


def main():
    st.set_page_config(
        page_title="AdaFuse · Console",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    st.title("AdaFuse · simulation & décision de fusion")
    st.caption("Ajustez les contraintes réseau, lancez une simulation, consultez le plan — puis discutez avec Llama (contexte conservé dans la session).")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # {role, content} user/assistant only
    if "simulation_digest" not in st.session_state:
        st.session_state.simulation_digest = "(aucune simulation encore)"

    with st.sidebar:
        st.header("Contraintes réseau")
        cap_max_mbps = st.slider("C_max (plafond lien) [Mbit/s]", 1.0, 50.0, 12.0, 0.5)
        cap_floor_mbps = st.slider(
            "Plancher lien [Mbit/s]",
            0.5,
            15.0,
            2.5,
            0.1,
            help="Doit être ≥ au débit intermediate (étoile) pour que les liens faibles restent faisables.",
        )
        d_ref = st.slider("Distance de référence d_ref [m]", 10.0, 150.0, 50.0, 5.0)
        eta = st.slider("Exposant affaiblissement η", 1.0, 4.0, 2.2, 0.1)
        gamma_dist = st.slider(
            "Couplage distance γ (liens : proches = meilleur réseau)",
            0.3,
            2.5,
            1.0,
            0.05,
            help="Dans C_ij, la distance est multipliée par γ avant l’affaiblissement. "
            "Plus γ est grand, plus les véhicules éloignés sont pénalisés (les proches gardent un meilleur lien relatif).",
        )
        inter_mbps = st.slider(
            "Débit requis intermediate [Mbit/s] (par lien hub↔agent)",
            0.5,
            20.0,
            2.5,
            0.5,
            help="Modèle en étoile : un flux intermediate par agent vers le hub du cluster.",
        )
        late_mbps = st.slider("Débit requis late [Mbit/s]", 0.05, 2.0, 0.2, 0.05)
        cap_scale = st.slider("Facteur global sur C_ij", 0.2, 2.0, 1.0, 0.05)

        st.divider()
        st.header("Scène")
        seed = st.number_input("Seed RNG", 0, 99999, 42)
        radius = st.slider("Rayon flotte [m]", 30.0, 200.0, 120.0, 5.0)
        n_agents = st.slider("Nombre d’agents", 2, NUM_LIDAR_AGENTS, NUM_LIDAR_AGENTS)

        use_llm = st.toggle("Décision via Llama (HF)", value=True)
        st.caption("Sans token HF, repli heuristique automatique.")
        st.divider()
        st.header("Ground truth (nMAP)")
        default_gt = os.environ.get("ADAFUSE_GT_FRAME", "")
        if not default_gt:
            r = resolve_default_gt_path()
            default_gt = str(r) if r else ""
        gt_frame_path = st.text_input(
            "Fichier ``ground_truth/*.json`` (CARLA)",
            value=default_gt,
            help="Pour calculer nMAP / mIoU vs GT. Ex. data/new_dataset_carla/ground_truth/000000.json",
        )

    bw_cfg = BandwidthConfig(
        intermediate_bps=inter_mbps * 1e6,
        late_bps=late_mbps * 1e6,
        capacity_max_bps=cap_max_mbps * 1e6,
        capacity_floor_bps=cap_floor_mbps * 1e6,
        reference_distance_m=d_ref,
        path_loss_exponent=eta,
        distance_coupling=float(gamma_dist),
    )

    agent_names = list_agent_dirnames()[: int(n_agents)]
    poses = random_poses_circle(
        n_agents=len(agent_names),
        radius_m=float(radius),
        prefix="Tesla",
        seed=int(seed),
    )
    for i, name in enumerate(agent_names):
        poses[i] = AgentPose(agent_id=name, x=poses[i].x, y=poses[i].y, yaw_rad=poses[i].yaw_rad)

    summary = build_scene_summary(poses, num_target_vehicles=NUM_TARGET_VEHICLES)
    scene = SceneContext(
        agent_poses=poses,
        scene_summary=summary,
        num_target_vehicles=NUM_TARGET_VEHICLES,
        extra_tags=[f"Dataset: {os.environ.get('ADAFUSE_CARLA_DATA', 'data/new_dataset_carla')}"],
    )

    model = BandwidthModel([p.agent_id for p in poses], config=bw_cfg)
    xy = np.array([[p.x, p.y] for p in poses], dtype=np.float64)
    dist = model.pairwise_distances_m(xy)
    cap = model.capacity_matrix_bps(dist) * cap_scale

    col_viz, col_dec = st.columns([1.1, 1.0])

    lbls = [p.agent_id for p in poses]
    last_clusters: Optional[List[List[str]]] = None
    if st.session_state.get("last_plan"):
        lp = st.session_state.last_plan
        if lp.get("mode") in ("hybrid", "intermediate"):
            ic = lp.get("intermediate_clusters") or []
            if ic:
                last_clusters = [list(x) for x in ic]

    with col_viz:
        st.subheader("Carte & capacités")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(_fig_positions(xy, lbls, clusters=last_clusters), clear_figure=True)
        with c2:
            st.pyplot(_fig_capacity(cap, lbls), clear_figure=True)

    with col_dec:
        st.subheader("Décision de fusion")
        if st.button("▶ Calculer la décision", type="primary", use_container_width=True):
            with st.spinner("Appel modèle…"):
                selector = FusionSelector(use_llm=None if use_llm else False)
                plan = selector.select_llm_position_only(scene, model, capacity_scale=cap_scale)
                rep = model.feasibility(plan, dist, capacity_scale=cap_scale)
                hint = estimate_objective_hint(rep)

                st.session_state.last_plan = plan.to_json_dict()
                st.session_state.last_diagnostics = selector.diagnostics_dict()
                st.session_state.last_feasible = rep.feasible
                st.session_state.last_overload = rep.overload_ratio
                st.session_state.last_hint = hint
                st.session_state.simulation_digest = _build_simulation_context(
                    st.session_state.last_plan,
                    rep.feasible,
                    rep.overload_ratio,
                    cap_scale,
                    bw_cfg,
                )

        if st.session_state.get("last_plan"):
            p = st.session_state.last_plan
            d = st.session_state.get("last_diagnostics", {})
            st.metric("Source", d.get("decision_source", "—"))
            st.metric("Mode", p.get("mode", "—"))
            ok = st.session_state.get("last_feasible", False)
            st.metric("Faisable (débit)", "oui" if ok else "non")
            with st.expander("JSON complet", expanded=False):
                st.json(
                    {
                        "fusion_plan": p,
                        "diagnostics": d,
                        "feasible": ok,
                        "overload_max_bps": st.session_state.get("last_overload"),
                        "objective_hint": st.session_state.get("last_hint"),
                    }
                )
        else:
            st.info("Cliquez sur **Calculer la décision** pour lancer Llama ou l’heuristique.")

        st.divider()
        st.markdown("##### Évaluation comparative (nMAP GT + réseau)")
        st.caption(
            "nMAP / mIoU : proxy basé sur le GT CARLA (bruit de localisation lié à la fusion). "
            "Indiquez un fichier ``ground_truth/*.json`` valide pour remplir ces colonnes. "
            "Sinon : métriques infrastructure uniquement."
        )
        if st.button("Comparer late / intermediate / LLM (clusters) / heuristique", use_container_width=True):
            infra_cfg = infrastructure_config_from_bandwidth_config(bw_cfg)
            agent_ids = [p.agent_id for p in poses]
            h_plan = heuristic_fusion_plan(scene, model, dist, capacity_scale=cap_scale)
            llm_pos = None
            load_dotenv_from_project()
            has_hf = bool(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
            sel_cmp = FusionSelector(use_llm=None if use_llm else False)
            if use_llm and has_hf:
                with st.spinner("LLM (positions + capacités, clusters)…"):
                    llm_pos = sel_cmp.select_llm_position_only(scene, model, capacity_scale=cap_scale)
            elif use_llm and not has_hf:
                st.warning(
                    "Token Hugging Face absent : comparaison sans appel LLM "
                    "(réglage du toggle ou HUGGINGFACEHUB_API_TOKEN)."
                )
            gt_boxes = load_gt_boxes_optional(gt_frame_path)
            cmp = compare_fusion_strategies(
                agent_ids=agent_ids,
                model=model,
                dist=dist,
                cap_scale=cap_scale,
                infra_cfg=infra_cfg,
                llm_position_only_plan=llm_pos,
                heuristic_plan=h_plan,
                gt_boxes=gt_boxes,
                agent_xy=xy,
                eval_seed=int(seed),
            )
            st.session_state.fusion_compare = cmp

        if st.session_state.get("fusion_compare"):
            cmp = st.session_state.fusion_compare
            order = [
                "late_fusion",
                "intermediate_fusion",
                "heuristique",
                "llm_clusters",
            ]
            rows = []
            for key in order:
                if key not in cmp:
                    continue
                ev = cmp[key]
                rows.append(
                    {
                        "Stratégie": key,
                        "Mode": ev.plan.mode.value,
                        "nMAP": "—" if ev.nmap is None else round(ev.nmap, 4),
                        "mIoU": "—" if ev.miou is None else round(ev.miou, 4),
                        "Faisable débit": ev.feasible_bandwidth,
                        "Surcharge max (bps)": round(ev.overload_max_bps, 1),
                        "T_comm (s)": round(ev.comm_time_s, 5),
                        "Faisable temps": ev.temporal_feasible,
                        "Bits/frame (UB)": int(ev.bits_upper_bound),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)
            with st.expander("Détail par stratégie (JSON + clusters intermédiaires)"):
                for key in order:
                    if key not in cmp:
                        continue
                    st.markdown(f"**{key}** — `{cmp[key].label}`")
                    st.json(cmp[key].to_display_dict())

    st.divider()
    st.subheader("Discussion avec Llama (mémoire de session)")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Posez une question sur la simulation ou la fusion…")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        messages = [
            {
                "role": "system",
                "content": CHAT_SYSTEM + "\n\n---\n## Contexte simulation (mis à jour après chaque calcul)\n"
                + st.session_state.simulation_digest,
            }
        ]
        for m in st.session_state.chat_messages:
            messages.append({"role": m["role"], "content": m["content"]})

        try:
            with st.chat_message("assistant"):
                with st.spinner("Llama…"):
                    reply = hf_router_chat_messages(messages, temperature=0.65, max_tokens=1024)
                st.markdown(reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            err = f"**Erreur API** : `{e}`"
            st.error(err)
            st.session_state.chat_messages.append({"role": "assistant", "content": err})

    with st.expander("Réinitialiser le chat"):
        if st.button("Effacer l’historique de conversation"):
            st.session_state.chat_messages = []
            st.rerun()


if __name__ == "__main__":
    main()
