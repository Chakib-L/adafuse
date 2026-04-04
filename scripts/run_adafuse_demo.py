#!/usr/bin/env python3
"""Demo: scene encoding + bandwidth model + fusion plan (heuristic or LLM)."""

import argparse
import json
import os
import sys

# Allow running without installing the package
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adafuse.bandwidth import BandwidthConfig, BandwidthModel, estimate_objective_hint, positions_to_xy
from adafuse.carla_constants import NUM_LIDAR_AGENTS, NUM_TARGET_VEHICLES, list_agent_dirnames
from adafuse.llm_selector import FusionSelector
from adafuse.scene import AgentPose, SceneContext, build_scene_summary, random_poses_circle


def main():
    parser = argparse.ArgumentParser(description="AdaFuse bandwidth + fusion decision demo")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic poses")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force heuristic only (default: Hugging Face Llama 3 when HUGGINGFACEHUB_API_TOKEN is set, .env supported)",
    )
    parser.add_argument("--agents", type=int, default=NUM_LIDAR_AGENTS, help="Number of agents (default: NUM_LIDAR_AGENTS)")
    args = parser.parse_args()

    agent_names = list_agent_dirnames()[: args.agents]
    poses = random_poses_circle(
        n_agents=len(agent_names),
        radius_m=120.0,
        prefix="Tesla",
        seed=args.seed,
    )
    for i, name in enumerate(agent_names):
        poses[i] = AgentPose(agent_id=name, x=poses[i].x, y=poses[i].y, yaw_rad=poses[i].yaw_rad)

    summary = build_scene_summary(poses, num_target_vehicles=NUM_TARGET_VEHICLES)
    scene = SceneContext(
        agent_poses=poses,
        scene_summary=summary,
        num_target_vehicles=NUM_TARGET_VEHICLES,
        extra_tags=[f"Data root (CARLA export): {os.environ.get('ADAFUSE_CARLA_DATA', 'data/new_dataset_carla')}"],
    )

    bw_cfg = BandwidthConfig()
    model = BandwidthModel([p.agent_id for p in poses], config=bw_cfg)
    xy = positions_to_xy(poses)
    dist = model.pairwise_distances_m(xy)

    use_llm = False if args.no_llm else None
    selector = FusionSelector(use_llm=use_llm)
    plan = selector.select(scene, model)
    rep = model.feasibility(plan, dist)
    hint = estimate_objective_hint(rep)

    out = {
        **selector.diagnostics_dict(),
        "carla_dataset": {
            "num_lidars": NUM_LIDAR_AGENTS,
            "num_target_vehicles": NUM_TARGET_VEHICLES,
            "agent_folders_example": agent_names[:5],
        },
        "fusion_plan": plan.to_json_dict(),
        "feasible_under_model": rep.feasible,
        "overload_max_bps": rep.overload_ratio,
        "objective_hint": hint,
        "bandwidth_config": {
            "intermediate_bps": bw_cfg.intermediate_bps,
            "late_bps": bw_cfg.late_bps,
            "capacity_max_bps": bw_cfg.capacity_max_bps,
        },
    }
    if args.no_llm:
        out["hf_model"] = None
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
