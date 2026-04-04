"""Constants for the CARLA multi-agent export under ``data/new_dataset_carla/``."""

import os

# Default: repository root contains ``data/new_dataset_carla`` (PLY + trajectoire + ground_truth)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CARLA_DATA_ROOT = os.environ.get(
    "ADAFUSE_CARLA_DATA",
    os.path.join(_REPO_ROOT, "data", "new_dataset_carla"),
)

NUM_LIDAR_AGENTS = 10
# Ordre de grandeur scène (le nombre exact d'objets GT varie par frame dans ``ground_truth/``)
NUM_TARGET_VEHICLES = 50
AGENT_PREFIX = "Tesla"


def agent_id_to_dirname(agent_index: int) -> str:
    """Return folder name e.g. ``Tesla_7`` for 1-based index."""
    if not 1 <= agent_index <= NUM_LIDAR_AGENTS:
        raise ValueError(f"agent_index must be in [1, {NUM_LIDAR_AGENTS}], got {agent_index}")
    return f"{AGENT_PREFIX}_{agent_index}"


def list_agent_dirnames() -> list:
    return [agent_id_to_dirname(i) for i in range(1, NUM_LIDAR_AGENTS + 1)]
