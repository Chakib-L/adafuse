# AdaFuse — Adaptive Fusion Strategy Selection for Cooperative Perception

> CentraleSupélec ST7 Research Project · 2025–2026

AdaFuse is a research framework for **cooperative multi-agent perception in autonomous vehicles**, built on top of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [CARLA](https://carla.org). Its core contribution is a **dynamic fusion strategy selector** that chooses between **Intermediate Fusion** and **Late Fusion** (and hybrid groupings of agents) so as to **maximize detection quality** while **respecting network bandwidth** between agents.

---

## Research Goal

The project uses OpenCOOD as the perception backbone. The open research question is **when to fuse at feature level (intermediate)** versus **when to exchange only detections (late)**, possibly **partitioning the fleet** into clusters (intermediate within a cluster, late fusion across clusters or among remaining agents).

### Objectives

1. **Model communication constraints** — Represent available bitrates between pairs of agents (V2V links), and estimate the **data volume** induced by intermediate fusion (feature maps) versus late fusion (bounding boxes / scores).
2. **Context for the selector** — Provide each decision step with:
   - **2D positions** (or poses) of the simulated robots / ego vehicles,
   - A **short textual scene summary** (density, layout, occlusion hints if available),
   - **Bandwidth matrix** and feasibility flags from the constraint model.
3. **LLM-based fusion policy** — An LLM reads this structured context and outputs a **fusion plan**: global intermediate, global late, or **mixed** (e.g. clusters in intermediate, then late fusion between clusters or selected agents).
4. **Bi-objective rationale** — Favor plans that improve **detection metrics** (e.g. AP from OpenCOOD evaluation) while **minimizing aggregate network load**; the LLM prompt encodes this trade-off; future work may **fine-tune a small LLM** on curated (context, decision, metrics) tuples.

### Scope Note

OpenCOOD’s reference `inference.py` runs **one** fusion method per pass (`late`, `early`, or `intermediate`). Implementing a **full hybrid forward pass** inside the same model for arbitrary agent partitions may require additional model or orchestration work. The **AdaFuse** code in this repository delivers the **constraint model**, **scene encoding**, and **LLM (or rule-based) policy** that decide *which* strategy (or *which* cluster assignment) to use; coupling this tightly to batched hybrid OpenCOOD inference is left as an integration step.

---

## Motivation

A single autonomous vehicle has a fundamentally limited field of view. Occlusions, long-range blind spots, and sensor noise are unavoidable when operating in isolation. Vehicle-to-Vehicle (V2V) communication enables agents to share their perceptions — but **how** they share it matters: intermediate fusion is richer but **heavier on the link**; late fusion is **lighter** but may miss complementary cues.

**AdaFuse** does not fix a single strategy. It adapts to **topology, bitrate limits, and scene semantics** using a reasoning layer (LLM at inference time; optional smaller fine-tuned model later).

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│  CARLA dataset (this repo): `data/new_dataset_carla/`               │
│  10 LiDAR agents · GT JSON + trajectoires · per-agent PLY            │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  Scene encoder                                                      │
│  · Agent positions (x, y, yaw, id)                                  │
│  · Textual summary of the scene                                      │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  Bandwidth & feasibility model (`adafuse.bandwidth`)                 │
│  · Pairwise capacity C_ij (bits/s)                                   │
│  · Estimated load: intermediate vs late per link / cluster            │
│  · Feasibility under constraints                                     │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  Fusion policy (LLM or rule-based fallback, `adafuse.llm_selector`) │
│  Output: intermediate / late / hybrid clusters                     │
│  Objective: ↑ detection metrics, ↓ network usage                    │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  OpenCOOD pipeline (train / eval / inference)                       │
│  PointPillar (or other) · Intermediate vs Late fusion modules       │
└────────────────────────────────────────────────────────────────────┘
```

---

## Dataset (CARLA, local)

The main multi-agent export is under:

`data/new_dataset_carla/`

- **10 LiDAR agents** (`Tesla_1` … `Tesla_10`), each with time-stamped point clouds (`.ply`) and a **`trajectoire.csv`** (pose par frame).
- **`ground_truth/NNNNNN.json`** : vérité terrain 3D par frame (objets, boîtes, classes CARLA).
- Variable number of vehicles per frame (see GT files). Override the root with **`ADAFUSE_CARLA_DATA`** if needed.

An older export may still live under `data/carla_simulator/`; the code defaults to **`new_dataset_carla`**.

---

## Components

### OpenCOOD

[OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) (ICRA 2022) provides multi-agent LiDAR cooperative 3D detection, including **intermediate** and **late** fusion baselines, training, and AP metrics.

### CARLA

[CARLA](https://carla.org) is used to generate **multi-agent** scenarios; this repository includes a **pre-exported** dataset under `data/new_dataset_carla/`.

### AdaFuse modules (`adafuse/`)

| Module | Role |
|--------|------|
| `bandwidth.py` | Pairwise capacities, estimated bitrates for intermediate vs late fusion, feasibility checks. |
| `scene.py` | Agent states, scene summary text from positions + optional tags. |
| `policy.py` | Structured fusion plan (clusters, modes). |
| `llm_selector.py` | Prompt construction, JSON plan parsing; Hugging Face Llama 3 by default when `HUGGINGFACEHUB_API_TOKEN` is set; heuristic fallback or `--no-llm`. |
| `carla_constants.py` | Paths and constants for the local CARLA export (10 agents by default). |

---

## Repository Structure

```
adafuse/
├── opencood/               # OpenCOOD codebase (integrated)
├── adafuse/                # AdaFuse: bandwidth, scene, LLM policy
├── data/
│   └── new_dataset_carla/  # CARLA: 10 agents, PLY + trajectoire + ground_truth/
├── scripts/
│   └── run_adafuse_demo.py # Demo: bandwidth + scene + fusion decision
├── adafuse/ui/
│   └── streamlit_app.py    # Console web (contraintes + décision + chat)
├── jobs/
│   └── adafuse_ui.batch    # SLURM: lance Streamlit sur le cluster
├── experiments/            # Logs, results, checkpoints
├── environment.yml
├── setup.py
└── README.md
```

---

## Installation

### Prerequisites

- Ubuntu 18.04+
- CUDA 11.3+, cuDNN
- GPU with ≥ 6 GB VRAM
- CARLA 0.9.13+ (for new data collection; optional if only using `data/new_dataset_carla/`)

### Setup

```bash
git clone git@github.com:TON-USERNAME/adafuse.git
cd adafuse

conda env create -f environment.yml
conda activate opencood

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install spconv-cu113

python opencood/utils/setup.py build_ext --inplace
python setup.py develop
```

### LLM (Hugging Face Inference) — default

The demo (`scripts/run_adafuse_demo.py`) calls **`meta-llama/Meta-Llama-3-8B-Instruct`** via the Hugging Face **Inference Providers router** (`https://router.huggingface.co/v1/chat/completions`, OpenAI-compatible) whenever **`HUGGINGFACEHUB_API_TOKEN`** is set (e.g. in a project `.env` loaded automatically). Accept the model license on the Hub and use a token with **Inference Providers** permissions.

Optional: **`ADAFUSE_HF_MODEL`** (Hub model id), **`ADAFUSE_HF_ROUTER_URL`** (override the chat endpoint if needed).

**Without a token**, or after API/parse failures, the code falls back to a **deterministic heuristic**. To **skip the LLM** on purpose, run with **`--no-llm`** (or set **`ADAFUSE_NO_LLM=1`** in `jobs/adafuse.batch`).

Diagnostics: the demo JSON includes **`fallback_reason`**, **`error_detail`**, and **`dotenv_file`**. Messages are also printed to **stderr** (SLURM: `jobs/logs/error/adafuse.err`). Set **`ADAFUSE_VERBOSE=1`** for a full Python traceback on HF/parse errors.

---

## Usage

### CARLA data layout

Point clouds are organized per agent, e.g. `data/new_dataset_carla/Tesla_<id>/<frame>.ply`, with `<id>` in `1 … 10`, plus `ground_truth/<frame>.json` and `trajectoire.csv` per agent.

### Demo: bandwidth model + fusion decision

```bash
python scripts/run_adafuse_demo.py
```

### Web UI (contraintes, visualisation, chat Llama)

Interactive console: adjust link capacities (sliders), view fleet layout and **C_ij** heatmap, run **fusion decision** (Llama or heuristic), and chat with **Llama** with **session memory** (context includes the latest simulation digest).

```bash
pip install -r requirements-ui.txt
streamlit run adafuse/ui/streamlit_app.py
```

On the DGX cluster (SLURM), use **`jobs/adafuse_ui.batch`**, then SSH port-forward to the compute node (see comments in the batch file). Set **`HUGGINGFACEHUB_API_TOKEN`** in `.env` or the environment.

### Visualize OPV2V (or compatible) data

```bash
# Edit validate_dir in opencood/hypes_yaml/visualization.yaml first
python opencood/visualization/vis_data_sequence.py --color_mode z-value
```

### OpenCOOD inference (fixed strategy)

```bash
python opencood/tools/inference.py \
    --model_dir <CHECKPOINT_FOLDER> \
    --fusion_method intermediate \
    --show_vis
```

Use `fusion_method` ∈ {`late`, `early`, `intermediate`} as in OpenCOOD.

### Training (OpenCOOD)

```bash
python opencood/tools/train.py --hypes_yaml <CONFIG_FILE>
```

---

## Future Work

- **Fine-tune a small LLM** on (scene encoding, bandwidth snapshot, fusion plan, achieved AP, bits transferred).
- **End-to-end hybrid inference** in OpenCOOD driven by `FusionPlan` (cluster-wise intermediate + late merge).

---

## References

```bibtex
@inproceedings{xu2022opencood,
  author    = {Runsheng Xu, Hao Xiang, Xin Xia, Xu Han, Jinlong Li, Jiaqi Ma},
  title     = {OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication},
  booktitle = {ICRA},
  year      = {2022}
}
```

---

## Acknowledgements

Built on top of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) by Runsheng Xu et al., and the [CARLA](https://carla.org) simulator. This project is conducted as part of the ST7 research track at CentraleSupélec.
