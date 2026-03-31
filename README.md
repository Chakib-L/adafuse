# AdaFuse — Adaptive Fusion Strategy Selection for Cooperative Perception

> CentraleSupélec ST7 Research Project · 2025–2026

AdaFuse is a research framework for **cooperative multi-agent perception in autonomous vehicles**, built on top of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [CARLA](https://carla.org). Its core contribution is a **dynamic fusion strategy selector** that adapts in real time to driving conditions — rather than committing to a fixed fusion approach for all scenarios.

---

## Motivation

A single autonomous vehicle has a fundamentally limited field of view. Occlusions, long-range blind spots, and sensor noise are unavoidable when operating in isolation. Vehicle-to-Vehicle (V2V) communication enables agents to share their perceptions — but how they share it matters enormously.

Current cooperative perception systems use a **fixed fusion strategy** (early, late, or intermediate) regardless of context. This is suboptimal: a high-occlusion urban intersection calls for richer feature sharing, while a low-bandwidth highway scenario may only afford lightweight detection exchange.

**AdaFuse addresses this gap** by learning to select the right fusion strategy dynamically, conditioned on the current driving context.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CARLA Simulator                       │
│         Multi-agent scenarios · Sensor data collection       │
└─────────────────────────┬───────────────────────────────────┘
                          │  LiDAR point clouds, metadata
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    OpenCOOD Pipeline                         │
│                                                              │
│  Agent 1       Agent 2       Agent N                         │
│  [Encoder] ── [Encoder] ── [Encoder]                        │
│       │            │            │                            │
│       └────────────┴────────────┘                            │
│                    │                                         │
│             [Fusion Module] ◄── AdaFuse selects strategy     │
│                    │                                         │
│             [3D Detection]                                   │
│                    │                                         │
│             [Bounding Boxes + AP Metrics]                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  AdaFuse — Core Contribution                 │
│                                                              │
│  Context:  { n_agents, bandwidth, occlusion_rate, ... }     │
│                        │                                     │
│              [LLM-based Strategy Selector]                   │
│                        │                                     │
│   ┌────────────────────┼────────────────────┐               │
│   ▼                    ▼                    ▼               │
│ Early Fusion    Intermediate Fusion    Late Fusion           │
│ (raw LiDAR)     (shared features)    (detections only)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### OpenCOOD

[OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) (ICRA 2022) is an open-source framework for cooperative 3D object detection. It provides:

- A unified pipeline for **multi-agent LiDAR-based perception**
- Multiple fusion strategies: Early, Late, and Intermediate (AttFuse, V2VNet, F-Cooper, CoBEVT…)
- The **OPV2V dataset** — a large-scale V2V benchmark collected in CARLA
- Training and evaluation tools with standard AP metrics

In AdaFuse, OpenCOOD provides the backbone perception pipeline. We extend its fusion module to accept dynamic strategy selection at inference time.

### CARLA

[CARLA](https://carla.org) is an open-source autonomous driving simulator built on Unreal Engine. It provides:

- High-fidelity **multi-sensor simulation** (LiDAR, cameras, GNSS, IMU)
- Configurable **multi-agent scenarios** with controllable traffic and weather
- A Python API for programmatic scenario scripting and data collection

In AdaFuse, CARLA is used to generate diverse cooperative driving scenarios — varying the number of agents, occlusion levels, traffic density, and communication constraints — to train and evaluate the fusion selector.

### AdaFuse — Adaptive Selector

The core contribution of this project. A **context-aware module** that observes the current driving scenario and selects the most appropriate fusion strategy:

| Input Context | Selected Strategy | Rationale |
|---|---|---|
| High occlusion, many agents, good bandwidth | Intermediate | Rich feature sharing needed |
| Low bandwidth, few agents, open road | Late | Lightweight, sufficient |
| Dense traffic, near-range interaction | Early | Maximum information sharing |

The selector is built as an **LLM-based reasoning module** that takes structured context as input and outputs a fusion decision — making the selection process interpretable and steerable.

---

## Repository Structure

```
adafuse/
├── opencood/               # OpenCOOD codebase (integrated)
│   ├── models/
│   │   └── fuse_modules/   # Fusion architectures
│   ├── data_utils/         # OPV2V data loading
│   ├── tools/              # train.py, inference.py
│   └── hypes_yaml/         # Experiment configs
├── adafuse/                # AdaFuse core modules
│   ├── llm_selector.py     # LLM-based strategy selector
│   ├── adaptive_fusion.py  # Dynamic fusion wrapper
│   └── agents/             # Multi-agent coordination
├── carla/                  # CARLA integration
│   ├── scenarios/          # Scenario definitions
│   ├── data_collection/    # Sensor data pipelines
│   └── configs/            # CARLA environment configs
├── experiments/            # Logs, results, checkpoints
├── scripts/
│   ├── train.sh
│   └── eval.sh
├── environment.yml
├── setup.py
└── .gitignore
```

---

## Installation

### Prerequisites

- Ubuntu 18.04+
- CUDA 11.3+, cuDNN
- GPU with ≥ 6 GB VRAM
- CARLA 0.9.13+

### Setup

```bash
# Clone the repo
git clone git@github.com:TON-USERNAME/AdaFuse.git
cd AdaFuse

# Create conda environment
conda env create -f environment.yml
conda activate opencood

# Install PyTorch (CUDA 11.3)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Install spconv
pip install spconv-cu113

# Build CUDA extensions
python opencood/utils/setup.py build_ext --inplace

# Install in dev mode
python setup.py develop
```

---

## Usage

### Visualize the OPV2V dataset

```bash
# Edit validate_dir in opencood/hypes_yaml/visualization.yaml first
python opencood/visualization/vis_data_sequence.py --color_mode z-value
```

### Run inference with a fixed strategy

```bash
python opencood/tools/inference.py \
    --model_dir <CHECKPOINT_FOLDER> \
    --fusion_method intermediate \
    --show_vis
```

### Run inference with AdaFuse adaptive selection

```bash
python scripts/eval.sh --model_dir <CHECKPOINT_FOLDER> --adaptive
```

### Train

```bash
python opencood/tools/train.py --hypes_yaml <CONFIG_FILE>

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --use_env \
    opencood/tools/train.py --hypes_yaml <CONFIG_FILE>
```

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