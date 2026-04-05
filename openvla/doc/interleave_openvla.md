# Interleave-OpenVLA

## Overview

This repository implements the interleaved version of [OpenVLA](https://openvla.github.io/), referred to as Interleave-OpenVLA. It extends the [OpenVLA codebase](https://github.com/openvla/openvla) by replacing the Prismatic VLM backbone with [InternVL2-2B](https://github.com/OpenGVLab/InternVL), enabling support for **interleaved image-text instructions** that guide robot manipulation.

Interleaved instructions mix natural language with inline object images (e.g., `"Put the <image> into the <image>"`), giving the model grounded visual references instead of relying on language alone.

---

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPUs with CUDA 12.1+ (tested on L40S 46GB and A100 80GB)
- Python 3.10

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (a fast Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Interleave-OpenVLA

```bash
cd openvla
uv sync
```

Install FlashAttention (recommended for faster training):

```bash
uv pip install flash-attn --no-build-isolation
```

### Install VimaBench

```bash
uv pip install pybullet "opencv-python<4.11" imageio transforms3d kornia hydra-core av black cloudpickle pyvirtualdisplay
uv pip install setuptools==65.5.0 pip==21 --index-strategy unsafe-best-match
uv pip install wheel==0.38.0 --index-strategy unsafe-best-match
uv run pip install gym==0.21.0  # If this fails, activate the venv and run: pip install gym==0.21.0
git clone https://github.com/vimalabs/VimaBench.git
uv pip install -e VimaBench --no-deps
```

### Verify Installation

```bash
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
uv run python -c "import vima_bench; print('VimaBench OK')"
uv run python -c "import internvl; print('InternVL OK')"
```

---

## Data Pipeline

Interleave-OpenVLA trains on datasets in [RLDS/TFDS](https://github.com/google-research/rlds) format.

VIMA is a simulation environment for tabletop pick-and-place tasks. The data generation script collects oracle demonstrations and saves them with interleaved image-text instructions.

#### Step 1: Generate Raw Episodes

```bash
cd openvla/scripts/vima_data_generation

# Full dataset (50,000 episodes per task, parallel across all CPU cores)
uv run generate_vima_data.py \
    num_episodes_per_task=50000 \
    save_path="vima_dataset_se2_full" \
    parallel=true \
    task_selection='["scene_understanding"]'
```

The script uses [Hydra](https://hydra.cc/) for configuration. See `conf.yaml` for all available parameters.
Available VIMA tasks: `scene_understanding`, `rotation`, `rearrange`, `rearrange_then_restore`, `novel_adj`, `novel_noun`, `twist`, `follow_order`, `sweep_without_exceeding`, `same_shape`, `manipulate_old_neighbor`, `pick_in_order_then_restore`, `visual_manipulation`.

Each generated episode is a `.npy` file containing per-step data:
- **action**: 12-dim SE2 vector `[pose0_xy(2), pose0_quat(4), pose1_xy(2), pose1_quat(4)]`
- **observation/image**: 128x256x3 RGB from the front camera
- **language_instruction**: Text with `<image>` placeholders (e.g., `"Put the <image> into the <image>"`)
- **image_instruction**: List of cropped object images (224x224x3) corresponding to each `<image>`

#### Step 2: Convert to TFDS Format

After generating raw `.npy` files, convert them into a TFDS dataset.

First, update `RAW_DATA_PATH` in the builder to point to your generated data:

```python
# Edit: tfds_dataset_builder/vima_interleave/vima_dataset_builder.py
RAW_DATA_PATH = "/path/to/your/vima_dataset_se2_full/*/"
```

Then build the dataset:

```bash
cd tfds_dataset_builder/vima_interleave
CUDA_VISIBLE_DEVICES="" tfds build --overwrite
```

The built dataset will be written to `~/tensorflow_datasets/vima_interleave/0.1.0/`.

---

## Model Checkpoint Setup

Interleave-OpenVLA uses InternVL2-2B as its vision-language backbone. Download it from Hugging Face and patch the config for VLA use:

```bash
# Download InternVL2-2B
cd openvla
huggingface-cli download OpenGVLab/InternVL2-2B --local-dir internvl_checkpoint/2b
```

---

## Training

### Configure Accelerate and DeepSpeed

Training uses [Accelerate](https://huggingface.co/docs/accelerate) with DeepSpeed ZeRO-2. The configs are in `vla-scripts/accelerate/` and `vla-scripts/deepspeed/`.

Edit `vla-scripts/accelerate/config.yaml` to match your GPU setup:

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: deepspeed/zero2.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
num_machines: 1
num_processes: 4          # <-- Set to your number of GPUs
```

### Launch Training

```bash
cd openvla/vla-scripts
bash finetune_internvl_dist.sh # remember to set the configs in this script before running
```

---

## Evaluation

### Evaluation on VIMA (Simulation)

Configure model paths and evaluation parameters (number of evaluations, tasks, partitions) at the top of the script or via command-line arguments.

```bash
uv run python experiments/robot/vima/eval_auto.py
```

---

## Troubleshooting

**`RuntimeError: Could not load library libcudnn_cnn_train.so.8`**

```bash
export LD_LIBRARY_PATH=${PWD}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
```
