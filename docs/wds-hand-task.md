# WDS Hand/Fingertip Task

This document describes the WDS hand/fingertip task added on 2026-05-05. It is an offline training path for Diffusion Policy that streams sliding windows directly from WebDataset shards.

## What It Trains

The task uses `DiffusionUnetHybridImagePolicy` to predict future hand motion from:
- RGB image history: `obs.image`
- self state history: `obs.state`

The model predicts:
- future hand action chunks: `action`

Tensor contract after collation:
- `obs.image`: `[B, n_obs_steps, 3, H, W]`, float32 in `[0, 1]`
- `obs.state`: `[B, n_obs_steps, 48]`, float32
- `action`: `[B, horizon, 48]`, float32

The 48D low-dimensional vector is:
- 18D wrist pose or action
- 30D fingertip points for both hands

Language instructions are not part of this task. WDS samples may omit instruction fields; the dataset adapter fills dummy metadata before calling the EgoVLA sliding-window composer.

## Data Requirements

Each WDS frame sample is expected to contain:
- `<key>.meta.json`
- `<key>.lowdim.npy`
- `<key>.image.jpg`

`lowdim.npy` must follow the EgoVLA base layout:
- `0:18`: `wrist_state`
- `18:48`: `hand_state`
- `48:66`: `wrist_action`
- `66:96`: `hand_action`
- `96:`: camera calibration blocks, including head camera extrinsic and intrinsic

Episode boundaries are read from `meta.json`:
- `dataset_name`
- `episode_index`

If these are missing, the adapter supplies defaults, but real training data should provide stable episode metadata so windows do not cross demonstrations.

## Representation

The dataset adapter uses the selected EgoVLA-style fingertip representation:
- Wrist state/action is transformed into the head-camera frame with the current head extrinsic.
- Fingertip points are transformed into their corresponding wrist frames.
- `use_relative_action: True` converts action chunks into relative actions from the latest observed state.

This representation is implemented in `process_wds_state_action()` inside `diffusion_policy/dataset/wds_hand_image_dataset.py`.

## Configuration

Task config:
- `diffusion_policy/config/task/wds_hand_image.yaml`

Training config:
- `diffusion_policy/config/train_diffusion_unet_hybrid_wds_workspace.yaml`

Important defaults:
- image shape: `[3, 224, 224]`
- `horizon: 16`
- `n_obs_steps: 2`
- `image_stride: 30`
- `state_stride: 30`
- `action_stride: 1`
- train shards: `data/wds/train/shard-*.tar`
- validation shards: `data/wds/val/shard-*.tar`
- normalizer cache: `data/wds/cache/wds_hand_normalizer.pt`
- checkpoint monitor: `val_loss`, `mode: min`

`training.steps_per_epoch` is required because the training dataset is streaming and has no meaningful length.

## Running

Install the environment with WDS dependencies. `webdataset` has been added to the conda environment pip sections; the task also requires OpenCV and Pillow because EgoVLA media decoding uses them.

Example training command:
```bash
python train.py --config-name=train_diffusion_unet_hybrid_wds_workspace \
  task.train_wds_datasets.0.shard_urls='/path/to/train/shard-*.tar' \
  task.val_wds_datasets.0.shard_urls='/path/to/val/shard-*.tar' \
  task.dataset.normalizer_cache_path='/path/to/cache/wds_hand_normalizer.pt' \
  training.steps_per_epoch=1000 \
  training.device=cuda:0
```

For a short smoke run:
```bash
python train.py --config-name=train_diffusion_unet_hybrid_wds_workspace \
  task.train_wds_datasets.0.shard_urls='/path/to/train/shard-*.tar' \
  task.val_wds_datasets.0.shard_urls='/path/to/val/shard-*.tar' \
  training.debug=True \
  training.steps_per_epoch=2 \
  training.max_val_steps=2 \
  dataloader.batch_size=2 \
  val_dataloader.batch_size=2 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0
```

## Testing

Synthetic WDS tests live in:
- `tests/test_wds_hand_image_dataset.py`

Intended checks:
- Missing instruction metadata does not fail.
- Dataset batches have Diffusion Policy-compatible shapes.
- Episode tail windows keep fixed tensor shapes under truncate padding.
- Optional policy smoke test runs `compute_loss()` when `robomimic` and `diffusers` are installed.

Known verification status on 2026-05-05:
- Static compile passed:
  ```bash
  python3 -m py_compile diffusion_policy/dataset/wds_hand_image_dataset.py diffusion_policy/workspace/train_diffusion_unet_hybrid_wds_workspace.py tests/test_wds_hand_image_dataset.py
  ```
- Runtime pytest was not executed in the implementation shell because that bare Python lacked `pytest`, `omegaconf`, `webdataset`, `cv2`, `PIL`, and `robomimic`.

Run the full WDS tests inside the project environment:
```bash
python -m pytest -q tests/test_wds_hand_image_dataset.py
```

## Current Limitations

- This is an offline training path only; there is no simulator or real robot env runner.
- Top-k checkpoints depend on validation producing `val_loss`; keep `val_wds_datasets` configured.
- The adapter assumes head-camera WDS data compatible with EgoVLA `wds_dataset.py`.
- The root `external/EgoVLA` checkout is required because the adapter reuses its WDS sliding-window utilities.

