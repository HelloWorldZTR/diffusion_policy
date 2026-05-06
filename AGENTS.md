# AGENTS.md

## WDS Hand/Fingertip Task

This repo now contains an offline WebDataset task for training Diffusion Policy on hand and fingertip motion windows.

Primary files:
- `diffusion_policy/dataset/wds_hand_image_dataset.py`
- `diffusion_policy/workspace/train_diffusion_unet_hybrid_wds_workspace.py`
- `diffusion_policy/workspace/train_diffusion_transformer_hybrid_wds_workspace.py`
- `diffusion_policy/config/task/wds_hand_image.yaml`
- `diffusion_policy/config/train_diffusion_unet_hybrid_wds_workspace.yaml`
- `diffusion_policy/config/train_diffusion_transformer_hybrid_wds_workspace.yaml`
- `tests/test_wds_hand_image_dataset.py`
- `tests/generate_wds_hand_normalizer.py`
- `tests/export_wds_hand_window.py`
- `docs/wds-hand-task.md`
- `train_torchrun.py`

Key behavior:
- The task can train `DiffusionUnetHybridImagePolicy` or `DiffusionTransformerHybridImagePolicy`.
- Observations are head RGB image history, breast/chest RGB image history, and a 48D low-dimensional state.
- Actions are 48D future hand motion chunks.
- Language is intentionally ignored. Missing `meta.json` instruction fields are filled with dummy values before EgoVLA sliding-window composition.
- The WDS lowdim layout is expected to match EgoVLA: `wrist_state[18]`, `hand_state[30]`, `wrist_action[18]`, `hand_action[30]`, then camera calibration blocks.
- The target representation follows the EgoVLA fingertips transform: wrist pose is transformed to the head-camera frame, fingertips are transformed to each wrist frame, and actions default to relative actions from the latest observed state.

## Training Command Runbook

Use the project conda environment for runtime checks and training:
```bash
conda activate robodiff
```

Set the shard and cache paths once, then reuse them in the commands below:
```bash
export TRAIN_SHARDS='/path/to/train/shard-*.tar'
export VAL_SHARDS='/path/to/val/shard-*.tar'
export NORMALIZER_CACHE='data/wds/cache/wds_hand_normalizer.pt'
```
or config them in `config/task/wds_hand_image.yaml`

Dataset unit checks:
```bash
python -m pytest -q tests/test_wds_hand_image_dataset.py
```

Real-shard data inspection. This exports one decoded WDS window, image contact sheet, tensor summary, and optional `.npz` payload under `outputs/wds_hand_windows/`:
```bash
python tests/export_wds_hand_window.py \
  --shards "$VAL_SHARDS" \
  --window-index 0
```

Generate or refresh the normalizer cache from the training shards:
```bash
python tests/generate_wds_hand_normalizer.py \
  --config-name=train_diffusion_transformer_hybrid_wds_workspace \
  --cache-mode refresh \
  --cache-path "$NORMALIZER_CACHE" \
  task.train_wds_datasets.0.shard_urls="$TRAIN_SHARDS" \
  task.val_wds_datasets.0.shard_urls="$VAL_SHARDS"
```

Single-GPU transformer smoke test. This checks Hydra instantiation, WDS loading, normalizer loading, one short train loop, validation, sampling, and checkpoint writing:
```bash
python train.py --config-name=train_diffusion_transformer_hybrid_wds_workspace \
  task.dataset.normalizer_cache_mode=readonly \
  training.device=cuda:0 \
  training.debug=True \
  training.num_epochs=1 \
  training.steps_per_epoch=2 \
  training.max_val_steps=2 \
  training.resume=False \
  logging.mode=offline \
  dataloader.batch_size=2 \
  val_dataloader.batch_size=2 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0
```

Single-GPU full transformer training:
```bash
python train.py --config-name=train_diffusion_transformer_hybrid_wds_workspace \
  task.dataset.normalizer_cache_mode=readonly \
  training.steps_per_epoch=1000 \
  training.device=cuda:0
```

Multi-GPU transformer training:
```bash
torchrun --standalone --nproc_per_node=8 train_torchrun.py \
  --config-name=train_diffusion_transformer_hybrid_wds_workspace \
  task.dataset.normalizer_cache_mode=readonly \
  training.steps_per_epoch=1000
```

UNet training entrypoint:
```bash
python train.py --config-name=train_diffusion_unet_hybrid_wds_workspace \
  task.train_wds_datasets.0.shard_urls="$TRAIN_SHARDS" \
  task.val_wds_datasets.0.shard_urls="$VAL_SHARDS" \
  task.dataset.normalizer_cache_path="$NORMALIZER_CACHE" \
  task.dataset.normalizer_cache_mode=readonly \
  training.steps_per_epoch=1000 \
  training.device=cuda:0
```

UNet multi-GPU torchrun entrypoint:
```bash
torchrun --standalone --nproc_per_node=2 train_torchrun.py \
  --config-name=train_diffusion_unet_hybrid_wds_workspace \
  task.train_wds_datasets.0.shard_urls="$TRAIN_SHARDS" \
  task.val_wds_datasets.0.shard_urls="$VAL_SHARDS" \
  task.dataset.normalizer_cache_path="$NORMALIZER_CACHE" \
  task.dataset.normalizer_cache_mode=readonly \
  training.steps_per_epoch=1000
```

Implementation notes:
- `WdsHandImageDataset` is an `IterableDataset`; do not rely on `len(dataloader)`.
- The WDS workspace uses `training.steps_per_epoch` for scheduler length and epoch boundaries.
- Under `torchrun`, the WDS workspaces wrap `policy.compute_loss()` with DDP via `PolicyLossWrapper`; do not call custom policy methods through raw DDP directly.
- Both WDS workspaces use the Diffusion Policy batch schema `obs.image`, `obs.breast_image`, `obs.state`, and `action`; they do not consume LegendVLA's `images`, `states`, `actions`, or `actions_valid_mask` batch keys.
- Only rank0 writes wandb logs and checkpoints. All ranks train; rank0 runs validation and sampling while the other ranks wait at barriers.
- WDS train dataloaders must keep `shuffle: False`; shard/sample shuffling is handled inside the WDS pipeline.
- There is no env runner or rollout metric for this task. Checkpoint top-k monitors `val_loss` with `mode: min`.
- Normalizer fitting scans lowdim-only WDS samples with streaming stats and caches a metadata-validated payload at `task.dataset.normalizer_cache_path`.
- `normalizer_cache_mode` supports `auto`, `refresh`, and `readonly`. Under torchrun, rank0 generates or refreshes cache while other ranks load it after the barrier.
- The WDS normalizer matches EgoVLA's wrist rot6d ignore policy for `state/action` dimensions `6:18`.
- Do not directly reuse an EgoVLA normalizer file as `normalizer_cache_path`. EgoVLA normalizers are keyed as `states/actions` or `motions`; this Diffusion Policy task requires `image/breast_image/state/action`. Refit from the same WDS shards with `tests/generate_wds_hand_normalizer.py`, or write an explicit converter that maps keys and verifies matching transform settings.

Validation status:
- `python3 -m py_compile diffusion_policy/dataset/wds_hand_image_dataset.py tests/test_wds_hand_image_dataset.py tests/export_wds_hand_window.py tests/generate_wds_hand_normalizer.py` passed on 2026-05-06 after the dual-image conversion.
- `conda run -n VLM python -m pytest -q tests/test_wds_hand_image_dataset.py` passed on 2026-05-06 with 6 passed, 2 skipped. The skipped policy smoke tests need an environment with `robomimic` and `diffusers`.
