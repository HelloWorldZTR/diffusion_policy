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
- `docs/wds-hand-task.md`
- `train_torchrun.py`

Key behavior:
- The task can train `DiffusionUnetHybridImagePolicy` or `DiffusionTransformerHybridImagePolicy`.
- Observations are RGB image history plus a 48D low-dimensional state.
- Actions are 48D future hand motion chunks.
- Language is intentionally ignored. Missing `meta.json` instruction fields are filled with dummy values before EgoVLA sliding-window composition.
- The WDS lowdim layout is expected to match EgoVLA: `wrist_state[18]`, `hand_state[30]`, `wrist_action[18]`, `hand_action[30]`, then camera calibration blocks.
- The target representation follows the EgoVLA fingertips transform: wrist pose is transformed to the head-camera frame, fingertips are transformed to each wrist frame, and actions default to relative actions from the latest observed state.

Training entrypoint:
```bash
python train.py --config-name=train_diffusion_unet_hybrid_wds_workspace \
  task.train_wds_datasets.0.shard_urls='/path/to/train/shard-*.tar' \
  task.val_wds_datasets.0.shard_urls='/path/to/val/shard-*.tar' \
  training.device=cuda:0
```

Transformer training entrypoint:
```bash
python train.py --config-name=train_diffusion_transformer_hybrid_wds_workspace \
  task.train_wds_datasets.0.shard_urls='/path/to/train/shard-*.tar' \
  task.val_wds_datasets.0.shard_urls='/path/to/val/shard-*.tar' \
  training.steps_per_epoch=1000 \
  training.device=cuda:0
```

Multi-GPU torchrun entrypoint:
```bash
torchrun --standalone --nproc_per_node=2 train_torchrun.py \
  --config-name=train_diffusion_transformer_hybrid_wds_workspace \
  task.train_wds_datasets.0.shard_urls='/path/to/train/shard-*.tar' \
  task.val_wds_datasets.0.shard_urls='/path/to/val/shard-*.tar'
```

Implementation notes:
- `WdsHandImageDataset` is an `IterableDataset`; do not rely on `len(dataloader)`.
- The WDS workspace uses `training.steps_per_epoch` for scheduler length and epoch boundaries.
- Under `torchrun`, the WDS workspaces wrap `policy.compute_loss()` with DDP via `PolicyLossWrapper`; do not call custom policy methods through raw DDP directly.
- Both WDS workspaces use the Diffusion Policy batch schema `obs.image`, `obs.state`, and `action`; they do not consume LegendVLA's `images`, `states`, `actions`, or `actions_valid_mask` batch keys.
- Only rank0 writes wandb logs and checkpoints. All ranks train; rank0 runs validation and sampling while the other ranks wait at barriers.
- WDS train dataloaders must keep `shuffle: False`; shard/sample shuffling is handled inside the WDS pipeline.
- There is no env runner or rollout metric for this task. Checkpoint top-k monitors `val_loss` with `mode: min`.
- Normalizer fitting scans lowdim-only WDS samples with streaming stats and caches a metadata-validated payload at `task.dataset.normalizer_cache_path`.
- `normalizer_cache_mode` supports `auto`, `refresh`, and `readonly`. Under torchrun, rank0 generates or refreshes cache while other ranks load it after the barrier.
- The WDS normalizer matches EgoVLA's wrist rot6d ignore policy for `state/action` dimensions `6:18`.
- Do not directly reuse an EgoVLA normalizer file as `normalizer_cache_path`. EgoVLA normalizers are keyed as `states/actions` or `motions`; this Diffusion Policy task requires `image/state/action`. Refit from the same WDS shards with `tests/generate_wds_hand_normalizer.py`, or write an explicit converter that maps keys and verifies matching transform settings.

Validation status:
- `python3 -m py_compile diffusion_policy/dataset/wds_hand_image_dataset.py diffusion_policy/workspace/train_diffusion_unet_hybrid_wds_workspace.py diffusion_policy/workspace/train_diffusion_transformer_hybrid_wds_workspace.py tests/test_wds_hand_image_dataset.py tests/generate_wds_hand_normalizer.py` passed on 2026-05-06.
- The local bare Python used during implementation lacked `pytest`, `omegaconf`, `webdataset`, `cv2`, `PIL`, and `robomimic`, so pytest and Hydra runtime smoke tests still need to be run inside the project conda environment.
