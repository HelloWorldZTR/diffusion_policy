# AGENTS.md

## WDS Hand/Fingertip Task

This repo now contains an offline WebDataset task for training Diffusion Policy on hand and fingertip motion windows.

Primary files:
- `diffusion_policy/dataset/wds_hand_image_dataset.py`
- `diffusion_policy/workspace/train_diffusion_unet_hybrid_wds_workspace.py`
- `diffusion_policy/config/task/wds_hand_image.yaml`
- `diffusion_policy/config/train_diffusion_unet_hybrid_wds_workspace.yaml`
- `tests/test_wds_hand_image_dataset.py`
- `docs/wds-hand-task.md`

Key behavior:
- The task trains `DiffusionUnetHybridImagePolicy`.
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

Implementation notes:
- `WdsHandImageDataset` is an `IterableDataset`; do not rely on `len(dataloader)`.
- The WDS workspace uses `training.steps_per_epoch` for scheduler length and epoch boundaries.
- WDS train dataloaders must keep `shuffle: False`; shard/sample shuffling is handled inside the WDS pipeline.
- There is no env runner or rollout metric for this task. Checkpoint top-k monitors `val_loss` with `mode: min`.
- Normalizer fitting scans lowdim-only WDS samples and caches stats at `task.dataset.normalizer_cache_path`.

Validation status:
- `python3 -m py_compile diffusion_policy/dataset/wds_hand_image_dataset.py diffusion_policy/workspace/train_diffusion_unet_hybrid_wds_workspace.py tests/test_wds_hand_image_dataset.py` passed on 2026-05-05.
- The local bare Python used during implementation lacked `pytest`, `omegaconf`, `webdataset`, `cv2`, `PIL`, and `robomimic`, so pytest and Hydra runtime smoke tests still need to be run inside the project conda environment.

