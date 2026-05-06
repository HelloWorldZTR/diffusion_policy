import io
import json
import os
import sys
import tarfile

import numpy as np
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

pytest.importorskip("webdataset")
pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

import torch
from torch.utils.data import DataLoader

from diffusion_policy.dataset.wds_hand_image_dataset import StreamingArrayStats, WdsHandImageDataset


def _encode_npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _encode_jpeg(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    return buffer.getvalue()


def _add_bytes(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _lowdim(frame_idx):
    wrist_state = np.zeros(18, dtype=np.float32)
    wrist_action = np.zeros(18, dtype=np.float32)
    identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    wrist_state[:3] = [frame_idx, 0, 0]
    wrist_state[3:6] = [0, frame_idx, 0]
    wrist_state[6:12] = identity_rot6d
    wrist_state[12:18] = identity_rot6d
    wrist_action[:3] = [frame_idx + 1, 0, 0]
    wrist_action[3:6] = [0, frame_idx + 1, 0]
    wrist_action[6:12] = identity_rot6d
    wrist_action[12:18] = identity_rot6d

    left_tips = np.array([[frame_idx, 0.1 * i, 0.2] for i in range(5)], dtype=np.float32).reshape(-1)
    right_tips = np.array([[0.1 * i, frame_idx, 0.3] for i in range(5)], dtype=np.float32).reshape(-1)
    hand_state = np.concatenate([left_tips, right_tips], axis=0)
    hand_action = hand_state + 0.01
    extrinsic = np.eye(4, dtype=np.float32).reshape(-1)
    intrinsic = np.array([32.0, 32.0, 16.0, 16.0], dtype=np.float32)
    return np.concatenate(
        [wrist_state, hand_state, wrist_action, hand_action, extrinsic, intrinsic],
        axis=0,
    ).astype(np.float32)


def _image(frame_idx, size=32):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = frame_idx * 20
    image[..., 1] = np.arange(size, dtype=np.uint8)[:, None]
    image[..., 2] = np.arange(size, dtype=np.uint8)[None, :]
    return image


def _write_shard(path, n_frames=4, include_instruction=False):
    with tarfile.open(path, "w") as tar:
        for frame_idx in range(n_frames):
            key = f"episode0_{frame_idx:06d}"
            meta = {
                "dataset_name": "synthetic",
                "episode_index": 0,
            }
            if include_instruction:
                meta["instruction"] = ["ignored"]
                meta["instruction_num"] = 1
            _add_bytes(tar, f"{key}.meta.json", json.dumps(meta).encode("utf-8"))
            _add_bytes(tar, f"{key}.lowdim.npy", _encode_npy(_lowdim(frame_idx)))
            _add_bytes(tar, f"{key}.image.jpg", _encode_jpeg(_image(frame_idx)))


def _shape_meta():
    return {
        "obs": {
            "image": {"shape": [3, 32, 32], "type": "rgb"},
            "state": {"shape": [48], "type": "low_dim"},
        },
        "action": {"shape": [48]},
    }


def test_wds_hand_dataset_shapes_and_missing_instruction(tmp_path):
    shard = tmp_path / "train.tar"
    _write_shard(shard, n_frames=4, include_instruction=False)
    dataset = WdsHandImageDataset(
        shape_meta=_shape_meta(),
        train_wds_datasets=[{"shard_urls": str(shard)}],
        val_wds_datasets=[{"shard_urls": str(shard)}],
        horizon=3,
        n_obs_steps=2,
        image_stride=1,
        state_stride=1,
        action_stride=1,
        shuffle_buffer=0,
        max_normalizer_samples=16,
    )

    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch["obs"]["image"].shape == (2, 2, 3, 32, 32)
    assert batch["obs"]["state"].shape == (2, 2, 48)
    assert batch["action"].shape == (2, 3, 48)
    assert batch["obs"]["image"].dtype == torch.float32
    assert torch.all(batch["obs"]["image"] >= 0)
    assert torch.all(batch["obs"]["image"] <= 1)

    normalizer = dataset.get_normalizer()
    assert set(normalizer.params_dict.keys()) == {"image", "state", "action"}
    assert torch.allclose(normalizer["state"].params_dict["scale"][6:18], torch.ones(12))
    assert torch.allclose(normalizer["state"].params_dict["offset"][6:18], torch.zeros(12))
    assert torch.allclose(normalizer["action"].params_dict["scale"][6:18], torch.ones(12))
    assert torch.allclose(normalizer["action"].params_dict["offset"][6:18], torch.zeros(12))


def test_streaming_array_stats_matches_numpy():
    array = np.arange(30, dtype=np.float32).reshape(10, 3)
    stats = StreamingArrayStats(3)
    stats.update(array[:4])
    stats.update(array[4:])
    result = stats.to_stats()

    assert stats.count == 10
    np.testing.assert_allclose(result["min"], array.min(axis=0))
    np.testing.assert_allclose(result["max"], array.max(axis=0))
    np.testing.assert_allclose(result["mean"], array.mean(axis=0))
    np.testing.assert_allclose(result["std"], array.std(axis=0), rtol=1e-6)


def test_wds_hand_normalizer_cache_payload_modes_and_legacy_load(tmp_path):
    shard = tmp_path / "train.tar"
    cache = tmp_path / "normalizer.pt"
    _write_shard(shard, n_frames=4, include_instruction=False)
    dataset = WdsHandImageDataset(
        shape_meta=_shape_meta(),
        train_wds_datasets=[{"shard_urls": str(shard)}],
        val_wds_datasets=[{"shard_urls": str(shard)}],
        horizon=3,
        n_obs_steps=2,
        image_stride=1,
        state_stride=1,
        action_stride=1,
        shuffle_buffer=0,
        normalizer_cache_path=str(cache),
        normalizer_cache_mode="auto",
        normalizer_max_rows=16,
    )

    normalizer = dataset.get_normalizer()
    payload = torch.load(cache, map_location="cpu")
    assert set(payload.keys()) == {"schema_version", "normalizer_state_dict", "stats", "metadata"}
    assert payload["metadata"]["normalizer_keys"] == ["image", "state", "action"]
    assert payload["metadata"]["effective_rows"]["state"] > 0
    assert payload["metadata"]["effective_rows"]["action"] > 0

    reloaded = dataset.get_normalizer()
    assert set(reloaded.params_dict.keys()) == {"image", "state", "action"}

    stale_payload = torch.load(cache, map_location="cpu")
    stale_payload["metadata"]["use_relative_action"] = False
    torch.save(stale_payload, cache)
    regenerated = dataset.get_normalizer()
    assert set(regenerated.params_dict.keys()) == {"image", "state", "action"}
    assert torch.load(cache, map_location="cpu")["metadata"]["use_relative_action"] is True

    stale_payload = torch.load(cache, map_location="cpu")
    stale_payload["metadata"]["use_relative_action"] = False
    torch.save(stale_payload, cache)
    dataset.normalizer_cache_mode = "readonly"
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        dataset.get_normalizer()

    legacy_cache = tmp_path / "legacy.pt"
    torch.save(normalizer.state_dict(), legacy_cache)
    legacy_dataset = WdsHandImageDataset(
        shape_meta=_shape_meta(),
        train_wds_datasets=[{"shard_urls": str(shard)}],
        val_wds_datasets=[{"shard_urls": str(shard)}],
        horizon=3,
        n_obs_steps=2,
        image_stride=1,
        state_stride=1,
        action_stride=1,
        shuffle_buffer=0,
        normalizer_cache_path=str(legacy_cache),
        normalizer_cache_mode="readonly",
    )
    with pytest.warns(UserWarning, match="legacy WDS normalizer cache"):
        legacy = legacy_dataset.get_normalizer()
    assert set(legacy.params_dict.keys()) == {"image", "state", "action"}


def test_wds_hand_dataset_fixed_shapes_at_episode_tail_with_truncate(tmp_path):
    shard = tmp_path / "val.tar"
    _write_shard(shard, n_frames=2, include_instruction=True)
    dataset = WdsHandImageDataset(
        shape_meta=_shape_meta(),
        train_wds_datasets=[{"shard_urls": str(shard)}],
        val_wds_datasets=[{"shard_urls": str(shard)}],
        horizon=4,
        n_obs_steps=3,
        image_stride=1,
        state_stride=1,
        action_stride=1,
        history_pad_mode="truncate",
        action_pad_mode="truncate",
        mode="val",
        shuffle_buffer=0,
    )

    samples = list(iter(dataset))
    assert len(samples) == 2
    for sample in samples:
        assert sample["obs"]["image"].shape == (3, 3, 32, 32)
        assert sample["obs"]["state"].shape == (3, 48)
        assert sample["action"].shape == (4, 48)


def test_wds_hand_batch_policy_compute_loss_smoke(tmp_path):
    pytest.importorskip("robomimic")
    pytest.importorskip("diffusers")
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy

    shard = tmp_path / "train.tar"
    _write_shard(shard, n_frames=4, include_instruction=False)
    dataset = WdsHandImageDataset(
        shape_meta=_shape_meta(),
        train_wds_datasets=[{"shard_urls": str(shard)}],
        val_wds_datasets=[{"shard_urls": str(shard)}],
        horizon=3,
        n_obs_steps=2,
        image_stride=1,
        state_stride=1,
        action_stride=1,
        shuffle_buffer=0,
        max_normalizer_samples=16,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))

    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=_shape_meta(),
        noise_scheduler=DDPMScheduler(
            num_train_timesteps=2,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        ),
        horizon=3,
        n_action_steps=2,
        n_obs_steps=2,
        num_inference_steps=2,
        obs_as_global_cond=True,
        crop_shape=None,
        diffusion_step_embed_dim=16,
        down_dims=[32, 64],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
    )
    policy.set_normalizer(dataset.get_normalizer())
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)


def test_wds_transformer_hybrid_config_smoke():
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(ROOT_DIR, "diffusion_policy", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_diffusion_transformer_hybrid_wds_workspace")

    assert (
        cfg._target_
        == "diffusion_policy.workspace.train_diffusion_transformer_hybrid_wds_workspace.TrainDiffusionTransformerHybridWdsWorkspace"
    )
    assert (
        cfg.policy._target_
        == "diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy"
    )
    assert cfg.dataloader.shuffle is False
    assert cfg.checkpoint.topk.monitor_key == "val_loss"
    assert cfg.checkpoint.topk.mode == "min"
    assert int(cfg.training.steps_per_epoch) > 0


def test_wds_hand_batch_transformer_policy_compute_loss_smoke(tmp_path):
    pytest.importorskip("robomimic")
    pytest.importorskip("diffusers")
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy

    shard = tmp_path / "train.tar"
    _write_shard(shard, n_frames=4, include_instruction=False)
    dataset = WdsHandImageDataset(
        shape_meta=_shape_meta(),
        train_wds_datasets=[{"shard_urls": str(shard)}],
        val_wds_datasets=[{"shard_urls": str(shard)}],
        horizon=3,
        n_obs_steps=2,
        image_stride=1,
        state_stride=1,
        action_stride=1,
        shuffle_buffer=0,
        max_normalizer_samples=16,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))

    policy = DiffusionTransformerHybridImagePolicy(
        shape_meta=_shape_meta(),
        noise_scheduler=DDPMScheduler(
            num_train_timesteps=2,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        ),
        horizon=3,
        n_action_steps=2,
        n_obs_steps=2,
        num_inference_steps=2,
        crop_shape=None,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        n_layer=2,
        n_cond_layers=0,
        n_head=2,
        n_emb=64,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
    )
    policy.set_normalizer(dataset.get_normalizer())
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
