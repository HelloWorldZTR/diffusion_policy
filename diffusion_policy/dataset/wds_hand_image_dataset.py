"""adapted version of EgoVLA's WebDataset dataloader and normalizer"""
from __future__ import annotations

import copy
import datetime
import os
import pathlib
import tempfile
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds

from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - OmegaConf is present in normal Hydra runs.
    OmegaConf = None


ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
EGOVLA_DIR = ROOT_DIR / "external" / "EgoVLA"
if EGOVLA_DIR.is_dir():
    import sys

    egovla_path = str(EGOVLA_DIR)
    if egovla_path not in sys.path:
        sys.path.insert(0, egovla_path)

try:
    from src.dataset.wds_dataset import (  # type: ignore
        WindowConfig,
        build_select_files,
        decode_sample_fields,
        expand_shard_patterns,
        materialize_sample_media,
        no_split,
        resolve_shuffle_initial,
        sliding_window_compose,
    )
except ImportError as e:  # pragma: no cover - exercised only in missing optional deps envs
    raise ImportError(
        "WdsHandImageDataset requires external/EgoVLA plus optional dependencies "
        "`webdataset` and `opencv`. Install the WDS environment before using this task."
    ) from e


NORMALIZER_CACHE_SCHEMA_VERSION = 1
WDS_HAND_TRANSFORM_VERSION = "egovla_wds_hand_v1"
EGOVLA_WRIST_ROT6D_SLICE = slice(6, 18)
NORMALIZER_METADATA_COMPARE_KEYS = (
    "schema_version",
    "shape_meta",
    "horizon",
    "n_obs_steps",
    "image_stride",
    "state_stride",
    "action_stride",
    "history_pad_mode",
    "action_pad_mode",
    "use_relative_action",
    "transform_version",
    "match_egovla_rot6d",
    "rot6d_ignore_slice",
    "normalizer_datasets",
    "expanded_shard_urls",
    "expanded_shard_count",
    "normalizer_keys",
)


class StreamingArrayStats:
    """Numerically stable enough per-dimension stats without retaining rows."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.min = np.full((self.dim,), np.inf, dtype=np.float64)
        self.max = np.full((self.dim,), -np.inf, dtype=np.float64)
        self.sum = np.zeros((self.dim,), dtype=np.float64)
        self.sum_sq = np.zeros((self.dim,), dtype=np.float64)

    def update(self, array: np.ndarray, max_new_rows: Optional[int] = None) -> int:
        array = np.asarray(array, dtype=np.float32).reshape(-1, self.dim)
        if max_new_rows is not None:
            array = array[: max(0, int(max_new_rows))]
        if array.shape[0] == 0:
            return 0
        if not np.all(np.isfinite(array)):
            raise ValueError("Normalizer scan encountered non-finite lowdim values.")
        self.min = np.minimum(self.min, array.min(axis=0))
        self.max = np.maximum(self.max, array.max(axis=0))
        array64 = array.astype(np.float64)
        self.sum += array64.sum(axis=0)
        self.sum_sq += np.square(array64).sum(axis=0)
        self.count += int(array.shape[0])
        return int(array.shape[0])

    def to_stats(self) -> Dict[str, np.ndarray]:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize empty streaming stats.")
        mean = self.sum / self.count
        var = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        return {
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
        }


def _normalizer_with_ignored_dim(stat: Dict[str, np.ndarray], ignore_dim: Optional[slice] = None):
    normalizer = get_range_normalizer_from_stat(stat)
    if ignore_dim is None:
        return normalizer
    params = normalizer.params_dict
    with torch.no_grad():
        params["scale"][ignore_dim] = 1.0
        params["offset"][ignore_dim] = 0.0
    ignored_dim_mask = torch.zeros_like(params["scale"])
    ignored_dim_mask[ignore_dim] = 1.0
    params["ignored_dim_mask"] = torch.nn.Parameter(ignored_dim_mask, requires_grad=False)
    return normalizer


def _jsonable(value: Any) -> Any:
    if OmegaConf is not None and OmegaConf.is_config(value):
        return _jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _stats_to_payload(stats: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, List[float]]]:
    return {
        key: {stat_name: stat_value.astype(np.float32).tolist() for stat_name, stat_value in value.items()}
        for key, value in stats.items()
    }


def _torch_load_cpu(path: str):
    return torch.load(path, map_location="cpu")


def _rot_matrix_from_6drot(rot: np.ndarray) -> np.ndarray:
    original_shape = rot.shape
    tensor = torch.from_numpy(rot.reshape(-1, 6))
    a = F.normalize(tensor[..., :3], dim=-1)
    b = tensor[..., 3:]
    b = b - torch.sum(a * b, dim=-1, keepdim=True) * a
    b = F.normalize(b, dim=-1)
    c = torch.cross(a, b, dim=-1)
    matrix = torch.stack([a, b, c], dim=-1)
    if len(original_shape) > 1:
        matrix = matrix.reshape(*original_shape[:-1], 3, 3)
    else:
        matrix = matrix.squeeze(0)
    return matrix.numpy()


def _rot_matrix_to_6drot(matrix: np.ndarray) -> np.ndarray:
    original_shape = matrix.shape
    matrix = matrix.reshape(-1, 3, 3)
    rot = np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)
    if len(original_shape) > 2:
        rot = rot.reshape(*original_shape[:-2], 6)
    else:
        rot = rot.squeeze(0)
    return rot.astype(np.float32)


def _homo_matrix_from_trans_6drot(trans: np.ndarray, rot_6d: np.ndarray) -> np.ndarray:
    rot_matrix = _rot_matrix_from_6drot(rot_6d.astype(np.float32))
    matrix = np.zeros(trans.shape[:-1] + (4, 4), dtype=np.float32)
    matrix[..., :3, :3] = rot_matrix
    matrix[..., :3, 3] = trans
    matrix[..., 3, 3] = 1.0
    return matrix


def _homo_matrix_to_trans_6drot(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix[..., :3, 3].astype(np.float32), _rot_matrix_to_6drot(matrix[..., :3, :3])


def _homo_matrix_from_wrist_pose(wrist_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if wrist_pose.shape[-1] != 18:
        raise ValueError(f"wrist_pose last dim must be 18, got {wrist_pose.shape}")
    left = _homo_matrix_from_trans_6drot(wrist_pose[..., :3], wrist_pose[..., 6:12])
    right = _homo_matrix_from_trans_6drot(wrist_pose[..., 3:6], wrist_pose[..., 12:18])
    return left, right


def _transform_wrist_to_target_frame(wrist_pose: np.ndarray, target_extrinsic: np.ndarray) -> np.ndarray:
    left, right = _homo_matrix_from_wrist_pose(wrist_pose)
    if wrist_pose.ndim > target_extrinsic.ndim - 1:
        target_extrinsic = np.expand_dims(target_extrinsic, axis=-3)
    left = np.matmul(target_extrinsic, left)
    right = np.matmul(target_extrinsic, right)
    left_trans, left_rot = _homo_matrix_to_trans_6drot(left)
    right_trans, right_rot = _homo_matrix_to_trans_6drot(right)
    return np.concatenate([left_trans, right_trans, left_rot, right_rot], axis=-1).astype(np.float32)


def _transform_points_to_target_frame(points: np.ndarray, target_extrinsic: np.ndarray) -> np.ndarray:
    point_shape = points.shape
    points = points.reshape(*point_shape[:-1], point_shape[-1] // 3, 3)
    ones = np.ones(points.shape[:-1] + (1,), dtype=np.float32)
    homo = np.concatenate([points, ones], axis=-1)[..., None]
    if len(point_shape) + 1 > target_extrinsic.ndim:
        target_extrinsic = np.expand_dims(target_extrinsic, axis=-3)
    target_extrinsic = np.expand_dims(target_extrinsic, axis=-3)
    result = np.matmul(target_extrinsic, homo).squeeze(-1)
    result = result[..., :3] / (result[..., 3:4] + 1e-6)
    return result.reshape(point_shape).astype(np.float32)


def _transform_hand_points_to_wrist_frame(hand_points: np.ndarray, wrist_pose: np.ndarray) -> np.ndarray:
    dim = hand_points.shape[-1]
    left_points = hand_points[..., : dim // 2]
    right_points = hand_points[..., dim // 2 :]
    left_wrist, right_wrist = _homo_matrix_from_wrist_pose(wrist_pose)
    left_world_to_wrist = np.linalg.pinv(left_wrist).astype(np.float32)
    right_world_to_wrist = np.linalg.pinv(right_wrist).astype(np.float32)
    left = _transform_points_to_target_frame(left_points, left_world_to_wrist)
    right = _transform_points_to_target_frame(right_points, right_world_to_wrist)
    return np.concatenate([left, right], axis=-1).astype(np.float32)


def _get_relative_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    action = action.copy()
    for hand_idx in range(2):
        trans_slice = slice(hand_idx * 3, hand_idx * 3 + 3)
        rot_slice = slice(6 + hand_idx * 6, 6 + hand_idx * 6 + 6)
        action_mat = _homo_matrix_from_trans_6drot(action[..., trans_slice], action[..., rot_slice])
        state_mat = _homo_matrix_from_trans_6drot(state[trans_slice], state[rot_slice])
        rel_mat = np.matmul(np.linalg.pinv(state_mat).astype(np.float32), action_mat)
        trans, rot = _homo_matrix_to_trans_6drot(rel_mat)
        action[..., trans_slice] = trans
        action[..., rot_slice] = rot
    action[..., 18:] = action[..., 18:] - state[18:]
    return action.astype(np.float32)


def process_wds_state_action(
    wrist_state: np.ndarray,
    hand_state: np.ndarray,
    wrist_action: np.ndarray,
    hand_action: np.ndarray,
    extrinsic: np.ndarray,
    use_relative_action: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Match EgoVLA's fingertips representation used by the new DP task."""
    extrinsic = extrinsic.reshape(4, 4).astype(np.float32)
    wrist_state = wrist_state.astype(np.float32)
    hand_state = hand_state.astype(np.float32)
    wrist_action = wrist_action.astype(np.float32)
    hand_action = hand_action.astype(np.float32)

    processed_wrist_state = _transform_wrist_to_target_frame(wrist_state, extrinsic)
    processed_wrist_action = _transform_wrist_to_target_frame(wrist_action, extrinsic)
    processed_hand_state = _transform_hand_points_to_wrist_frame(hand_state, wrist_state)
    processed_hand_action = _transform_hand_points_to_wrist_frame(hand_action, wrist_action)

    state = np.concatenate([processed_wrist_state, processed_hand_state], axis=-1).astype(np.float32)
    action = np.concatenate([processed_wrist_action, processed_hand_action], axis=-1).astype(np.float32)
    if use_relative_action:
        action = _get_relative_action(state[-1], action)
    return state, action


DatasetSpec = Union[Sequence[Dict], Sequence[str], str]


def _as_dataset_list(datasets: DatasetSpec) -> List[Dict]:
    if isinstance(datasets, (str, bytes, os.PathLike)):
        return [{"shard_urls": os.fspath(datasets), "name": "wds"}]
    result = []
    for item in datasets:
        if isinstance(item, dict) or hasattr(item, "items"):
            result.append(dict(item))
        else:
            result.append({"shard_urls": os.fspath(item), "name": "wds"})
    return result


def _expand_dataset_shards(datasets: DatasetSpec) -> List[str]:
    urls: List[str] = []
    for dataset_cfg in _as_dataset_list(datasets):
        shard_urls, _ = expand_shard_patterns(dataset_cfg["shard_urls"])
        urls.extend(shard_urls)
    if not urls:
        raise ValueError(f"No WebDataset shards found for {datasets!r}")
    return urls


def _ensure_instruction_meta(sample: Dict) -> Dict:
    meta = sample.get("meta.json")
    if meta is None:
        meta = {}
    meta.setdefault("instruction", ["wds hand task"])
    meta.setdefault("instruction_num", 1)
    meta.setdefault("dataset_name", "wds_hand")
    meta.setdefault("episode_index", 0)
    meta.setdefault("presence", 3)
    sample["meta.json"] = meta
    return sample


def _pad_first_axis(array: np.ndarray, length: int, mode: str = "edge") -> np.ndarray:
    if array.shape[0] == length:
        return array
    if array.shape[0] > length:
        return array[:length]
    if array.shape[0] == 0:
        raise ValueError("Cannot pad an empty array")
    if mode == "zero":
        pad = np.zeros((length - array.shape[0],) + array.shape[1:], dtype=array.dtype)
    else:
        pad = np.repeat(array[-1:], length - array.shape[0], axis=0)
    return np.concatenate([array, pad], axis=0)


class WdsHandImageDataset(torch.utils.data.IterableDataset):
    """Stream hand/fingertip WDS windows as Diffusion Policy image batches."""

    def __init__(
        self,
        shape_meta: Dict,
        train_wds_datasets: DatasetSpec,
        val_wds_datasets: Optional[DatasetSpec] = None,
        horizon: int = 16,
        n_obs_steps: int = 2,
        image_stride: int = 30,
        state_stride: int = 30,
        action_stride: int = 1,
        history_pad_mode: str = "repeat",
        action_pad_mode: str = "repeat",
        mode: str = "train",
        shuffle_buffer: int = 16384,
        shuffle_initial: Optional[int] = 4096,
        keep_ratio: float = 1.0,
        use_relative_action: bool = True,
        normalizer_cache_path: Optional[str] = None,
        normalizer_wds_datasets: Optional[DatasetSpec] = None,
        max_normalizer_samples: int = 100000,
        normalizer_max_rows: Optional[int] = None,
        normalizer_cache_mode: str = "auto",
        normalizer_match_egovla_rot6d: bool = True,
    ):
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be train or val, got {mode!r}")
        if normalizer_cache_mode not in {"auto", "refresh", "readonly"}:
            raise ValueError(
                "normalizer_cache_mode must be one of auto, refresh, readonly; "
                f"got {normalizer_cache_mode!r}"
            )
        self.shape_meta = shape_meta
        self.train_wds_datasets = _as_dataset_list(train_wds_datasets)
        self.val_wds_datasets = _as_dataset_list(val_wds_datasets) if val_wds_datasets is not None else None
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.image_stride = int(image_stride)
        self.state_stride = int(state_stride)
        self.action_stride = int(action_stride)
        self.history_pad_mode = history_pad_mode
        self.action_pad_mode = action_pad_mode
        self.mode = mode
        self.shuffle_buffer = int(shuffle_buffer)
        self.shuffle_initial = shuffle_initial
        self.keep_ratio = float(keep_ratio)
        self.use_relative_action = bool(use_relative_action)
        self.normalizer_cache_path = normalizer_cache_path
        self.normalizer_wds_datasets = (
            _as_dataset_list(normalizer_wds_datasets)
            if normalizer_wds_datasets is not None
            else self.train_wds_datasets
        )
        if normalizer_max_rows is None:
            normalizer_max_rows = max_normalizer_samples
        self.max_normalizer_samples = int(normalizer_max_rows)
        self.normalizer_max_rows = int(normalizer_max_rows)
        if self.normalizer_max_rows <= 0:
            raise ValueError("normalizer_max_rows must be a positive integer.")
        self.normalizer_cache_mode = normalizer_cache_mode
        self.normalizer_match_egovla_rot6d = bool(normalizer_match_egovla_rot6d)

        self.rgb_keys = []
        self.image_hw_by_key = {}
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type != "rgb":
                continue
            image_shape = tuple(attr["shape"])
            if len(image_shape) != 3 or image_shape[0] != 3:
                raise ValueError(f"shape_meta.obs.{key}.shape must be [3,H,W], got {image_shape}")
            self.rgb_keys.append(str(key))
            self.image_hw_by_key[str(key)] = (int(image_shape[1]), int(image_shape[2]))
        required_rgb_keys = {"image", "breast_image"}
        missing_rgb_keys = required_rgb_keys - set(self.rgb_keys)
        if missing_rgb_keys:
            raise ValueError(
                "WdsHandImageDataset dual-image mode requires rgb obs keys "
                f"{sorted(required_rgb_keys)}, missing {sorted(missing_rgb_keys)}"
            )
        action_shape = tuple(shape_meta["action"]["shape"])
        if action_shape != (48,):
            raise ValueError(f"shape_meta.action.shape must be [48], got {action_shape}")
        self.normalizer_keys = [*self.rgb_keys, "state", "action"]

        self.window_config = WindowConfig(
            action_horizon=self.horizon,
            action_stride=self.action_stride,
            state_horizon=self.n_obs_steps,
            state_stride=self.state_stride,
            image_horizon=self.n_obs_steps,
            image_stride=self.image_stride,
            history_pad_mode=self.history_pad_mode,
            action_pad_mode=self.action_pad_mode,
        )

    def get_validation_dataset(self):
        if self.val_wds_datasets is None:
            warnings.warn("val_wds_datasets is not set; validation dataset will be empty.")
            return WdsHandImageEmptyDataset(self.shape_meta)
        val_set = copy.copy(self)
        val_set.mode = "val"
        return val_set

    def _active_datasets(self):
        if self.mode == "val":
            return self.val_wds_datasets if self.val_wds_datasets is not None else []
        return self.train_wds_datasets

    def _build_pipeline(self, load_image: bool, preprocess_fn=None, finite: Optional[bool] = None):
        datasets = self._active_datasets()
        if not datasets:
            return iter(())
        shard_urls = _expand_dataset_shards(datasets)
        is_train = self.mode == "train" if finite is None else not finite
        pipeline = wds.WebDataset(
            shard_urls,
            shardshuffle=False,
            nodesplitter=no_split if is_train else wds.split_by_node,
            workersplitter=no_split if is_train else wds.shardlists.split_by_worker,
            resampled=is_train,
            empty_check=False,
            select_files=build_select_files(load_image=load_image, load_depth=False, load_breast=load_image),
        )
        stages = [
            wds.map(decode_sample_fields),
            wds.map(_ensure_instruction_meta),
            lambda src: sliding_window_compose(src, self.window_config),
        ]
        if is_train and self.keep_ratio < 1.0:
            stages.append(wds.select(lambda _s: np.random.random() < self.keep_ratio))
        if load_image:
            if is_train and self.shuffle_buffer > 0:
                stages.append(
                    wds.shuffle(
                        self.shuffle_buffer,
                        initial=resolve_shuffle_initial(self.shuffle_buffer, self.shuffle_initial),
                    )
                )
            stages.append(wds.map(materialize_sample_media))
        if preprocess_fn is not None:
            stages.append(wds.map(preprocess_fn))
        for stage in stages:
            pipeline.append(stage)
        return pipeline

    def _resize_image_sequence(self, images: np.ndarray, key: str) -> np.ndarray:
        target_h, target_w = self.image_hw_by_key[key]
        if images.shape[1] == target_h and images.shape[2] == target_w:
            return images
        import cv2

        return np.stack(
            [cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA) for frame in images],
            axis=0,
        )

    def _prepare_image_obs(self, sample: Dict, key: str) -> np.ndarray:
        if key not in sample:
            raise KeyError(
                f"Dual-image WDS hand task requires sample[{key!r}]. "
                "Check that shards contain breast_image.jpg and that load_breast=True is used."
            )
        image = sample[key].astype(np.uint8)
        image = _pad_first_axis(image, self.n_obs_steps)
        image = self._resize_image_sequence(image, key)
        image = np.moveaxis(image, -1, 1).astype(np.float32) / 255.0 # normalized to [0,1]
        return image.astype(np.float32)

    def _sample_to_dp(self, sample: Dict) -> Dict[str, torch.Tensor]:
        state, action = process_wds_state_action(
            wrist_state=sample["wrist_state"],
            hand_state=sample["hand_state"],
            wrist_action=sample["wrist_action"],
            hand_action=sample["hand_action"],
            extrinsic=sample["extrinsic"],
            use_relative_action=self.use_relative_action,
        )
        # pad to fixed length
        state = _pad_first_axis(state, self.n_obs_steps)
        action = _pad_first_axis(action, self.horizon)

        # normalize and resize observations
        obs = {key: self._prepare_image_obs(sample, key) for key in self.rgb_keys}
        obs["state"] = state.astype(np.float32)
        data = {
            "obs": obs,
            "action": action.astype(np.float32),
        }
        return dict_apply(data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)

    def __iter__(self):
        return iter(self._build_pipeline(load_image=True, preprocess_fn=self._sample_to_dp))

    def get_all_actions(self) -> torch.Tensor:
        actions = []
        count = 0
        for sample in self._iter_lowdim_samples():
            _, action = self._lowdim_sample_to_arrays(sample)
            actions.append(action)
            count += action.shape[0]
            if count >= self.normalizer_max_rows:
                break
        if not actions:
            return torch.empty((0, 48), dtype=torch.float32)
        return torch.from_numpy(np.concatenate(actions, axis=0).astype(np.float32))

    def _lowdim_sample_to_arrays(self, sample: Dict) -> tuple[np.ndarray, np.ndarray]:
        return process_wds_state_action(
            wrist_state=sample["wrist_state"],
            hand_state=sample["hand_state"],
            wrist_action=sample["wrist_action"],
            hand_action=sample["hand_action"],
            extrinsic=sample["extrinsic"],
            use_relative_action=self.use_relative_action,
        )

    def _iter_lowdim_samples(self) -> Iterable[Dict]:
        scan_set = copy.copy(self)
        scan_set.mode = "val"
        scan_set.val_wds_datasets = self.normalizer_wds_datasets
        return scan_set._build_pipeline(load_image=False, finite=True)

    def _normalizer_cache_path(self) -> Optional[str]:
        if self.normalizer_cache_path is None:
            return None
        return os.path.expanduser(self.normalizer_cache_path)

    def _expanded_normalizer_shards(self) -> List[str]:
        return _expand_dataset_shards(self.normalizer_wds_datasets)

    def _normalizer_expected_metadata(self, expanded_shards: Optional[List[str]] = None) -> Dict[str, Any]:
        if expanded_shards is None:
            expanded_shards = self._expanded_normalizer_shards()
        return {
            "schema_version": NORMALIZER_CACHE_SCHEMA_VERSION,
            "shape_meta": _jsonable(self.shape_meta),
            "horizon": self.horizon,
            "n_obs_steps": self.n_obs_steps,
            "image_stride": self.image_stride,
            "state_stride": self.state_stride,
            "action_stride": self.action_stride,
            "history_pad_mode": self.history_pad_mode,
            "action_pad_mode": self.action_pad_mode,
            "use_relative_action": self.use_relative_action,
            "transform_version": WDS_HAND_TRANSFORM_VERSION,
            "match_egovla_rot6d": self.normalizer_match_egovla_rot6d,
            "rot6d_ignore_slice": [EGOVLA_WRIST_ROT6D_SLICE.start, EGOVLA_WRIST_ROT6D_SLICE.stop],
            "normalizer_datasets": _jsonable(self.normalizer_wds_datasets),
            "expanded_shard_urls": [os.fspath(x) for x in expanded_shards],
            "expanded_shard_count": len(expanded_shards),
            "normalizer_max_rows": self.normalizer_max_rows,
            "normalizer_keys": list(self.normalizer_keys),
        }

    def _metadata_mismatch_reason(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> Optional[str]:
        for key in NORMALIZER_METADATA_COMPARE_KEYS:
            if actual.get(key) != expected.get(key):
                return f"{key}: cache={actual.get(key)!r} current={expected.get(key)!r}"
        return None

    def _load_normalizer_cache(
        self,
        cache_path: str,
        expected_metadata: Dict[str, Any],
    ) -> Optional[LinearNormalizer]:
        if not os.path.exists(cache_path):
            if self.normalizer_cache_mode == "readonly":
                raise FileNotFoundError(f"Normalizer cache does not exist: {cache_path}")
            return None

        normalizer = LinearNormalizer()
        payload = _torch_load_cpu(cache_path)
        if isinstance(payload, dict) and "normalizer_state_dict" in payload:
            if payload.get("schema_version") != NORMALIZER_CACHE_SCHEMA_VERSION:
                reason = (
                    f"schema_version: cache={payload.get('schema_version')!r} "
                    f"current={NORMALIZER_CACHE_SCHEMA_VERSION!r}"
                )
                if self.normalizer_cache_mode == "readonly":
                    raise RuntimeError(f"Normalizer cache metadata mismatch ({reason}).")
                warnings.warn(f"Regenerating WDS normalizer cache because metadata mismatched ({reason}).")
                return None
            actual_metadata = payload.get("metadata", {})
            reason = self._metadata_mismatch_reason(actual_metadata, expected_metadata)
            if reason is not None:
                if self.normalizer_cache_mode == "readonly":
                    raise RuntimeError(f"Normalizer cache metadata mismatch ({reason}).")
                warnings.warn(f"Regenerating WDS normalizer cache because metadata mismatched ({reason}).")
                return None
            normalizer.load_state_dict(payload["normalizer_state_dict"])
            return normalizer

        # Backward compatibility for the first WDS adapter cache format, which
        # saved only LinearNormalizer.state_dict() and has no metadata.
        normalizer.load_state_dict(payload)
        expected_keys = set(expected_metadata["normalizer_keys"])
        actual_keys = set(normalizer.params_dict.keys())
        if actual_keys != expected_keys:
            reason = f"normalizer_keys: cache={sorted(actual_keys)!r} current={sorted(expected_keys)!r}"
            if self.normalizer_cache_mode == "readonly":
                raise RuntimeError(f"Legacy WDS normalizer cache metadata mismatch ({reason}).")
            warnings.warn(f"Regenerating legacy WDS normalizer cache because metadata mismatched ({reason}).")
            return None
        warnings.warn(
            "Loaded legacy WDS normalizer cache without metadata validation. "
            "Run with normalizer_cache_mode=refresh to rewrite it in the new format."
        )
        return normalizer

    def _save_normalizer_cache(
        self,
        cache_path: str,
        normalizer: LinearNormalizer,
        stats: Dict[str, Dict[str, np.ndarray]],
        metadata: Dict[str, Any],
    ) -> None:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        payload = {
            "schema_version": NORMALIZER_CACHE_SCHEMA_VERSION,
            "normalizer_state_dict": normalizer.state_dict(),
            "stats": _stats_to_payload(stats),
            "metadata": _jsonable(metadata),
        }
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".tmp-wds-hand-normalizer-",
                suffix=".pt",
                dir=cache_dir or ".",
            )
            with os.fdopen(fd, "wb") as f:
                torch.save(payload, f)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _fit_normalizer_from_wds(
        self,
        expected_metadata: Dict[str, Any],
    ) -> tuple[LinearNormalizer, Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
        state_stats = StreamingArrayStats(48)
        action_stats = StreamingArrayStats(48)
        window_count = 0
        for sample in self._iter_lowdim_samples():
            state, action = self._lowdim_sample_to_arrays(sample)
            state_stats.update(state, max_new_rows=self.normalizer_max_rows - state_stats.count)
            action_stats.update(action, max_new_rows=self.normalizer_max_rows - action_stats.count)
            window_count += 1
            if state_stats.count >= self.normalizer_max_rows and action_stats.count >= self.normalizer_max_rows:
                break
        if state_stats.count == 0 or action_stats.count == 0:
            raise RuntimeError("No WDS samples available for normalizer fitting.")

        stats = {
            "state": state_stats.to_stats(),
            "action": action_stats.to_stats(),
        }
        normalizer = LinearNormalizer()
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        ignore_dim = EGOVLA_WRIST_ROT6D_SLICE if self.normalizer_match_egovla_rot6d else None
        normalizer["state"] = _normalizer_with_ignored_dim(stats["state"], ignore_dim=ignore_dim)
        normalizer["action"] = _normalizer_with_ignored_dim(stats["action"], ignore_dim=ignore_dim)
        metadata = dict(expected_metadata)
        metadata.update(
            {
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "windows_scanned": window_count,
                "effective_rows": {
                    "state": state_stats.count,
                    "action": action_stats.count,
                },
                "normalizer_keys": list(self.normalizer_keys),
            }
        )
        return normalizer, stats, metadata

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        cache_path = self._normalizer_cache_path()
        if cache_path is None and self.normalizer_cache_mode == "readonly":
            raise ValueError("normalizer_cache_mode=readonly requires normalizer_cache_path.")

        expanded_shards = self._expanded_normalizer_shards()
        expected_metadata = self._normalizer_expected_metadata(expanded_shards)
        if self.normalizer_cache_mode != "refresh" and cache_path is not None:
            cached = self._load_normalizer_cache(cache_path, expected_metadata)
            if cached is not None:
                return cached

        normalizer, stats, metadata = self._fit_normalizer_from_wds(expected_metadata)
        if cache_path is not None:
            self._save_normalizer_cache(cache_path, normalizer, stats, metadata)
        return normalizer


class WdsHandImageEmptyDataset(torch.utils.data.IterableDataset):
    def __init__(self, shape_meta: Dict):
        super().__init__()
        self.shape_meta = shape_meta

    def __iter__(self):
        return iter(())
