from __future__ import annotations

import copy
import io
import os
import pathlib
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer


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
    ):
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be train or val, got {mode!r}")
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
        self.max_normalizer_samples = int(max_normalizer_samples)

        image_shape = tuple(shape_meta["obs"]["image"]["shape"])
        if len(image_shape) != 3 or image_shape[0] != 3:
            raise ValueError(f"shape_meta.obs.image.shape must be [3,H,W], got {image_shape}")
        self.image_hw = (int(image_shape[1]), int(image_shape[2]))
        action_shape = tuple(shape_meta["action"]["shape"])
        if action_shape != (48,):
            raise ValueError(f"shape_meta.action.shape must be [48], got {action_shape}")

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
        stages = [
            wds.WebDataset(
                shard_urls,
                shardshuffle=False,
                nodesplitter=no_split if is_train else wds.split_by_node,
                workersplitter=no_split if is_train else wds.shardlists.split_by_worker,
                resampled=is_train,
                empty_check=False,
                select_files=build_select_files(load_image=load_image, load_depth=False, load_breast=False),
            ),
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
        return wds.DataPipeline(*stages)

    def _resize_image_sequence(self, images: np.ndarray) -> np.ndarray:
        target_h, target_w = self.image_hw
        if images.shape[1] == target_h and images.shape[2] == target_w:
            return images
        import cv2

        return np.stack(
            [cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA) for frame in images],
            axis=0,
        )

    def _sample_to_dp(self, sample: Dict) -> Dict[str, torch.Tensor]:
        state, action = process_wds_state_action(
            wrist_state=sample["wrist_state"],
            hand_state=sample["hand_state"],
            wrist_action=sample["wrist_action"],
            hand_action=sample["hand_action"],
            extrinsic=sample["extrinsic"],
            use_relative_action=self.use_relative_action,
        )
        state = _pad_first_axis(state, self.n_obs_steps)
        action = _pad_first_axis(action, self.horizon)

        image = sample["image"].astype(np.uint8)
        image = _pad_first_axis(image, self.n_obs_steps)
        image = self._resize_image_sequence(image)
        image = np.moveaxis(image, -1, 1).astype(np.float32) / 255.0

        data = {
            "obs": {
                "image": image.astype(np.float32),
                "state": state.astype(np.float32),
            },
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
            if count >= self.max_normalizer_samples:
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

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        if self.normalizer_cache_path and os.path.exists(self.normalizer_cache_path):
            normalizer.load_state_dict(torch.load(self.normalizer_cache_path, map_location="cpu"))
            return normalizer

        states = []
        actions = []
        count = 0
        for sample in self._iter_lowdim_samples():
            state, action = self._lowdim_sample_to_arrays(sample)
            states.append(state)
            actions.append(action)
            count += state.shape[0] + action.shape[0]
            if count >= self.max_normalizer_samples:
                break
        if not states or not actions:
            raise RuntimeError("No WDS samples available for normalizer fitting.")

        state_arr = np.concatenate(states, axis=0).astype(np.float32)
        action_arr = np.concatenate(actions, axis=0).astype(np.float32)
        normalizer["image"] = get_image_range_normalizer()
        normalizer["state"] = get_range_normalizer_from_stat(array_to_stats(state_arr))
        normalizer["action"] = get_range_normalizer_from_stat(array_to_stats(action_arr))

        if self.normalizer_cache_path:
            cache_path = os.path.expanduser(self.normalizer_cache_path)
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            buffer = io.BytesIO()
            torch.save(normalizer.state_dict(), buffer)
            with open(cache_path, "wb") as f:
                f.write(buffer.getvalue())
        return normalizer


class WdsHandImageEmptyDataset(torch.utils.data.IterableDataset):
    def __init__(self, shape_meta: Dict):
        super().__init__()
        self.shape_meta = shape_meta

    def __iter__(self):
        return iter(())
