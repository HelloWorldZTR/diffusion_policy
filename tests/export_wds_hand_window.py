#!/usr/bin/env python3
"""Export one real WDS hand/fingertip dataset window for inspection.

Example:
    python tests/export_wds_hand_window.py \
      --shards '/path/to/wds/val/shard-*.tar' \
      --window-index 0

Outputs are written under outputs/wds_hand_windows/<timestamp>/ by default:
    - contact_sheet.png: all observation frames in one image
    - obs_image_*.png: decoded observation image frames
    - window.npz: obs image/state/action arrays, unless --images-only is set
    - summary.json: shapes, dtypes, min/max, source args
"""

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime

import numpy as np
from torch.utils.data import DataLoader


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)

from diffusion_policy.dataset.wds_hand_image_dataset import WdsHandImageDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        default="data/wds/val/shard-*.tar",
        help="WDS shard glob/path. Use quotes around globs so Python expands them.",
    )
    parser.add_argument("--output-dir", default="outputs/wds_hand_windows")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only export PNG images and summary.json; skip window.npz.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write directly into --output-dir instead of creating a timestamped subdirectory.",
    )
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--image-stride", type=int, default=30)
    parser.add_argument("--state-stride", type=int, default=30)
    parser.add_argument("--action-stride", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--history-pad-mode", choices=["repeat", "truncate"], default="repeat")
    parser.add_argument("--action-pad-mode", choices=["repeat", "truncate"], default="repeat")
    parser.add_argument(
        "--absolute-action",
        action="store_true",
        help="Disable relative action conversion and export absolute transformed action.",
    )
    return parser.parse_args()


def build_shape_meta(height, width):
    return {
        "obs": {
            "image": {"shape": [3, height, width], "type": "rgb"},
            "state": {"shape": [48], "type": "low_dim"},
        },
        "action": {"shape": [48]},
    }


def to_numpy(batch_value):
    if hasattr(batch_value, "detach"):
        return batch_value.detach().cpu().numpy()
    return np.asarray(batch_value)


def summarize_array(name, array):
    finite = np.isfinite(array)
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(finite.all()),
    }
    if finite.any():
        summary.update(
            {
                "min": float(array[finite].min()),
                "max": float(array[finite].max()),
                "mean": float(array[finite].mean()),
            }
        )
    else:
        summary.update({"min": None, "max": None, "mean": None})
    return name, summary


def save_obs_images(obs_image, output_dir):
    # obs_image: [T, C, H, W], float in [0, 1]
    import cv2

    paths = []
    frames = np.moveaxis(obs_image, 1, -1)
    frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    for idx, frame in enumerate(frames):
        path = output_dir / f"obs_image_{idx:02d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(str(path))

    if len(frames) > 0:
        sheet = np.concatenate(frames, axis=1)
        sheet_path = output_dir / "contact_sheet.png"
        cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        paths.insert(0, str(sheet_path))
    return paths


def main():
    args = parse_args()
    if args.window_index < 0:
        raise ValueError("--window-index must be >= 0")

    dataset = WdsHandImageDataset(
        shape_meta=build_shape_meta(args.image_height, args.image_width),
        train_wds_datasets=[{"name": "export", "shard_urls": args.shards}],
        val_wds_datasets=[{"name": "export", "shard_urls": args.shards}],
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        image_stride=args.image_stride,
        state_stride=args.state_stride,
        action_stride=args.action_stride,
        history_pad_mode=args.history_pad_mode,
        action_pad_mode=args.action_pad_mode,
        mode="val",
        shuffle_buffer=0,
        use_relative_action=not args.absolute_action,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    sample = None
    for idx, batch in enumerate(loader):
        if idx == args.window_index:
            sample = batch
            break
    if sample is None:
        raise RuntimeError(f"No window found at index {args.window_index} for shards={args.shards!r}")

    obs_image = to_numpy(sample["obs"]["image"][0])
    obs_state = to_numpy(sample["obs"]["state"][0])
    action = to_numpy(sample["action"][0])

    output_dir = pathlib.Path(args.output_dir)
    if not args.no_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = None
    if not args.images_only:
        npz_path = output_dir / "window.npz"
        np.savez_compressed(
            npz_path,
            obs_image=obs_image,
            obs_state=obs_state,
            action=action,
        )
    image_paths = save_obs_images(obs_image, output_dir)

    summary = {
        "args": vars(args),
        "window_index": args.window_index,
        "npz_path": str(npz_path) if npz_path is not None else None,
        "image_paths": image_paths,
        "arrays": dict(
            [
                summarize_array("obs_image", obs_image),
                summarize_array("obs_state", obs_state),
                summarize_array("action", action),
            ]
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Exported WDS hand window to {output_dir}")
    print(f"Images: {output_dir}")
    if npz_path is not None:
        print(f"NPZ: {npz_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
