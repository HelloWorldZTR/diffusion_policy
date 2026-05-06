#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)

import hydra
import torch
from hydra import compose, initialize_config_dir


def _dataset_cfg(cfg):
    if "task" in cfg and "dataset" in cfg.task:
        return cfg.task.dataset
    if "dataset" in cfg:
        return cfg.dataset
    raise KeyError("Config must contain either task.dataset or dataset.")


def _summarize_payload(cache_path: str) -> None:
    if not cache_path or not os.path.exists(cache_path):
        return
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or "normalizer_state_dict" not in payload:
        print(f"legacy_cache: {cache_path}")
        return
    metadata = payload.get("metadata", {})
    print(f"cache_path: {cache_path}")
    print(f"schema_version: {payload.get('schema_version')}")
    print(f"normalizer_keys: {metadata.get('normalizer_keys')}")
    print(f"windows_scanned: {metadata.get('windows_scanned')}")
    print(f"effective_rows: {metadata.get('effective_rows')}")
    print(f"expanded_shard_count: {metadata.get('expanded_shard_count')}")
    for key, stats in payload.get("stats", {}).items():
        min_preview = stats["min"][:3]
        max_preview = stats["max"][:3]
        print(f"{key}.min[:3]: {min_preview}")
        print(f"{key}.max[:3]: {max_preview}")


def main():
    parser = argparse.ArgumentParser(description="Generate the WDS hand Diffusion Policy normalizer cache.")
    parser.add_argument(
        "--config-name",
        default="train_diffusion_unet_hybrid_wds_workspace",
        help="Hydra config name under diffusion_policy/config.",
    )
    parser.add_argument("--cache-path", default=None, help="Override dataset.normalizer_cache_path.")
    parser.add_argument(
        "--cache-mode",
        default=None,
        choices=["auto", "refresh", "readonly"],
        help="Override dataset.normalizer_cache_mode.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Override dataset.normalizer_max_rows.")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides.")
    args = parser.parse_args()

    config_dir = str(ROOT_DIR / "diffusion_policy" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    dataset = hydra.utils.instantiate(_dataset_cfg(cfg))
    if args.cache_path is not None:
        dataset.normalizer_cache_path = args.cache_path
    if args.cache_mode is not None:
        dataset.normalizer_cache_mode = args.cache_mode
    if args.max_rows is not None:
        dataset.normalizer_max_rows = int(args.max_rows)
        dataset.max_normalizer_samples = int(args.max_rows)

    normalizer = dataset.get_normalizer()
    print(f"normalizer_keys: {list(normalizer.params_dict.keys())}")
    cache_path = dataset.normalizer_cache_path
    if cache_path is not None:
        _summarize_payload(os.path.expanduser(cache_path))
    else:
        print("cache_path: <not configured>")


if __name__ == "__main__":
    main()
