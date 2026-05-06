"""
Torchrun entrypoint for distributed WDS training.

Example:
torchrun --standalone --nproc_per_node=8 train_torchrun.py \
  --config-name=train_diffusion_transformer_hybrid_wds_workspace \
  task.train_wds_datasets.0.shard_urls='/path/to/train/shard-*.tar' \
  task.val_wds_datasets.0.shard_urls='/path/to/val/shard-*.tar'

torchrun --standalone --nproc_per_node=8 train_torchrun.py \
  --config-name=train_diffusion_unet_hybrid_wds_workspace \
  task.train_wds_datasets.0.shard_urls='/path/to/train/shard-*.tar' \
  task.val_wds_datasets.0.shard_urls='/path/to/val/shard-*.tar'
"""

import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace


sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        print("WORLD_SIZE<=1; running without DistributedDataParallel.")
    else:
        print(f"torchrun rank setup: local_rank={local_rank}, world_size={world_size}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
