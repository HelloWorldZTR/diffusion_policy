if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import pathlib
import random

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_unet_hybrid_wds_workspace import (
    TrainDiffusionUnetHybridWdsWorkspace,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionTransformerHybridWdsWorkspace(TrainDiffusionUnetHybridWdsWorkspace):
    """
    Streaming WDS workspace for DiffusionTransformerHybridImagePolicy.
    All the streaming, ckpt, DDP details is the same as DiffusionUnetHybridWdsWorkspace
    """

    def __init__(self, cfg: OmegaConf, output_dir=None):
        BaseWorkspace.__init__(self, cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0
        self.ddp_model = None
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_rank0 = True


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWdsWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
