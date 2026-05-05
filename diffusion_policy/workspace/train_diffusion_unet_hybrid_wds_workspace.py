if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetHybridWdsWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0

    @staticmethod
    def _next_batch(iterator, dataloader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(dataloader)
            try:
                return next(iterator), iterator
            except StopIteration as e:
                raise RuntimeError("Dataloader produced no batches.") from e

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        steps_per_epoch = int(cfg.training.steps_per_epoch)
        if steps_per_epoch <= 0:
            raise ValueError("training.steps_per_epoch must be a positive integer for WDS training.")

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(steps_per_epoch * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.steps_per_epoch = min(steps_per_epoch, 3)
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            steps_per_epoch = int(cfg.training.steps_per_epoch)

        train_iter = iter(train_dataloader)
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = {}
                train_losses = []

                with tqdm.tqdm(
                    range(steps_per_epoch),
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx in tepoch:
                        batch, train_iter = self._next_batch(train_iter, train_dataloader)
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == steps_per_epoch - 1
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                step_log["train_loss"] = float(np.mean(train_losses))

                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()

                if (self.epoch % cfg.training.val_every) == 0:
                    val_losses = []
                    with torch.no_grad():
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                val_losses.append(self.model.compute_loss(batch))
                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx >= cfg.training.max_val_steps - 1
                                ):
                                    break
                    if val_losses:
                        step_log["val_loss"] = torch.mean(torch.stack(val_losses)).item()

                if (self.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        result = policy.predict_action(batch["obs"])
                        mse = torch.nn.functional.mse_loss(result["action_pred"], batch["action"])
                        step_log["train_action_mse_error"] = mse.item()

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                    if cfg.checkpoint.topk.monitor_key in metric_dict:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                policy.train()
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWdsWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
