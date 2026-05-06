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
import torch.distributed as dist
import torch.nn as nn
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


class PolicyLossWrapper(nn.Module):
    def __init__(self, policy: DiffusionUnetHybridImagePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)


class TrainDiffusionUnetHybridWdsWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    exclude_keys = ["ddp_model"]

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
        self.ddp_model = None
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_rank0 = True

    def _setup_distributed(self, cfg):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        self.is_rank0 = self.rank == 0

        if not self.distributed:
            return

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            cfg.training.device = f"cuda:{self.local_rank}"
            backend = "nccl"
        else:
            cfg.training.device = "cpu"
            backend = "gloo"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")

    def _barrier(self):
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _cleanup_distributed(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

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
        self._setup_distributed(cfg)
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
        normalizer_cache_mode = getattr(dataset, "normalizer_cache_mode", "auto")
        if (
            self.distributed
            and getattr(dataset, "normalizer_cache_path", None)
            and normalizer_cache_mode != "readonly"
        ):
            if self.is_rank0:
                normalizer = dataset.get_normalizer()
                self._barrier()
            else:
                self._barrier()
                old_cache_mode = dataset.normalizer_cache_mode
                dataset.normalizer_cache_mode = "readonly"
                try:
                    normalizer = dataset.get_normalizer()
                finally:
                    dataset.normalizer_cache_mode = old_cache_mode
        else:
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

        wandb_run = None
        if self.is_rank0:
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
        if self.distributed:
            self.ddp_model = DDP(
                PolicyLossWrapper(self.model),
                device_ids=[self.local_rank] if device.type == "cuda" else None,
                output_device=self.local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

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
        json_logger_context = JsonLogger(log_path) if self.is_rank0 else None
        if json_logger_context is None:
            class _NullLogger:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def log(self, data):
                    return None

            json_logger_context = _NullLogger()

        try:
            with json_logger_context as json_logger:
                for _ in range(cfg.training.num_epochs):
                    step_log = {}
                    train_losses = []

                    iterator = range(steps_per_epoch)
                    if self.is_rank0:
                        iterator = tqdm.tqdm(
                            iterator,
                            desc=f"Training epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        )

                    for batch_idx in iterator:
                        batch, train_iter = self._next_batch(train_iter, train_dataloader)
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        if self.ddp_model is not None:
                            raw_loss = self.ddp_model(batch)
                        else:
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
                        if self.is_rank0 and hasattr(iterator, "set_postfix"):
                            iterator.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == steps_per_epoch - 1
                        if not is_last_batch:
                            if self.is_rank0:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                    epoch_train_loss = torch.tensor(
                        float(np.mean(train_losses)),
                        device=device,
                        dtype=torch.float32,
                    )
                    if self.distributed:
                        dist.all_reduce(epoch_train_loss, op=dist.ReduceOp.AVG)
                    step_log["train_loss"] = epoch_train_loss.item()

                    self._barrier()
                    if self.is_rank0:
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

                    self._barrier()
                    self.global_step += 1
                    self.epoch += 1
        finally:
            self._cleanup_distributed()


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
