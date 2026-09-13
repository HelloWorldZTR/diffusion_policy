import json

import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.wds_training_util import batch_scaled_learning_rate


@pytest.mark.parametrize("batch,world,accum,expected", [
    (128, 1, 1, 1e-4 / 2**0.5),
    (128, 2, 1, 1e-4),
    (32, 8, 1, 1e-4),
    (64, 2, 2, 1e-4),
    (128, 4, 1, 1e-4 * 2**0.5),
    (128, 16, 1, 2e-4),
])
def test_lr_scales_by_effective_batch_and_caps(batch, world, accum, expected):
    lr, effective = batch_scaled_learning_rate(1e-4, batch, world, accum,
                                              reference_batch_size=256, max_lr=2e-4)
    assert lr == pytest.approx(expected)
    assert effective == batch * world * accum


def test_lr_scaling_is_opt_in_and_supports_linear():
    assert batch_scaled_learning_rate(1e-4, 128, 8)[0] == 1e-4
    assert batch_scaled_learning_rate(1e-4, 128, 4, reference_batch_size=256, rule="linear")[0] == 2e-4
    with pytest.raises(ValueError):
        batch_scaled_learning_rate(1e-4, 0, 2)


class _TinyDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        for _ in range(8):
            yield {"obs": {"state": torch.ones(1, 1)}, "action": torch.zeros(1, 1), "__key__": "test-frame"}

    def get_validation_dataset(self):
        return self

    def get_normalizer(self):
        return None


class _TinyPolicy(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen_weights = []
        self.nonfinite = nonfinite

    def set_normalizer(self, normalizer):
        pass

    def compute_loss(self, batch):
        if torch.is_grad_enabled():
            self.seen_weights.append(self.weight.item())
        loss = (self.weight * batch["obs"]["state"] - batch["action"]).square().mean()
        return loss * float("nan") if self.nonfinite else loss

    def predict_action(self, obs):
        return {"action_pred": self.weight * obs["state"]}


def _workspace_cfg(nonfinite=False):
    cfg = OmegaConf.create({
        "policy": {"_target_": f"{__name__}._TinyPolicy", "nonfinite": nonfinite},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
        "ema": {"_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel"},
        "task": {"dataset": {"_target_": f"{__name__}._TinyDataset"}},
        "dataloader": {"batch_size": 2, "num_workers": 0},
        "val_dataloader": {"batch_size": 2, "num_workers": 0},
        "training": {"seed": 42, "device": "cpu", "resume": False, "debug": False,
                     "num_epochs": 1, "steps_per_epoch": 4, "gradient_accumulate_every": 2,
                     "lr_scheduler": "constant", "lr_warmup_steps": 0, "use_ema": True,
                     "val_every": 1, "sample_every": 1, "checkpoint_every": 5,
                     "max_val_steps": 1, "tqdm_interval_sec": 10,
                     "max_grad_norm": 1.0, "fail_on_nonfinite": True,
                     "lr_scale": {"reference_batch_size": 4, "rule": "sqrt"}},
        "logging": {"mode": "disabled", "project": "dp-unit-tests"},
        "checkpoint": {"topk": {"monitor_key": "val_loss", "mode": "min", "k": 1,
                                "format_str": "epoch={epoch:04d}-val_loss={val_loss:.6f}.ckpt"},
                       "save_last_ckpt": True, "save_last_snapshot": False},
    })
    OmegaConf.set_struct(cfg, True)  # match Hydra-composed training configs
    return cfg


def test_workspace_accumulation_clipping_ema_and_validation(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    from diffusion_policy.workspace.train_diffusion_unet_hybrid_wds_workspace import TrainDiffusionUnetHybridWdsWorkspace
    steps = []
    original_step = EMAModel.step

    def record_step(self, model):
        steps.append(self.optimization_step)
        return original_step(self, model)

    monkeypatch.setattr(EMAModel, "step", record_step)
    workspace = TrainDiffusionUnetHybridWdsWorkspace(_workspace_cfg(), output_dir=str(tmp_path))
    workspace.run()
    if workspace._saving_thread is not None:
        workspace._saving_thread.join()
    assert workspace.model.seen_weights == pytest.approx([1, 1, 0.9, 0.9])
    assert workspace.model.weight.item() == pytest.approx(0.8)
    assert steps == [0, 1]
    logs = [json.loads(line) for line in (tmp_path / "logs.json.txt").read_text().splitlines()]
    assert logs[1]["grad_norm"] == pytest.approx(2)
    assert logs[1]["grad_clipped"] == 1
    assert "val_action_mse_error" in logs[-1]
    assert (tmp_path / "checkpoints/latest.ckpt").exists()
    import dill
    checkpoint = torch.load(tmp_path / "checkpoints/latest.ckpt", pickle_module=dill)
    metadata = dill.loads(checkpoint["pickles"]["optimizer_run_metadata"])
    assert metadata["effective_batch_size"] == 4
    assert metadata["world_size"] == 1
    assert metadata["scaled_learning_rate"] == 0.1

    # Resume keeps the optimizer/EMA age and honors total epochs, rather than
    # training num_epochs additional epochs with a restarted EMA warmup.
    resumed_cfg = _workspace_cfg()
    resumed_cfg.training.resume = True
    resumed_cfg.training.num_epochs = 2
    resumed = TrainDiffusionUnetHybridWdsWorkspace(resumed_cfg, output_dir=str(tmp_path))
    resumed.run()
    if resumed._saving_thread is not None:
        resumed._saving_thread.join()
    assert resumed.global_step == 8
    assert resumed.epoch == 2
    assert resumed.model.weight.item() == pytest.approx(0.6)
    assert steps == [0, 1, 2, 3]


def test_nonfinite_loss_stops_before_weights_change(tmp_path):
    pytest.importorskip("diffusers")
    from diffusion_policy.workspace.train_diffusion_unet_hybrid_wds_workspace import TrainDiffusionUnetHybridWdsWorkspace
    workspace = TrainDiffusionUnetHybridWdsWorkspace(_workspace_cfg(nonfinite=True), output_dir=str(tmp_path))
    with pytest.raises(FloatingPointError, match="test-frame"):
        workspace.run()
    assert workspace.model.weight.item() == 1
    assert not workspace.optimizer.state
