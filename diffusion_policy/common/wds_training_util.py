"""Optimizer setup for streaming WDS training."""
import math


def batch_scaled_learning_rate(base_lr, batch_size, world_size, accumulation=1,
                               reference_batch_size=None, rule="sqrt", max_lr=None):
    """Scale against the actual effective batch; absent reference disables scaling."""
    if min(batch_size, world_size, accumulation) <= 0:
        raise ValueError("Batch size, world size, and accumulation must be positive")
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("Base learning rate must be finite and positive")
    effective_batch = int(batch_size) * int(world_size) * int(accumulation)
    lr = float(base_lr)
    if reference_batch_size is not None:
        if reference_batch_size <= 0:
            raise ValueError("reference_batch_size must be positive")
        powers = {"linear": 1.0, "sqrt": 0.5, "none": 0.0}
        if rule not in powers:
            raise ValueError(f"Unsupported LR scaling rule: {rule!r}")
        lr *= (effective_batch / reference_batch_size) ** powers[rule]
        if max_lr is not None:
            if not math.isfinite(max_lr) or max_lr <= 0:
                raise ValueError("max_lr must be finite and positive")
            lr = min(lr, max_lr)
    return lr, effective_batch
