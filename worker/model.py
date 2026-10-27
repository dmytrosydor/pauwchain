from typing import Dict
import torch
import torch.nn as nn


def mnist_cnn() -> nn.Module:
    # No dropout or batchnorm (D3): a transition between checkpoints depends only on weights and batches.
    return nn.Sequential(
        nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(800, 100), nn.ReLU(), nn.Linear(100, 10),
    )


def init_state(task_seed: int) -> Dict[str, torch.Tensor]:
    """Genesis model state, deterministically derived from task_seed (D1)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(task_seed)
        model = mnist_cnn()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}
