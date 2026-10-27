from typing import Dict
import torch
import torch.nn as nn


def mnist_cnn() -> nn.Module:
    # Без dropout і batchnorm (Р3): перехід між точками залежить лише від ваг і батчів.
    return nn.Sequential(
        nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(800, 100), nn.ReLU(), nn.Linear(100, 10),
    )


def init_state(task_seed: int) -> Dict[str, torch.Tensor]:
    """Генезис-стан моделі, детерміновано виведений з task_seed (Р1)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(task_seed)
        model = mnist_cnn()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}
