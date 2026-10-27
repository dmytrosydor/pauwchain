import hashlib
from dataclasses import dataclass
from functools import lru_cache
import torch

MEAN, STD = 0.1307, 0.3081


@dataclass(frozen=True)
class Dataset:
    x_train: torch.Tensor  # uint8 [N, 28, 28]
    y_train: torch.Tensor  # int64 [N]
    x_test: torch.Tensor
    y_test: torch.Tensor


@lru_cache(maxsize=None)
def load_dataset(name: str, root: str = "data") -> Dataset:
    from torchvision import datasets

    cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[name]
    train = cls(root, train=True, download=True)
    test = cls(root, train=False, download=True)
    return Dataset(train.data, train.targets, test.data, test.targets)


def dataset_hash(ds: Dataset) -> bytes:
    """Хеш канонічної копії набору даних, фіксується в завданні (Р12)."""
    h = hashlib.sha256()
    for t in (ds.x_train, ds.y_train, ds.x_test, ds.y_test):
        h.update(t.contiguous().numpy().tobytes())
    return h.digest()


def val_set(ds: Dataset):
    """Перша половина тестового набору: на ній майнер заявляє метрику, а механізм A її перевіряє."""
    n = ds.x_test.shape[0] // 2
    return ds.x_test[:n], ds.y_test[:n]


def test_set(ds: Dataset):
    """Друга половина: лише для звітної точності, у консенсусі не бере участі."""
    n = ds.x_test.shape[0] // 2
    return ds.x_test[n:], ds.y_test[n:]


def to_input(x_uint8: torch.Tensor) -> torch.Tensor:
    return ((x_uint8.float() / 255.0 - MEAN) / STD).unsqueeze(1)


def permutation(seed: int, n: int) -> torch.Tensor:
    """Порядок даних виводиться з seed, майнер його не обирає (Р2)."""
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def batch_indices(perm: torch.Tensor, step: int, batch_size: int) -> torch.Tensor:
    # Індекси за модулем: блок може мати K·B > N.
    n = perm.numel()
    return perm[(step * batch_size + torch.arange(batch_size)) % n]
