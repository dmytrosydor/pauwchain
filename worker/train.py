import hashlib
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from worker.canonical import canonical_weight_hash
from worker.data import Dataset, batch_indices, permutation, to_input
from worker.model import mnist_cnn

State = Dict[str, torch.Tensor]


class Cancelled(Exception):
    """Навчання перервано ззовні (наприклад, вузол отримав нову вершину ланцюга)."""


def _cpu_copy(model: torch.nn.Module) -> State:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_steps(
    state: State,
    ds: Dataset,
    seed: int,
    step_from: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: str = "cpu",
    ckpt_every: Optional[int] = None,
    stop: Optional[Callable[[], bool]] = None,
) -> Tuple[State, List[State]]:
    """Кроки step_from .. step_from+steps−1 звичайного SGD (без моментуму, Р3).

    Повертає кінцевий стан і точки після кожних ckpt_every кроків (якщо задано).
    Той самий виклик на тому самому пристрої дає побітово ті самі ваги.
    stop перевіряється перед кожним кроком; якщо повертає True — Cancelled.
    """
    model = mnist_cnn()
    model.load_state_dict(state)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    perm = permutation(seed, ds.x_train.shape[0])

    ckpts = []
    for i in range(steps):
        if stop is not None and stop():
            raise Cancelled()
        idx = batch_indices(perm, step_from + i, batch_size)
        x = to_input(ds.x_train[idx]).to(device)
        y = ds.y_train[idx].to(device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ckpt_every and (i + 1) % ckpt_every == 0:
            ckpts.append(_cpu_copy(model))
    return _cpu_copy(model), ckpts


def train_block(
    w_start: State,
    ds: Dataset,
    seed: int,
    K: int,
    k: int,
    batch_size: int,
    lr: float,
    device: str = "cpu",
    stop: Optional[Callable[[], bool]] = None,
) -> List[State]:
    """Робота майнера в блоці (Р4): K кроків, точки C₀..C_m, m = K/k, C₀ = W_start."""
    if K % k != 0:
        raise ValueError(f"K={K} must be a multiple of k={k}")
    start = {n: t.detach().cpu().clone() for n, t in w_start.items()}
    _, ckpts = train_steps(start, ds, seed, 0, K, batch_size, lr, device, ckpt_every=k, stop=stop)
    return [start] + ckpts


def ckpt_root(ckpt_hashes: List[bytes]) -> bytes:
    """SHA-256 від конкатенації хешів точок C₀..C_m (поле заголовка ckpt_root)."""
    return hashlib.sha256(b"".join(ckpt_hashes)).digest()


def hash_states(states: List[State]) -> List[bytes]:
    return [canonical_weight_hash(s) for s in states]


@torch.no_grad()
def accuracy(
    state: State,
    x_uint8: torch.Tensor,
    y: torch.Tensor,
    device: str = "cpu",
    chunk: int = 2000,
) -> float:
    model = mnist_cnn()
    model.load_state_dict(state)
    model.to(device).eval()
    correct = 0
    for i in range(0, x_uint8.shape[0], chunk):
        x = to_input(x_uint8[i:i + chunk]).to(device)
        pred = model(x).argmax(1).cpu()
        correct += int((pred == y[i:i + chunk]).sum())
    return correct / x_uint8.shape[0]


def to_micro(value: float) -> int:
    """Fixed-point ×10⁶ для metric_micro у заголовку блоку."""
    return round(value * 1_000_000)
