"""Cross-process determinism: the same block in different processes and with different CPU thread counts
must give bitwise identical weights. Check B on a single device relies on exactly this."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT = """
import sys, torch
torch.set_num_threads(int(sys.argv[1]))
from worker.data import Dataset
from worker.model import init_state
from worker.train import train_block, hash_states
g = torch.Generator().manual_seed(0)
x = torch.randint(0, 256, (600, 28, 28), dtype=torch.uint8, generator=g)
y = torch.randint(0, 10, (600,), generator=g)
ds = Dataset(x[:500], y[:500], x[500:], y[500:])
ckpts = train_block(init_state(3), ds, seed=12345, K=30, k=10, batch_size=32, lr=0.05)
print(hash_states(ckpts)[-1].hex())
"""


def run(threads: int) -> str:
    env = {**os.environ, "PYTHONHASHSEED": str(threads)}
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(threads)], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_bitwise_equal_across_processes_and_threads():
    hashes = {run(t) for t in (1, 4)}
    assert len(hashes) == 1, hashes
