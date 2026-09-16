"""Feasibility probe: timing, determinism/divergence, safetensors byte-stability.
Synthetic data (random images, fixed labels) -- FP divergence behaviour does not
depend on data semantics; timing does not depend on content.
"""
import hashlib, os, time, json
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import save_file, save

def mnist_cnn():
    return nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(800, 100), nn.ReLU(), nn.Linear(100, 10))

def cifar_cnn():
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(2048, 128), nn.ReLU(), nn.Linear(128, 10))

def data(shape, n=60000, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn((n,) + shape, generator=g)
    # learnable labels: sign pattern of a fixed random projection
    W = torch.randn(int(torch.tensor(shape).prod()), 10, generator=g)
    y = (X.flatten(1) @ W).argmax(1)
    return X, y

def flat(m):
    return torch.cat([p.detach().float().cpu().flatten() for p in m.state_dict().values()])

def rel(a, b):
    return ((a - b).norm() / b.norm()).item()

def train(make, X, y, device, steps, init_seed=0, order_seed=1, bs=64, ckpt_every=None,
          start_state=None, start_step=0):
    torch.manual_seed(init_seed)
    m = make()
    if start_state is not None:
        m.load_state_dict(start_state)
    m.to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.0)
    g = torch.Generator().manual_seed(order_seed)
    perm = torch.randperm(len(X), generator=g)  # data order fully fixed by order_seed
    Xd, yd = X.to(device), y.to(device)
    ckpts = {}
    for s in range(start_step, start_step + steps):
        idx = perm[(s * bs) % len(X): (s * bs) % len(X) + bs].to(device)
        loss = F.cross_entropy(m(Xd[idx]), yd[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ckpt_every and (s + 1) % ckpt_every == 0:
            ckpts[s + 1] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    if device == "mps": torch.mps.synchronize()
    return m, ckpts

def timeit(make, shape, device, steps):
    X, y = data(shape, n=steps * 64 if steps * 64 < 60000 else 60000)
    train(make, X, y, device, 20)  # warmup
    t = time.perf_counter(); train(make, X, y, device, steps); return time.perf_counter() - t

R = {"torch": torch.__version__, "threads_default": torch.get_num_threads(),
     "mps": torch.backends.mps.is_available()}
devs = ["cpu"] + (["mps"] if R["mps"] else [])

# ---- timing ----
R["params"] = {"mnist_cnn": sum(p.numel() for p in mnist_cnn().parameters()),
               "cifar_cnn": sum(p.numel() for p in cifar_cnn().parameters())}
R["timing_s"] = {}
for name, make, shape in [("mnist_cnn", mnist_cnn, (1, 28, 28)), ("cifar_cnn", cifar_cnn, (3, 32, 32))]:
    for d in devs:
        for steps in (100, 1000):
            R["timing_s"][f"{name}/{d}/{steps}"] = round(timeit(make, shape, d, steps), 3)

# ---- file sizes & safetensors determinism ----
m = cifar_cnn(); sd = {k: v.contiguous() for k, v in m.state_dict().items()}
b1 = save(sd); b2 = save(dict(reversed(list(sd.items()))))
b3 = save(sd, metadata={"a": "1", "b": "2", "c": "3", "round": "7"})
b4 = save(sd, metadata={"round": "7", "c": "3", "b": "2", "a": "1"})
R["safetensors"] = {
    "cifar_bytes": len(b1), "mnist_bytes": len(save({k: v.contiguous() for k, v in mnist_cnn().state_dict().items()})),
    "same_bytes_reordered_dict": hashlib.sha256(b1).hexdigest() == hashlib.sha256(b2).hexdigest(),
    "same_bytes_metadata_reordered": b3 == b4,
    "header_prefix": b1[8:120].decode(errors="replace"),
}

# ---- determinism / divergence (MNIST CNN, 1000 steps, ckpt every 100) ----
X, y = data((1, 28, 28))
T, K = 1000, 100
D = {}
torch.set_num_threads(R["threads_default"])
a, ca = train(mnist_cnn, X, y, "cpu", T, ckpt_every=K)
b, cb = train(mnist_cnn, X, y, "cpu", T, ckpt_every=K)
D["cpu_repeat_bitwise_equal"] = torch.equal(flat(a), flat(b))
torch.set_num_threads(1)
c, cc = train(mnist_cnn, X, y, "cpu", T, ckpt_every=K)
torch.set_num_threads(R["threads_default"])
D["cpu_threads1_vs_default_bitwise_equal"] = torch.equal(flat(a), flat(c))
D["relL2_threads1_vs_default_by_step"] = {s: rel(torch.cat([v.flatten() for v in cc[s].values()]),
                                                torch.cat([v.flatten() for v in ca[s].values()])) for s in ca}
# honest-progress scale references
w0 = flat(train(mnist_cnn, X, y, "cpu", 0)[0])
D["relL2_honest_update_k100 (W200 vs W100)"] = rel(torch.cat([v.flatten() for v in ca[200].values()]),
                                                  torch.cat([v.flatten() for v in ca[100].values()]))
D["relL2_W1000_vs_W0"] = rel(flat(a), w0)
o, _ = train(mnist_cnn, X, y, "cpu", T, order_seed=2)
D["relL2_different_data_order_after_1000"] = rel(flat(o), flat(a))
if "mps" in devs:
    p1, cp1 = train(mnist_cnn, X, y, "mps", T, ckpt_every=K)
    p2, _ = train(mnist_cnn, X, y, "mps", T, ckpt_every=K)
    D["mps_repeat_bitwise_equal"] = torch.equal(flat(p1), flat(p2))
    D["relL2_mps_repeat_after_1000"] = rel(flat(p2), flat(p1))
    D["relL2_mps_vs_cpu_by_step"] = {s: rel(torch.cat([v.flatten() for v in cp1[s].values()]),
                                           torch.cat([v.flatten() for v in ca[s].values()])) for s in ca}
    # mechanism B: verifier recomputes ONE transition (k=100) from prover's checkpoint on other device
    t0 = 500
    rec, _ = train(mnist_cnn, X, y, "mps", K, start_state=ca[t0], start_step=t0)
    D["mechB_single_transition_k100_cpu_prover_mps_verifier_relL2"] = rel(flat(rec), torch.cat([v.flatten() for v in ca[t0 + K].values()]))
torch.set_num_threads(1)
rec1, _ = train(mnist_cnn, X, y, "cpu", K, start_state=ca[500], start_step=500)
torch.set_num_threads(R["threads_default"])
D["mechB_single_transition_k100_threads_mismatch_relL2"] = rel(flat(rec1), torch.cat([v.flatten() for v in ca[600].values()]))
# lazy replay reference: previous state + tiny noise
prev = torch.cat([v.flatten() for v in ca[500].values()])
D["mechB_lazy_replay_relL2_vs_true_W600"] = rel(prev, torch.cat([v.flatten() for v in ca[600].values()]))
# verification cost: one k=100 recompute vs full 1000-step round is ~ K/T by construction
R["divergence"] = D
print(json.dumps(R, indent=1))
