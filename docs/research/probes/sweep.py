import json, time, torch, os
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")).read()
exec(_src.split('R = {"torch"')[0])  # function defs only, skip bench's main body

R = {}
X, y = data((1, 28, 28))
# trajectory on CPU with ckpt every step up to 700
base, ck = train(mnist_cnn, X, y, "cpu", 700, ckpt_every=1)
cat = lambda sd: torch.cat([v.flatten() for v in sd.values()])
t0 = 500
for k in (1, 10, 50, 100, 200):
    rec, _ = train(mnist_cnn, X, y, "mps", k, start_state=ck[t0], start_step=t0)
    true = cat(ck[t0 + k])
    R[f"k={k}"] = {"honest_cross_device_err": rel(flat(rec), true),
                   "lazy_replay_err": rel(cat(ck[t0]), true)}
    R[f"k={k}"]["margin_x"] = R[f"k={k}"]["lazy_replay_err"] / max(R[f"k={k}"]["honest_cross_device_err"], 1e-12)

# thread counts 1 / 4 / 10 on CIFAR CNN (bigger reductions)
Xc, yc = data((3, 32, 32), n=20000)
outs = {}
for th in (1, 4, 10):
    torch.set_num_threads(th)
    m, _ = train(cifar_cnn, Xc, yc, "cpu", 200)
    outs[th] = flat(m)
torch.set_num_threads(4)
R["cifar_threads_relL2_1v4"] = rel(outs[1], outs[4])
R["cifar_threads_relL2_4v10"] = rel(outs[10], outs[4])

# deterministic flag on MPS: does it run, speed
torch.use_deterministic_algorithms(True)
try:
    t = time.perf_counter(); m1, _ = train(mnist_cnn, X, y, "mps", 300); R["mps_det_flag_300steps_s"] = round(time.perf_counter() - t, 2)
    m2, _ = train(mnist_cnn, X, y, "mps", 300)
    R["mps_det_flag_bitwise_repeat"] = torch.equal(flat(m1), flat(m2))
except Exception as e:
    R["mps_det_flag_error"] = repr(e)[:200]
torch.use_deterministic_algorithms(False)
print(json.dumps(R, indent=1))
