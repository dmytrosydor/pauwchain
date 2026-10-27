"""Adaptive 5c against direction test R: shift = target + orthogonal noise of equal norm,
so the pairwise cosine of residuals is ≈ 0.5·(target) and stays below τ. The cost is less drift toward the target."""
from sim.core import add, sub
from sim.attacks import _norm, _unit

NAME = "tolerance_camouflage"


def produce(ctx):
    alpha, tau = ctx.verifier["alpha"], ctx.verifier["residual_tau"]
    # cos(r_i, r_j) ≈ a² / (a² + b²) for r = a·d + b·n; aim slightly below τ.
    a2 = 0.8 * tau
    a, b = a2 ** 0.5, (1 - a2) ** 0.5
    ckpts = [ctx.w_start]
    for j in range(ctx.m):
        cur = ckpts[-1]
        good = ctx.honest(cur, j)
        evil = ctx.poisoned(cur, j)
        d = _unit(sub(evil, good))
        noise = _unit(ctx.noise_like(good, 1.0))
        direction = _unit({n: a * d[n] + b * noise[n] for n in d})
        ckpts.append(add(good, direction, 0.9 * alpha * _norm(sub(good, cur))))
    return ckpts
