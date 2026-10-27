"""Адаптивна 5c проти тесту напрямку R: зсув = ціль + рівний за нормою ортогональний шум,
щоб попарний косинус залишків був ≈ 0,5·(ціль) і не перевищував τ. Ціна — менший дрейф до цілі."""
from sim.core import add, sub
from sim.attacks import _norm, _unit

NAME = "tolerance_camouflage"


def produce(ctx):
    alpha, tau = ctx.verifier["alpha"], ctx.verifier["residual_tau"]
    # cos(r_i, r_j) ≈ a² / (a² + b²) для r = a·d + b·n; беремо трохи нижче τ.
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
