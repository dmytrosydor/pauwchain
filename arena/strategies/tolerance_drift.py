"""Tolerance attack (5c): each checkpoint is shifted toward a poisoned model by 0.9·α·‖Δ‖, just under B's threshold."""
from sim.core import add, sub
from sim.attacks import _norm, _unit

NAME = "tolerance_drift"


def produce(ctx):
    alpha = ctx.verifier["alpha"]
    ckpts = [ctx.w_start]
    for j in range(ctx.m):
        cur = ckpts[-1]
        good = ctx.honest(cur, j)
        evil = ctx.poisoned(cur, j)
        shift = 0.9 * alpha * _norm(sub(good, cur))
        ckpts.append(add(good, _unit(sub(evil, good)), shift))
    return ckpts
