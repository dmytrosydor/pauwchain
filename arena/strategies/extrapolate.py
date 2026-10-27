"""Partial work: two honest transitions, the rest is linear extrapolation of their mean step."""
from sim.core import add, sub

NAME = "extrapolate"


def produce(ctx):
    ckpts = [ctx.w_start]
    for j in range(2):
        ckpts.append(ctx.honest(ckpts[-1], j))
    step = {n: t / 2 for n, t in sub(ckpts[-1], ckpts[0]).items()}
    while len(ckpts) < ctx.m + 1:
        ckpts.append(add(ckpts[-1], step))
    return ckpts
