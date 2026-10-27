"""Часткова робота: два чесні переходи, решта — лінійна екстраполяція їхнього середнього кроку."""
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
