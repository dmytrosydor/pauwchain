"""Attack 1: does not train, repeats the starting weights with tiny noise."""
from sim.core import add

NAME = "lazy"


def produce(ctx):
    ckpts = [ctx.w_start]
    for _ in range(ctx.m):
        ckpts.append(add(ckpts[-1], ctx.noise_like(ckpts[-1], 1e-4)))
    return ckpts
