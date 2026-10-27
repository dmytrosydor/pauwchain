"""Honest miner, the control: must pass everything."""
NAME = "honest"


def produce(ctx):
    ckpts = [ctx.w_start]
    for j in range(ctx.m):
        ckpts.append(ctx.honest(ckpts[-1], j))
    return ckpts
