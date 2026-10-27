"""Один стрибок: усі переходи чесні, крім одного випадкового — там отруєння з великим кроком."""
NAME = "one_jump"


def produce(ctx):
    j_star = ctx.rng.randrange(ctx.m)
    ckpts = [ctx.w_start]
    for j in range(ctx.m):
        nxt = ctx.poisoned(ckpts[-1], j, p=1.0, lr_scale=10.0) if j == j_star else ctx.honest(ckpts[-1], j)
        ckpts.append(nxt)
    return ckpts
