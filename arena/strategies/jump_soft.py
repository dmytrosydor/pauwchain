"""Adaptive jump: the validator samples q of m transitions at random, so the jump position does not matter,
but the attacker can make the jump smaller (lr_scale 3) so that even a checked transition
looks "almost honest" to the α threshold. Tests whether B has headroom against such attacks."""
NAME = "jump_soft"


def produce(ctx):
    j_star = ctx.rng.randrange(ctx.m)
    ckpts = [ctx.w_start]
    for j in range(ctx.m):
        nxt = ctx.poisoned(ckpts[-1], j, p=0.3, lr_scale=3.0) if j == j_star else ctx.honest(ckpts[-1], j)
        ckpts.append(nxt)
    return ckpts
