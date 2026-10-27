"""Адаптивний стрибок: валідатор бере q із m переходів випадково, тож місце стрибка не важить —
але атакувальник може зробити стрибок меншим (lr_scale 3), щоб навіть спійманий перехід
виглядав «майже чесним» для порогу α. Перевіряє, чи має B запас на такі атаки."""
NAME = "jump_soft"


def produce(ctx):
    j_star = ctx.rng.randrange(ctx.m)
    ckpts = [ctx.w_start]
    for j in range(ctx.m):
        nxt = ctx.poisoned(ckpts[-1], j, p=0.3, lr_scale=3.0) if j == j_star else ctx.honest(ckpts[-1], j)
        ckpts.append(nxt)
    return ckpts
