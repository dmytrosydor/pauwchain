"""Шаблон стратегії: скопіюйте файл без підкреслення в імені й змініть produce."""
from sim.core import add, sub  # допоміжні дії над вагами: add(a, b, scale), sub(a, b)

NAME = "template"


def produce(ctx):
    """Повертає точки C₀..C_m. Тут — сумлінна робота: кожен перехід чесний."""
    ckpts = [ctx.w_start]
    for j in range(ctx.m):
        ckpts.append(ctx.honest(ckpts[-1], j))
    return ckpts


# Необов'язково: що заявити як точність (×10⁶). Без цієї функції арена заявляє чесно виміряну.
# def claim_metric(ctx, w_end):
#     return 990000

_ = (add, sub)
