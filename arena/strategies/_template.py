"""Strategy template: copy the file to a name without the underscore and change produce."""
from sim.core import add, sub  # weight helpers: add(a, b, scale), sub(a, b)

NAME = "template"


def produce(ctx):
    """Returns checkpoints C₀..C_m. Here: honest work, every transition is genuine."""
    ckpts = [ctx.w_start]
    for j in range(ctx.m):
        ckpts.append(ctx.honest(ckpts[-1], j))
    return ckpts


# Optional: what to claim as accuracy (×10⁶). Without this function the arena claims the honestly measured one.
# def claim_metric(ctx, w_end):
#     return 990000

_ = (add, sub)
