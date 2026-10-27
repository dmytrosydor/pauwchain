# Attack arena

You can write your own cheating strategy as a single file and check whether it gets past mechanisms A, B, R, C and D.

```sh
.venv/bin/python -m sim.arena --trials 3                     # all strategies in arena/strategies/
.venv/bin/python -m sim.arena --validators 3 --q 2 --alpha 0.05
```

The template is `arena/strategies/_template.py` (the arena skips files starting with `_`). A strategy receives `ctx`:

| What | Meaning |
|---|---|
| `ctx.w_start`, `ctx.m`, `ctx.k`, `ctx.K`, `ctx.seed` | starting weights, number of transitions, steps per transition and per block, data-order seed |
| `ctx.verifier` | verification parameters: `q`, `alpha`, `residual_tau`, `validators` (known to the attacker) |
| `ctx.honest(state, j, steps=None)` | honest steps of transition j |
| `ctx.poisoned(state, j, p=1.0, lr_scale=1.0)` | transition j on corrupted labels |
| `ctx.noise_like(state, scale)` | Gaussian noise of the same shape |
| `ctx.rng` | the strategy's random number generator |

The arena counts training steps made through `ctx` (the "work" column, as a fraction of K) and the damage, i.e. the accuracy drop
compared with an honest block. Scoreboard: the fraction of blocks that passed each mechanism, with a 95% Wilson interval.

Strategy code runs inside the arena process: only run strategies you trust.
