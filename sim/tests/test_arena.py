import random

from sim.arena import check_shape, load_strategies, play, scoreboard
from sim.core import Task, block_seed, miner_id
from sim.mechanisms import run
from worker.data import synthetic
from worker.model import init_state

TASK = Task(synthetic(), K=40, k=10, batch_size=32, lr=0.05)
W0 = init_state(1)
SEED = block_seed(bytes(32), miner_id(0), 0)
VERIFIER = {"q": 2, "alpha": 0.05, "residual_tau": 0.5, "validators": 1}


def strategies():
    return {s.name: s for s in load_strategies("arena/strategies")}


def test_builtin_strategies_load_and_template_is_skipped():
    names = set(strategies())
    assert {"honest", "lazy", "one_jump", "tolerance_drift", "tolerance_camouflage"} <= names
    assert "template" not in names


def test_honest_passes_and_lazy_fails_b():
    st = strategies()
    for name, should_pass in (("honest", True), ("lazy", False)):
        blk, ctx, err = play(st[name], TASK, W0, SEED, VERIFIER, random.Random(0))
        assert err is None
        assert (not run("B", TASK, blk, W0, random.Random(1)).rejected) == should_pass
    assert ctx.steps_used == 0  # лінивий не зробив жодного кроку


def test_work_is_counted():
    blk, ctx, _ = play(strategies()["honest"], TASK, W0, SEED, VERIFIER, random.Random(0))
    assert ctx.steps_used == TASK.K and blk.cost_steps == TASK.K


def test_bad_strategy_is_disqualified():
    class Broken:
        name, claim_metric, description, path = "broken", None, "", ""

        @staticmethod
        def produce(ctx):
            return [ctx.w_start]  # замало точок

    blk, _, err = play(Broken, TASK, W0, SEED, VERIFIER, random.Random(0))
    assert blk is None and "checkpoints" in err
    assert check_shape([W0] * (TASK.m + 1), W0, TASK.m) is None


def test_scoreboard_rates():
    rows = [{"strategy": "x", "mechanism": "B", "passed": p, "work": 0.0, "damage": 0.5, "error": ""}
            for p in (1, 0, 0, 0)]
    board = scoreboard(rows, ["x"], ["B"])
    assert board["x"]["B"]["pass_rate"] == 0.25
    assert abs(board["x"]["B"]["expected_damage"] - 0.125) < 1e-9
