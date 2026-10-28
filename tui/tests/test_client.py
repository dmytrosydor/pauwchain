from tui.client import parse_sse
from tui.view import chart, describe, short


def test_parse_sse_frames_and_keepalive():
    lines = [
        "id: 1", "event: tip", 'data: {"seq": 1, "type": "tip", "data": {"height": 1}}', "",
        ": keepalive", "",
        "id: 2", "event: verdict", 'data: {"seq": 2, "type": "verdict",', 'data: "data": {}}', "",
    ]
    evs = list(parse_sse(lines))
    assert [e["seq"] for e in evs] == [1, 2]
    assert evs[0]["data"]["height"] == 1


def test_describe_verdict_names_cheater_in_plain_words():
    names = {"aa" * 32: "2"}
    ev = {"type": "verdict", "data": {"accepted": False, "height": 4, "miner": "aa" * 32,
                                      "reason": "B: transition 3 failed",
                                      "checks": [{"mech": "B", "j": 3, "distance": 0.48, "threshold": 0.02,
                                                  "verdict": "reject"}]}}
    text = describe(ev, names)
    assert "Caught node 2 cheating" in text and "#4" in text and "24×" in text


def test_routine_events_stay_out_of_the_feed():
    for t in ("mined", "lost_race", "mining_aborted", "peer", "genesis"):
        assert describe({"type": t, "data": {}}, {}) is None
    accepted = {"type": "verdict", "data": {"accepted": True, "height": 1, "miner": "aa" * 32}}
    assert describe(accepted, {}) is None


def test_short_hash():
    assert short("abcdef0123456789") == "abcdef01"


def test_describe_unavailable_is_not_a_rejection():
    ev = {"type": "unavailable",
          "data": {"height": 3, "miner": "ab" * 32, "reason": "data unavailable: checkpoint 1234"}}
    text = describe(ev, {"ab" * 32: "2"})
    assert "Could not download" in text and "node 2" in text and "Caught" not in text


def test_chart_scales_to_the_box():
    lines = chart([85.0, 95.0, 98.0], width=20, height=4).plain.splitlines()
    assert len(lines) == 5 and lines[0].startswith(" 100%")
