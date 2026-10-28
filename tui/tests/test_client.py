from tui.client import parse_sse
from tui.view import describe, short


def test_parse_sse_frames_and_keepalive():
    lines = [
        "id: 1", "event: tip", 'data: {"seq": 1, "type": "tip", "data": {"height": 1}}', "",
        ": keepalive", "",
        "id: 2", "event: verdict", 'data: {"seq": 2, "type": "verdict",', 'data: "data": {}}', "",
    ]
    evs = list(parse_sse(lines))
    assert [e["seq"] for e in evs] == [1, 2]
    assert evs[0]["data"]["height"] == 1


def test_describe_verdict_names_miner_and_checks():
    names = {"aa" * 32: "2"}
    ev = {"type": "verdict", "data": {"accepted": False, "height": 4, "miner": "aa" * 32,
                                      "reason": "B: transition 3 failed",
                                      "checks": [{"mech": "B", "j": 3, "distance": 0.47, "threshold": 0.024,
                                                  "verdict": "reject"}]}}
    text = describe(ev, names)
    assert "node 2" in text and "h=4" in text and "j=3" in text and "0.47" in text


def test_short_hash():
    assert short("abcdef0123456789") == "abcdef01"


def test_describe_unavailable_is_not_a_rejection():
    ev = {"type": "unavailable",
          "data": {"height": 3, "miner": "ab" * 32, "reason": "data unavailable: checkpoint 1234"}}
    text = describe(ev, {"ab" * 32: "2"})
    assert "skipped" in text and "node 2" in text and "rejected" not in text
