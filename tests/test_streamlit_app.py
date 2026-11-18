"""Streamlit app (R26): first screen, both chat surfaces in replay, live mode with a stub agent; offline, no key."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "src" / "ui"
sys.path[:0] = [str(UI), str(ROOT)]
pytest.importorskip("streamlit")
import agent as A  # noqa: E402
import aha_agents  # noqa: E402
import showcase  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(UI / "dashboard.py")


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(ROOT)                          # the app reads data/ relative to the repo, as on Streamlit Cloud
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


PAGES = ["overview", "explore", "maps", "data", "sentiment", "discovery", "agent"]


def body(at) -> str:
    return " ".join(e.proto.body for e in at.get("html"))


def app(page="overview", **state):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def test_first_screen_has_results_and_brief():
    at = app()
    h = showcase.headline()
    assert [m.value for m in at.metric][:2] == [f"{h['ndcg_after']:.3f}", f"{h['grounded_new']:.0%}"]
    assert [m.value for m in at.metric][2:4] == ["27,945", "444,182"]
    assert any("12 hours" in c.value for c in at.caption)
    for w in ["What", "Why", "Who", "Where", "When it fails", "How"]:          # the 5W1H brief (R22)
        assert f'class="f-w">{w}<' in body(at), w
    assert [b.key for b in at.sidebar.button] == [f"nav_{k}" for k in PAGES]


def test_every_page_opens_with_its_business_question():
    for key in PAGES[1:]:
        assert "Business question" in body(app(key)), key


def test_owner_key_is_dropped_at_start(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk" + "-" + "owner-key-must-not-be-used")   # fake, built at runtime
    app("discovery")
    import os
    assert os.environ.get("OPENAI_API_KEY") is None


@pytest.mark.parametrize("name,agent", [("data_agent", aha_agents.DataAgent), ("reviews", aha_agents.ReviewAgent)])
def test_replays_recorded_grounded_and_keyless(name, agent):
    p = UI / f"replays_{name}.json"
    d = json.loads(p.read_text())
    assert [r["question"] for r in d["runs"]] == agent.QUESTIONS and all(r["grounded"] for r in d["runs"])
    raw = p.read_text()
    assert "sk-" not in raw and "OPENAI" not in raw
    for r in d["runs"]:
        assert all((UI / img).exists() for img in r.get("images", []))


def test_replay_both_surfaces_without_a_key():
    at = app("agent")
    assert at.radio(key="data_chat_mode").value.startswith("Replay") and at.radio(key="rag_chat_mode").value.startswith("Replay")
    d = json.loads((UI / "replays_data_agent.json").read_text())["runs"][1]
    at.selectbox(key="data_replay_pick").set_value(d["question"]).run()
    assert not at.exception and any(d["answer"][:40] in m.value for m in at.markdown)
    r = json.loads((UI / "replays_reviews.json").read_text())["runs"][0]
    at.selectbox(key="rag_replay_pick").set_value(r["question"]).run()
    assert not at.exception and any(r["answer"][:40] in m.value for m in at.markdown)
    assert any("Cited sources" in m.value for m in at.markdown)


def test_live_mode_uses_viewer_key_and_never_echoes_it(monkeypatch):
    key = "sk" + "-" + "test-viewerkey-123456"                                # fake, built at runtime
    seen = {}

    def fake(question, k):
        seen["key"] = k
        return {"question": question, "answer": "Live answer.", "tool_calls": [], "tool_results": [], "images": [], "steps": 1,
                "grounded": True, "ungrounded_numbers": [], "model": "stub", "tokens": {"input": 1, "cached_input": 0, "output": 1},
                "cost_usd": 0.0, "latency_ms": 1.0}
    monkeypatch.setattr(aha_agents.DataAgent, "live", staticmethod(fake))
    at = app("agent")
    at.radio(key="data_chat_mode").set_value("Live: use your own OpenAI key").run()
    at.text_input(key="data_byo_key").input(key).run()
    at.chat_input(key="data_live_input").set_value("Anything").run()
    assert seen["key"] == key and any("Live answer." in m.value for m in at.markdown)
    assert key not in " ".join(m.value for m in at.markdown)
    assert A.redact(f"Incorrect API key provided: {key}", key) == "Incorrect API key provided: [your key]"


def test_frame_is_the_shared_copy():
    """The app frame is identical in every portfolio app; change it everywhere at once (checksum of the shared copy)."""
    import hashlib
    assert hashlib.sha256((UI / "frame.py").read_bytes()).hexdigest()[:16] == "385e92329f20405a"
