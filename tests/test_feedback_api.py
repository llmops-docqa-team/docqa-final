"""POST /feedback: thumbs up/down stored on the request row."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeEmbedder
from tests.fakes import FakeLLM


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    settings.retrieval.theta = -2.0
    llm = FakeLLM(settings, general=["Paris is the capital of France."])
    with TestClient(create_app(embedder=FakeEmbedder(), llm=llm)) as c:
        yield c


def stored(client, trace_id):
    return client.app.state.request_store.get(trace_id)


def test_thumbs_up_and_down_are_stored(client):
    assert client.post("/feedback", json={"trace_id": "t-up", "value": 1}).status_code == 200
    assert client.post("/feedback", json={"trace_id": "t-down", "value": -1}).status_code == 200
    assert stored(client, "t-up")["feedback"] == 1
    assert stored(client, "t-down")["feedback"] == -1


def test_feedback_updates_an_existing_row_and_keeps_its_other_fields(client):
    import sqlite3

    path = client.app.state.settings.sqlite_path
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO requests (trace_id, route, status, t_total_ms) VALUES ('t-1','DOCUMENT','answered',1234)"
    )
    conn.commit()
    conn.close()

    r = client.post("/feedback", json={"trace_id": "t-1", "value": 1})
    assert r.json() == {"trace_id": "t-1", "value": 1}
    row = stored(client, "t-1")
    assert (row["feedback"], row["route"], row["status"], row["t_total_ms"]) == (
        1,
        "DOCUMENT",
        "answered",
        1234,
    )


def test_a_later_click_replaces_the_earlier_one(client):
    client.post("/feedback", json={"trace_id": "t-2", "value": 1})
    client.post("/feedback", json={"trace_id": "t-2", "value": -1})
    assert stored(client, "t-2")["feedback"] == -1


def test_only_plus_or_minus_one_is_accepted(client):
    for bad in (0, 2, -2, "up", None):
        r = client.post("/feedback", json={"trace_id": "t-3", "value": bad})
        assert r.status_code == 422, bad
    assert stored(client, "t-3") is None


@pytest.mark.parametrize(
    "body", [{"value": 1}, {"trace_id": "", "value": 1}, {"trace_id": "x" * 200, "value": 1}]
)
def test_a_missing_or_unreasonable_trace_id_is_rejected(client, body):
    assert client.post("/feedback", json=body).status_code == 422


def test_feedback_for_the_trace_id_returned_by_query(client):
    q = client.post(
        "/query", json={"question": "What is the capital of France?"}, headers={"X-Request-ID": "trace-abc"}
    ).json()
    assert q["trace_id"] == "trace-abc"
    assert client.post("/feedback", json={"trace_id": q["trace_id"], "value": -1}).status_code == 200
    assert stored(client, "trace-abc")["feedback"] == -1


def test_the_trace_id_is_stored_as_data_not_as_sql(client):
    nasty = "x'); DROP TABLE requests;--"
    assert client.post("/feedback", json={"trace_id": nasty, "value": 1}).status_code == 200
    assert stored(client, nasty)["feedback"] == 1
    assert client.post("/feedback", json={"trace_id": "after", "value": 1}).status_code == 200


def test_feedback_log_line_has_no_content(client, capsys):
    client.post("/feedback", json={"trace_id": "t-log", "value": 1})
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines() if line.startswith("{") and '"feedback"' in line]
    assert any(line.get("trace_id") == "t-log" and line.get("value") == 1 for line in lines)
