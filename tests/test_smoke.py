import sqlite3

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from app.storage.db import init_db


def _tmp_settings(tmp_path):
    s = load_settings()
    s.paths.sqlite_path = str(tmp_path / "db.sqlite")
    s.paths.upload_dir = str(tmp_path / "uploads")
    s.paths.chroma_dir = str(tmp_path / "chroma")
    s.paths.model_cache_dir = str(tmp_path / "models")
    return s


def test_config_loads():
    s = load_settings()
    assert s.chunking.size_tokens == 400
    assert s.retrieval.top_k == 8
    assert s.embedding.backend == "model2vec" and s.embedding.model == "minishlab/potion-retrieval-32M"
    assert s.prompts.router == "router_v1"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    s = load_settings()
    assert s.llm.base_url == "http://localhost:11434/v1"
    assert s.groq_api_key == "x"


def test_db_tables_created(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    init_db(db)  # idempotent
    conn = sqlite3.connect(db)
    names = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    conn.close()
    assert {"documents", "requests"} <= names


def test_db_under_old_project_name_is_adopted(tmp_path):
    old = tmp_path / "docqa.sqlite"
    init_db(old)
    conn = sqlite3.connect(old)
    conn.execute("insert into requests (trace_id) values ('kept')")
    conn.commit()
    conn.close()  # Windows cannot rename a file that is still open
    new = tmp_path / "finchat.sqlite"
    init_db(new)
    assert new.exists() and not old.exists()
    conn = sqlite3.connect(new)
    rows = conn.execute("select trace_id from requests").fetchall()
    conn.close()
    assert rows == [("kept",)]
    init_db(old)  # a fresh old-name file next to the new one is left alone
    assert new.exists() and old.exists()


def test_health_and_request_id(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.get_settings", lambda: _tmp_settings(tmp_path))
    with TestClient(create_app()) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}
        assert r.headers["x-request-id"]
        assert client.get("/health", headers={"X-Request-ID": "abc"}).headers["x-request-id"] == "abc"
    assert (tmp_path / "db.sqlite").exists()


def test_embedder_backend_is_chosen_from_config():
    from app.ingestion.embedder import FastEmbedder, StaticEmbedder, embedder_from_settings

    s = load_settings()
    assert isinstance(embedder_from_settings(s), StaticEmbedder)
    s.embedding.backend = "fastembed"
    assert isinstance(embedder_from_settings(s), FastEmbedder)
