"""Run metadata (git hash, eval-set hash) and optional MLflow logging for eval runs.

MLflow is optional: it is imported only when a run is logged, so CI and unit tests do not need it
(`pip install -r requirements-eval.txt` to use it). Local file store under ./mlruns (git-ignored).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_DIR = ROOT / "mlruns"


def git_hash() -> str:
    """Short commit hash, with '+dirty' if the working tree has uncommitted changes. 'unknown' outside git."""
    try:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()

        return git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def file_hash(path: str | Path, length: int = 12) -> str:
    """SHA-256 of a file's bytes (CRLF folded to LF so Windows checkouts hash like Linux ones)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:length]


def log_mlflow_run(
    *,
    experiment: str,
    run_name: str,
    params: dict,
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    artifacts: list[str | Path] | None = None,
    tracking_dir: str | Path | None = None,
) -> str:
    """Log one run to a local MLflow file store. Returns the run id. Raises ImportError if not installed."""
    # MLflow 3.x calls the file store "maintenance mode" and refuses it unless this is set. The design
    # asks for a local file store, and mlflow-skinny (no SQL backend) keeps the install small.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    store = Path(tracking_dir or DEFAULT_TRACKING_DIR)
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store.resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: ("" if v is None else v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.set_tags(tags or {})
        for artifact in artifacts or []:
            mlflow.log_artifact(str(artifact))
        return run.info.run_id
