"""Judge the last N answered requests and store the scores for the Metrics page's Quality section.

    python scripts/judge_recent.py --last 50 [--rejudge] [--config config.yaml]

Needs GEMINI_API_KEY (from the environment or .env) and requests whose text was kept, i.e.
`observability.log_content: true` when they were asked (the request log itself holds no text). Run it by
hand or on a schedule (daily is plenty); it only touches `requests.judge_correct` / `judge_grounded`.
See eval/online_judge.py for what "correct" means without a gold answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.observability.tracing import build_tracer, set_tracer  # noqa: E402
from app.storage.db import init_db  # noqa: E402
from app.storage.requests import RequestStore  # noqa: E402
from eval.env import load_env_file  # noqa: E402
from eval.judge import judge_from_settings  # noqa: E402
from eval.online_judge import judge_recent, summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Judge recent answered requests (online evaluation).")
    ap.add_argument("--last", type=int, default=50, help="how many recent answered requests to look at")
    ap.add_argument("--rejudge", action="store_true", help="include requests that already have scores")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args(argv)

    load_env_file()
    settings = load_settings(args.config)
    judge = judge_from_settings(settings, use_cache=False)  # live traffic: no replay from the dev cache
    if judge is None:
        print("error: GEMINI_API_KEY is not set (environment or .env), so there is no judge", file=sys.stderr)
        return 2
    init_db(settings.sqlite_path)  # adds the judge columns / content table to an older database
    # With LANGFUSE_* set, each verdict is also attached to the request's trace as a score.
    tracer = build_tracer(capture_content=False)
    set_tracer(tracer)
    try:
        result = judge_recent(RequestStore(settings.sqlite_path), judge, args.last, rejudge=args.rejudge)
        print(summary(result))
    finally:
        tracer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
