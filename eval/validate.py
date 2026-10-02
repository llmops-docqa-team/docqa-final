"""Validate eval/questions.jsonl.  Usage: python -m eval.validate [path] [--docs path] [--strict]"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from pydantic import ValidationError

from eval.schema import EvalRow, Route

HERE = Path(__file__).parent
DEFAULT_QUESTIONS = HERE / "questions.jsonl"
DEFAULT_DOCS = HERE / "docs.yaml"


class Report:
    def __init__(self) -> None:
        self.rows: list[EvalRow] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        # Rows with gold pages that nobody else has checked. Informational: a second look is recommended
        # (eval/README.md), never a blocker, so it is a count and not a warning per row.
        self.unverified: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def load_doc_keys(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return set((data.get("docs") or {}).keys())


def _is_todo(row: EvalRow) -> bool:
    texts = [row.question, row.gold_answer or "", row.gold_general_answer or ""]
    return any("TODO" in t for t in texts)


def validate(questions_path: Path = DEFAULT_QUESTIONS, docs_path: Path = DEFAULT_DOCS) -> Report:
    rep = Report()
    try:
        doc_keys = load_doc_keys(docs_path)
    except (OSError, yaml.YAMLError) as e:
        rep.errors.append(f"{docs_path}: cannot load docs map: {e}")
        doc_keys = set()
    try:
        lines = Path(questions_path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        rep.errors.append(f"{questions_path}: cannot read: {e}")
        return rep

    seen: dict[str, int] = {}
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = EvalRow.model_validate(json.loads(line))
        except json.JSONDecodeError as e:
            rep.errors.append(f"line {n}: invalid JSON: {e.msg}")
            continue
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(p) for p in err["loc"]) or "row"
                rep.errors.append(f"line {n}: {loc}: {err['msg']}")
            continue
        if row.id in seen:
            rep.errors.append(f"line {n}: duplicate id {row.id!r} (first on line {seen[row.id]})")
            continue
        seen[row.id] = n
        rep.rows.append(row)
        _warn(rep, row, n, doc_keys)
    return rep


def _warn(rep: Report, row: EvalRow, n: int, doc_keys: set[str]) -> None:
    w = rep.warnings.append
    tag = f"line {n} ({row.id})"
    if _is_todo(row):
        w(f"{tag}: placeholder row (contains TODO)")
    if row.route is Route.DOCUMENT:
        if row.answerable and not row.gold_pages:
            w(f"{tag}: answerable DOCUMENT row has no gold_pages")
        if row.answerable and not row.gold_answer:
            w(f"{tag}: answerable DOCUMENT row has no gold_answer")
        if row.answerable is False and row.gold_pages:
            w(f"{tag}: unanswerable row has gold_pages (expected ABSTAIN)")
    elif row.gold_pages and row.route is not Route.MIXED:  # (MIXED: gold pages are for the document part)
        w(f"{tag}: {row.route.value} row has gold_pages")
    if row.gold_pages and not row.verified_by:
        rep.unverified += 1
    for gp in row.gold_pages:
        if doc_keys and gp.doc not in doc_keys:
            w(f"{tag}: unknown doc key {gp.doc!r}")
        if gp.pdf_page is None:
            w(f"{tag}: gold page {gp.doc}:{gp.page} has no pdf_page")


def _fmt(c: Counter) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "-"


def summary(rep: Report) -> str:
    rows = rep.rows
    doc = [r for r in rows if r.route is Route.DOCUMENT]
    return "\n".join(
        [
            f"{len(rows)} valid rows",
            "by route:       " + _fmt(Counter(r.route.value for r in rows)),
            "by slice:       " + _fmt(Counter(r.slice.value for r in rows)),
            "by type:        " + _fmt(Counter(r.type.value for r in rows)),
            "doc answerable: "
            + _fmt(Counter("answerable" if r.answerable else "unanswerable" for r in doc)),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the eval question set.")
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = ap.parse_args(argv)

    rep = validate(args.path, args.docs)
    print(summary(rep))
    for e in rep.errors:
        print(f"ERROR   {e}")
    for w in rep.warnings:
        print(f"WARNING {w}")
    print(f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    if rep.unverified:
        print(
            f"info: {rep.unverified} row(s) with gold pages have no verified_by yet "
            "(recommended before quoting numbers, not required)"
        )
    if rep.errors or (args.strict and rep.warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
