"""Judge calibration: do the judge's verdicts agree with people's? (design §14: below 80%, fix the prompt.)

    python -m eval.calibrate export --run default --n 20     # writes a sheet for humans + a hidden key
    python -m eval.calibrate score --sheet eval/results/calibration_sheet.csv

`export` samples answers the judge has already graded, half it called correct and half it called wrong (a
random sample of a system that is mostly right would hold two wrong answers and tell us nothing about
whether the judge can spot them). The sheet shows the question, the reference answer, the system's answer and
the cited text, but never the judge's verdict, so the labellers are not nudged. Two people fill in
`human_a_*` and `human_b_*` with 1/0 (or yes/no); `correct` = the answer matches the reference, `grounded` =
every claim is supported by the cited text (leave it blank for general-knowledge answers, which cite nothing).
`score` reports, per verdict: judge vs each person, judge vs the two people when they agree, and person vs
person, as % agree and Cohen's kappa.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from eval.answer_metrics import agreement
from eval.records import RunFile, judge_targets, verdict_of

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"
DEFAULT_PREFIX = RESULTS_DIR / "calibration"
THRESHOLD = 0.8
SHEET_COLUMNS = [
    "item",
    "question",
    "reference_answer",
    "system_answer",
    "cited_sources",
    "human_a_correct",
    "human_a_grounded",
    "human_b_correct",
    "human_b_grounded",
]
_YES, _NO = {"1", "y", "yes", "true", "t"}, {"0", "n", "no", "false", "f"}


def judged_items(records: list[dict]) -> list[dict]:
    """Every answer in the run that has a judge verdict, as a flat item."""
    items = []
    for rec in records:
        for t in judge_targets(rec):
            v = verdict_of(rec, t.part)
            if v is None:
                continue
            items.append(
                {
                    "item": f"{rec['id']}:{t.part}",
                    "question": t.question,
                    "reference": t.reference or "",
                    "answer": t.answer,
                    "sources": [f"[p.{s.label.removeprefix('p.')}] {s.text}" for s in t.sources],
                    "judge": {"correct": bool(v["correct"]), "grounded": v.get("grounded")},
                }
            )
    return items


def sample_items(items: list[dict], n: int, seed: int) -> list[dict]:
    """Up to `n` items, half judged correct and half judged wrong (as far as each side has them), shuffled."""
    rng = random.Random(seed)
    right = [i for i in items if i["judge"]["correct"]]
    wrong = [i for i in items if not i["judge"]["correct"]]
    rng.shuffle(right)
    rng.shuffle(wrong)
    take_wrong = min(len(wrong), n // 2)
    take_right = min(len(right), n - take_wrong)
    take_wrong = min(len(wrong), n - take_right)  # the right side ran short: give the slack to the wrong side
    chosen = right[:take_right] + wrong[:take_wrong]
    rng.shuffle(chosen)
    return chosen


def write_sheet(items: list[dict], sheet: Path, key: Path) -> None:
    sheet.parent.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows ₹ and ± properly
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for i in items:
            sources = (
                "\n\n".join(s[:700] for s in i["sources"])[:2200] or "(none: a general-knowledge answer)"
            )
            w.writerow(
                {
                    "item": i["item"],
                    "question": i["question"],
                    "reference_answer": i["reference"] or "(none)",
                    "system_answer": i["answer"],
                    "cited_sources": sources,
                }
            )
    key.write_text(
        json.dumps({i["item"]: i["judge"] for i in items}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_label(raw: str | None, *, where: str) -> int | None:
    value = (raw or "").strip().lower()
    if not value:
        return None
    if value in _YES:
        return 1
    if value in _NO:
        return 0
    raise ValueError(f"{where}: '{raw}' is not 1/0/yes/no")


def read_sheet(sheet: Path) -> dict[str, dict[str, int | None]]:
    """{item: {a_correct, a_grounded, b_correct, b_grounded}} with None for a blank cell."""
    out: dict[str, dict[str, int | None]] = {}
    with sheet.open(newline="", encoding="utf-8-sig") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            item = (row.get("item") or "").strip()
            if not item:
                continue
            out[item] = {
                f"{who}_{crit}": parse_label(row.get(f"human_{who}_{crit}"), where=f"row {n} ({item})")
                for who in ("a", "b")
                for crit in ("correct", "grounded")
            }
    return out


def _paired(pairs: list[tuple[int | None, int | None]]) -> tuple[list[int], list[int]]:
    both = [(a, b) for a, b in pairs if a is not None and b is not None]
    return [a for a, _ in both], [b for _, b in both]


def score(labels: dict[str, dict[str, int | None]], key: dict[str, dict]) -> dict[str, Any]:
    """Agreement between the judge and the humans, per criterion."""
    result: dict[str, Any] = {"n_items": len(labels), "criteria": {}}
    for crit in ("correct", "grounded"):
        judge = {item: key[item][crit] for item in labels if item in key and key[item][crit] is not None}
        judge = {k: int(v) for k, v in judge.items()}

        def human(who: str, item: str, crit: str = crit) -> int | None:
            return labels[item][f"{who}_{crit}"]

        block: dict[str, Any] = {}
        for who in ("a", "b"):
            j, h = _paired([(judge.get(i), human(who, i)) for i in labels])
            block[f"judge_vs_{who}"] = agreement(j, h)
        j, h = _paired(
            [(judge.get(i), human("a", i) if human("a", i) == human("b", i) else None) for i in labels]
        )
        block["judge_vs_consensus"] = agreement(j, h)
        a, b = _paired([(human("a", i), human("b", i)) for i in labels])
        block["human_a_vs_b"] = agreement(a, b)
        result["criteria"][crit] = block
    return result


def format_score(result: dict[str, Any]) -> str:
    lines = [f"Judge calibration: {result['n_items']} labelled items"]
    worst = 1.0
    for crit, block in result["criteria"].items():
        lines.append(f"  {crit}:")
        for name, label in (
            ("judge_vs_a", "judge vs human A"),
            ("judge_vs_b", "judge vs human B"),
            ("judge_vs_consensus", "judge vs both (when A = B)"),
            ("human_a_vs_b", "human A vs human B"),
        ):
            a = block[name]
            if not a["n"]:
                lines.append(f"    {label:<28} no labels")
                continue
            kappa = "n/a (one label throughout)" if a["kappa"] is None else f"{a['kappa']:.2f}"
            lines.append(
                f"    {label:<28} {a['agree']['value'] * 100:5.1f}% agree "
                f"({a['agree']['k']}/{a['n']}), kappa {kappa}"
            )
            if name.startswith("judge_vs"):
                worst = min(worst, a["agree"]["value"])
    if worst < THRESHOLD:
        lines.append(
            f"  WARNING: judge agreement is below {THRESHOLD * 100:.0f}%. Read the disagreements and fix "
            "prompts/judge_v1.yaml (bump the version) before trusting the judged numbers."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Judge calibration: export a sheet for humans, score the result."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="sample judged answers into a CSV for two people to label")
    ex.add_argument("--run", default="default")
    ex.add_argument("--n", type=int, default=20)
    ex.add_argument("--seed", type=int, default=7)
    ex.add_argument(
        "--out", type=Path, default=DEFAULT_PREFIX, help="path prefix for <prefix>_sheet.csv/_key.json"
    )
    sc = sub.add_parser("score", help="compare the labelled sheet with the judge")
    sc.add_argument("--sheet", type=Path, default=Path(f"{DEFAULT_PREFIX}_sheet.csv"))
    sc.add_argument("--key", type=Path, default=Path(f"{DEFAULT_PREFIX}_key.json"))
    args = ap.parse_args(argv)

    if args.cmd == "export":
        _, records = RunFile(RUNS_DIR / f"{args.run}.jsonl").load()
        items = judged_items(list(records.values()))
        if not items:
            print(f"error: run '{args.run}' has no judged answers yet", file=sys.stderr)
            return 2
        chosen = sample_items(items, args.n, args.seed)
        sheet, key = Path(f"{args.out}_sheet.csv"), Path(f"{args.out}_key.json")
        write_sheet(chosen, sheet, key)
        wrong = sum(1 for i in chosen if not i["judge"]["correct"])
        print(
            f"wrote {len(chosen)} items to {sheet} ({wrong} the judge called wrong).\n"
            f"The judge's answers are in {key}: do not show that file to the labellers."
        )
        if len(chosen) < args.n:
            print(f"note: only {len(chosen)} judged answers exist so far (asked for {args.n})")
        return 0

    try:
        labels = read_sheet(args.sheet)
        key = json.loads(args.key.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = score(labels, key)
    out = Path(f"{args.sheet}".removesuffix("_sheet.csv") + "_result.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(format_score(result))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
