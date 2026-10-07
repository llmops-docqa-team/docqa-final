"""Judge calibration: sampling, sheet and key files, reading labels, agreement, kappa, the 80% warning."""

from __future__ import annotations

import csv
import json

import pytest

from eval import calibrate as cal
from eval.records import RunFile
from tests.evalrecs import rec


def judged_records(n_right=8, n_wrong=4) -> list[dict]:
    out = []
    for i in range(n_right):
        out.append(rec(f"R{i}", judge={"document": (True, True)}))
    for i in range(n_wrong):
        out.append(rec(f"W{i}", judge={"document": (False, False)}))
    out.append(rec("N0"))  # not judged: never offered
    out.append(
        rec(
            "G0",
            route="GENERAL",
            slice="general",
            answerable=None,
            gold_answer="Paris.",
            pages=(),
            status=None,
            general_status="answered",
            judge={"general": (True, None)},
        )
    )
    return out


# ---------------------------------------------------------------- sampling


def test_only_judged_answers_are_offered_with_their_references_and_sources():
    items = cal.judged_items(judged_records())
    assert len(items) == 8 + 4 + 1
    r0 = next(i for i in items if i["item"] == "R0:document")
    assert r0["reference"].startswith("Rs 100.50") and r0["sources"] == ["[p.56] source text of page 56"]
    assert r0["judge"] == {"correct": True, "grounded": True}
    g = next(i for i in items if i["item"] == "G0:general")
    assert g["sources"] == [] and g["judge"]["grounded"] is None


def test_the_sample_is_half_judged_wrong_when_there_are_enough_of_them():
    items = cal.judged_items(judged_records(n_right=30, n_wrong=30))
    chosen = cal.sample_items(items, 20, seed=1)
    assert len(chosen) == 20
    assert sum(1 for i in chosen if not i["judge"]["correct"]) == 10


def test_a_short_side_gives_its_slack_to_the_other_and_the_total_is_capped():
    items = cal.judged_items(judged_records(n_right=30, n_wrong=3))
    chosen = cal.sample_items(items, 20, seed=1)
    assert len(chosen) == 20 and sum(1 for i in chosen if not i["judge"]["correct"]) == 3
    few = cal.sample_items(cal.judged_items(judged_records(2, 1)), 20, seed=1)
    assert len(few) == 4  # 2 + 1 + the general one


def test_the_sample_is_reproducible_and_shuffled():
    items = cal.judged_items(judged_records(30, 30))
    a, b = cal.sample_items(items, 20, seed=7), cal.sample_items(items, 20, seed=7)
    assert [i["item"] for i in a] == [i["item"] for i in b]
    assert [i["item"] for i in a] != [i["item"] for i in cal.sample_items(items, 20, seed=8)]


# ---------------------------------------------------------------- the sheet


def test_the_sheet_hides_the_judges_verdict_and_the_key_holds_it(tmp_path):
    items = cal.sample_items(cal.judged_items(judged_records()), 6, seed=3)
    sheet, key = tmp_path / "s_sheet.csv", tmp_path / "s_key.json"
    cal.write_sheet(items, sheet, key)
    text = sheet.read_text(encoding="utf-8-sig")
    assert "judge" not in text.lower().split("\n")[0]  # no judge column
    with sheet.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["item"] for r in rows] == [i["item"] for i in items]
    assert list(rows[0]) == cal.SHEET_COLUMNS
    assert all(r["human_a_correct"] == "" and r["human_b_grounded"] == "" for r in rows)
    assert all(r["system_answer"] and r["reference_answer"] for r in rows)
    k = json.loads(key.read_text(encoding="utf-8"))
    assert set(k) == {i["item"] for i in items} and all({"correct", "grounded"} == set(v) for v in k.values())


def test_a_general_answer_says_it_has_no_sources(tmp_path):
    items = [i for i in cal.judged_items(judged_records()) if i["item"] == "G0:general"]
    cal.write_sheet(items, tmp_path / "a.csv", tmp_path / "a.json")
    with (tmp_path / "a.csv").open(newline="", encoding="utf-8-sig") as f:
        [row] = list(csv.DictReader(f))
    assert "none" in row["cited_sources"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1", 1),
        ("Y", 1),
        ("yes", 1),
        (" TRUE ", 1),
        ("0", 0),
        ("n", 0),
        ("No", 0),
        ("false", 0),
        ("", None),
        (None, None),
        ("  ", None),
    ],
)
def test_parse_label(raw, expected):
    assert cal.parse_label(raw, where="x") == expected


def test_a_bad_label_names_the_row():
    with pytest.raises(ValueError, match="row 3 .*'maybe'"):
        cal.parse_label("maybe", where="row 3 (R1:document)")


def test_read_sheet_returns_blank_cells_as_none(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text(
        "item,question,human_a_correct,human_a_grounded,human_b_correct,human_b_grounded\n"
        "R1:document,q,1,0,yes,\n,ignored,1,1,1,1\nG1:general,q,0,,0,\n",
        encoding="utf-8-sig",
    )
    got = cal.read_sheet(p)
    assert got == {
        "R1:document": {"a_correct": 1, "a_grounded": 0, "b_correct": 1, "b_grounded": None},
        "G1:general": {"a_correct": 0, "a_grounded": None, "b_correct": 0, "b_grounded": None},
    }


# ---------------------------------------------------------------- agreement


KEY = {
    f"i{n}": {"correct": c, "grounded": g}
    for n, (c, g) in enumerate([(True, True), (True, True), (False, False), (False, True), (True, None)])
}


def labels(a, b):
    out = {}
    for n, ((ac, ag), (bc, bg)) in enumerate(zip(a, b, strict=True)):
        out[f"i{n}"] = {"a_correct": ac, "a_grounded": ag, "b_correct": bc, "b_grounded": bg}
    return out


def test_score_computes_agreement_per_criterion_and_per_rater():
    a = [(1, 1), (1, 1), (0, 0), (1, 1), (1, None)]  # human A disagrees with the judge on item 3 (correct)
    b = [(1, 1), (1, 0), (0, 0), (0, 1), (1, None)]  # human B disagrees on item 1 (grounded)
    res = cal.score(labels(a, b), KEY)
    c = res["criteria"]["correct"]
    assert c["judge_vs_a"]["n"] == 5 and c["judge_vs_a"]["agree"]["k"] == 4
    assert c["judge_vs_b"]["agree"]["k"] == 5 and c["judge_vs_b"]["kappa"] == 1.0
    # when A and B agree (items 0, 1, 2, 4) the judge matches them on all four
    assert c["judge_vs_consensus"]["n"] == 4 and c["judge_vs_consensus"]["agree"]["k"] == 4
    assert c["human_a_vs_b"]["agree"]["k"] == 4
    g = res["criteria"]["grounded"]
    assert g["judge_vs_a"]["n"] == 4  # item 4 has no grounded verdict (a general answer)
    assert g["judge_vs_b"]["agree"]["k"] == 3 and g["human_a_vs_b"]["agree"]["k"] == 3


def test_a_person_who_has_not_labelled_yet_gives_no_numbers_not_an_error():
    res = cal.score(labels([(1, 1)] * 5, [(None, None)] * 5), KEY)
    assert res["criteria"]["correct"]["judge_vs_b"]["n"] == 0
    assert res["criteria"]["correct"]["human_a_vs_b"]["n"] == 0
    assert "no labels" in cal.format_score(res)


def test_the_report_warns_below_80_percent_and_not_above():
    bad = cal.score(labels([(0, 0)] * 5, [(0, 0)] * 5), KEY)  # humans say wrong where the judge said right
    assert "below 80%" in cal.format_score(bad)
    good = cal.score(
        labels([(1, 1), (1, 1), (0, 0), (0, 1), (1, None)], [(1, 1), (1, 1), (0, 0), (0, 1), (1, None)]), KEY
    )
    text = cal.format_score(good)
    assert "WARNING" not in text and "100.0% agree" in text
    assert "kappa 1.00" in text


def test_kappa_is_reported_as_undefined_when_everyone_used_one_label():
    key = {f"i{n}": {"correct": True, "grounded": True} for n in range(4)}
    res = cal.score(
        {f"i{n}": {"a_correct": 1, "a_grounded": 1, "b_correct": 1, "b_grounded": 1} for n in range(4)}, key
    )
    assert res["criteria"]["correct"]["judge_vs_a"]["kappa"] is None
    assert "n/a (one label throughout)" in cal.format_score(res)


# ---------------------------------------------------------------- the commands


def test_export_then_score_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cal, "RUNS_DIR", tmp_path / "runs")
    rf = RunFile(tmp_path / "runs" / "default.jsonl")
    for r in judged_records():
        rf.append(r)
    prefix = tmp_path / "cal"
    assert cal.main(["export", "--n", "8", "--seed", "1", "--out", str(prefix)]) == 0
    out = capsys.readouterr().out
    assert "do not show that file to the labellers" in out
    sheet, key = tmp_path / "cal_sheet.csv", tmp_path / "cal_key.json"
    assert sheet.exists() and key.exists()

    # two labellers who always agree with the judge
    judge = json.loads(key.read_text(encoding="utf-8"))
    with sheet.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        v = judge[r["item"]]
        for who in "ab":
            r[f"human_{who}_correct"] = str(int(v["correct"]))
            r[f"human_{who}_grounded"] = "" if v["grounded"] is None else str(int(v["grounded"]))
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cal.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert cal.main(["score", "--sheet", str(sheet), "--key", str(key)]) == 0
    text = capsys.readouterr().out
    assert "judge vs human A" in text and "100.0% agree" in text and "WARNING" not in text
    assert (tmp_path / "cal_result.json").exists()


def test_export_without_judged_answers_is_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cal, "RUNS_DIR", tmp_path / "runs")
    RunFile(tmp_path / "runs" / "default.jsonl").append(rec("N0"))
    assert cal.main(["export", "--out", str(tmp_path / "cal")]) == 2
    assert "no judged answers" in capsys.readouterr().err


def test_score_with_a_missing_file_is_an_error(tmp_path, capsys):
    assert (
        cal.main(["score", "--sheet", str(tmp_path / "nope.csv"), "--key", str(tmp_path / "nope.json")]) == 2
    )
    assert "error" in capsys.readouterr().err
