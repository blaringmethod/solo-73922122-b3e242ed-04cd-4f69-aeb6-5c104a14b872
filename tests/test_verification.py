"""Tests for independent-timer leap-second adjudication.

Covers the difference-constraint solver (uniqueness, multiple solutions,
contradictions, 23:59:60 forcing and limits), the persistence/confirmation
workflow (re-check against latest committed data, stored-UTC-text guard,
active-table fallback) and the Flask pages.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from ledger.app import create_app
from ledger.store import Ledger, ValidationError
from ledger.timecode import DEFAULT_LEAP_TABLE, LeapTable, utc_to_real
from ledger.verification import (
    MAX_CANDIDATES,
    MAX_INTERVALS,
    VerificationError,
    adjudicate,
    parse_candidates,
    parse_intervals,
)

ACTIVE = DEFAULT_LEAP_TABLE


def header():
    return "telescope,exposure_id,start,duration_s\n"


def make_ledger(tmp_path, name="verify.sqlite3"):
    return Ledger(tmp_path / name)


# ---------------------------------------------------------------------------
# Parsing and limits
# ---------------------------------------------------------------------------


def test_parse_candidates_validates_range_and_duplicates():
    assert parse_candidates("2020-12-31\n# comment\n\n2019-01-01\n") == (
        dt.date(2019, 1, 1),
        dt.date(2020, 12, 31),
    )
    for bad in ("", "2020-13-01\n", "2020-12-31\n2020-12-31\n", "1999-12-31\n"):
        with pytest.raises(VerificationError):
            parse_candidates(bad)


def test_candidate_limit_enforced():
    text = "\n".join(
        (dt.date(2030, 1, 1) + dt.timedelta(days=i)).isoformat()
        for i in range(MAX_CANDIDATES + 1)
    )
    with pytest.raises(VerificationError, match="120"):
        parse_candidates(text)


def test_parse_intervals_accepts_separators_and_header():
    text = (
        "start,end,measured\n"
        "2020-12-31 00:00:00,2021-01-01 00:00:00,86401\n"
        "2020-12-31 00:00:00;2021-01-01 00:00:00;86401\n"
    )
    intervals = parse_intervals(text)
    assert len(intervals) == 2
    assert intervals[0].measured == 86401


def test_parse_intervals_rejects_bad_rows():
    bad_rows = [
        "2020-12-31 00:00:00,2020-12-31 00:00:00,5\n",       # equal labels
        "2020-12-31 00:00:01,2020-12-31 00:00:00,5\n",       # reversed
        "2020-12-31 00:00:00,2020-12-31 00:00:01,0\n",       # non-positive
        "2020-12-31 00:00:00 2020-12-31 00:00:01 5\n",       # wrong field count
        "2020-12-31 23:58:60,2020-12-31 23:59:00,1\n",       # :60 not at end of day
        "2020-12-31 00:00:00,2020-12-31 00:00:01,x\n",       # non-integer measure
    ]
    for row in bad_rows:
        with pytest.raises(VerificationError):
            parse_intervals(row)


def test_interval_limit_enforced():
    row = "2030-01-01 00:00:00,2030-01-02 00:00:00,86400\n"
    with pytest.raises(VerificationError, match="500"):
        parse_intervals(row * (MAX_INTERVALS + 1))


def test_sixty_label_on_candidate_day_parses_free_of_table():
    intervals = parse_intervals(
        "2030-12-31 23:59:60,2031-01-01 00:00:00,1\n"
    )
    assert intervals[0].start_flag60 is True
    assert intervals[0].start_day == dt.date(2030, 12, 31)


# ---------------------------------------------------------------------------
# Solver verdicts
# ---------------------------------------------------------------------------


def test_unique_solution_detects_new_leap_day():
    result = adjudicate(
        "2020-12-31\n",
        "2020-12-31 00:00:00,2021-01-01 00:00:00,86401\n",
        ACTIVE,
    )
    assert result.status == "unique"
    assert result.verdicts[0].determined is True
    assert result.verdicts[0].value == 1
    assert dt.date(2020, 12, 31) in result.candidate_table.dates
    # non-candidate active leap days are carried into the candidate table
    assert dt.date(2016, 12, 31) in result.candidate_table.dates


def test_unique_solution_can_remove_existing_leap_day():
    result = adjudicate(
        "2016-12-31\n",
        "2016-12-31 00:00:00,2017-01-01 00:00:00,86400\n",
        ACTIVE,
    )
    assert result.status == "unique"
    assert result.verdicts[0].value == 0
    assert dt.date(2016, 12, 31) not in result.candidate_table.dates


def test_multiple_solutions_list_undetermined_dates():
    measured = (dt.date(2022, 1, 1) - dt.date(2020, 12, 31)).days * 86400 + 1
    result = adjudicate(
        "2020-12-31\n2021-06-30\n",
        f"2020-12-31 00:00:00,2022-01-01 00:00:00,{measured}\n",
        ACTIVE,
    )
    assert result.status == "multiple"
    assert set(result.undetermined) == {
        dt.date(2020, 12, 31),
        dt.date(2021, 6, 30),
    }
    assert result.candidate_table is None


def test_no_solution_when_sixty_label_has_no_possible_leap_day():
    result = adjudicate(
        "2020-12-31\n",
        "2022-12-31 23:59:60,2023-01-01 00:00:00,1\n",
        ACTIVE,
    )
    assert result.status == "no_solution"
    assert result.contradictions


def test_sixty_label_forces_candidate_to_leap_day():
    # 23:59:60 -> next 00:00:00 measured 1: civil delta 0, so rhs = 1.
    result = adjudicate(
        "2020-12-31\n",
        "2020-12-31 23:59:60,2021-01-01 00:00:00,1\n",
        ACTIVE,
    )
    assert result.status == "unique"
    assert result.verdicts[0].value == 1


def test_forcing_constraint_contradiction_is_no_solution():
    # :60 forces x=1 but a whole-day measurement of 86400 forces x=0.
    result = adjudicate(
        "2020-12-31\n",
        "2020-12-31 23:59:60,2021-01-01 00:00:00,1\n"
        "2020-12-31 00:00:00,2021-01-01 00:00:00,86400\n",
        ACTIVE,
    )
    assert result.status == "no_solution"


def test_fixed_active_leap_seconds_are_counted():
    # 2016 is an active leap day outside the candidate set; the measured two
    # seconds across 23:59:59 -> 00:00:00 are explained without any candidate.
    result = adjudicate(
        "2020-12-31\n",
        "2016-12-31 23:59:59,2017-01-01 00:00:00,2\n",
        ACTIVE,
    )
    assert result.status == "multiple"
    assert result.verdicts[0].determined is False


def test_solver_matches_independent_truth_table_fuzz():
    import random

    rng = random.Random(7)
    for _ in range(200):
        base = dt.date(2010, 1, 1)
        candidates = sorted(
            {base + dt.timedelta(days=rng.randint(0, 3000)) for _ in range(rng.randint(1, 8))}
        )
        truth_x = {day: rng.randint(0, 1) for day in candidates}
        truth_dates = {
            day for day in ACTIVE.dates if day not in truth_x
        } | {day for day, x in truth_x.items() if x}
        truth = LeapTable(tuple(sorted(truth_dates)))
        lines = []
        for _ in range(rng.randint(1, 10)):
            d1 = base + dt.timedelta(days=rng.randint(0, 3000))
            d2 = d1 + dt.timedelta(days=rng.randint(0, 5))

            def label(day):
                if rng.random() < 0.2 and day in truth.dates:
                    return f"{day.isoformat()} 23:59:60"
                sec = rng.choice([0, 3600, 23 * 3600 + 3599])
                return (
                    dt.datetime.combine(day, dt.time()) + dt.timedelta(seconds=sec)
                ).strftime("%Y-%m-%d %H:%M:%S")

            l1, l2 = label(d1), label(d2)
            r1, r2 = utc_to_real(l1, truth), utc_to_real(l2, truth)
            if r2 > r1:
                lines.append(f"{l1},{l2},{r2 - r1}")
        if not lines:
            continue
        result = adjudicate(
            "\n".join(day.isoformat() for day in candidates) + "\n",
            "\n".join(lines) + "\n",
            ACTIVE,
        )
        assert result.status != "no_solution"
        for verdict in result.verdicts:
            if verdict.determined:
                assert verdict.value == truth_x[verdict.day]
        if result.status == "unique":
            assert set(result.candidate_table.dates) == set(truth_dates)


# ---------------------------------------------------------------------------
# Persistence and confirmation workflow
# ---------------------------------------------------------------------------


def test_run_verification_persists_and_confirm_activates_unique_table(tmp_path):
    ledger = make_ledger(tmp_path)
    meta, result = ledger.run_verification(
        "2020-12-31\n",
        "2020-12-31 00:00:00,2021-01-01 00:00:00,86401\n",
    )
    assert result.status == "unique"
    stored = ledger.get_verification(meta["run_id"])
    assert stored["status"] == "unique" and stored["interval_count"] == 1

    outcome = ledger.confirm_verification(meta["run_id"])
    assert outcome["activated"] is True
    version, table = ledger.active_table()
    assert version == 2 and dt.date(2020, 12, 31) in table.dates
    assert ledger.get_verification(meta["run_id"])["status"] == "confirmed"


def test_confirm_fails_when_stored_utc_text_would_break(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.import_csv(header() + "T1,KEEP,2016-12-31 23:59:60,1\n")
    meta, result = ledger.run_verification(
        "2016-12-31\n",
        "2016-12-31 00:00:00,2017-01-01 00:00:00,86400\n",
    )
    assert result.status == "unique"
    outcome = ledger.confirm_verification(meta["run_id"])
    assert outcome["activated"] is False
    # The then-active table stays in force.
    version, table = ledger.active_table()
    assert version == 1 and dt.date(2016, 12, 31) in table.dates
    assert ledger.get_verification(meta["run_id"])["status"] == "confirm_failed"


def test_confirm_rechecks_latest_committed_data_and_detects_divergence(tmp_path):
    ledger = make_ledger(tmp_path)
    meta, _ = ledger.run_verification(
        "2020-12-31\n",
        "2020-12-31 00:00:00,2021-01-01 00:00:00,86401\n",
    )
    # Manual table swap during the initial adjudication window changes the
    # fixed leap set, so the re-check no longer matches the original verdict.
    ledger.replace_leap_table(
        "2005-12-31\n2008-12-31\n2012-06-30\n2015-06-30\n"
        "2016-12-31\n2021-06-30\n"
    )
    outcome = ledger.confirm_verification(meta["run_id"])
    assert outcome["activated"] is False
    assert "different candidate table" in outcome["message"]
    version, _ = ledger.active_table()
    assert version == 2  # manual swap remains the active table


def test_confirm_still_succeeds_when_latest_data_matches_initial_verdict(tmp_path):
    ledger = make_ledger(tmp_path)
    meta, _ = ledger.run_verification(
        "2020-12-31\n",
        "2020-12-31 00:00:00,2021-01-01 00:00:00,86401\n",
    )
    # An unrelated import lands before confirmation; the re-check is unchanged.
    ledger.import_csv(header() + "T1,NEW,2020-12-30 12:00:00,60\n")
    outcome = ledger.confirm_verification(meta["run_id"])
    assert outcome["activated"] is True
    version, table = ledger.active_table()
    assert version == 2 and dt.date(2020, 12, 31) in table.dates


def test_non_unique_and_unknown_runs_cannot_be_confirmed(tmp_path):
    ledger = make_ledger(tmp_path)
    measured = (dt.date(2022, 1, 1) - dt.date(2020, 12, 31)).days * 86400 + 1
    meta, result = ledger.run_verification(
        "2020-12-31\n2021-06-30\n",
        f"2020-12-31 00:00:00,2022-01-01 00:00:00,{measured}\n",
    )
    assert result.status == "multiple"
    with pytest.raises(ValidationError):
        ledger.confirm_verification(meta["run_id"])
    with pytest.raises(ValidationError):
        ledger.confirm_verification(9999)


def test_verification_history_records_every_run(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.run_verification(
        "2020-12-31\n",
        "2020-12-31 00:00:00,2021-01-01 00:00:00,86401\n",
    )
    ledger.run_verification(
        "2020-12-31\n",
        "2022-12-31 23:59:60,2023-01-01 00:00:00,1\n",
    )
    history = ledger.verification_history()
    assert [row["status"] for row in history] == ["no_solution", "unique"]


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    app = create_app({"TESTING": True, "LEDGER_DB": str(tmp_path / "web.sqlite3")})
    app.secret_key = "test"
    return app.test_client()


def test_web_adjudication_unique_then_confirm(client):
    response = client.post(
        "/leap",
        data={
            "action": "adjudicate",
            "candidate_text": "2020-12-31\n",
            "interval_text": "2020-12-31 00:00:00,2021-01-01 00:00:00,86401\n",
        },
    )
    text = response.get_data(as_text=True)
    assert "唯一解" in text
    match = re.search(r'name="run_id" value="(\d+)"', text)
    assert match
    run_id = match.group(1)

    confirmed = client.post(
        "/leap",
        data={
            "action": "confirm",
            "run_id": run_id,
            "candidate_text": "2020-12-31\n",
            "interval_text": "2020-12-31 00:00:00,2021-01-01 00:00:00,86401\n",
        },
    )
    body = confirmed.get_data(as_text=True)
    assert "确认启用成功" in body and "v2" in body


def test_web_multiple_solutions_list_undetermined(client):
    measured = (dt.date(2031, 1, 1) - dt.date(2030, 1, 1)).days * 86400 + 1
    response = client.post(
        "/leap",
        data={
            "action": "adjudicate",
            "candidate_text": "2030-06-30\n2030-12-31\n",
            "interval_text": f"2030-01-01 00:00:00,2031-01-01 00:00:00,{measured}\n",
        },
    )
    body = response.get_data(as_text=True)
    assert "多解" in body and "不能确定" in body
    assert "2030-06-30" in body and "2030-12-31" in body


def test_web_no_solution_and_bad_input(client):
    body = client.post(
        "/leap",
        data={
            "action": "adjudicate",
            "candidate_text": "2030-06-30\n",
            "interval_text": "2031-12-31 23:59:60,2032-01-01 00:00:00,1\n",
        },
    ).get_data(as_text=True)
    assert "无解" in body

    body = client.post(
        "/leap",
        data={"action": "adjudicate", "candidate_text": "", "interval_text": ""},
    ).get_data(as_text=True)
    assert "核定输入未通过校验" in body


def test_web_interval_template_download(client):
    response = client.get("/interval_template.csv")
    assert response.status_code == 200
    assert b"start_utc,end_utc,measured_seconds" in response.data
