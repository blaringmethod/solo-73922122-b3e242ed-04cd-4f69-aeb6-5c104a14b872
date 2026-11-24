"""Leap-second adjudication from independent timer pulse intervals.

The survey team hands in up to :data:`MAX_CANDIDATES` dates whose leap-second
status is unknown plus up to :data:`MAX_INTERVALS` pulse intervals.  Each
interval gives two UTC labels (``23:59:60`` allowed on a leap day) and the
integer number of real seconds an independent timer measured between them.

Mathematics
-----------
For a civil label ``t`` define its civil ordinal ``o(t)`` as the number of
nominal UTC seconds from the epoch, counting ``23:59:60`` as one extra ordinal
(a ``:60`` label parses to ``23:59:59`` with a leap flag, so
``o = ordinal(datetime) + flag``).  With a leap table the true elapsed real
seconds between ``t1`` and ``t2`` are

    m = o(t2) - o(t1) + number of positive leap seconds in the half-open
        civil span [date(t1), date(t2)).

A leap second inserted at the end of day ``d`` is counted exactly when
``date(t1) <= d < date(t2)``: the interval starting at ``d 23:59:60`` and
ending at ``d+1 00:00:00`` still contains that leap second.

Leap days already fixed by the active table (and not under adjudication)
contribute a known term; the candidate dates are binary variables
``x_j in {0, 1}``.  Every interval becomes one linear equation

    sum(x_j for candidate day j inside [date1, date2)) = b_i

and every ``23:59:60`` label on a candidate day forces ``x_j = 1``.  The
resulting 0/1 system is solved through prefix sums and difference
constraints: with ``s_k = x_1 + ... + x_k`` (``s_0 = 0``) every equation is
of the form ``s_v - s_u = w`` plus the chain ``0 <= s_k - s_{k-1} <= 1``.
All-pairs shortest paths (Floyd-Warshall) decide feasibility (a negative
cycle means no solution) and give tight per-variable bounds; a variable is
determined only when both bounds agree.
"""

from __future__ import annotations

import bisect
import datetime as _dt
from dataclasses import dataclass, field

from .timecode import (
    DATE_RE,
    EARLIEST_DAY,
    EPOCH,
    LATEST_DAY,
    LeapTable,
    TimecodeError,
    normalize_utc_text,
    parse_utc,
)

MAX_CANDIDATES = 120
MAX_INTERVALS = 500
MAX_MEASURED_DIGITS = 19


class VerificationError(ValueError):
    """Raised when an adjudication submission cannot be processed."""


@dataclass(frozen=True)
class PulseInterval:
    """One measured interval, kept for display and later re-checks."""

    start_text: str
    end_text: str
    measured: int
    start_dt: _dt.datetime
    end_dt: _dt.datetime
    start_day: _dt.date
    end_day: _dt.date
    start_flag60: bool
    end_flag60: bool


@dataclass(frozen=True)
class CandidateVerdict:
    """The adjudication verdict for one candidate date."""

    day: _dt.date
    determined: bool
    value: int  # 1 = positive leap day, 0 = ordinary day; valid when determined


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of solving one adjudication submission against one table."""

    status: str  # "no_solution" | "multiple" | "unique"
    candidates: tuple[_dt.date, ...]
    intervals: tuple[PulseInterval, ...]
    fixed_leap_dates: tuple[_dt.date, ...]
    verdicts: tuple[CandidateVerdict, ...] = field(default_factory=tuple)
    contradictions: tuple[str, ...] = field(default_factory=tuple)
    candidate_table: LeapTable | None = None
    undetermined: tuple[_dt.date, ...] = field(default_factory=tuple)

    @property
    def is_unique(self) -> bool:
        return self.status == "unique"


# Labels are parsed "free" of any leap table: a 23:59:60 on a candidate day is
# legal even before the adjudication confirms that day.  Non-:60 labels use the
# ordinary parser, which never consults the table for seconds 0..59.
_FREE_LABEL_TABLE = LeapTable(())


# ---------------------------------------------------------------------------
# Submission parsing
# ---------------------------------------------------------------------------


def _split_line(line: str) -> list[str]:
    # Accept comma, semicolon or tab separated fields; plain spaces alone are
    # never separators because the UTC labels themselves contain spaces.
    for sep in (",", ";", "\t"):
        if sep in line:
            return [part.strip() for part in line.split(sep)]
    return [part.strip() for part in line.split()]


def _is_integer(token: str) -> bool:
    return bool(token) and (
        token.isdigit() or (token[0] in "+-" and token[1:].isdigit())
    )


def parse_candidates(text: str) -> tuple[_dt.date, ...]:
    """Parse the candidate date list (one ``YYYY-MM-DD`` per line)."""

    if not isinstance(text, str) or not text.strip():
        raise VerificationError("请至少填写一个待查日期。")
    days: list[_dt.date] = []
    seen: set[_dt.date] = set()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if len(days) >= MAX_CANDIDATES:
            raise VerificationError(
                f"待查日期最多 {MAX_CANDIDATES} 个（第 {lineno} 行超出）。"
            )
        if not DATE_RE.match(line):
            raise VerificationError(
                f"待查日期第 {lineno} 行：应为 YYYY-MM-DD，实际为 {raw.strip()!r}。"
            )
        try:
            day = _dt.date.fromisoformat(line)
        except ValueError as exc:
            raise VerificationError(
                f"待查日期第 {lineno} 行：日期 {line!r} 不存在。"
            ) from exc
        if day < EARLIEST_DAY or day > LATEST_DAY:
            raise VerificationError(
                f"待查日期第 {lineno} 行：{line} 超出台账支持范围 "
                f"[{EARLIEST_DAY.isoformat()}, {LATEST_DAY.isoformat()}]。"
            )
        if day in seen:
            raise VerificationError(f"待查日期第 {lineno} 行：{line} 重复。")
        seen.add(day)
        days.append(day)
    if not days:
        raise VerificationError("请至少填写一个待查日期。")
    return tuple(sorted(days))


def _parse_label(token: str, line_no: int, which: str) -> tuple[_dt.datetime, bool, _dt.date, str]:
    try:
        normalized = normalize_utc_text(token)
    except TimecodeError as exc:
        raise VerificationError(
            f"脉冲间隔第 {line_no} 行：{which}时刻 {token!r} 无法识别（{exc}）。"
        ) from exc
    flag60 = normalized.endswith(":60")
    if flag60:
        date_part, time_part = normalized.split(" ")
        hour, minute, second = (int(part) for part in time_part.split(":"))
        if (hour, minute, second) != (23, 59, 60):
            raise VerificationError(
                f"脉冲间隔第 {line_no} 行：{which}时刻 {token!r} 的 :60 只能出现在 23:59:60。"
            )
        try:
            day = _dt.date.fromisoformat(date_part)
            dt_value = _dt.datetime(day.year, day.month, day.day, 23, 59, 59)
        except ValueError as exc:
            raise VerificationError(
                f"脉冲间隔第 {line_no} 行：{which}时刻 {token!r} 的日期不存在。"
            ) from exc
    else:
        try:
            dt_value = parse_utc(normalized, _FREE_LABEL_TABLE)
        except TimecodeError as exc:
            raise VerificationError(
                f"脉冲间隔第 {line_no} 行：{which}时刻 {token!r} 无法识别（{exc}）。"
            ) from exc
        day = dt_value.date()
    return dt_value, flag60, day, normalized


def parse_intervals(text: str) -> tuple[PulseInterval, ...]:
    """Parse up to 500 pulse interval lines.

    Each line holds three fields: two UTC labels and one measured integer
    second count.  An optional header line whose third field is not an integer
    is skipped.
    """

    if not isinstance(text, str) or not text.strip():
        raise VerificationError("请至少填写一条脉冲间隔。")
    intervals: list[PulseInterval] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = _split_line(line)
        if len(fields) != 3:
            raise VerificationError(
                f"脉冲间隔第 {lineno} 行：需要 3 个字段（开始时刻, 结束时刻, 实测秒数），"
                f"实际 {len(fields)} 个。"
            )
        start_tok, end_tok, measure_tok = fields
        if not _is_integer(measure_tok):
            # A header row is accepted before any data row.
            if not intervals:
                continue
            raise VerificationError(
                f"脉冲间隔第 {lineno} 行：实测秒数必须是整数，实际为 {measure_tok!r}。"
            )
        if len(intervals) >= MAX_INTERVALS:
            raise VerificationError(
                f"脉冲间隔最多 {MAX_INTERVALS} 条（第 {lineno} 行超出）。"
            )
        if len(measure_tok.lstrip("+-")) > MAX_MEASURED_DIGITS:
            raise VerificationError(f"脉冲间隔第 {lineno} 行：实测秒数位数过多。")
        measured = int(measure_tok)
        if measured <= 0:
            raise VerificationError(
                f"脉冲间隔第 {lineno} 行：实测秒数必须为正整数，实际为 {measured}。"
            )
        start_dt, start_flag, start_day, start_text = _parse_label(
            start_tok, lineno, "开始"
        )
        end_dt, end_flag, end_day, end_text = _parse_label(
            end_tok, lineno, "结束"
        )
        if start_day < EARLIEST_DAY or end_day > LATEST_DAY:
            raise VerificationError(
                f"脉冲间隔第 {lineno} 行：时刻超出台账支持范围 "
                f"[{EARLIEST_DAY.isoformat()}, {LATEST_DAY.isoformat()}]。"
            )
        # Civil label ordering: a :60 label ticks after 23:59:59 and before
        # the next day's 00:00:00.  Ordinal values can be equal across that
        # boundary (the leap term supplies the elapsed second), so compare the
        # (datetime, leap-flag) label pair instead.
        if (end_dt, end_flag) <= (start_dt, start_flag):
            raise VerificationError(
                f"脉冲间隔第 {lineno} 行：结束时刻必须晚于开始时刻（按 UTC 民用时钟次序）。"
            )
        intervals.append(
            PulseInterval(
                start_text=start_text,
                end_text=end_text,
                measured=measured,
                start_dt=start_dt,
                end_dt=end_dt,
                start_day=start_day,
                end_day=end_day,
                start_flag60=start_flag,
                end_flag60=end_flag,
            )
        )
    if not intervals:
        raise VerificationError("请至少填写一条脉冲间隔。")
    return tuple(intervals)


def _civil_ordinal(dt_value: _dt.datetime, flag60: bool) -> int:
    """Nominal UTC seconds from the epoch, counting ``23:59:60`` itself."""

    return int((dt_value - EPOCH).total_seconds()) + (1 if flag60 else 0)


def _label_ordinal(iv: PulseInterval, start: bool) -> int:
    if start:
        return _civil_ordinal(iv.start_dt, iv.start_flag60)
    return _civil_ordinal(iv.end_dt, iv.end_flag60)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _fixed_leaps_between(
    day_a: _dt.date, day_b: _dt.date, active: LeapTable, candidate_set: set[_dt.date]
) -> int:
    """Active-table leap days in the half-open civil span ``[day_a, day_b)``."""

    dates = active.dates
    lo = bisect.bisect_left(dates, day_a)
    hi = bisect.bisect_left(dates, day_b)
    return sum(1 for idx in range(lo, hi) if dates[idx] not in candidate_set)


def adjudicate(
    candidate_text: str, interval_text: str, active: LeapTable
) -> VerificationResult:
    """Solve the adjudication submission against the active leap table.

    Only the candidate dates may gain (or keep) a positive leap second; every
    other date keeps its active-table status.  Returns the verdict, including
    the unique candidate table when the data admits exactly one solution.
    """

    candidates = parse_candidates(candidate_text)
    intervals = parse_intervals(interval_text)
    candidate_set = set(candidates)
    n = len(candidates)

    fixed_tuple = tuple(
        sorted(day for day in active.dates if day not in candidate_set)
    )

    contradictions: list[str] = []
    # Edges of the difference-constraint graph over prefix sums s_0..s_n.
    # An edge (u -> v, w) encodes s_v - s_u <= w.
    edges: list[tuple[int, int, int, str]] = []

    # Binary chain: x_k = s_k - s_{k-1} in {0, 1}.
    for k in range(1, n + 1):
        edges.append((k - 1, k, 1, "候选日期取值 0/1"))
        edges.append((k, k - 1, 0, "候选日期取值 0/1"))

    def cand_index(day: _dt.date) -> int:
        return bisect.bisect_left(candidates, day)

    # Forcing constraints from 23:59:60 labels on candidate days.
    for line_no, iv in enumerate(intervals, 1):
        for token, day, flag in (
            (iv.start_text, iv.start_day, iv.start_flag60),
            (iv.end_text, iv.end_day, iv.end_flag60),
        ):
            if not flag:
                continue
            if day in candidate_set:
                j = cand_index(day) + 1  # x_j uses prefix indices j-1 and j
                reason = f"第 {line_no} 行 {token} 要求该日为闰秒日"
                edges.append((j, j - 1, -1, reason))
                edges.append((j - 1, j, 1, reason))
            elif day not in active:
                contradictions.append(
                    f"第 {line_no} 行：{token} 的 23:59:60 所在日期 {day.isoformat()} "
                    "既不在待查日期中，也不是现行表闰秒日，任何候选表都无法使其有效。"
                )

    # Interval equations.
    for line_no, iv in enumerate(intervals, 1):
        civil_delta = _label_ordinal(iv, start=False) - _label_ordinal(iv, start=True)
        fixed_term = _fixed_leaps_between(
            iv.start_day, iv.end_day, active, candidate_set
        )
        rhs = iv.measured - civil_delta - fixed_term
        lo = cand_index(iv.start_day)
        hi = cand_index(iv.end_day)  # half-open [start_day, end_day)
        in_window = hi - lo
        if in_window == 0:
            if rhs != 0:
                contradictions.append(
                    f"第 {line_no} 行：区间 [{iv.start_text} → {iv.end_text}] "
                    f"不含任何待查日期，固定项要求实测 {civil_delta + fixed_term} 秒，"
                    f"实际填写 {iv.measured} 秒，无法满足。"
                )
            continue
        if rhs < 0 or rhs > in_window:
            contradictions.append(
                f"第 {line_no} 行：区间内只有 {in_window} 个待查日期，"
                f"却需要其中 {rhs} 个为闰秒日，无解。"
            )
            continue
        reason = f"第 {line_no} 行脉冲间隔"
        edges.append((lo, hi, rhs, reason))      # s_hi - s_lo <= rhs
        edges.append((hi, lo, -rhs, reason))     # s_lo - s_hi <= -rhs

    if contradictions:
        return VerificationResult(
            status="no_solution",
            candidates=candidates,
            intervals=intervals,
            fixed_leap_dates=fixed_tuple,
            contradictions=tuple(contradictions),
        )

    feasible, dist = _floyd(n + 1, edges)
    if not feasible:
        witness = _negative_cycle_witness(n + 1, edges)
        contradictions.append(
            "约束之间互相矛盾（差分约束出现负环），不存在候选表。"
            + (f" 相关约束：{witness}" if witness else "")
        )
        return VerificationResult(
            status="no_solution",
            candidates=candidates,
            intervals=intervals,
            fixed_leap_dates=fixed_tuple,
            contradictions=tuple(contradictions),
        )

    # Tight bounds with s_0 = 0.  For x_j = s_j - s_{j-1}:
    #   max over feasible assignments of (s_j - s_{j-1}) = dist[j-1][j]
    #   min = -dist[j][j-1]
    verdicts: list[CandidateVerdict] = []
    undetermined: list[_dt.date] = []
    chosen: list[_dt.date] = []
    for j in range(1, n + 1):
        x_low = -dist[j][j - 1]
        x_high = dist[j - 1][j]
        day = candidates[j - 1]
        if x_low == x_high:
            verdicts.append(CandidateVerdict(day, True, x_low))
            if x_low == 1:
                chosen.append(day)
        else:
            verdicts.append(CandidateVerdict(day, False, 0))
            undetermined.append(day)

    if undetermined:
        return VerificationResult(
            status="multiple",
            candidates=candidates,
            intervals=intervals,
            fixed_leap_dates=fixed_tuple,
            verdicts=tuple(verdicts),
            undetermined=tuple(undetermined),
        )

    candidate_table = LeapTable(tuple(sorted(set(fixed_tuple) | set(chosen))))
    return VerificationResult(
        status="unique",
        candidates=candidates,
        intervals=intervals,
        fixed_leap_dates=fixed_tuple,
        verdicts=tuple(verdicts),
        candidate_table=candidate_table,
    )


def _floyd(
    size: int, edges: list[tuple[int, int, int, str]]
) -> tuple[bool, list[list[int]]]:
    """All-pairs shortest paths; ``False`` when a negative cycle exists."""

    inf = 10**18
    dist = [[inf] * size for _ in range(size)]
    for i in range(size):
        dist[i][i] = 0
    for u, v, w, _label in edges:
        if w < dist[u][v]:
            dist[u][v] = w
    for k in range(size):
        dk = dist[k]
        for i in range(size):
            di = dist[i]
            dik = di[k]
            if dik == inf:
                continue
            for j in range(size):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd
        if any(dist[i][i] < 0 for i in range(size)):
            return False, dist
    return True, dist


def _negative_cycle_witness(
    size: int, edges: list[tuple[int, int, int, str]]
) -> str:
    """Bellman-Ford from a virtual source to name constraints on a bad cycle."""

    dist = [0] * size
    parent: list[int | None] = [None] * size
    parent_edge: list[str | None] = [None] * size
    relaxed_node = -1
    for _ in range(size):
        relaxed_node = -1
        for u, v, w, label in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                parent_edge[v] = label
                relaxed_node = v
    if relaxed_node == -1:
        return ""
    node = relaxed_node
    for _ in range(size):  # walk back onto the cycle
        node = parent[node]  # type: ignore[assignment]
    cycle_labels: list[str] = []
    cur = node
    seen: set[int] = set()
    while cur not in seen and cur is not None:
        seen.add(cur)
        label = parent_edge[cur]
        if label and label not in cycle_labels:
            cycle_labels.append(label)
        cur = parent[cur]  # type: ignore[assignment]
    return "；".join(cycle_labels[:6])


def intervals_to_text(intervals: tuple[PulseInterval, ...]) -> str:
    """Serialise parsed intervals back to the textarea storage format."""

    return "".join(
        f"{iv.start_text},{iv.end_text},{iv.measured}\n" for iv in intervals
    )


def candidates_to_text(days: tuple[_dt.date, ...]) -> str:
    return "".join(f"{day.isoformat()}\n" for day in days)
