"""Flask web app for the exposure ledger."""

from __future__ import annotations

import csv
import datetime as _dt
import io
import os
from pathlib import Path

from flask import (
    Flask,
    Response,
    redirect,
    render_template,
    request,
    url_for,
)

from .store import (
    CSV_FIELDS,
    Ledger,
    LedgerError,
    LeapSwapRejected,
    ValidationError,
)
from .timecode import (
    TimecodeError,
    format_leap_table,
    parse_date,
)
from .verification import VerificationError

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _csv_report(report) -> str:
    """Render one day report as a CSV attachment (raw values preserved)."""

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "telescope",
            "exposure_id",
            "start_utc",
            "end_utc_exclusive",
            "seconds",
            "source_start",
            "source_duration_s",
        ]
    )
    for row in report.rows:
        writer.writerow(
            [
                row["telescope"],
                row["exposure_id"],
                row["window_start_utc"],
                row["window_end_utc"],
                row["window_seconds"],
                row["source_start"],
                row["source_duration_s"],
            ]
        )
    return buffer.getvalue()


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.secret_key = os.environ.get("LEDGER_SECRET", "dev-only-change-me")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["LEDGER_DB"] = os.environ.get(
        "LEDGER_DB", str(Path(app.root_path).parent / "var" / "ledger.sqlite3")
    )
    if config:
        app.config.update(config)
    app.ledger = Ledger(app.config["LEDGER_DB"])
    app.jinja_env.filters["leap_table_text"] = format_leap_table

    @app.get("/")
    def index():
        return redirect(url_for("day_view"))

    @app.get("/day")
    def day_view():
        raw_day = request.args.get("date", "").strip()
        day = None
        report = None
        error = None
        if raw_day:
            try:
                day = parse_date(raw_day)
                report = app.ledger.day_report(day)
            except (TimecodeError, ValidationError) as exc:
                error = str(exc)
        else:
            today = _dt.datetime.now(_dt.timezone.utc).date()
            raw_day = today.isoformat()
        return render_template(
            "day.html",
            active="day",
            raw_day=raw_day,
            error=error,
            report=report,
            stats=app.ledger.stats(),
            leap_version=app.ledger.active_table()[0],
        )

    @app.get("/day.csv")
    def day_csv():
        raw_day = request.args.get("date", "").strip()
        try:
            day = parse_date(raw_day)
            report = app.ledger.day_report(day)
        except (TimecodeError, ValidationError) as exc:
            return Response(str(exc), status=400, mimetype="text/plain")
        body = _csv_report(report)
        filename = f"exposures_{report.day.isoformat()}.csv"
        return Response(
            body,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.route("/import", methods=("GET", "POST"))
    def import_view():
        result = None
        errors: list[str] = []
        message = None
        if request.method == "POST":
            uploaded = request.files.get("csv_file")
            pasted = request.form.get("csv_text", "")
            source_name = "pasted.csv"
            text = pasted
            if uploaded is not None and uploaded.filename:
                source_name = os.path.basename(uploaded.filename)
                raw = uploaded.read()
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    errors.append("the uploaded file is not valid UTF-8")
                if len(raw) > MAX_UPLOAD_BYTES:
                    errors.append("the uploaded file is larger than 5 MiB")
            if not errors:
                try:
                    result = app.ledger.import_csv(text, source_name=source_name)
                except ValidationError as exc:
                    message = str(exc)
                    errors = exc.details
                except LedgerError as exc:
                    message = str(exc)
        return render_template(
            "import.html",
            active="import",
            result=result,
            errors=errors,
            message=message,
            stats=app.ledger.stats(),
            leap_version=app.ledger.active_table()[0],
            csv_fields=",".join(CSV_FIELDS),
        )

    @app.get("/template.csv")
    def template_csv():
        sample = (
            "telescope,exposure_id,start,duration_s\n"
            "T1,EXP-0001,2026-11-24 00:00:00,600\n"
            "T2,EXP-0002,2026-11-24 12:30:00,1200\n"
            "T3,EXP-0003,848793605,900\n"
        )
        return Response(
            sample,
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="exposures_template.csv"'
            },
        )

    @app.route("/leap", methods=("GET", "POST"))
    def leap_view():
        offending: list[dict] | None = None
        result = None
        message = None
        if request.method == "POST":
            action = request.form.get("action", "replace")
            if action == "adjudicate":
                return _adjudication_response()
            if action == "confirm":
                return _confirmation_response()
            text = request.form.get("leap_text", "")
            try:
                result = app.ledger.replace_leap_table(text)
            except LeapSwapRejected as exc:
                offending = exc.offending
                message = str(exc)
            except ValidationError as exc:
                message = str(exc)
                offending = [{"error": detail} for detail in exc.details]
            except LedgerError as exc:
                message = str(exc)
        return _render_leap(
            offending=offending, result=result, message=message
        )

    def _render_leap(
        offending=None,
        result=None,
        message=None,
        verification=None,
        verify_error=None,
        verify_run_id=None,
        form_candidates="",
        form_intervals="",
        confirm_message=None,
        confirm_ok=None,
        confirm_outcome=None,
    ):
        version, table = app.ledger.active_table()
        return render_template(
            "leap.html",
            active="leap",
            version=version,
            table=table,
            table_text=format_leap_table(table),
            history=app.ledger.leap_history(),
            offending=offending,
            result=result,
            message=message,
            verification=verification,
            verify_error=verify_error,
            verify_run_id=verify_run_id,
            form_candidates=form_candidates,
            form_intervals=form_intervals,
            confirm_message=confirm_message,
            confirm_ok=confirm_ok,
            confirm_outcome=confirm_outcome,
            verification_history=app.ledger.verification_history(),
            stats=app.ledger.stats(),
        )

    def _adjudication_response():
        candidate_text = request.form.get("candidate_text", "")
        interval_text = request.form.get("interval_text", "")
        try:
            meta, verification = app.ledger.run_verification(
                candidate_text, interval_text
            )
        except VerificationError as exc:
            return _render_leap(
                verify_error=str(exc),
                form_candidates=candidate_text,
                form_intervals=interval_text,
            )
        return _render_leap(
            verification=verification,
            verify_run_id=meta["run_id"],
            form_candidates=candidate_text,
            form_intervals=interval_text,
        )

    def _confirmation_response():
        raw_id = request.form.get("run_id", "").strip()
        candidate_text = request.form.get("candidate_text", "")
        interval_text = request.form.get("interval_text", "")
        if not raw_id.isdigit():
            return _render_leap(
                confirm_message="确认失败：缺少有效的核定编号。",
                confirm_ok=False,
                form_candidates=candidate_text,
                form_intervals=interval_text,
            )
        try:
            outcome = app.ledger.confirm_verification(int(raw_id))
        except ValidationError as exc:
            return _render_leap(
                confirm_message="；".join(exc.details) or str(exc),
                confirm_ok=False,
                form_candidates=candidate_text,
                form_intervals=interval_text,
            )
        return _render_leap(
            confirm_message=outcome["message"],
            confirm_ok=outcome["activated"],
            confirm_outcome=outcome,
            form_candidates=candidate_text,
            form_intervals=interval_text,
        )

    @app.get("/interval_template.csv")
    def interval_template_csv():
        sample = (
            "start_utc,end_utc,measured_seconds\n"
            "2020-12-31 00:00:00,2021-01-01 00:00:00,86401\n"
            "2020-12-31 23:59:59,2021-01-01 00:00:00,2\n"
        )
        return Response(
            sample,
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="pulse_interval_template.csv"'
            },
        )

    @app.errorhandler(413)
    def too_large(_exc):
        return (
            render_template(
                "import.html",
                active="import",
                result=None,
                errors=["the upload is larger than 5 MiB"],
                message="upload rejected",
                stats=app.ledger.stats(),
                leap_version=app.ledger.active_table()[0],
                csv_fields=",".join(CSV_FIELDS),
            ),
            413,
        )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "ledger": app.ledger.stats()}

    return app
