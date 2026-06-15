from __future__ import annotations

import sys
import uuid
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.validation_design_agent import (  # noqa: E402
    _load_demo_data,
    _write_cv_benchmark_csv,
    build_plan,
)
from validation_design import (  # noqa: E402
    evaluate_cv_designs,
    generate_split_code,
    write_json_plan,
    write_markdown_report,
)

app = FastAPI(title="Sentinelle Validation Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACT_ROOT = ROOT / "artifacts" / "web_runs"
RUN_ID_RE = re.compile(r"^[a-f0-9]{12}$")
ALLOWED_ARTIFACTS = {
    "validation_design_report.md",
    "validation_design_report.pdf",
    "validation_design_plan.json",
    "split_code.py",
    "cv_benchmark_results.csv",
}
ARTIFACT_MEDIA_TYPES = {
    "validation_design_report.pdf": "application/pdf",
    "validation_design_report.md": "text/markdown",
    "validation_design_plan.json": "application/json",
    "split_code.py": "text/x-python",
    "cv_benchmark_results.csv": "text/csv",
}


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_preview(df: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    preview = df.head(n).copy()
    preview = preview.where(pd.notnull(preview), None)
    return preview.to_dict(orient="records")


def detect_submission_like_table(df: pd.DataFrame) -> str | None:
    cols = [str(c).lower() for c in df.columns]
    col_set = set(cols)

    has_row_id = "row_id" in col_set
    has_few_columns = len(cols) <= 3
    prediction_like = any(
        name in col_set
        for name in {
            "prediction",
            "pred",
            "target",
            "rate_per_10000_ed_visits",
            "label",
            "y",
        }
    )

    if has_row_id and has_few_columns and prediction_like:
        return (
            "This looks like a submission-style file with only an identifier and prediction/target column. "
            "Sentinelle works best with a training table that includes feature columns and, when available, the target column."
        )

    if has_row_id and has_few_columns:
        return (
            "This table has very few columns and includes row_id. "
            "If this is a submission file, upload a training table with feature columns instead."
        )

    return None


def _clean_inline_markdown(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return text.strip()


def _is_markdown_separator_row(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell or "") is not None for cell in cells)


def _split_markdown_row(line: str) -> list[str]:
    return [_clean_inline_markdown(cell) for cell in line.strip().strip("|").split("|")]


def _collect_markdown_table(lines: list[str], start: int) -> tuple[list[list[str]] | None, int]:
    table_lines: list[str] = []
    i = start

    while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i].rstrip())
        i += 1

    if len(table_lines) < 2 or not _is_markdown_separator_row(table_lines[1]):
        return None, start

    rows = [_split_markdown_row(table_lines[0])]
    rows.extend(_split_markdown_row(line) for line in table_lines[2:])

    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    return normalized, i


def _make_pdf_table(rows: list[list[str]], doc: SimpleDocTemplate, styles) -> Table:
    max_cols = max(len(row) for row in rows)
    usable_width = doc.width

    if max_cols <= 4:
        col_widths = [usable_width / max_cols] * max_cols
    else:
        base = usable_width / (max_cols + 1)
        col_widths = [base] * (max_cols - 1) + [base * 2]

    body_style = ParagraphStyle(
        name="SentinelleTableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#374151"),
    )
    header_style = ParagraphStyle(
        name="SentinelleTableHeader",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#111827"),
    )

    formatted_rows = []
    for row_idx, row in enumerate(rows):
        style = header_style if row_idx == 0 else body_style
        formatted_rows.append([Paragraph(escape(cell), style) for cell in row])

    table = Table(
        formatted_rows,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=True,
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
            ]
        )
    )
    return table


def write_pdf_report(markdown_text: str, output_path: Path) -> None:
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="SentinelleTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#111827"),
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SentinelleHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SentinelleBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#374151"),
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SentinelleMono",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#374151"),
            backColor=colors.HexColor("#F3F4F6"),
            borderColor=colors.HexColor("#E5E7EB"),
            borderWidth=0.5,
            borderPadding=6,
            spaceAfter=8,
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="Sentinelle Validation Report",
    )

    story = [
        Paragraph("Sentinelle Validation Report", styles["SentinelleTitle"]),
        Spacer(1, 0.08 * inch),
    ]

    lines = markdown_text.splitlines()
    in_code_block = False
    code_lines: list[str] = []
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            if in_code_block:
                code_text = "<br/>".join(escape(x) for x in code_lines) or " "
                story.append(Paragraph(code_text, styles["SentinelleMono"]))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        if line.strip().startswith("|"):
            table_rows, next_i = _collect_markdown_table(lines, i)
            if table_rows is not None:
                story.append(_make_pdf_table(table_rows, doc, styles))
                story.append(Spacer(1, 0.10 * inch))
                i = next_i
                continue

        if not line.strip():
            story.append(Spacer(1, 0.05 * inch))
            i += 1
            continue

        if line.startswith("#"):
            heading = _clean_inline_markdown(line.lstrip("#").strip())
            if heading:
                story.append(Paragraph(escape(heading), styles["SentinelleHeading"]))
            i += 1
            continue

        if line.startswith("- "):
            story.append(
                Paragraph("• " + escape(_clean_inline_markdown(line[2:].strip())), styles["SentinelleBody"])
            )
            i += 1
            continue

        story.append(Paragraph(escape(_clean_inline_markdown(line)), styles["SentinelleBody"]))
        i += 1

    if in_code_block and code_lines:
        code_text = "<br/>".join(escape(x) for x in code_lines)
        story.append(Paragraph(code_text, styles["SentinelleMono"]))

    doc.build(story)


def summarize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    recommendation = plan.get("recommendation", {}) or {}
    return {
        "task_type": plan.get("task_type", "unknown"),
        "detected_structure": recommendation.get("detected_structure", "unknown"),
        "primary_cv": recommendation.get("primary_cv", "unknown"),
        "risk_level": recommendation.get("risk_level", "unknown"),
    }


def benchmark_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = plan.get("cv_benchmark") or {}
    results = benchmark.get("results") or {}
    rows: list[dict[str, Any]] = []

    for name, result in results.items():
        rows.append(
            {
                "validation_design": name,
                "mean_score": result.get("mean_score"),
                "std_score": result.get("std_score"),
                "actual_n_folds": result.get("actual_n_folds"),
                "risk": result.get("risk"),
                "interpretation": result.get("interpretation"),
            }
        )

    return rows


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo")
def demo() -> dict[str, Any]:
    df = _load_demo_data()
    return {
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "preview": safe_preview(df),
        "default_target": "target" if "target" in df.columns else "",
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(default=None),
    use_demo: str = Form(default="false"),
    target_col: str = Form(default=""),
    run_cv_benchmark: str = Form(default="false"),
    model_name: str = Form(default="random_forest"),
    n_splits: int = Form(default=3),
) -> dict[str, Any]:
    use_demo_flag = parse_bool(use_demo)
    run_benchmark_flag = parse_bool(run_cv_benchmark)

    if use_demo_flag:
        df = _load_demo_data()
        source_label = "sample dataset"
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="Upload a CSV file or set use_demo=true.")
        try:
            df = pd.read_csv(file.file)
            source_label = file.filename or "uploaded.csv"
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}") from exc

    input_warnings = []
    submission_warning = detect_submission_like_table(df)
    if submission_warning:
        input_warnings.append(submission_warning)

    clean_target = target_col.strip() or None

    run_id = uuid.uuid4().hex[:12]
    output_dir = ARTIFACT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plan = build_plan(df, target_col=clean_target, data_description=None)

        if run_benchmark_flag:
            if not clean_target:
                plan["cv_benchmark_warning"] = (
                    "Select a target column to compare CV strategies. "
                    "Structural diagnostics can still run without a target."
                )
            elif plan.get("task_type") not in {"regression", "classification"}:
                plan["cv_benchmark_warning"] = f"Unsupported task type for CV strategy comparison: {plan.get('task_type')}"
            else:
                plan["cv_benchmark"] = evaluate_cv_designs(
                    df,
                    target_col=clean_target,
                    plan=plan,
                    task_type=plan["task_type"],
                    model_name=model_name,
                    n_splits=int(n_splits),
                    random_state=42,
                    max_rows=50000,
                )
                _write_cv_benchmark_csv(
                    plan.get("cv_benchmark", {}),
                    output_dir / "cv_benchmark_results.csv",
                )

        report_md_path = output_dir / "validation_design_report.md"
        report_pdf_path = output_dir / "validation_design_report.pdf"

        write_markdown_report(plan, str(report_md_path))
        write_pdf_report(report_md_path.read_text(encoding="utf-8"), report_pdf_path)
        write_json_plan(plan, str(output_dir / "validation_design_plan.json"))
        (output_dir / "split_code.py").write_text(generate_split_code(plan), encoding="utf-8")

        report_markdown = report_md_path.read_text(encoding="utf-8")

        artifact_urls = {
            "report_pdf": f"/api/artifacts/{run_id}/validation_design_report.pdf",
            "report_markdown": f"/api/artifacts/{run_id}/validation_design_report.md",
            "json_plan": f"/api/artifacts/{run_id}/validation_design_plan.json",
            "split_code": f"/api/artifacts/{run_id}/split_code.py",
        }
        if (output_dir / "cv_benchmark_results.csv").exists():
            artifact_urls["benchmark_csv"] = f"/api/artifacts/{run_id}/cv_benchmark_results.csv"

        return {
            "run_id": run_id,
            "source_label": source_label,
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "preview": safe_preview(df),
            "summary": summarize_plan(plan),
            "recommendation": plan.get("recommendation", {}),
            "time_cols": plan.get("time_cols", []),
            "group_cols": plan.get("group_cols", []),
            "risks": plan.get("risks", []),
            "input_warnings": input_warnings,
            "benchmark": {
                "metric_name": (plan.get("cv_benchmark") or {}).get("metric_name"),
                "rows": benchmark_rows(plan),
                "optimism_warning": (plan.get("cv_benchmark") or {}).get("optimism_warning"),
                "warning": plan.get("cv_benchmark_warning"),
            },
            "report_markdown": report_markdown,
            "artifact_urls": artifact_urls,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.get("/api/artifacts/{run_id}/{filename}")
def download_artifact(run_id: str, filename: str) -> FileResponse:
    if not RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="Artifact not found.")

    if filename not in ALLOWED_ARTIFACTS:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    run_dir = (ARTIFACT_ROOT / run_id).resolve()
    artifact_path = (run_dir / filename).resolve()

    try:
        artifact_path.relative_to(run_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid artifact path.")

    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")

    return FileResponse(
        path=artifact_path,
        media_type=ARTIFACT_MEDIA_TYPES.get(filename, "application/octet-stream"),
        filename=filename,
    )


DIST_DIR = ROOT / "frontend" / "dist"

if DIST_DIR.exists():
    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    def serve_react_app(full_path: str) -> FileResponse:
        requested = DIST_DIR / full_path
        if requested.exists() and requested.is_file():
            return FileResponse(requested)
        return FileResponse(DIST_DIR / "index.html")
