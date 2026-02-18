from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.domain.ir.pipeline import run_ir_analysis


@dataclass
class BatchRunConfig:
    input_dir: Path
    output_root: Path
    pitch_type: str = "COMPETITION"
    glob_pattern: str = "*.pdf"
    max_files: Optional[int] = None
    use_chunking: bool = True
    skip_notice_like: bool = True


def run_ir_batch(config: BatchRunConfig) -> Dict[str, Any]:
    input_dir = config.input_dir
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob(config.glob_pattern))
    if config.skip_notice_like:
        pdfs = [p for p in pdfs if "notice" not in p.name.lower()]
    if config.max_files is not None:
        pdfs = pdfs[: max(0, config.max_files)]

    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows: List[Dict[str, Any]] = []

    for index, pdf in enumerate(pdfs, start=1):
        started = time.perf_counter()
        case_output_dir = output_root / pdf.stem
        case_output_dir.mkdir(parents=True, exist_ok=True)
        row: Dict[str, Any] = {
            "index": index,
            "file": str(pdf),
            "status": "FAILED",
            "elapsed_sec": 0.0,
            "final_path": "",
            "total_score": None,
            "covered_groups": 0,
            "partial_groups": 0,
            "not_covered_groups": 0,
            "error": "",
        }
        try:
            result = run_ir_analysis(
                ir_pdf=pdf,
                output_dir=case_output_dir,
                strategy=None,
                use_chunking=config.use_chunking,
                pitch_type=config.pitch_type,
            )
            final_path = Path(result["final_path"])
            payload = json.loads(final_path.read_text(encoding="utf-8"))
            criteria_scores = payload.get("criteria_scores", [])
            statuses = [str(c.get("coverage_status", "")) for c in criteria_scores]
            row.update(
                {
                    "status": "COMPLETED",
                    "final_path": str(final_path),
                    "total_score": int(payload.get("deck_score", {}).get("total_score", 0)),
                    "covered_groups": sum(1 for s in statuses if s == "COVERED"),
                    "partial_groups": sum(1 for s in statuses if s == "PARTIALLY_COVERED"),
                    "not_covered_groups": sum(1 for s in statuses if s == "NOT_COVERED"),
                }
            )
        except Exception as exc:
            row["error"] = str(exc)
        finally:
            row["elapsed_sec"] = round(time.perf_counter() - started, 2)
            rows.append(row)

    finished_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    completed = [r for r in rows if r["status"] == "COMPLETED"]
    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "pitch_type": config.pitch_type,
        "total_files": len(rows),
        "completed_files": len(completed),
        "failed_files": len(rows) - len(completed),
        "avg_score": round(sum(r["total_score"] for r in completed) / len(completed), 2) if completed else None,
        "results": rows,
    }

    summary_json = output_root / "batch_summary.json"
    summary_csv = output_root / "batch_summary.csv"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(summary_csv, rows)
    summary["summary_json"] = str(summary_json)
    summary["summary_csv"] = str(summary_csv)
    return summary


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "file",
        "status",
        "elapsed_sec",
        "total_score",
        "covered_groups",
        "partial_groups",
        "not_covered_groups",
        "final_path",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
