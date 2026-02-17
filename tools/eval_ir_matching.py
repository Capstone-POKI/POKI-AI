#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.ir.tuning_metrics import aggregate_eval, evaluate_label, find_result_for_label, load_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate IR rubric matching quality against GT labels.")
    parser.add_argument("--dataset", type=Path, default=Path("data/config/pitchcoach_tuning_dataset.json"))
    parser.add_argument("--results-root", type=Path, default=Path("data/output/ir_benchmark"))
    parser.add_argument("--out", type=Path, default=Path("data/output/ir_benchmark/eval_report.json"))
    args = parser.parse_args()

    labels = load_labels(args.dataset)
    if not labels:
        raise SystemExit(f"No labels found in dataset: {args.dataset}")

    rows = []
    missing_results = []
    for label in labels:
        result_path = find_result_for_label(
            args.results_root,
            str(label.get("filename", "")),
            aliases=label.get("filename_aliases"),
        )
        if not result_path:
            missing_results.append(str(label.get("filename", "")))
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        row = evaluate_label(label, payload)
        row["result_path"] = str(result_path)
        rows.append(row)

    summary = aggregate_eval(rows)
    report = {
        "dataset": str(args.dataset),
        "results_root": str(args.results_root),
        "summary": summary,
        "evaluated_cases": rows,
        "missing_results": missing_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
