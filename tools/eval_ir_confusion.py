#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.ir.tuning_metrics import (
    build_confusion,
    extract_slide_category_pairs,
    find_result_for_label,
    load_labels,
    normalize_category_for_report,
)


def _parse_focus(v: str) -> List[str]:
    return [normalize_category_for_report(x.strip()) for x in v.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build slide category confusion matrix from GT labels.")
    parser.add_argument("--dataset", type=Path, default=Path("data/config/pitchcoach_tuning_dataset.json"))
    parser.add_argument("--results-root", type=Path, default=Path("data/output/ir_benchmark"))
    parser.add_argument("--focus", type=str, default="PLAN,PRODUCT,TRACTION")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("data/output/ir_benchmark/confusion_report.json"))
    args = parser.parse_args()

    labels = load_labels(args.dataset)
    if not labels:
        raise SystemExit(f"No labels found in dataset: {args.dataset}")

    all_pairs = []
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
        all_pairs.extend(extract_slide_category_pairs(label, payload))

    overall = build_confusion(all_pairs)
    focus_set = set(_parse_focus(args.focus))
    focus_pairs = [p for p in all_pairs if p["expected"] in focus_set or p["predicted"] in focus_set]
    focus_conf = build_confusion(focus_pairs)

    report = {
        "dataset": str(args.dataset),
        "results_root": str(args.results_root),
        "focus_categories": sorted(focus_set),
        "pair_count": len(all_pairs),
        "missing_results": missing_results,
        "overall_matrix": overall["matrix"],
        "overall_top_errors": overall["top_errors"][: args.top_n],
        "focus_matrix": focus_conf["matrix"],
        "focus_top_errors": focus_conf["top_errors"][: args.top_n],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

