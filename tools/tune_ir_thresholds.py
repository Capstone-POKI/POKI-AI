#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.ir.rag_pipeline import run_rag_ir_analysis
from src.domain.ir.tuning_metrics import (
    aggregate_eval,
    evaluate_label,
    find_docai_for_label,
    load_labels,
    score_for_ranking,
)


def _parse_float_list(v: str) -> List[float]:
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def _parse_int_list(v: str) -> List[int]:
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def _set_env(overrides: Dict[str, str]) -> Dict[str, str]:
    before = {}
    for k, val in overrides.items():
        before[k] = os.environ.get(k)
        os.environ[k] = str(val)
    return before


def _restore_env(before: Dict[str, str]) -> None:
    for k, old in before.items():
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-search IR_SIM_HIGH/IR_SIM_MID/IR_TOP_K with GT labels.")
    parser.add_argument("--dataset", type=Path, default=Path("data/config/pitchcoach_tuning_dataset.json"))
    parser.add_argument(
        "--search-roots",
        type=Path,
        nargs="+",
        default=[Path("data/output/ir_benchmark"), Path("data/output/ir_analysis"), Path("data/output")],
    )
    parser.add_argument("--sim-high", type=str, default="0.65,0.68,0.72")
    parser.add_argument("--sim-mid", type=str, default="0.55,0.60")
    parser.add_argument("--top-k", type=str, default="2,3,4")
    parser.add_argument("--out-json", type=Path, default=Path("data/output/ir_benchmark/tuning_report.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/output/ir_benchmark/tuning_report.csv"))
    args = parser.parse_args()

    labels = load_labels(args.dataset)
    if not labels:
        raise SystemExit(f"No labels found in dataset: {args.dataset}")

    highs = _parse_float_list(args.sim_high)
    mids = _parse_float_list(args.sim_mid)
    topks = _parse_int_list(args.top_k)

    rows = []
    with TemporaryDirectory(prefix="ir_tuning_") as tmp:
        tmp_dir = Path(tmp)
        for high in highs:
            for mid in mids:
                if mid >= high:
                    continue
                for topk in topks:
                    before = _set_env(
                        {
                            "IR_SIM_HIGH": str(high),
                            "IR_SIM_MID": str(mid),
                            "IR_TOP_K": str(topk),
                            "IR_FAST_MODE": "1",
                            "IR_LLM_SLIDE_LIMIT": "0",
                        }
                    )
                    try:
                        eval_rows = []
                        skipped = 0
                        for label in labels:
                            docai_path = find_docai_for_label(
                                args.search_roots,
                                str(label.get("filename", "")),
                                aliases=label.get("filename_aliases"),
                            )
                            if not docai_path:
                                skipped += 1
                                continue
                            docai = json.loads(docai_path.read_text(encoding="utf-8"))
                            out_path = tmp_dir / f"{Path(label['filename']).stem}_h{high}_m{mid}_k{topk}.json"
                            pred = run_rag_ir_analysis(
                                docai_result=docai,
                                output_path=str(out_path),
                                strategy=None,
                                analysis_version=1,
                                pitch_type=label.get("pitch_type"),
                            )
                            eval_rows.append(evaluate_label(label, pred))

                        summary = aggregate_eval(eval_rows)
                        rows.append(
                            {
                                "sim_high": high,
                                "sim_mid": mid,
                                "top_k": topk,
                                "cases": summary["cases"],
                                "skipped": skipped,
                                "pitch_type_accuracy": summary["pitch_type_accuracy"],
                                "group_coverage_accuracy": summary["group_coverage_accuracy"],
                                "related_slide_hit_rate": summary["related_slide_hit_rate"],
                                "slide_category_accuracy": summary["slide_category_accuracy"],
                                "coverage_macro_f1": summary.get("coverage_macro_f1", 0.0),
                            }
                        )
                    finally:
                        _restore_env(before)

    if not rows:
        raise SystemExit("No tuning results produced. Check dataset/search roots.")

    best = max(rows, key=score_for_ranking)
    report = {
        "dataset": str(args.dataset),
        "search_roots": [str(p) for p in args.search_roots],
        "trials": rows,
        "best": best,
        "recommended_env": {
            "IR_SIM_HIGH": str(best["sim_high"]),
            "IR_SIM_MID": str(best["sim_mid"]),
            "IR_TOP_K": str(best["top_k"]),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
