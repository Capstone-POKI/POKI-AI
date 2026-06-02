#!/usr/bin/env python3
"""
run_full_benchmark.py

기존 7개 GT(data/config/pitchcoach_tuning_dataset.json)와
새로 만든 8개 GT(data/gt_labels/*.json)를 합쳐서
전체 최대 15개 기준으로 벤치마크를 실행하는 스크립트.

사용법:
    cd /path/to/Pitchcoach-AI
    python3 tests/run_full_benchmark.py [--out-json PATH] [--out-csv PATH]

옵션:
    --dataset       기존 GT 파일 경로 (기본: data/config/pitchcoach_tuning_dataset.json)
    --gt-dir        새 GT 라벨 폴더 (기본: data/gt_labels/)
    --search-roots  결과 JSON 검색 루트 (기본: data/output/ir_benchmark ...)
    --out-json      결과 JSON 출력 경로
    --out-csv       결과 CSV 출력 경로
    --sim-high      SIM_HIGH 오버라이드 (기본: 코드/config 값 사용)
    --sim-mid       SIM_MID 오버라이드
    --top-k         TOP_K 오버라이드
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional


def _nfc(s: str) -> str:
    """Unicode NFC 정규화 — 플랫폼 간 파일명 비교에 사용."""
    return unicodedata.normalize("NFC", s or "")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.ir import rag_pipeline as _rp
from src.domain.ir.rag_pipeline import run_rag_ir_analysis
from src.domain.ir.tuning_metrics import (
    aggregate_eval,
    evaluate_label,
    find_docai_for_label,
    load_labels,
    normalize_category,
    extract_slide_category_pairs,
    build_confusion,
)


# ---------------------------------------------------------------------------
# GT Loader helpers
# ---------------------------------------------------------------------------

def load_gt_labels_from_dir(gt_dir: Path) -> List[Dict[str, Any]]:
    """
    data/gt_labels/*.json 형식의 GT 파일을 읽어서
    tuning_dataset과 동일한 label dict 형태로 변환한다.
    """
    labels: List[Dict[str, Any]] = []
    if not gt_dir.exists():
        return labels

    for gt_file in sorted(gt_dir.glob("gt_*.json")):
        try:
            data = json.loads(gt_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] GT 파일 로드 실패: {gt_file} — {e}")
            continue

        deck_name = data.get("deck_name", gt_file.stem)
        folder = data.get("folder", deck_name)
        pitch_type = data.get("pitch_type", "STARTUP_CONTEST")
        slide_labels = data.get("slides", [])

        # slide_classification_labels 형태로 변환
        slide_classification_labels = [
            {
                "slide_number": int(s.get("slide_index", s.get("slide_number", 0))),
                "expected_category": str(s.get("gt_category", s.get("model_predicted", "OTHER"))),
            }
            for s in slide_labels
            if s.get("slide_index") or s.get("slide_number")
        ]

        # NFC 정규화로 macOS NFD 파일명 문제 해결
        folder_nfc = _nfc(folder)
        label = {
            "label_id": f"GT_EXT_{gt_file.stem}",
            "filename": f"{folder_nfc}.pdf",
            "filename_aliases": [f"{folder_nfc}.pdf"],
            "pitch_type": pitch_type,
            "group_labels": [],  # 슬라이드 분류 전용 GT (group coverage 레이블 없음)
            "slide_classification_labels": slide_classification_labels,
            "_source": str(gt_file),
            "_deck_name": deck_name,
        }
        labels.append(label)

    return labels


def merge_and_deduplicate(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    primary(기존 tuning_dataset) 우선으로 합친다.
    같은 filename이 있으면 primary를 유지하고 secondary는 스킵.
    NFC 정규화로 한국어 파일명 플랫폼 간 비교 문제를 해결한다.
    """
    seen_files: set = set()
    merged: List[Dict[str, Any]] = []

    for lbl in primary:
        fn = _nfc(lbl.get("filename", ""))
        seen_files.add(fn)
        # aliases도 추가
        for alias in lbl.get("filename_aliases", []):
            seen_files.add(_nfc(alias))
        merged.append(lbl)

    for lbl in secondary:
        fn = _nfc(lbl.get("filename", ""))
        aliases = [_nfc(a) for a in lbl.get("filename_aliases", [])]
        if fn in seen_files or any(a in seen_files for a in aliases):
            continue
        merged.append(lbl)
        seen_files.add(fn)

    return merged


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_full_benchmark(
    dataset_path: Path,
    gt_dir: Path,
    search_roots: List[Path],
    sim_high: Optional[float] = None,
    sim_mid: Optional[float] = None,
    top_k: Optional[int] = None,
    fast_mode: bool = True,
) -> Dict[str, Any]:
    """
    GT 라벨 로드 → 파이프라인 실행 → 평가 → 결과 반환.
    """
    # 1. 레이블 로드
    primary_labels = load_labels(dataset_path)
    ext_labels = load_gt_labels_from_dir(gt_dir)
    all_labels = merge_and_deduplicate(primary_labels, ext_labels)

    print(f"[벤치마크] GT 레이블: 기존 {len(primary_labels)}개 + 신규 {len(ext_labels)}개")
    print(f"           중복 제거 후 총 {len(all_labels)}개")

    # 2. 환경변수 오버라이드 설정
    env_overrides: Dict[str, str] = {}
    if sim_high is not None:
        env_overrides["IR_SIM_HIGH"] = str(sim_high)
    if sim_mid is not None:
        env_overrides["IR_SIM_MID"] = str(sim_mid)
    if top_k is not None:
        env_overrides["IR_TOP_K"] = str(top_k)
    if fast_mode:
        env_overrides["IR_FAST_MODE"] = "1"
        env_overrides["IR_LLM_SLIDE_LIMIT"] = "0"

    saved_env: Dict[str, Optional[str]] = {}
    for k, v in env_overrides.items():
        saved_env[k] = os.environ.get(k)
        os.environ[k] = v

    # config 캐시 초기화
    _rp._PIPELINE_CONFIG_CACHE = None

    try:
        eval_rows: List[Dict[str, Any]] = []
        missing_results: List[str] = []
        all_pairs: List[Dict[str, Any]] = []

        with TemporaryDirectory(prefix="full_bench_") as tmp:
            tmp_dir = Path(tmp)

            for lbl in all_labels:
                docai_path = find_docai_for_label(
                    search_roots,
                    str(lbl.get("filename", "")),
                    aliases=lbl.get("filename_aliases"),
                )
                if not docai_path:
                    print(f"  [SKIP] docai 없음: {lbl['filename']}")
                    missing_results.append(lbl["filename"])
                    continue

                stem = Path(lbl["filename"]).stem
                out_path = tmp_dir / f"{stem}_bench.json"
                print(f"  [실행] {lbl['label_id']} — {lbl['filename']}")

                try:
                    docai = json.loads(docai_path.read_text(encoding="utf-8"))
                    pred = run_rag_ir_analysis(
                        docai_result=docai,
                        output_path=str(out_path),
                        strategy=None,
                        analysis_version=1,
                        pitch_type=lbl.get("pitch_type"),
                    )
                except Exception as e:
                    print(f"  [ERROR] {lbl['filename']}: {e}")
                    continue

                row = evaluate_label(lbl, pred)
                row["docai_path"] = str(docai_path)
                eval_rows.append(row)

                pairs = extract_slide_category_pairs(lbl, pred)
                all_pairs.extend(pairs)

        summary = aggregate_eval(eval_rows)
        confusion = build_confusion(all_pairs)

        return {
            "total_labels": len(all_labels),
            "evaluated": len(eval_rows),
            "skipped": len(missing_results),
            "missing_results": missing_results,
            "summary": summary,
            "evaluated_cases": eval_rows,
            "confusion": confusion,
            "env_used": {
                k: os.environ.get(k, "(default)") for k in
                ["IR_SIM_HIGH", "IR_SIM_MID", "IR_TOP_K", "IR_FAST_MODE"]
            },
        }

    finally:
        for k, old in saved_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        _rp._PIPELINE_CONFIG_CACHE = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="기존 7개 GT + 신규 GT를 합쳐서 전체 벤치마크 실행"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/config/pitchcoach_tuning_dataset.json",
        help="기존 GT 파일 경로",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=ROOT / "data/gt_labels",
        help="신규 GT 라벨 폴더 (gt_*.json)",
    )
    parser.add_argument(
        "--search-roots",
        type=Path,
        nargs="+",
        default=[
            ROOT / "data/output/ir_benchmark",
            ROOT / "data/output/ir_analysis",
            ROOT / "data/output",
        ],
        help="docai/final.json 검색 루트 디렉토리",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data/output/ir_benchmark/full_benchmark_report.json",
        help="결과 JSON 저장 경로",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "data/output/ir_benchmark/full_benchmark_report.csv",
        help="결과 CSV 저장 경로",
    )
    parser.add_argument("--sim-high", type=float, default=None, help="SIM_HIGH 오버라이드")
    parser.add_argument("--sim-mid", type=float, default=None, help="SIM_MID 오버라이드")
    parser.add_argument("--top-k", type=int, default=None, help="TOP_K 오버라이드")
    parser.add_argument(
        "--no-fast-mode",
        action="store_true",
        help="IR_FAST_MODE 비활성화 (LLM 실사용, API 키 필요)",
    )

    args = parser.parse_args()

    result = run_full_benchmark(
        dataset_path=args.dataset,
        gt_dir=args.gt_dir,
        search_roots=args.search_roots,
        sim_high=args.sim_high,
        sim_mid=args.sim_mid,
        top_k=args.top_k,
        fast_mode=not args.no_fast_mode,
    )

    # JSON 저장
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 JSON 저장: {args.out_json}")

    # CSV 저장 (요약 + per-case)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        summary = result["summary"]
        writer = csv.writer(f)
        writer.writerow(["# 요약"])
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])
        writer.writerow([])
        writer.writerow(["# 케이스별 결과"])
        cases = result.get("evaluated_cases", [])
        if cases:
            fieldnames = [k for k in cases[0].keys() if k not in ("coverage_stats",)]
            writer.writerow(fieldnames)
            for row in cases:
                writer.writerow([row.get(f, "") for f in fieldnames])

    print(f"결과 CSV 저장: {args.out_csv}")

    # 콘솔 출력
    print("\n" + "=" * 60)
    print("전체 벤치마크 결과 요약")
    print("=" * 60)
    summary = result["summary"]
    print(f"  평가 케이스 수:          {result['evaluated']}/{result['total_labels']}")
    print(f"  pitch_type_accuracy:     {summary.get('pitch_type_accuracy', 0):.4f}")
    print(f"  slide_category_accuracy: {summary.get('slide_category_accuracy', 0):.4f}")
    print(f"  related_slide_hit_rate:  {summary.get('related_slide_hit_rate', 0):.4f}")
    print(f"  group_coverage_accuracy: {summary.get('group_coverage_accuracy', 0):.4f}")
    print(f"  coverage_macro_f1:       {summary.get('coverage_macro_f1', 0):.4f}")
    print("=" * 60)

    if result["missing_results"]:
        print(f"\n[SKIP] docai 없어서 제외된 파일 ({len(result['missing_results'])}개):")
        for fn in result["missing_results"]:
            print(f"  - {fn}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
