#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.ir.batch_runner import BatchRunConfig, run_ir_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IR pipeline for multiple sample PDFs.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"))
    parser.add_argument("--output-root", type=Path, default=Path("data/output/ir_benchmark"))
    parser.add_argument("--pitch-type", type=str, default="COMPETITION")
    parser.add_argument("--glob", type=str, default="*.pdf")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--no-chunking", action="store_true")
    parser.add_argument("--include-notice-like", action="store_true")
    args = parser.parse_args()

    config = BatchRunConfig(
        input_dir=args.input_dir,
        output_root=args.output_root,
        pitch_type=args.pitch_type,
        glob_pattern=args.glob,
        max_files=args.max_files,
        use_chunking=not args.no_chunking,
        skip_notice_like=not args.include_notice_like,
    )
    summary = run_ir_batch(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
