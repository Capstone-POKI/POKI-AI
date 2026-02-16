import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.domain.ir.pipeline import resolve_strategy, run_ir_analysis
from src.domain.notice.pipeline import init_gemini, run_notice_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POKI-AI CLI")
    parser.add_argument("--mode", choices=["notice", "ir", "all"], default="all")
    parser.add_argument("--notice-pdf", default="data/input/sample_notice.pdf")
    parser.add_argument("--ir-pdf", default="data/input/sample_irdeck.pdf")
    parser.add_argument("--notice-output-dir", default="data/output/notice_analysis")
    parser.add_argument("--ir-output-dir", default="data/output/ir_analysis")
    parser.add_argument("--no-gemini", action="store_true")
    parser.add_argument("--no-auto-strategy", action="store_true")
    parser.add_argument("--no-chunking", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    strategy = None
    if args.mode in ["all", "notice"]:
        gemini = None if args.no_gemini else init_gemini()
        notice_result = run_notice_analysis(
            notice_pdf=Path(args.notice_pdf),
            output_dir=Path(args.notice_output_dir),
            gemini=gemini,
        )
        strategy = notice_result.get("strategy")

    if args.mode in ["all", "ir"]:
        if strategy is None:
            strategy, _ = resolve_strategy(
                strategy_json=None,
                notice_output_dir=Path(args.notice_output_dir),
                auto_use_latest=not args.no_auto_strategy,
            )
        run_ir_analysis(
            ir_pdf=Path(args.ir_pdf),
            output_dir=Path(args.ir_output_dir),
            strategy=strategy,
            use_chunking=not args.no_chunking,
        )


if __name__ == "__main__":
    main()
