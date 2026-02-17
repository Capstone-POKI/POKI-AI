from pathlib import Path

from fastapi import APIRouter

from app.schemas.ir_schema import IRAnalyzeRequest
from src.domain.ir.pipeline import resolve_strategy, run_ir_analysis

router = APIRouter(prefix="/ir", tags=["ir"])


@router.post("/analyze")
def analyze_ir(payload: IRAnalyzeRequest):
    strategy_json = Path(payload.strategy_json) if payload.strategy_json else None
    strategy, _ = resolve_strategy(
        strategy_json=strategy_json,
        notice_output_dir=Path(payload.notice_output_dir),
        auto_use_latest=not payload.no_auto_strategy,
    )
    return run_ir_analysis(
        ir_pdf=Path(payload.ir_pdf),
        output_dir=Path(payload.output_dir),
        strategy=strategy,
        use_chunking=not payload.no_chunking,
        pitch_type=payload.pitch_type.value if payload.pitch_type else None,
    )
