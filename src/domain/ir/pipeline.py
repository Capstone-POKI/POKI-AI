from pathlib import Path
from typing import Dict, Optional, Tuple

from src.common.utils import find_latest_strategy, load_strategy
from src.domain.ir.rag_pipeline import run_rag_ir_analysis
from src.domain.ir.scorer import export_final_json
from src.infrastructure.document_ai.pipeline import run_document_ai_pipeline


def resolve_strategy(
    strategy_json: Optional[Path],
    notice_output_dir: Path,
    auto_use_latest: bool = True,
) -> Tuple[Optional[Dict], Optional[Path]]:
    if strategy_json:
        strategy = load_strategy(strategy_json)
        return strategy, strategy_json if strategy else None

    if auto_use_latest:
        latest = find_latest_strategy(notice_output_dir)
        if latest:
            strategy = load_strategy(latest)
            return strategy, latest if strategy else None

    return None, None


def run_ir_analysis(
    ir_pdf: Path,
    output_dir: Path,
    strategy: Optional[Dict] = None,
    use_chunking: bool = True,
    pitch_type: Optional[str] = None,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n📊 [IR Analysis] IR Deck 분석 시작")

    if not ir_pdf.exists():
        raise FileNotFoundError(f"IR Deck 파일이 없습니다: {ir_pdf}")

    ocr_result = run_document_ai_pipeline(ir_pdf, output_dir, use_chunking=use_chunking)
    if not ocr_result and not use_chunking:
        print("⚠️ OCR 결과가 비어 있어 chunking 모드로 재시도합니다.")
        ocr_result = run_document_ai_pipeline(ir_pdf, output_dir, use_chunking=True)
    if not ocr_result:
        raise RuntimeError("IR OCR 단계 실패: 결과가 비어 있습니다.")

    final_path = output_dir / f"{ir_pdf.stem}_final.json"
    try:
        # Primary engine: B-plan RAG pipeline.
        run_rag_ir_analysis(
            docai_result=ocr_result,
            output_path=str(final_path),
            strategy=strategy,
            analysis_version=1,
            pitch_type=pitch_type,
            source_pdf_path=str(ir_pdf),
        )
    except Exception as e:
        print(f"⚠️ B안 파이프라인 실패, 기존 엔진으로 폴백: {e}")
        export_final_json(ocr_result, str(final_path), strategy)
    print(f"✅ IR 분석 결과 저장 완료: {final_path}")

    return {
        "final_path": str(final_path),
        "ocr_output": str(output_dir / f"{ir_pdf.stem}_docai.json"),
    }


# Backward compatibility
def run_ir_deck_analysis(
    ir_pdf: Path,
    output_dir: Path,
    strategy: Optional[Dict] = None,
    use_chunking: bool = True,
) -> Dict:
    return run_ir_analysis(
        ir_pdf=ir_pdf,
        output_dir=output_dir,
        strategy=strategy,
        use_chunking=use_chunking,
    )
