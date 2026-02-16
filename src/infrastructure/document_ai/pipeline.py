from pathlib import Path
from typing import Dict
import logging

from src.utils.io_utils import read_json
from src.infrastructure.document_ai.processor import (
    merge_chunk_results,
    process_document,
    process_pdf_ocr_in_chunks,
)


logger = logging.getLogger("POKI")


def run_document_ai_pipeline(
    pdf_path: Path,
    output_dir: Path,
    use_chunking: bool = False,
    pages_per_chunk: int = 15,
) -> Dict:
    print(f"\n📄 [OCR] {pdf_path.name}")
    output_path = output_dir / f"{pdf_path.stem}_docai.json"

    if output_path.exists():
        print(f"⚡️ 기존 결과 재사용: {output_path.name}")
        return read_json(str(output_path))

    try:
        if use_chunking:
            print("   ⚙️ 대용량 분할 처리 중...")
            chunk_dir = output_dir / f"{pdf_path.stem}_chunks"
            chunk_results = process_pdf_ocr_in_chunks(
                str(pdf_path),
                str(chunk_dir),
                pages_per_chunk=pages_per_chunk,
            )
            return merge_chunk_results(chunk_results, str(output_path))
        return process_document(str(pdf_path), "OCR", str(output_path))
    except Exception as e:
        logger.error(f"❌ OCR 실패: {e}")
        return {}
