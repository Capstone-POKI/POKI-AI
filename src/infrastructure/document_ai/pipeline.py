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


def _is_page_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "page_limit_exceeded" in msg or "document pages" in msg or "page limit" in msg


def _get_pdf_page_count(pdf_path: Path) -> int | None:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return None

    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return None


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

    # Proactive switch: if page count exceeds non-chunking practical limit, force chunking first.
    page_count = _get_pdf_page_count(pdf_path)
    if page_count is not None and page_count > pages_per_chunk and not use_chunking:
        print(f"   ⚙️ 페이지 수 {page_count}p 감지 -> chunking 자동 전환 ({pages_per_chunk}p 단위)")
        use_chunking = True

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
        # Root fix: fallback to chunking automatically when page limit is exceeded.
        if not use_chunking and _is_page_limit_error(e):
            logger.warning(f"⚠️ OCR 페이지 제한 감지, chunking 모드로 자동 전환: {e}")
            try:
                chunk_dir = output_dir / f"{pdf_path.stem}_chunks"
                chunk_results = process_pdf_ocr_in_chunks(
                    str(pdf_path),
                    str(chunk_dir),
                    pages_per_chunk=pages_per_chunk,
                )
                return merge_chunk_results(chunk_results, str(output_path))
            except Exception as chunk_err:
                logger.error(f"❌ OCR chunking 재시도 실패: {chunk_err}")
                return {}

        logger.error(f"❌ OCR 실패: {e}")
        return {}
