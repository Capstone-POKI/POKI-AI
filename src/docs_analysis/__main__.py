print("--- [시스템] 초기화 시작 ---")

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# ----------------------------------------------------------------
# [Mac 필수 설정] 이것만 있으면 됩니다 (버전 맞추면 해결됨)
# ----------------------------------------------------------------
os.environ["GRPC_DNS_RESOLVER"] = "native"
os.environ["GRPC_POLL_STRATEGY"] = "poll"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 환경 설정
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("POKI")

# 2. 경로 설정
if "__file__" in locals():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
else:
    PROJECT_ROOT = Path.cwd()

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# 3. 라이브러리 로드 (이제 멈추지 않을 것입니다)
print("⚙️ [시스템] AI 엔진 로딩 중...")
try:
    import torch
    from transformers import LayoutLMv3Processor
    from src.docs_analysis.layoutlm.preprocess import prepare_layoutlm_input, load_docai_json
    from src.docs_analysis.document_ai.processor import process_document, process_pdf_ocr_in_chunks, merge_chunk_results
    from src.utils.io_utils import save_json, read_json
except ImportError as e:
    print(f"❌ 필수 라이브러리가 없습니다: {e}")
    sys.exit(1)


def run_document_ai_pipeline(pdf_path: Path, use_chunking: bool = False) -> Dict:
    print(f"\n📄 [Step 1] Document AI (OCR): {pdf_path.name}")
    output_path = OUTPUT_DIR / f"{pdf_path.stem}_docai.json"
    
    if output_path.exists():
        print(f"⚡️ 기존 결과 재사용: {output_path.name}")
        return read_json(str(output_path))
    
    try:
        if use_chunking:
            print("   ⚙️ 대용량 분할 처리 중...")
            chunk_dir = OUTPUT_DIR / f"{pdf_path.stem}_chunks"
            chunk_results = process_pdf_ocr_in_chunks(str(pdf_path), str(chunk_dir), pages_per_chunk=15)
            return merge_chunk_results(chunk_results, str(output_path))
        return process_document(str(pdf_path), "OCR", str(output_path))
    except Exception as e:
        logger.error(f"❌ OCR 실패: {e}")
        return {}

def run_layoutlm_pipeline(pdf_path: Path, docai_json_path: Path) -> Dict:
    print(f"\n🤖 [Step 2] LayoutLM 구조 분석")
    try:
        docai_result = load_docai_json(str(docai_json_path))
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        layoutlm_input = prepare_layoutlm_input(docai_result, str(pdf_path), processor)
        
        result = {
            "doc_type": "ir_deck",
            "status": "success",
            "input_shape": str(layoutlm_input["input_ids"].shape)
        }
        
        save_path = OUTPUT_DIR / f"{pdf_path.stem}_layoutlm.json"
        save_json(result, str(save_path))
        print(f"   ✅ 분석 완료: {save_path.name}")
        return result
    except Exception as e:
        logger.error(f"❌ LayoutLM 실패: {e}")
        return {}

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n🚀 POKI-AI 파이프라인 시작")
    
    # Gemini 초기화
    gemini = None
    try:
        from src.docs_analysis.llm.gemini_client import GeminiAnalyst
        from src.docs_analysis.post_processing.exporter import export_final_json
        gemini = GeminiAnalyst()
        print("☁️ Gemini 연결 성공!")
    except:
        print("⚠️ Gemini 연결 실패 (기본 분석만 진행)")

    # [Phase 1] 공고문
    notice_pdf = INPUT_DIR / "sample_notice.pdf"
    strategy = {"type": "general", "focus_point": "기본"}
    if notice_pdf.exists():
        res = run_document_ai_pipeline(notice_pdf)
        if gemini and res.get("text"):
            strategy = gemini.analyze_notice(res.get("text", ""))
            print(f"🎯 전략: {strategy.get('focus_point')}")

    # [Phase 2] IR Deck
    ir_pdf = INPUT_DIR / "sample_irdeck.pdf"
    if ir_pdf.exists():
        ocr_res = run_document_ai_pipeline(ir_pdf, use_chunking=True)
        lm_res = run_layoutlm_pipeline(ir_pdf, OUTPUT_DIR / f"{ir_pdf.stem}_docai.json")
        
        final_path = OUTPUT_DIR / f"{ir_pdf.stem}_final.json"
        if gemini:
            export_final_json(ocr_res, lm_res, str(final_path), strategy)
        print(f"\n✨ 전체 완료! 결과: {final_path}")
    else:
        print(f"❌ 파일 없음: {ir_pdf}")

if __name__ == "__main__":
    main()