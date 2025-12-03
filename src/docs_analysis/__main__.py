print("--- [시스템] 라이브러리 로딩 시작 (시간이 걸릴 수 있음) ---")
"""
통합 Document AI + LayoutLM + Gemini(RAG) 파이프라인
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv  # ✅ [추가됨]

# ✅ [추가됨] .env 파일 로드 (가장 먼저 실행하여 환경 변수 등록)
load_dotenv()

from src.utils.io_utils import save_json, read_json
from src.utils.pdf_split import split_pdf 

from src.docs_analysis.document_ai.processor import (
    process_document,
    process_pdf_ocr_in_chunks,
    merge_chunk_results
)
from src.docs_analysis.layoutlm.preprocess import (
    prepare_layoutlm_input,
    load_docai_json,
    get_labels,
    get_label_info,
    print_label_statistics
)
from src.docs_analysis.layoutlm.inference import run_inference, aggregate_entities
from src.docs_analysis.layoutlm.config import LAYOUTLM_MODEL_PATH

# 🔥 [NEW] Gemini 및 후처리 모듈 추가
from src.docs_analysis.llm.gemini_client import GeminiAnalyst
from src.docs_analysis.post_processing.exporter import export_final_json


INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"


def detect_document_type(docai_result: Dict) -> str:
    """Document AI 결과로 문서 타입 추정"""
    metadata = docai_result.get("metadata", {})
    detected_sections = metadata.get("detected_sections", [])
    full_text = docai_result.get("text", "")
    
    if "예산" in full_text or "발주기관" in full_text or "입찰" in full_text:
        return "notice"
    
    section_keywords = ["background", "problem", "solution", "team", "market"]
    if any(s in detected_sections for s in section_keywords):
        return "pitch_deck"
    
    numbers = docai_result.get("extracted_numbers", {})
    currency_count = len(numbers.get("currency", []))
    if currency_count >= 5:
        return "ir_deck"
    
    return "pitch_deck"


def run_document_ai_pipeline(
    pdf_path: str,
    processor_type: str = "OCR",
    output_path: Optional[str] = None,
    enable_enhancement: bool = True,
    use_chunking: bool = False,
    pages_per_chunk: int = 15
) -> Dict:
    """Document AI 실행 (단일 또는 청크 처리)"""
    
    print("\n" + "=" * 80)
    print("📄 Step 1: Document AI 처리")
    print("=" * 80)
    
    pdf_name = Path(pdf_path).stem
    
    if not output_path:
        output_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_docai_{processor_type.lower()}.json")
    
    # 이미 분석된 파일이 있으면 재사용 (시간 절약)
    if os.path.exists(output_path):
        print(f"⚡️ 기존 분석 결과 발견! ({output_path}) - 재사용합니다.")
        return read_json(output_path)
    
    if use_chunking:
        chunk_dir = os.path.join(OUTPUT_DIR, f"{pdf_name}_chunks")
        chunk_results = process_pdf_ocr_in_chunks(
            file_path=pdf_path,
            output_dir=chunk_dir,
            pages_per_chunk=pages_per_chunk,
            enable_enhancement=enable_enhancement
        )
        result = merge_chunk_results(chunk_results, output_path)
    else:
        result = process_document(
            file_path=pdf_path,
            processor_type=processor_type,
            output_path=output_path,
            enable_enhancement=enable_enhancement
        )
    
    return result


def run_layoutlm_pipeline(
    pdf_path: str,
    docai_json_path: str,
    doc_type: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """LayoutLM 분석 실행"""
    
    print("\n" + "=" * 80)
    print("🤖 Step 2: LayoutLM 엔티티 추출")
    print("=" * 80)
    
    docai_result = load_docai_json(docai_json_path)
    
    if not doc_type:
        doc_type = detect_document_type(docai_result)
        print(f"  🔍 문서 타입 자동 감지: {doc_type}")
    else:
        print(f"  📋 문서 타입: {doc_type}")
    
    labels = get_labels(doc_type)
    print(f"  🏷️ 사용 라벨: {len(labels)}개")
    
    from transformers import LayoutLMv3Processor
    
    # 🔥 apply_ocr=False 적용
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False
    )
    
    layoutlm_input = prepare_layoutlm_input(
        doc_json=docai_result,
        pdf_path=pdf_path,
        processor=processor,
        max_length=512
    )
    
    print(f"\n  🎯 LayoutLM 추론 실행...")
    
    result = {
        "doc_type": doc_type,
        "num_labels": len(labels),
        "labels_sample": labels[:20],
        "input_shape": str(layoutlm_input["input_ids"].shape),
    }
    
    if not output_dir:
        output_dir = OUTPUT_DIR
    
    pdf_name = Path(pdf_path).stem
    result_path = os.path.join(output_dir, f"{pdf_name}_layoutlm_result.json")
    save_json(result, result_path)
    
    print(f"  ✅ 결과 저장: {result_path}\n")
    
    return result


def main():
    """메인 파이프라인 실행 (RAG Workflow)"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("🚀 POKI-AI Intelligent RAG Pipeline (Gemini Powered)")
    print("=" * 80)
    
    # 0. Gemini 초기화
    gemini = GeminiAnalyst()

    # -------------------------------------------------------------------------
    # 1단계: 공고문(Criteria) 분석 - 심사 기준 수립
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📢 [Phase 1] 공고문 분석 및 심사 전략 수립")
    print("=" * 80)
    
    notice_pdf = os.path.join(INPUT_DIR, "sample_notice.pdf")
    strategy = None

    if os.path.exists(notice_pdf):
        # 1-1. Document AI로 텍스트 추출
        notice_result = run_document_ai_pipeline(
            pdf_path=notice_pdf,
            processor_type="OCR",
            enable_enhancement=True
        )
        
        # 1-2. Gemini에게 전략 수립 요청
        print(f"\n🧠 Gemini가 공고문을 읽고 심사 기준을 세우는 중...")
        notice_text = notice_result.get("text", "")
        strategy = gemini.analyze_notice(notice_text)
        
        print(f"\n🎯 [AI 전략 수립 결과]")
        print(f"   • 피칭 타입: {strategy.get('type', 'Unknown')}")
        print(f"   • 핵심 포인트: {strategy.get('focus_point', 'N/A')}")
        print(f"   • 필수 섹션: {strategy.get('required_sections', [])}")

        # 엘리베이터 피치인 경우 중단 (사용자 요청 사항 반영)
        if strategy.get("type") == "elevator":
            print("\n⛔️ 엘리베이터 피치(1분 미만)는 심층 분석 대상이 아닙니다. 프로그램을 종료합니다.")
            return

    else:
        print(f"⚠️ 공고문 파일 없음 ({notice_pdf}). 기본 전략(General)으로 진행합니다.")
        strategy = {"type": "general", "required_sections": [], "focus_point": "일반적인 사업성 평가"}

    # -------------------------------------------------------------------------
    # 2단계: IR Deck(Target) 분석 - 구조 및 데이터 추출
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 [Phase 2] IR Deck 심층 분석")
    print("=" * 80)

    ir_pdf = os.path.join(INPUT_DIR, "sample_irdeck.pdf")
    
    if os.path.exists(ir_pdf):
        # 2-1. Document AI (청크 처리)
        docai_result = run_document_ai_pipeline(
            pdf_path=ir_pdf,
            processor_type="OCR",
            enable_enhancement=True,
            use_chunking=True,  # IR Deck은 보통 기니까 청크 처리
            pages_per_chunk=15
        )
        
        docai_json_path = os.path.join(OUTPUT_DIR, "sample_irdeck_docai_ocr.json")
        
        # 2-2. LayoutLM (구조 분석)
        layoutlm_result = run_layoutlm_pipeline(
            pdf_path=ir_pdf,
            docai_json_path=docai_json_path,
            doc_type="ir_deck"
        )
        
        # 2-3. [핵심] 최종 진단 및 JSON 내보내기 (Gemini 전략 적용)
        print("\n" + "=" * 80)
        print("🏁 [Phase 3] 맞춤형 진단 리포트 생성")
        print("=" * 80)
        
        final_json_path = os.path.join(OUTPUT_DIR, "sample_irdeck_final_analysis.json")
        
        # 🔥 여기서 Gemini가 만든 strategy를 함께 넘겨줍니다!
        export_final_json(
            docai_result=docai_result,
            layoutlm_result=layoutlm_result,
            output_path=final_json_path,
            pitch_strategy=strategy  # <--- RAG의 핵심 연결 고리
        )
        
        print(f"\n✨ 모든 분석이 완료되었습니다!")
        print(f"📂 최종 결과물: {final_json_path}")
        
    else:
        print(f"⚠️ IR Deck 파일 없음: {ir_pdf}")


if __name__ == "__main__":
    main()