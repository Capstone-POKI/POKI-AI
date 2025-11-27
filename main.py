# main.py
"""
통합 Document AI + LayoutLM 파이프라인
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.io_utils import save_json, read_json
# pdf_split은 이제 processor.py 내부 로직을 따르므로, 직접 import할 필요가 없거나 
# 배치 처리를 위해 필요하다면 단순 유틸로만 사용합니다.
from src.utils.pdf_split import split_pdf 

from src.document_ai.processor import (
    process_document,
    process_pdf_ocr_in_chunks,
    merge_chunk_results
)
from src.layoutlm.preprocess import (
    prepare_layoutlm_input,
    load_docai_json,
    get_labels,
    get_label_info,
    print_label_statistics
)
from src.layoutlm.inference import run_inference, aggregate_entities
from src.layoutlm.config import LAYOUTLM_MODEL_PATH


INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"


def detect_document_type(docai_result: Dict) -> str:
    """Document AI 결과로 문서 타입 추정"""
    
    metadata = docai_result.get("metadata", {})
    detected_sections = metadata.get("detected_sections", [])
    full_text = docai_result.get("text", "")
    
    # 공고문 패턴
    if "예산" in full_text or "발주기관" in full_text or "입찰" in full_text:
        return "notice"
    
    # Pitch Deck 패턴
    section_keywords = ["background", "problem", "solution", "team", "market"]
    if any(s in detected_sections for s in section_keywords):
        return "pitch_deck"
    
    # IR Deck 패턴
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
    
    if enable_enhancement and "metadata" in result:
        print(f"\n📊 Document AI 분석 결과:")
        metadata = result["metadata"]
        print(f"  - 총 페이지: {metadata.get('total_pages', 0)}개")
        print(f"  - 감지된 섹션: {', '.join(metadata.get('detected_sections', []))}")
        
        numbers = result.get("extracted_numbers", {})
        total_numbers = sum(len(v) for v in numbers.values())
        print(f"  - 추출된 숫자: {total_numbers}개")
        
        if numbers.get("currency"):
            print(f"    • 화폐: {[n['text'] for n in numbers['currency'][:3]]}")
        if numbers.get("percentage"):
            print(f"    • 백분율: {[n['text'] for n in numbers['percentage'][:3]]}")
    
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
    
    # 🔥 [수정됨] apply_ocr=False 옵션 추가
    # Document AI가 이미 OCR 좌표(bbox)를 제공하므로, LayoutLM 내부의 Tesseract OCR을 끕니다.
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
    print(f"  ⚠️ 주의: 실제 모델 가중치가 학습되지 않았으므로 결과는 랜덤할 수 있습니다.")
    
    # 실제 추론 로직 연결 (더미 실행)
    # 학습된 모델이 있다면 여기서 run_inference()를 호출합니다.
    # 현재는 데이터 파이프라인 점검용으로 입력 형태만 확인합니다.
    
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


def generate_comprehensive_report(
    pdf_path: str,
    docai_result: Dict,
    layoutlm_result: Dict,
    output_path: str
):
    """상세 분석 리포트 생성"""
    
    pdf_name = Path(pdf_path).stem
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"📊 문서 분석 종합 리포트\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"📄 문서명: {pdf_name}\n")
        f.write(f"📋 문서 타입: {layoutlm_result.get('doc_type', 'unknown')}\n")
        f.write(f"🏷️ 사용 라벨 수: {layoutlm_result.get('num_labels', 0)}개\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("🔍 Document AI 분석 결과\n")
        f.write("-" * 80 + "\n")
        
        metadata = docai_result.get("metadata", {})
        f.write(f"총 페이지: {metadata.get('total_pages', 0)}개\n")
        f.write(f"총 블록: {metadata.get('total_blocks', 0)}개\n")
        f.write(f"총 문단: {metadata.get('total_paragraphs', 0)}개\n\n")
        
        detected_sections = docai_result.get("detected_sections", [])
        if detected_sections:
            f.write("📍 감지된 섹션:\n")
            for section in detected_sections:
                f.write(f"  • 페이지 {section['page']}: {section['section']}\n")
                if 'preview' in section:
                    f.write(f"    {section['preview'][:80]}...\n")
            f.write("\n")
        
        numbers = docai_result.get("extracted_numbers", {})
        if numbers:
            f.write("💰 추출된 숫자/통계:\n")
            
            if numbers.get("currency"):
                f.write(f"  화폐 ({len(numbers['currency'])}개):\n")
                for num in numbers["currency"][:10]:
                    f.write(f"    - {num['text']}\n")
            
            if numbers.get("percentage"):
                f.write(f"  백분율 ({len(numbers['percentage'])}개):\n")
                for num in numbers["percentage"][:10]:
                    f.write(f"    - {num['text']}\n")
            
            if numbers.get("quantity"):
                f.write(f"  수량 ({len(numbers['quantity'])}개):\n")
                for num in numbers["quantity"][:10]:
                    f.write(f"    - {num['text']}\n")
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("🤖 LayoutLM 엔티티 추출 결과\n")
        f.write("-" * 80 + "\n")
        f.write(f"사용 모델: LayoutLMv3\n")
        f.write(f"입력 형태: {layoutlm_result.get('input_shape', 'N/A')}\n")
        f.write(f"라벨 샘플 (20개):\n")
        for label in layoutlm_result.get('labels_sample', [])[:20]:
            f.write(f"  - {label}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"📄 리포트 생성 완료: {output_path}")


def main():
    """메인 파이프라인 실행"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("🚀 통합 Document AI + LayoutLM 파이프라인")
    print("=" * 80)
    
    print_label_statistics()
    
    # 예제 1: 피칭 자료 (단일 처리)
    print("\n" + "=" * 80)
    print("📄 예제 1: 피칭 자료 분석 (단일 처리)")
    print("=" * 80)
    
    pitch_pdf = os.path.join(INPUT_DIR, "sample_pitch.pdf")
    
    if os.path.exists(pitch_pdf):
        docai_result = run_document_ai_pipeline(
            pdf_path=pitch_pdf,
            processor_type="OCR",
            enable_enhancement=True,
            use_chunking=False
        )
        
        docai_json = os.path.join(OUTPUT_DIR, "sample_pitch_docai_ocr.json")
        layoutlm_result = run_layoutlm_pipeline(
            pdf_path=pitch_pdf,
            docai_json_path=docai_json,
            doc_type="pitch_deck"
        )
        
        report_path = os.path.join(OUTPUT_DIR, "sample_pitch_report.txt")
        generate_comprehensive_report(
            pdf_path=pitch_pdf,
            docai_result=docai_result,
            layoutlm_result=layoutlm_result,
            output_path=report_path
        )
    else:
        print(f"⚠️ 파일 없음: {pitch_pdf}")
    
    # 예제 2: IR Deck (청크 처리)
    print("\n" + "=" * 80)
    print("📄 예제 2: IR Deck 분석 (청크 처리)")
    print("=" * 80)
    
    irdeck_pdf = os.path.join(INPUT_DIR, "sample_irdeck.pdf")
    
    if os.path.exists(irdeck_pdf):
        docai_result = run_document_ai_pipeline(
            pdf_path=irdeck_pdf,
            processor_type="OCR",
            enable_enhancement=True,
            use_chunking=True,
            pages_per_chunk=15
        )
        
        docai_json = os.path.join(OUTPUT_DIR, "sample_irdeck_docai_ocr.json")
        layoutlm_result = run_layoutlm_pipeline(
            pdf_path=irdeck_pdf,
            docai_json_path=docai_json,
            doc_type="ir_deck"
        )
        
        report_path = os.path.join(OUTPUT_DIR, "sample_irdeck_report.txt")
        generate_comprehensive_report(
            pdf_path=irdeck_pdf,
            docai_result=docai_result,
            layoutlm_result=layoutlm_result,
            output_path=report_path
        )
    else:
        print(f"⚠️ 파일 없음: {irdeck_pdf}")
    
    # 예제 3: 공고문
    print("\n" + "=" * 80)
    print("📄 예제 3: 공고문 분석")
    print("=" * 80)
    
    notice_pdf = os.path.join(INPUT_DIR, "sample_notice.pdf")
    
    if os.path.exists(notice_pdf):
        docai_result = run_document_ai_pipeline(
            pdf_path=notice_pdf,
            processor_type="OCR",
            enable_enhancement=True
        )
        
        docai_json = os.path.join(OUTPUT_DIR, "sample_notice_docai_ocr.json")
        layoutlm_result = run_layoutlm_pipeline(
            pdf_path=notice_pdf,
            docai_json_path=docai_json,
            doc_type="notice"
        )
        
        report_path = os.path.join(OUTPUT_DIR, "sample_notice_report.txt")
        generate_comprehensive_report(
            pdf_path=notice_pdf,
            docai_result=docai_result,
            layoutlm_result=layoutlm_result,
            output_path=report_path
        )
    else:
        print(f"⚠️ 파일 없음: {notice_pdf}")
    
    print("\n" + "=" * 80)
    print("✅ 모든 파이프라인 완료!")
    print("=" * 80)
    print(f"\n📁 결과 확인: {OUTPUT_DIR}/")
    print(f"  - *_docai_ocr.json: Document AI 결과 (강화)")
    print(f"  - *_layoutlm_result.json: LayoutLM 엔티티 추출")
    print(f"  - *_report.txt: 종합 리포트")
    print(f"  - *_chunks/: 청크 처리 결과 (대용량 PDF)\n")


def batch_process_documents(
    pdf_list: List[str],
    doc_type: Optional[str] = None,
    use_chunking: bool = False
):
    """여러 문서 배치 처리"""
    
    print(f"\n🔄 배치 처리 시작: {len(pdf_list)}개 문서")
    
    results = []
    
    for idx, pdf_path in enumerate(pdf_list, 1):
        print(f"\n{'='*80}")
        print(f"📄 [{idx}/{len(pdf_list)}] {Path(pdf_path).name}")
        print(f"{'='*80}")
        
        try:
            docai_result = run_document_ai_pipeline(
                pdf_path=pdf_path,
                processor_type="OCR",
                enable_enhancement=True,
                use_chunking=use_chunking
            )
            
            pdf_name = Path(pdf_path).stem
            docai_json = os.path.join(OUTPUT_DIR, f"{pdf_name}_docai_ocr.json")
            
            layoutlm_result = run_layoutlm_pipeline(
                pdf_path=pdf_path,
                docai_json_path=docai_json,
                doc_type=doc_type
            )
            
            report_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_report.txt")
            generate_comprehensive_report(
                pdf_path, docai_result, layoutlm_result, report_path
            )
            
            results.append({
                "pdf": pdf_path,
                "status": "success",
                "doc_type": layoutlm_result.get("doc_type")
            })
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            results.append({
                "pdf": pdf_path,
                "status": "failed",
                "error": str(e)
            })
    
    print(f"\n{'='*80}")
    print(f"📊 배치 처리 완료")
    print(f"{'='*80}")
    
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"✅ 성공: {success}/{len(pdf_list)}")
    print(f"❌ 실패: {failed}/{len(pdf_list)}")
    
    if failed > 0:
        print(f"\n실패한 문서:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {Path(r['pdf']).name}: {r['error']}")
    
    return results


if __name__ == "__main__":
    main()