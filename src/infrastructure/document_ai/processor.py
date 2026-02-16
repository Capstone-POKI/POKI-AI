"""
Document AI API 호출 + 강화 기능
"""

import json
import re
import os
from typing import Dict, List
from google.cloud import documentai_v1beta3 as documentai

from src.utils.io_utils import save_json, read_bytes
from src.utils.pdf_split import split_pdf

PROJECT_ID = os.getenv("PROJECT_ID", "pitchcoachai")
LOCATION = os.getenv("LOCATION", "us")
PROCESSORS = {
    "OCR": os.getenv("OCR_PROCESSOR_ID", "e41bb5d1cae96184"),
    "LAYOUT": os.getenv("LAYOUT_PROCESSOR_ID", "82698693210d7aa8"),
    "FORM": os.getenv("FORM_PROCESSOR_ID", "662d7f1f1e179648"),
}


# 섹션 감지 패턴
SECTION_KEYWORDS = {
    "cover": ["경진대회", "pitch deck", "ir deck", "발표자료"],
    "background": ["background", "배경", "현황", "문제제기", "시장 배경"],
    "problem": ["problem", "문제점", "pain point", "해결하고자", "불편함"],
    "solution": ["solution", "솔루션", "해결방안", "우리의 답"],
    "product": ["product", "제품", "서비스", "핵심 기능"],
    "market": ["market", "시장", "tam", "sam", "som", "시장 규모", "트렌드"],
    "competition": ["competition", "경쟁", "competitive", "차별점", "경쟁우위"],
    "business_model": ["business model", "비즈니스 모델", "수익 모델", "revenue"],
    "finance": ["finance", "재무", "매출", "투자", "unit economics"],
    "team": ["team", "팀", "구성원", "경력", "멤버"],
    "growth": ["growth", "성장", "확장", "계획", "roadmap", "milestone"],
}


# 숫자 추출 패턴
NUMBER_PATTERNS = {
    "currency_korean": [
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*억\s*원?",
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*조\s*원?",
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*만\s*원?",
    ],
    "percentage": [
        r"(\d+(?:\.\d+)?)\s*%",
    ],
    "quantity": [
        r"(\d+(?:,\d{3})*)\s*억?\s*개",
        r"(\d+(?:,\d{3})*)\s*대",
        r"(\d+(?:,\d{3})*)\s*명",
    ],
}


def process_document(
    file_path: str,
    processor_type: str,
    output_path: str,
    enable_enhancement: bool = True
) -> Dict:
    """Document AI API 호출 + 강화 기능"""
    
    processor_id = PROCESSORS[processor_type]
    
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(PROJECT_ID, LOCATION, processor_id)
    
    print(f"📄 [{processor_type}] {file_path} 분석 시작...")
    
    # 기존 유틸 사용
    content = read_bytes(file_path)
    
    raw_document = documentai.RawDocument(
        content=content,
        mime_type="application/pdf",
    )
    
    # OCR 옵션 강화
    if processor_type == "OCR":
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                compute_style_info=True,
                enable_native_pdf_parsing=True,
                enable_image_quality_scores=True,
                enable_symbol=True,
            )
        )
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document,
            process_options=process_options
        )
    else:
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document
        )
    
    result = client.process_document(request=request)
    doc = result.document
    
    # Document AI Document → dict
    doc_dict = json.loads(documentai.Document.to_json(doc))
    
    # 강화 기능 적용
    if enable_enhancement:
        print(f"🔧 강화 기능 적용 중...")
        doc_dict = detect_sections(doc_dict)
        doc_dict = extract_numbers(doc_dict)
        doc_dict = generate_metadata(doc_dict)
        
        print(f"✅ 강화 완료: {len(doc_dict.get('detected_sections', []))}개 섹션, "
              f"{sum(len(v) for v in doc_dict.get('extracted_numbers', {}).values())}개 숫자 추출")
    
    # 기존 유틸 사용
    save_json(doc_dict, output_path)
    print(f"✅ [{processor_type}] 결과 저장 완료 → {output_path}\n")
    
    return doc_dict


def process_pdf_ocr_in_chunks(
    file_path: str,
    output_dir: str,
    pages_per_chunk: int = 15,
    enable_enhancement: bool = True
) -> List[Dict]:
    """대용량 PDF를 청크로 나누어 OCR 처리"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📄 대용량 PDF 청크 처리: {file_path}")
    print(f"  - 청크 크기: {pages_per_chunk}페이지")
    print(f"  - 출력 디렉토리: {output_dir}")
    
    # 기존 pdf_split 사용
    chunk_files = split_pdf(file_path, output_dir, pages_per_chunk)
    
    print(f"  ✅ {len(chunk_files)}개 청크로 분할 완료\n")
    
    results = []
    
    for idx, chunk_path in enumerate(chunk_files, 1):
        print(f"📄 청크 {idx}/{len(chunk_files)} 처리 중...")
        
        chunk_name = os.path.splitext(os.path.basename(chunk_path))[0]
        output_path = os.path.join(output_dir, f"{chunk_name}_ocr.json")
        
        result = process_document(
            file_path=chunk_path,
            processor_type="OCR",
            output_path=output_path,
            enable_enhancement=enable_enhancement
        )
        
        result["chunk_info"] = {
            "chunk_index": idx,
            "total_chunks": len(chunk_files),
            "chunk_file": chunk_path,
        }
        
        results.append(result)
    
    print(f"\n✅ 전체 {len(results)}개 청크 처리 완료\n")
    
    return results


def detect_sections(doc_dict: Dict) -> Dict:
    """페이지별 섹션 자동 감지"""
    
    pages = doc_dict.get("pages", [])
    full_text = doc_dict.get("text", "")
    detected_sections = []
    
    for page_idx, page in enumerate(pages):
        blocks = page.get("blocks", [])
        if not blocks:
            continue
        
        first_block = blocks[0]
        block_text = _extract_block_text(first_block, full_text).lower()
        
        section_type = "unknown"
        for section, keywords in SECTION_KEYWORDS.items():
            if any(keyword.lower() in block_text for keyword in keywords):
                section_type = section
                break
        
        detected_sections.append({
            "page": page_idx + 1,
            "section": section_type,
            "preview": block_text[:100]
        })
        
        page["detected_section"] = section_type
    
    doc_dict["detected_sections"] = detected_sections
    return doc_dict


def extract_numbers(doc_dict: Dict) -> Dict:
    """숫자/통계 데이터 자동 추출"""
    
    full_text = doc_dict.get("text", "")
    extracted = {
        "currency": [],
        "percentage": [],
        "quantity": [],
    }
    
    # 화폐 추출
    for pattern in NUMBER_PATTERNS["currency_korean"]:
        for match in re.finditer(pattern, full_text):
            extracted["currency"].append({
                "text": match.group(0),
                "value": match.group(1),
                "position": match.start()
            })
    
    # 백분율 추출
    for pattern in NUMBER_PATTERNS["percentage"]:
        for match in re.finditer(pattern, full_text):
            extracted["percentage"].append({
                "text": match.group(0),
                "value": match.group(1),
                "position": match.start()
            })
    
    # 수량 추출
    for pattern in NUMBER_PATTERNS["quantity"]:
        for match in re.finditer(pattern, full_text):
            extracted["quantity"].append({
                "text": match.group(0),
                "value": match.group(1).replace(",", ""),
                "position": match.start()
            })
    
    doc_dict["extracted_numbers"] = extracted
    return doc_dict


def generate_metadata(doc_dict: Dict) -> Dict:
    """메타데이터 자동 생성"""
    
    pages = doc_dict.get("pages", [])
    
    metadata = {
        "total_pages": len(pages),
        "total_blocks": sum(len(p.get("blocks", [])) for p in pages),
        "total_paragraphs": sum(
            len(b.get("paragraphs", []))
            for p in pages
            for b in p.get("blocks", [])
        ),
        "detected_sections": list(set(
            p.get("detected_section", "unknown") for p in pages
        )),
        "has_currency": len(doc_dict.get("extracted_numbers", {}).get("currency", [])) > 0,
        "has_percentage": len(doc_dict.get("extracted_numbers", {}).get("percentage", [])) > 0,
        "language": "ko",
    }
    
    doc_dict["metadata"] = metadata
    return doc_dict


def _extract_block_text(block: Dict, full_text: str) -> str:
    """블록에서 텍스트 추출"""
    
    layout = block.get("layout", {})
    text_anchor = layout.get("textAnchor", {})
    segments = text_anchor.get("textSegments", [])
    
    texts = []
    for segment in segments:
        start = int(segment.get("startIndex", 0))
        end = int(segment.get("endIndex", 0))
        texts.append(full_text[start:end])
    
    return " ".join(texts).strip()


def merge_chunk_results(chunk_results: List[Dict], output_path: str) -> Dict:
    """여러 청크 결과를 하나로 병합"""
    
    if not chunk_results:
        raise ValueError("❌ 병합할 청크 결과가 없습니다.")
    
    print(f"\n🔗 {len(chunk_results)}개 청크 결과 병합 중...")
    
    merged = {
        "text": "",
        "pages": [],
        "detected_sections": [],
        "extracted_numbers": {
            "currency": [],
            "percentage": [],
            "quantity": []
        },
        "metadata": {
            "total_chunks": len(chunk_results)
        }
    }
    
    page_offset = 0
    
    for chunk in chunk_results:
        merged["text"] += chunk.get("text", "")
        
        for page in chunk.get("pages", []):
            page["original_page_number"] = page_offset + page.get("pageNumber", 0)
            merged["pages"].append(page)
        
        page_offset += len(chunk.get("pages", []))
        
        for section in chunk.get("detected_sections", []):
            section["page"] += page_offset - len(chunk.get("pages", []))
            merged["detected_sections"].append(section)
        
        numbers = chunk.get("extracted_numbers", {})
        for num_type in ["currency", "percentage", "quantity"]:
            merged["extracted_numbers"][num_type].extend(numbers.get(num_type, []))
    
    merged["metadata"]["total_pages"] = len(merged["pages"])
    merged["metadata"]["total_blocks"] = sum(
        len(p.get("blocks", [])) for p in merged["pages"]
    )
    
    save_json(merged, output_path)
    print(f"✅ 병합 완료: {output_path}\n")
    
    return merged
