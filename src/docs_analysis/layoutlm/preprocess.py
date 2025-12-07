# src/docs_analysis/layoutlm/preprocess.py

import os
from typing import Dict, List, Tuple, Any
from pdf2image import convert_from_path
from src.utils.io_utils import read_json

# -------------------------------------------------------------------------
# 1. 라벨 정의 (기존 코드 유지)
# -------------------------------------------------------------------------
ANNOUNCEMENT_LABELS = [
    "O",
    "B-공고제목", "I-공고제목", "B-공고번호", "I-공고번호",
    "B-예산금액", "I-예산금액", "B-발주기관", "I-발주기관",
    "B-사업내용", "I-사업내용", "B-계약기간", "I-계약기간",
    "B-제출마감", "I-제출마감",
]

PITCH_DECK_LABELS = [
    "O",
    "B-회사명", "I-회사명", "B-슬로건", "I-슬로건",
    "B-대표자", "I-대표자", "B-연락처", "I-연락처",
    "B-제품명", "I-제품명", "B-제품설명", "I-제품설명",
    "B-핵심기능", "I-핵심기능", "B-특장점", "I-특장점",
    "B-가격", "I-가격", "B-시장규모", "I-시장규모",
    "B-성장률", "I-성장률", "B-타겟시장", "I-타겟시장",
    "B-시장트렌드", "I-시장트렌드", "B-규제정보", "I-규제정보",
    "B-매출액", "I-매출액", "B-투자금액", "I-투자금액",
    "B-비용", "I-비용", "B-가격정책", "I-가격정책",
    "B-기술명", "I-기술명", "B-기술설명", "I-기술설명",
    "B-특허", "I-특허", "B-기술키워드", "I-기술키워드",
    "B-팀원명", "I-팀원명", "B-직책", "I-직책", "B-경력", "I-경력",
    "B-고객사", "I-고객사", "B-파트너사", "I-파트너사", "B-제휴", "I-제휴",
    "B-경쟁사명", "I-경쟁사명", "B-경쟁우위", "I-경쟁우위", "B-차별점", "I-차별점",
    "B-문제점", "I-문제점", "B-솔루션", "I-솔루션", "B-배경", "I-배경", "B-비전", "I-비전",
    "B-날짜", "I-날짜", "B-기간", "I-기간", "B-마일스톤", "I-마일스톤", "B-연혁", "I-연혁",
    "B-통계수치", "I-통계수치",
]

IR_DECK_LABELS = [
    "O",
    "B-회사명", "I-회사명", "B-설립일", "I-설립일", "B-대표자", "I-대표자",
    "B-사업영역", "I-사업영역", "B-제품명", "I-제품명",
    "B-매출액", "I-매출액", "B-영업이익", "I-영업이익", "B-순이익", "I-순이익",
    "B-투자금액", "I-투자금액", "B-투자자", "I-투자자",
    "B-시장규모", "I-시장규모", "B-TAM", "I-TAM", "B-SAM", "I-SAM", "B-SOM", "I-SOM",
    "B-기술역량", "I-기술역량", "B-특허", "I-특허", "B-R&D", "I-R&D",
    "B-팀원명", "I-팀원명", "B-직책", "I-직책", "B-경력", "I-경력", "B-학력", "I-학력",
    "B-고객사", "I-고객사", "B-사용자수", "I-사용자수",
    "B-통계수치", "I-통계수치",
]

# -------------------------------------------------------------------------
# 2. 유틸리티 함수
# -------------------------------------------------------------------------

def get_labels(doc_type: str) -> List[str]:
    doc_type = doc_type.lower()
    if doc_type in ["announcement", "notice"]: return ANNOUNCEMENT_LABELS
    elif doc_type in ["pitch_deck", "pitch"]: return PITCH_DECK_LABELS
    elif doc_type in ["ir_deck", "ir"]: return IR_DECK_LABELS
    else: return PITCH_DECK_LABELS

def get_label_info(doc_type: str = None) -> Dict:
    info = {
        "announcement": {"count": len(ANNOUNCEMENT_LABELS), "labels": ANNOUNCEMENT_LABELS},
        "pitch_deck": {"count": len(PITCH_DECK_LABELS), "labels": PITCH_DECK_LABELS},
        "ir_deck": {"count": len(IR_DECK_LABELS), "labels": IR_DECK_LABELS}
    }
    if doc_type: return info.get(doc_type.lower().replace("_", ""), info)
    return info

def load_docai_json(path: str) -> Dict:
    return read_json(path)

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def convert_bounding_poly(bounding_poly: Dict, width: int, height: int) -> List[int]:
    """좌표 변환 및 0~1000 범위 클램핑 (LayoutLM 필수)"""
    if "normalizedVertices" in bounding_poly:
        verts = bounding_poly["normalizedVertices"]
        xs = [v.get("x", 0) * 1000 for v in verts]
        ys = [v.get("y", 0) * 1000 for v in verts]
    elif "vertices" in bounding_poly:
        verts = bounding_poly["vertices"]
        xs = [v.get("x", 0) / width * 1000 for v in verts]
        ys = [v.get("y", 0) / height * 1000 for v in verts]
    else:
        return [0, 0, 0, 0]
    
    return [
        int(clamp(min(xs), 0, 1000)),
        int(clamp(min(ys), 0, 1000)),
        int(clamp(max(xs), 0, 1000)),
        int(clamp(max(ys), 0, 1000)),
    ]

def extract_text_from_segment(full_text: str, segment: Dict) -> str:
    start = int(segment.get("startIndex", 0))
    end = int(segment.get("endIndex", 0))
    if start >= len(full_text): return "" 
    return full_text[start:end].strip()

# -------------------------------------------------------------------------
# 3. 핵심 전처리 로직 (에러 수정 완료)
# -------------------------------------------------------------------------

def prepare_layoutlm_input(
    doc_json: Dict,
    pdf_path: str,
    processor,
    max_length: int = 512
) -> Dict:
    """Document AI JSON + PDF → LayoutLMv3 입력 텐서"""
    
    pages = doc_json.get("pages", [])
    if not pages:
        raise ValueError("❌ OCR JSON에 pages가 없습니다.")
    
    full_text = doc_json.get("text", "")
    
    print(f"📄 PDF → 이미지 변환 중...")
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"⚠️ 이미지 변환 실패 (Poppler 확인 필요): {e}")
        # 실패 시 빈 이미지 생성 (코드 중단 방지)
        from PIL import Image
        images = [Image.new('RGB', (100, 100)) for _ in range(len(pages))]

    # 이미지 모드 변환 (RGB 강제)
    images = [img.convert("RGB") for img in images]
    
    all_page_tokens = []
    all_page_boxes = []
    all_page_images = []
    
    # 페이지 처리 루프
    loop_count = min(len(pages), len(images))
    
    for idx in range(loop_count):
        page = pages[idx]
        image = images[idx]
        dim = page.get("dimension", {})
        width = dim.get("width", 1)
        height = dim.get("height", 1)
        
        page_tokens = []
        page_boxes = []
        
        # 블록 단위 파싱
        for block in page.get("blocks", []):
            block_layout = block.get("layout", {})
            block_bbox = block_layout.get("boundingPoly")
            
            # 텍스트 세그먼트 추출 로직
            segments_to_process = []
            if "paragraphs" in block:
                for para in block["paragraphs"]:
                    segments_to_process.extend(para.get("layout", {}).get("textAnchor", {}).get("textSegments", []))
                    # paragraph bbox가 있으면 사용, 없으면 block bbox 사용
                    current_bbox = para.get("layout", {}).get("boundingPoly", block_bbox)
            else:
                segments_to_process = block_layout.get("textAnchor", {}).get("textSegments", [])
                current_bbox = block_bbox
            
            if not current_bbox: continue
            
            # 텍스트와 좌표 매핑
            for segment in segments_to_process:
                text = extract_text_from_segment(full_text, segment)
                if not text or text.isspace(): continue
                
                norm_bbox = convert_bounding_poly(current_bbox, width, height)
                
                # 단어 단위로 쪼개서 추가
                for word in text.split():
                    if word.strip():
                        page_tokens.append(word)
                        page_boxes.append(norm_bbox)
        
        # 🔥 [핵심 수정] 빈 페이지(텍스트 없는 슬라이드) 처리
        # 이게 없으면 Processor가 텐서를 만들다가 멈춥니다.
        if not page_tokens:
            # print(f"  ⚠️ {idx+1}페이지는 텍스트가 없습니다. (Empty Placeholder 추가)")
            page_tokens = ["<IMAGE>"]
            page_boxes = [[0, 0, 0, 0]]
        
        all_page_tokens.append(page_tokens)
        all_page_boxes.append(page_boxes)
        all_page_images.append(image)
    
    print(f"\n🔍 전처리 완료:")
    print(f"  - 총 페이지: {len(all_page_images)}")
    print(f"  - 총 토큰 수: {sum(len(t) for t in all_page_tokens)}")
    
    print(f"\n🤖 LayoutLM Encoding (Padding=True)...")
    
    # 🔥 Processor 호출 (안전장치 포함)
    try:
        encoding = processor(
            images=all_page_images,
            text=all_page_tokens,
            boxes=all_page_boxes,
            return_tensors="pt",
            padding="max_length",  # 배치 처리 시 필수
            truncation=True,
            max_length=max_length
        )
        return encoding
        
    except Exception as e:
        print(f"❌ Processor Encoding 오류: {e}")
        raise e

def print_label_statistics():
    pass