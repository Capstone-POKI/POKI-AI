# src/layoutlm/preprocess.py
"""
LayoutLM 전처리 및 라벨 정의
"""

from typing import Dict, List, Tuple
from pdf2image import convert_from_path
from src.utils.io_utils import read_json


# 공고문 라벨 (17개)
ANNOUNCEMENT_LABELS = [
    "O",
    "B-공고제목", "I-공고제목",
    "B-공고번호", "I-공고번호",
    "B-예산금액", "I-예산금액",
    "B-발주기관", "I-발주기관",
    "B-사업내용", "I-사업내용",
    "B-계약기간", "I-계약기간",
    "B-제출마감", "I-제출마감",
]

# Pitch Deck 라벨 (85개)
PITCH_DECK_LABELS = [
    "O",
    # 회사 기본 정보
    "B-회사명", "I-회사명",
    "B-슬로건", "I-슬로건",
    "B-대표자", "I-대표자",
    "B-연락처", "I-연락처",
    
    # 제품/서비스
    "B-제품명", "I-제품명",
    "B-제품설명", "I-제품설명",
    "B-핵심기능", "I-핵심기능",
    "B-특장점", "I-특장점",
    "B-가격", "I-가격",
    
    # 시장 정보
    "B-시장규모", "I-시장규모",
    "B-성장률", "I-성장률",
    "B-타겟시장", "I-타겟시장",
    "B-시장트렌드", "I-시장트렌드",
    "B-규제정보", "I-규제정보",
    
    # 재무 정보
    "B-매출액", "I-매출액",
    "B-투자금액", "I-투자금액",
    "B-비용", "I-비용",
    "B-가격정책", "I-가격정책",
    
    # 기술 정보
    "B-기술명", "I-기술명",
    "B-기술설명", "I-기술설명",
    "B-특허", "I-특허",
    "B-기술키워드", "I-기술키워드",
    
    # 팀 정보
    "B-팀원명", "I-팀원명",
    "B-직책", "I-직책",
    "B-경력", "I-경력",
    
    # 고객/파트너
    "B-고객사", "I-고객사",
    "B-파트너사", "I-파트너사",
    "B-제휴", "I-제휴",
    
    # 경쟁사
    "B-경쟁사명", "I-경쟁사명",
    "B-경쟁우위", "I-경쟁우위",
    "B-차별점", "I-차별점",
    
    # 문제/솔루션
    "B-문제점", "I-문제점",
    "B-솔루션", "I-솔루션",
    "B-배경", "I-배경",
    "B-비전", "I-비전",
    
    # 마일스톤
    "B-날짜", "I-날짜",
    "B-기간", "I-기간",
    "B-마일스톤", "I-마일스톤",
    "B-연혁", "I-연혁",
    
    # 통계/기타
    "B-통계수치", "I-통계수치",
]

# IR Deck 라벨 (47개)
IR_DECK_LABELS = [
    "O",
    # 회사 정보
    "B-회사명", "I-회사명",
    "B-설립일", "I-설립일",
    "B-대표자", "I-대표자",
    
    # 사업 영역
    "B-사업영역", "I-사업영역",
    "B-제품명", "I-제품명",
    
    # 재무 정보 (상세)
    "B-매출액", "I-매출액",
    "B-영업이익", "I-영업이익",
    "B-순이익", "I-순이익",
    "B-투자금액", "I-투자금액",
    "B-투자자", "I-투자자",
    
    # 시장 정보
    "B-시장규모", "I-시장규모",
    "B-TAM", "I-TAM",
    "B-SAM", "I-SAM",
    "B-SOM", "I-SOM",
    
    # 기술 역량
    "B-기술역량", "I-기술역량",
    "B-특허", "I-특허",
    "B-R&D", "I-R&D",
    
    # 팀 정보
    "B-팀원명", "I-팀원명",
    "B-직책", "I-직책",
    "B-경력", "I-경력",
    "B-학력", "I-학력",
    
    # 고객 정보
    "B-고객사", "I-고객사",
    "B-사용자수", "I-사용자수",
    
    # 통계
    "B-통계수치", "I-통계수치",
]


def get_labels(doc_type: str) -> List[str]:
    """문서 타입에 맞는 라벨 반환"""
    
    doc_type = doc_type.lower()
    
    if doc_type in ["announcement", "notice"]:
        return ANNOUNCEMENT_LABELS
    elif doc_type in ["pitch_deck", "pitch"]:
        return PITCH_DECK_LABELS
    elif doc_type in ["ir_deck", "ir"]:
        return IR_DECK_LABELS
    else:
        print(f"⚠️ 알 수 없는 문서 타입: {doc_type}, 기본값(pitch_deck) 사용")
        return PITCH_DECK_LABELS


def get_label_info(doc_type: str = None) -> Dict:
    """라벨 정보 딕셔너리 반환"""
    
    info = {
        "announcement": {
            "count": len(ANNOUNCEMENT_LABELS),
            "labels": ANNOUNCEMENT_LABELS,
            "description": "공고문 (예산금액, 발주기관, 사업내용 등)"
        },
        "pitch_deck": {
            "count": len(PITCH_DECK_LABELS),
            "labels": PITCH_DECK_LABELS,
            "description": "피칭 자료 (제품, 시장, 팀, 재무, 경쟁사 등)"
        },
        "ir_deck": {
            "count": len(IR_DECK_LABELS),
            "labels": IR_DECK_LABELS,
            "description": "IR 자료 (매출, 영업이익, TAM/SAM/SOM 등)"
        }
    }
    
    if doc_type:
        return info.get(doc_type.lower().replace("_", ""), info)
    
    return info


def load_docai_json(path: str) -> Dict:
    """Document AI JSON 로드"""
    return read_json(path)


def convert_bounding_poly(bounding_poly: Dict, width: int, height: int) -> List[int]:
    """Document AI boundingPoly → LayoutLM normalized bbox (0-1000)"""
    
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
        int(min(xs)),
        int(min(ys)),
        int(max(xs)),
        int(max(ys)),
    ]


def extract_text_from_segment(full_text: str, segment: Dict) -> str:
    """textAnchor segment에서 실제 텍스트 추출"""
    
    start = int(segment.get("startIndex", 0))
    end = int(segment.get("endIndex", 0))
    return full_text[start:end].strip()


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
    images = convert_from_path(pdf_path)
    
    if len(images) != len(pages):
        print(f"⚠️ 경고: PDF 페이지 수({len(images)})와 OCR 페이지 수({len(pages)})가 다릅니다.")
    
    all_page_tokens: List[List[str]] = []
    all_page_boxes: List[List[List[int]]] = []
    all_page_images: List = []
    
    for idx, page in enumerate(pages):
        dim = page.get("dimension", {})
        width = dim.get("width", 1)
        height = dim.get("height", 1)
        
        page_tokens = []
        page_boxes = []
        
        for block in page.get("blocks", []):
            block_layout = block.get("layout", {})
            block_text_anchor = block_layout.get("textAnchor", {})
            block_bbox = block_layout.get("boundingPoly")
            
            if not block_bbox:
                continue
            
            if "paragraphs" in block and block.get("paragraphs"):
                for paragraph in block.get("paragraphs", []):
                    para_layout = paragraph.get("layout", {})
                    para_text_anchor = para_layout.get("textAnchor", {})
                    para_bbox = para_layout.get("boundingPoly")
                    
                    if not para_bbox:
                        continue
                    
                    for segment in para_text_anchor.get("textSegments", []):
                        text = extract_text_from_segment(full_text, segment)
                        
                        if not text or text.isspace():
                            continue
                        
                        words = text.split()
                        norm_bbox = convert_bounding_poly(para_bbox, width, height)
                        
                        for word in words:
                            if word.strip():
                                page_tokens.append(word)
                                page_boxes.append(norm_bbox)
            else:
                for segment in block_text_anchor.get("textSegments", []):
                    text = extract_text_from_segment(full_text, segment)
                    
                    if not text or text.isspace():
                        continue
                    
                    words = text.split()
                    norm_bbox = convert_bounding_poly(block_bbox, width, height)
                    
                    for word in words:
                        if word.strip():
                            page_tokens.append(word)
                            page_boxes.append(norm_bbox)
        
        all_page_tokens.append(page_tokens)
        all_page_boxes.append(page_boxes)
        
        if idx < len(images):
            all_page_images.append(images[idx])
        else:
            all_page_images.append(images[-1])
    
    total_tokens = sum(len(t) for t in all_page_tokens)
    print(f"\n🔍 전처리 결과:")
    print(f"  - 페이지 수: {len(all_page_images)}")
    print(f"  - 총 토큰 수: {total_tokens}")
    
    if total_tokens > 0:
        print(f"  - 첫 페이지 토큰 샘플: {all_page_tokens[0][:10]}")
        print(f"  - 첫 페이지 bbox 샘플: {all_page_boxes[0][:2]}")
    else:
        print("  ⚠️ 경고: 추출된 토큰이 없습니다!")
    
    print(f"\n🤖 LayoutLM Processor 인코딩 중...")
    encoding = processor(
        images=all_page_images,
        text=all_page_tokens,
        boxes=all_page_boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    
    print(f"  ✅ 인코딩 완료")
    print(f"  - input_ids shape: {encoding['input_ids'].shape}")
    print(f"  - bbox shape: {encoding['bbox'].shape}")
    print(f"  - pixel_values shape: {encoding['pixel_values'].shape}\n")
    
    return encoding


def print_label_statistics():
    """라벨 통계 출력"""
    
    print("\n" + "=" * 80)
    print("📊 라벨 시스템 통계")
    print("=" * 80)
    
    info = get_label_info()
    
    total_labels = sum(v["count"] for v in info.values())
    print(f"\n✅ 전체 라벨 수: {total_labels}개")
    
    print(f"\n📋 문서 타입별:")
    for doc_type, data in info.items():
        print(f"  {doc_type:15s}: {data['count']:3d}개 - {data['description']}")
    
    print("\n" + "=" * 80)