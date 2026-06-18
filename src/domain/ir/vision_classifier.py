"""Gemini Vision 기반 슬라이드 유형 분류 모듈."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from src.infrastructure.gemini.client import GeminiJSONClient

VALID_CATEGORIES = {
    "COVER", "PROBLEM", "SOLUTION", "PRODUCT", "MARKET",
    "BUSINESS_MODEL", "TRACTION", "COMPETITION", "TEAM",
    "FINANCE", "ASK", "GTM", "OTHER",
}

_CLASSIFICATION_PROMPT = """
다음 IR 피치덱 슬라이드 이미지를 보고 슬라이드 유형을 분류하세요.
슬라이드 헤더, 제목, 아이콘, 레이아웃을 모두 참고하세요.

## 분류 기준
- COVER: 표지, 회사 로고+제품 이미지, 목차, 마지막 감사 슬라이드
- PROBLEM: 고객 문제, pain point, 현황 분석, 시장의 불편함
- SOLUTION: 해결책, 핵심 기능, 서비스 설명, 솔루션 개요
- PRODUCT: 제품 데모, UI/UX 화면, 기술 아키텍처, 기술 상세
- MARKET: TAM/SAM/SOM, 시장 규모, 성장률, 시장 트렌드 수치
- BUSINESS_MODEL: 수익 모델, 가격 정책, BM 구조도
- TRACTION: 현재까지 달성한 성과 수치. 미래 목표가 포함되어 있어도 실제 달성 이력(MAU, 매출, 계약, MOU, 앱 다운로드, 이벤트 완료 등)이 하나라도 있으면 TRACTION
  [TRACTION O] "현재 MAU 3만명, 목표 10만명" → 현재 수치 있으므로 TRACTION
  [TRACTION O] "앱 런칭 이벤트 완료, 다운로드 2천건" → 완료된 마케팅 성과이므로 TRACTION
  [TRACTION O] "대학교 MOU 40EA, 벤처기업 인증 완료" → 달성 이력 있으므로 TRACTION
  [TRACTION O] "영업이익 증가 추이 그래프 + 2025 목표" → 추이(과거) 있으므로 TRACTION
  [TRACTION X] "2027년까지 매출 1000억 달성 목표만 있음" → 현재 수치 전혀 없으면 FINANCE
- COMPETITION: 경쟁사 비교표, 차별점, 포지셔닝 맵
- TEAM: 팀원 프로필 사진, 경력, 조직도
- FINANCE: 순수하게 미래 재무 계획만 있는 슬라이드. 투자금 사용처, 손익분기점 예측, 향후 매출 목표만 있고 현재 달성 수치가 전혀 없는 경우에만 FINANCE
  [FINANCE O] "2026년 매출 목표 50억, BEP 달성 예정" → 미래 계획만 있으므로 FINANCE
  [FINANCE O] "투자금 30억: R&D 50%, 마케팅 30%, 인건비 20%" → 자금 사용 계획만 있으므로 FINANCE
  [FINANCE X] "영업이익 증가 추이 + 2025 목표" → 추이(과거 성과) 있으므로 TRACTION
- ASK: 투자 요청 금액·지분 구조·자금 사용 계획 명시. 핵심: '우리가 얼마를 원한다'
  [긍정] "시리즈A 30억 요청, R&D 50%/마케팅 30%" → ASK
  [부정] "2025→2027 연도별 매출 로드맵" → TRACTION 또는 FINANCE
- GTM: 미래 고객 획득 전략, 마케팅 채널, 영업 계획, 글로벌 진출 전략. 핵심: '어떻게 고객을 확보할 것인가'
  [긍정] "초기 타겟: 베트남 대도시, 패스트트랙 인허가로 진입" → GTM
  [긍정] "마케팅 추진 전략: 초기/중기/후기 단계별 채널 계획" → GTM
  [부정] "TAM 6조, SAM 4113억 시장 규모" → MARKET
  [부정] "MOU 40개 체결 완료" → 이미 완료된 실적이므로 TRACTION
- OTHER: 전환 슬라이드(훅), 영상 캡처, 이미지만 있는 슬라이드

## 주의사항
- TRACTION vs GTM: 이미 체결된 파트너십/성과 → TRACTION, 앞으로 할 계획 → GTM
- TRACTION vs ASK: 달성 수치 → TRACTION, 투자 요청 금액 → ASK
- GTM vs MARKET: 채널/전략 → GTM, 시장 규모 수치(TAM/SAM/SOM) → MARKET
- 슬라이드 헤더(제목) 텍스트를 최우선으로 참고하세요

## 출력 형식 (JSON만 반환)
{"category": "...", "category_confidence": 0.0~1.0, "reason": "한 문장 근거"}
""".strip()


def pdf_to_images(pdf_path: str, dpi: int = 100) -> Dict[int, str]:
    """PDF를 페이지별 JPG로 변환. {page_number: image_path} 반환."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = Path(tmp_dir) / "slide"
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-jpeg", pdf_path, str(prefix)],
            check=True, capture_output=True,
        )
        images = {}
        for img_file in sorted(Path(tmp_dir).glob("slide-*.jpg")):
            parts = img_file.stem.split("-")
            page_num = int(parts[-1])
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                dest = Path(tmp_file.name)
            shutil.copy(img_file, dest)
            images[page_num] = str(dest)
    return images


def classify_slides_vision(
    pdf_path: str,
    gemini: Optional[GeminiJSONClient] = None,
    existing_images: Optional[Dict[int, str]] = None,
) -> Dict[int, Dict]:
    """
    PDF 슬라이드를 Vision으로 분류.
    existing_images가 있으면 PDF 변환 스킵.
    반환: {slide_index: {category, confidence}}
    """
    if gemini is None:
        gemini = GeminiJSONClient()
    if not gemini.model:
        raise RuntimeError("Gemini API 키가 설정되지 않았습니다.")

    if existing_images:
        images = existing_images
    else:
        print(f"   📷 [Vision] PDF → 이미지 변환 중: {Path(pdf_path).name}")
        images = pdf_to_images(pdf_path)

    results: Dict[int, Dict] = {}
    total = len(images)
    for i, (page_num, img_path) in enumerate(sorted(images.items()), 1):
        if i % 5 == 0 or i == total:
            print(f"   📷 [Vision] 분류 진행: {i}/{total}")
        try:
            out = gemini.generate_json_with_image(_CLASSIFICATION_PROMPT, img_path, temperature=0.1)
            cat = str(out.get("category", "OTHER")).upper()
            if cat not in VALID_CATEGORIES:
                cat = "OTHER"
            conf = float(out.get("category_confidence", 0.7))
            results[page_num] = {
                "category": cat,
                "confidence": max(0.0, min(1.0, conf)),
                "reason": str(out.get("reason", "")),
            }
        except Exception as e:
            results[page_num] = {"category": "OTHER", "confidence": 0.2, "reason": str(e)}

    return results


def apply_vision_to_slides(
    slides: list,
    vision_results: Dict[int, Dict],
    confidence_threshold: float = 0.6,
) -> int:
    """
    Vision 분류 결과를 슬라이드에 적용.
    confidence >= threshold 인 경우만 오버라이드.
    반환: 오버라이드된 슬라이드 수
    """
    overridden = 0
    for slide in slides:
        page_num = slide.get("page_number") or slide.get("slide_number") or slide.get("page_num")
        if page_num is None:
            continue
        vision = vision_results.get(page_num)
        if vision and vision["confidence"] >= confidence_threshold:
            slide["category"] = vision["category"]
            slide["category_confidence"] = vision["confidence"]
            slide["vision_reason"] = vision["reason"]
            slide["vision_classified"] = True
            overridden += 1
    return overridden
