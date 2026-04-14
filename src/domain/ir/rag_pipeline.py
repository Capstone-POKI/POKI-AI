import json
import os
import re
from io import BytesIO
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdf2image import convert_from_path
from PIL import Image

from src.infrastructure.embedding.client import EmbeddingClient, GeminiEmbeddingClient
from src.infrastructure.gemini.client import GeminiJSONClient
from src.infrastructure.storage.s3_adapter import S3Adapter


_RESAMPLING = getattr(Image, "Resampling", Image)
THUMBNAIL_RESAMPLE_FILTER = _RESAMPLING.LANCZOS


DEFAULT_TOP_K = 3
SIM_HIGH = 0.72
SIM_MID = 0.60
DEFAULT_LLM_SLIDE_LIMIT = 50
GROUP_CATEGORY_PRIORS = {
    "PROBLEM": {"PROBLEM", "MARKET"},
    "SOLUTION": {"SOLUTION", "PRODUCT"},
    "MARKET_BM": {"MARKET", "BUSINESS_MODEL", "COMPETITION", "GTM"},
    "TRACTION": {"TRACTION", "MARKET"},
    "TEAM": {"TEAM"},
    "FINANCE": {"FINANCE", "BUSINESS_MODEL", "ASK"},
}

CATEGORY_KEYWORDS = {
    "PROBLEM": [
        "문제",
        "pain",
        "불편",
        "한계",
        "리스크",
        "왜 필요한",
        "니즈",
        "현황",
    ],
    "SOLUTION": [
        "해결",
        "솔루션",
        "solution",
        "as-is",
        "to-be",
        "개선",
        "제안",
        "approach",
    ],
    "PRODUCT": [
        "제품",
        "프로세스",
        "아키텍처",
        "ui",
        "ux",
        "화면",
        "인터페이스",
        "스크린샷",
        "데모",
        "flow",
        "작동 방식",
        "사용 흐름",
        "사용자 여정",
        "시연",
    ],
    "MARKET": [
        "tam",
        "sam",
        "som",
        "시장",
        "시장규모",
        "cagr",
        "성장률",
        "수요",
        "고객수",
    ],
    "BUSINESS_MODEL": [
        "비즈니스 모델",
        "bm",
        "수익",
        "수수료",
        "구독",
        "pricing",
        "ltv",
        "arpu",
        "arr",
        "unit economics",
    ],
    "TRACTION": [
        "mou",
        "loi",
        "poc",
        "매출",
        "실매출",
        "mrr",
        "arr",
        "활성 사용자",
        "mau",
        "dau",
        "런칭",
        "베타",
        "파일럿",
        "계약",
        "재계약",
        "선정",
        "인증",
        "특허 등록",
        "고객사",
        "지표",
    ],
    "COMPETITION": [
        "경쟁",
        "경쟁사",
        "차별",
        "비교",
        "포지셔닝",
        "moat",
    ],
    "TEAM": [
        "팀",
        "ceo",
        "cto",
        "coo",
        "cso",
        "cmo",
        "founder",
        "자문",
        "경력",
        "학력",
    ],
    "FINANCE": [
        "재무",
        "손익",
        "bep",
        "burn",
        "runway",
        "투자",
        "자금",
        "cashflow",
        "ipo",
    ],
    "ASK": [
        "로드맵",
        "roadmap",
        "계획",
        "milestone",
        "마일스톤",
        "phase",
        "일정",
        "분기",
        "q1",
        "q2",
        "q3",
        "q4",
        "2026",
        "2027",
        "2028",
        "next step",
        "요청",
        "문의",
    ],
    "GTM": [
        "go-to-market",
        "gtm",
        "진입 전략",
        "초기 시장",
        "영업 전략",
        "제휴",
        "파트너십",
        "partnership",
        "가맹",
        "제휴 완료",
        "협업",
        "확장 전략",
        "초기 고객",
        "채널",
    ],
}

CATEGORY_PRIORITY = [
    "TRACTION",
    "FINANCE",
    "BUSINESS_MODEL",
    "GTM",
    "MARKET",
    "TEAM",
    "COMPETITION",
    "SOLUTION",
    "PRODUCT",
    "PROBLEM",
    "ASK",
    "COVER",
    "OTHER",
]

VALID_SLIDE_MIN_TEXT_LEN = 24
INVALID_SLIDE_DUPLICATE_SIM = 0.92
INVALID_SLIDE_MIXED_SECTION_SIM = 0.88

DISPLAY_CATEGORY_LABELS = {
    "COVER": "표지/구성",
    "PROBLEM": "문제정의",
    "SOLUTION": "솔루션",
    "PRODUCT": "제품/데모",
    "MARKET": "시장분석",
    "BUSINESS_MODEL": "비즈니스모델",
    "TRACTION": "실적",
    "COMPETITION": "경쟁분석",
    "TEAM": "팀 소개",
    "FINANCE": "재무/자금",
    "ASK": "요청사항",
    "GTM": "진입전략",
    "OTHER": "기타",
}

_PIPELINE_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _load_pipeline_config() -> Dict[str, Any]:
    global _PIPELINE_CONFIG_CACHE
    if _PIPELINE_CONFIG_CACHE is not None:
        return _PIPELINE_CONFIG_CACHE
    default_path = Path("data/config/pitchcoach_pipeline_config.json")
    cfg_path = Path(os.getenv("PITCHCOACH_PIPELINE_CONFIG_PATH", str(default_path)))
    if cfg_path.exists():
        try:
            _PIPELINE_CONFIG_CACHE = json.loads(cfg_path.read_text(encoding="utf-8"))
            return _PIPELINE_CONFIG_CACHE
        except Exception:
            pass
    _PIPELINE_CONFIG_CACHE = {}
    return _PIPELINE_CONFIG_CACHE


def _sim_high() -> float:
    cfg = _load_pipeline_config()
    cfg_val = (
        cfg.get("matching", {})
        .get("similarity_threshold", {})
        .get("high", SIM_HIGH)
    )
    try:
        return float(os.getenv("IR_SIM_HIGH", str(cfg_val)))
    except Exception:
        return SIM_HIGH


def _sim_mid() -> float:
    cfg = _load_pipeline_config()
    cfg_val = (
        cfg.get("matching", {})
        .get("similarity_threshold", {})
        .get("mid", SIM_MID)
    )
    try:
        return float(os.getenv("IR_SIM_MID", str(cfg_val)))
    except Exception:
        return SIM_MID


def _sim_low() -> float:
    cfg = _load_pipeline_config()
    cfg_val = cfg.get("matching", {}).get("similarity_threshold", {}).get("low")
    if cfg_val is None:
        cfg_val = max(0.0, _sim_mid() - 0.10)
    try:
        return float(os.getenv("IR_SIM_LOW", str(cfg_val)))
    except Exception:
        return max(0.0, _sim_mid() - 0.10)


def _top_k() -> int:
    cfg = _load_pipeline_config()
    cfg_val = cfg.get("matching", {}).get("top_k", DEFAULT_TOP_K)
    try:
        v = int(os.getenv("IR_TOP_K", str(cfg_val)))
        return max(1, min(10, v))
    except Exception:
        return DEFAULT_TOP_K


def run_rag_ir_analysis(
    docai_result: Dict[str, Any],
    output_path: str,
    strategy: Optional[Dict[str, Any]] = None,
    analysis_version: int = 1,
    pitch_type: Optional[str] = None,
    source_pdf_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not docai_result:
        raise RuntimeError("OCR 결과가 비어 있습니다.")

    print("🧠 [RAG] 분석 엔진 시작")
    gemini = GeminiJSONClient()
    slides = _build_slides(docai_result)
    pitch_type = _resolve_pitch_type(strategy, pitch_type, slides)
    rubric = _load_rubric(pitch_type)
    rubric = _apply_strategy_to_rubric(rubric, strategy)
    print(f"🧾 [RAG] 슬라이드 로드 완료: {len(slides)}장")

    print("🏷️ [RAG] 슬라이드 분류/요약 진행")
    _classify_and_summarize_slides(slides, gemini)
    _mark_slide_validity(slides)
    valid_slides = [slide for slide in slides if slide.get("is_valid", True)]
    excluded_slides = [
        {
            "slide_number": slide["slide_number"],
            "reason": slide.get("invalid_reason", ""),
        }
        for slide in slides
        if not slide.get("is_valid", True)
    ]
    print(f"🧹 [RAG] 유효 슬라이드 {len(valid_slides)}장, 제외 {len(excluded_slides)}장")

    print("🔢 [RAG] 임베딩 생성 진행")
    embed_client = _init_embedding_client()
    _embed_slides(valid_slides, embed_client)
    _embed_rubric_items(rubric, embed_client)

    print("📚 [RAG] 루브릭 매칭 및 기준별 점수 계산")
    criteria_scores = _score_criteria_with_rag(
        slides=valid_slides,
        rubric=rubric,
        gemini=gemini,
    )

    print("🧩 [RAG] 종합 점수/가이드 생성")
    total_score = _calculate_total_score(criteria_scores, rubric)
    deck_summary_bundle = _build_deck_summary_bundle_llm(
        slides=valid_slides,
        criteria_scores=criteria_scores,
        total_score=total_score,
        strategy=strategy,
        gemini=gemini,
    )
    deck_score = _build_deck_score(
        criteria_scores=criteria_scores,
        rubric=rubric,
        total_score=total_score,
        llm_bundle=deck_summary_bundle,
    )
    presentation_guide = _build_presentation_guide(
        slides=valid_slides,
        criteria_scores=criteria_scores,
        strategy=strategy,
        llm_bundle=deck_summary_bundle,
    )
    deck_id = Path(output_path).stem.replace("_final", "")
    thumbnail_urls = _build_slide_thumbnail_urls(
        source_pdf_path=source_pdf_path,
        deck_id=deck_id,
        slide_numbers=[int(s.get("slide_number", 0) or 0) for s in valid_slides],
    )
    slide_cards = _build_slide_cards(
        valid_slides,
        criteria_scores,
        gemini,
        thumbnail_urls=thumbnail_urls,
    )

    final_output: Dict[str, Any] = {
        "analysis_version": analysis_version,
        "analysis_method": "RAG+LLM" if gemini.model else "RAG+RuleBased",
        "pitch_type": pitch_type,
        "deck_score": deck_score,
        "criteria_scores": criteria_scores,
        "presentation_guide": presentation_guide,
        "slides": slide_cards,
        "meta": {
            "filename": docai_result.get("metadata", {}).get("filename", "unknown"),
            "total_slides": len(valid_slides),
            "raw_total_slides": len(slides),
            "excluded_slides": excluded_slides,
            "analysis_model": gemini.model_name if gemini.model else None,
            "embedding_model": "gemini-embedding-001",
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("✅ [RAG] 최종 JSON 생성 완료")
    return final_output


def _resolve_pitch_type(
    strategy: Optional[Dict[str, Any]],
    explicit_pitch_type: Optional[str],
    slides: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if explicit_pitch_type:
        mapped = _normalize_pitch_type(explicit_pitch_type)
        if mapped:
            return mapped

    if not strategy:
        inferred = _infer_pitch_type_from_slides(slides or [])
        return inferred or "VC_DEMO"
    raw = str(strategy.get("type", "")).lower()
    if "government" in raw or "정부" in raw or "grant" in raw:
        return "GOV_SUPPORT"
    if "competition" in raw or "경진" in raw:
        return "STARTUP_CONTEST"
    return "VC_DEMO"


def _normalize_pitch_type(value: str) -> Optional[str]:
    v = value.strip().upper()
    mapping = {
        "VC_DEMO": "VC_DEMO",
        "GOVERNMENT": "GOV_SUPPORT",
        "GOV_SUPPORT": "GOV_SUPPORT",
        "STARTUP_CONTEST": "STARTUP_CONTEST",
        "COMPETITION": "STARTUP_CONTEST",
        # Elevator pitch is closest to short-form VC story in current v1 rubric set.
        "ELEVATOR": "VC_DEMO",
    }
    return mapping.get(v)


def _infer_pitch_type_from_slides(slides: List[Dict[str, Any]]) -> Optional[str]:
    text = " ".join((s.get("clean_text", "") or "") for s in slides[:10]).lower()
    if not text:
        return None

    gov_keywords = ["정부", "지원사업", "정책", "지자체", "공공", "과제", "k-startup", "창업패키지"]
    comp_keywords = ["경진대회", "contest", "해커톤", "수상", "데모데이 외 대회"]

    gov_hits = sum(1 for k in gov_keywords if k in text)
    comp_hits = sum(1 for k in comp_keywords if k in text)

    if gov_hits >= 2:
        return "GOV_SUPPORT"
    if comp_hits >= 2:
        return "STARTUP_CONTEST"
    return "VC_DEMO"


def _build_slides(docai_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = docai_result.get("pages", [])
    detected_sections = docai_result.get("detected_sections", [])
    section_map = {s.get("page"): s.get("section", "unknown") for s in detected_sections if isinstance(s, dict)}
    full_text = docai_result.get("text", "")

    slides: List[Dict[str, Any]] = []
    for idx, page in enumerate(pages):
        page_num = idx + 1
        page_text = _extract_page_text(page, full_text).strip()
        slides.append(
            {
                "slide_number": page_num,
                "raw_text": page_text,
                "clean_text": _clean_text(page_text),
                "short_summary": "",
                "key_claims": [],
                "category": section_map.get(page_num, "OTHER").upper(),
                "category_confidence": 0.5,
                "text_deficiency_flag": len(page_text) < 20,
                "ocr_noise_ratio": _estimate_ocr_noise_ratio(page_text),
                "is_valid": True,
                "invalid_reason": "",
                "embedding": [],
            }
        )
    return slides


def _extract_page_text(page: Dict[str, Any], full_text: str) -> str:
    parts: List[str] = []
    for block in page.get("blocks", []):
        layout = block.get("layout", {})
        for segment in layout.get("textAnchor", {}).get("textSegments", []):
            start = int(segment.get("startIndex", 0))
            end = int(segment.get("endIndex", 0))
            parts.append(full_text[start:end])
    return " ".join(parts)


def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", " ", text)
    text = re.sub(r"\\+pm\^\{?\*?\}?\d*(?:\.\d+)?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"([가-힣A-Za-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([가-힣A-Za-z])", r"\1 \2", text)
    text = re.sub(r"[|]{2,}", " ", text)
    text = re.sub(r"[_=~]{2,}", " ", text)
    text = re.sub(r"\b[a-zA-Z]{1}\b", " ", text)
    text = re.sub(r"(\d+)\s+(원|명|건|회|개|배|곳|년|월|일|%)\b", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _estimate_ocr_noise_ratio(text: str) -> float:
    tokens = re.findall(r"\S+", text or "")
    if not tokens:
        return 1.0
    noisy = 0
    for token in tokens:
        if len(token) <= 1:
            noisy += 1
            continue
        if re.search(r"[\\^*_]{2,}", token):
            noisy += 1
            continue
        if re.fullmatch(r"[\W_]+", token):
            noisy += 1
            continue
    return noisy / max(1, len(tokens))


def _split_meaningful_lines(text: str) -> List[str]:
    raw = re.split(r"[\n\r•●■\-]+", text or "")
    lines = []
    for line in raw:
        cleaned = _clean_text(line)
        if len(cleaned) < 6:
            continue
        if cleaned in lines:
            continue
        lines.append(cleaned)
    return lines


def _summarize_slide_text(text: str, category: str) -> str:
    lines = _split_meaningful_lines(text)
    if not lines:
        return "OCR 텍스트가 불안정해 핵심 메시지를 추출하기 어렵습니다."

    title = lines[0]
    evidence_lines = []
    for line in lines[1:]:
        has_number = bool(re.search(r"\d", line))
        long_enough = len(line) >= 14
        if has_number or long_enough:
            evidence_lines.append(line)
        if len(evidence_lines) == 2:
            break

    category_hint = {
        "COMPETITION": "경쟁 비교 기준을 제시합니다.",
        "BUSINESS_MODEL": "수익 구조와 과금 방식을 설명합니다.",
        "GTM": "초기 진입 및 확장 전략을 설명합니다.",
        "TRACTION": "실제 검증 지표 또는 고객 반응을 제시합니다.",
        "MARKET": "시장 규모 또는 성장 근거를 설명합니다.",
        "TEAM": "핵심 팀 구성과 역할을 설명합니다.",
    }.get(category)

    parts = [title]
    if evidence_lines:
        parts.append(" / ".join(evidence_lines))
    elif category_hint:
        parts.append(category_hint)
    summary = " ".join(parts)
    return summary[:280]


def _classify_and_summarize_slides(slides: List[Dict[str, Any]], gemini: GeminiJSONClient) -> None:
    llm_slide_limit = int(os.getenv("IR_LLM_SLIDE_LIMIT", str(DEFAULT_LLM_SLIDE_LIMIT)))
    use_llm_count = min(len(slides), llm_slide_limit) if gemini.model else 0
    if gemini.model:
        print(f"   - Gemini 분류 대상: {use_llm_count}/{len(slides)}장 (나머지 규칙 기반)")

    for idx, slide in enumerate(slides, start=1):
        if idx % 5 == 0 or idx == len(slides):
            print(f"   - 분류 진행: {idx}/{len(slides)}")
        if not slide["clean_text"]:
            slide["short_summary"] = "텍스트가 거의 없는 슬라이드입니다."
            slide["key_claims"] = []
            slide["category"], slide["category_confidence"] = "OTHER", 0.2
            continue

        if gemini.model and idx <= use_llm_count:
            try:
                prompt = (
                    "다음 IR 피치덱 슬라이드의 OCR 텍스트를 분석하여 JSON만 반환하세요.\n\n"
                    "## 분류 규칙\n"
                    "category는 다음 중 하나를 선택하세요:\n"
                    "- COVER: 표지, 마지막 감사 슬라이드, 목차\n"
                    "- PROBLEM: 고객 문제, pain point, 현황 분석\n"
                    "- SOLUTION: 해결책, 핵심 기능, 서비스 설명\n"
                    "- PRODUCT: 제품 데모, UI/UX, 화면 설명, 기술 아키텍처\n"
                    "- MARKET: TAM/SAM/SOM, 시장 규모, 성장률, 시장 트렌드\n"
                    "- BUSINESS_MODEL: 수익 모델, 가격 정책, BM 구조\n"
                    "- TRACTION: 실적, 매출, 사용자 지표, PoC 결과, 고객사\n"
                    "- COMPETITION: 경쟁사 비교표, 차별점, 포지셔닝\n"
                    "- TEAM: 팀 구성, 멤버 경력, 조직도\n"
                    "- FINANCE: 재무 계획, 투자 요청, 자금 사용 계획\n"
                    "- ASK: 로드맵, 마일스톤, 향후 계획, 투자 요청\n"
                    "- GTM: 시장 진입 전략, 초기 고객 확보, 영업 전략, 제휴 전략\n"
                    "- OTHER: 위 어디에도 해당하지 않는 경우\n\n"
                    "## 주의사항\n"
                    "- 경쟁사 비교표가 있으면 COMPETITION (TEAM 아님)\n"
                    "- 'Business Model'이 제목에 있으면 BUSINESS_MODEL (TRACTION 아님)\n"
                    "- 시장 진입/영업/제휴 전략은 GTM (MARKET 아님)\n"
                    "- OCR 노이즈(깨진 글자, 잘못된 띄어쓰기)는 무시하고 의미를 파악하세요\n\n"
                    "## short_summary 작성 규칙\n"
                    "- OCR 원문을 그대로 복사하지 마세요\n"
                    "- 슬라이드의 핵심 메시지를 1~2문장으로 재작성하세요\n"
                    "- 브랜드명, 수치는 정확히 유지하세요\n\n"
                    "## 출력 형식\n"
                    "{\"category\":\"...\",\"category_confidence\":0.0~1.0,"
                    "\"short_summary\":\"핵심 메시지 1~2문장\","
                    "\"key_claims\":[\"주장1\", \"주장2\"]}\n\n"
                    f"[슬라이드 {idx}/{len(slides)} OCR 텍스트]\n{slide['clean_text'][:4000]}"
                )
                out = gemini.generate_json(prompt, temperature=0.1)
                category = str(out.get("category", "OTHER")).upper()
                if category not in {
                    "COVER",
                    "PROBLEM",
                    "SOLUTION",
                    "PRODUCT",
                    "MARKET",
                    "BUSINESS_MODEL",
                    "TRACTION",
                    "COMPETITION",
                    "TEAM",
                    "FINANCE",
                    "ASK",
                    "GTM",
                    "OTHER",
                }:
                    category = "OTHER"
                slide["category"] = category
                slide["category_confidence"] = _clamp01(float(out.get("category_confidence", 0.7)))
                raw_summary = str(out.get("short_summary", "")).strip()
                slide["short_summary"] = _sanitize_summary(raw_summary) or _summarize_slide_text(slide["clean_text"], category)
                claims = out.get("key_claims", [])
                if isinstance(claims, list):
                    slide["key_claims"] = [_sanitize_claim(str(c)) for c in claims if _sanitize_claim(str(c))][:5]
                else:
                    slide["key_claims"] = []
                _postprocess_slide_category(slide)
                continue
            except Exception:
                pass

        # Fallback classification and summary
        category, conf = _keyword_classify_with_confidence(
            slide["clean_text"],
            slide_number=int(slide.get("slide_number", 0)),
            total_slides=len(slides),
        )
        slide["category"] = category
        slide["category_confidence"] = conf
        slide["short_summary"] = _summarize_slide_text(slide["clean_text"], category)
        slide["key_claims"] = _extract_claims(slide["clean_text"])
        _postprocess_slide_category(slide)


def _sanitize_summary(text: str) -> str:
    cleaned = _clean_text(text)
    cleaned = re.sub(r"(원시민|위시민|워시편|셀 프 세차)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 12:
        return ""
    return cleaned[:280]


def _sanitize_claim(text: str) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) < 8:
        return ""
    return cleaned[:120]


def _postprocess_slide_category(slide: Dict[str, Any]) -> None:
    text = (slide.get("clean_text", "") or "").lower()
    summary = (slide.get("short_summary", "") or "").lower()
    merged = f"{text} {summary}"

    # 슬라이드 제목 추출 (첫 줄 또는 첫 50자)
    first_line = (text.split("\n")[0] if "\n" in text else text[:80]).strip()

    # 1. 경쟁분석 - 최우선 (competitive, 비교표 등 명확한 시그널)
    competition_signals = ["competitive", "경쟁사", "경쟁 분석", "비교표", "vs ", "포지셔닝", "moat"]
    if any(k in merged for k in competition_signals):
        # 비교 대상이 2개 이상이면 거의 확실한 경쟁분석
        competitor_count = sum(1 for k in ["불가", "가능", "○", "×", "✓", "✗"] if k in text)
        if competitor_count >= 2 or any(k in first_line for k in ["competitive", "경쟁"]):
            slide["category"] = "COMPETITION"
            slide["category_confidence"] = max(float(slide.get("category_confidence", 0.0)), 0.88)
            return

    # 2. 실적/트랙션 - PoC, 고객 만족도, 실매출 등 검증 지표
    traction_signals = ["poc", "고객 만족", "추천 의사", "시장 검증", "실매출", "mrr", "arr",
                        "mau", "dau", "런칭", "베타", "파일럿", "재계약", "누적", "성장률",
                        "전월 대비", "객단가", "사용량 증가"]
    traction_title_signals = ["poc", "traction", "실적", "성과", "검증"]
    if any(k in first_line for k in traction_title_signals) or sum(1 for k in traction_signals if k in merged) >= 2:
        slide["category"] = "TRACTION"
        slide["category_confidence"] = max(float(slide.get("category_confidence", 0.0)), 0.85)
        return

    # 3. 시장 - TAM/SAM/SOM, 시장 규모 데이터가 핵심
    market_signals = ["tam", "sam", "som", "시장규모", "시장 규모", "cagr", "시장 성장"]
    market_title_signals = ["market", "시장"]
    if any(k in first_line for k in market_title_signals) or sum(1 for k in market_signals if k in merged) >= 2:
        slide["category"] = "MARKET"
        slide["category_confidence"] = max(float(slide.get("category_confidence", 0.0)), 0.85)
        return

    # 4. 비즈니스 모델
    if any(k in merged for k in ["business model", "비즈니스 모델", "수수료", "구독", "pricing", "수익 모델", "과금"]):
        if any(k in first_line for k in ["business model", "비즈니스", "bm", "수익"]):
            slide["category"] = "BUSINESS_MODEL"
            slide["category_confidence"] = max(float(slide.get("category_confidence", 0.0)), 0.85)
            return

    # 5. GTM / 사업 전략
    if any(k in merged for k in ["go-to-market", "gtm", "영업 전략", "초기 고객", "제휴", "파트너십", "진입 전략", "확장 전략"]):
        slide["category"] = "GTM"
        slide["category_confidence"] = max(float(slide.get("category_confidence", 0.0)), 0.8)
        return

    # 6. 자금 계획
    if any(k in merged for k in ["투자금", "자금 사용", "use of proceeds", "runway", "bep", "손익", "투자 유치"]):
        if any(k in first_line for k in ["investment", "투자", "자금", "ask"]):
            slide["category"] = "FINANCE"
            slide["category_confidence"] = max(float(slide.get("category_confidence", 0.0)), 0.82)
            return


def _mark_slide_validity(slides: List[Dict[str, Any]]) -> None:
    previous_valid_text = ""
    for index, slide in enumerate(slides):
        text = slide.get("clean_text", "") or ""
        token_count = len(re.findall(r"[a-zA-Z0-9가-힣]+", text))
        noise_ratio = float(slide.get("ocr_noise_ratio", 0.0))

        if token_count < VALID_SLIDE_MIN_TEXT_LEN and noise_ratio >= 0.22:
            slide["is_valid"] = False
            slide["invalid_reason"] = "OCR 텍스트가 너무 적고 노이즈 비율이 높습니다."
            continue

        heading_hits = sum(
            1
            for keyword in ["problem", "solution", "team", "market", "traction", "business model", "finance"]
            if keyword in text.lower()
        )
        if heading_hits >= 3:
            slide["is_valid"] = False
            slide["invalid_reason"] = "여러 슬라이드 요소가 합쳐진 혼합 페이지로 판단됩니다."
            continue

        if previous_valid_text:
            sim_prev = _lexical_similarity(text, previous_valid_text)
            if sim_prev >= INVALID_SLIDE_DUPLICATE_SIM:
                slide["is_valid"] = False
                slide["invalid_reason"] = "이전 슬라이드와 내용이 거의 동일한 중복 페이지입니다."
                continue
            if sim_prev >= INVALID_SLIDE_MIXED_SECTION_SIM and any(k in text.lower() for k in ["problem", "solution", "team", "market"]):
                slide["is_valid"] = False
                slide["invalid_reason"] = "여러 슬라이드 요소가 합쳐진 혼합 페이지로 판단됩니다."
                continue

        if slide.get("category") == "OTHER" and token_count < 40 and noise_ratio > 0.3:
            slide["is_valid"] = False
            slide["invalid_reason"] = "의미 있는 텍스트가 부족해 분석 신뢰도가 낮습니다."
            continue

        previous_valid_text = text


def _keyword_classify(text: str) -> str:
    return _keyword_classify_with_confidence(text)[0]


def _keyword_classify_with_confidence(
    text: str,
    slide_number: int = 0,
    total_slides: int = 0,
) -> Tuple[str, float]:
    t = (text or "").lower()
    if not t.strip():
        return "OTHER", 0.2
    token_count = len(re.findall(r"[a-zA-Z0-9가-힣]+", t))
    line_count = len([ln for ln in re.split(r"[\r\n]+", t) if ln.strip()])
    num_cnt = len(re.findall(r"\d", t))

    has_market_core = any(k in t for k in ["tam", "sam", "som", "시장규모", "cagr", "성장률", "시장 점유", "시장 성장"])
    has_plan_core = any(k in t for k in ["로드맵", "roadmap", "마일스톤", "q1", "q2", "q3", "q4", "2026", "2027", "2028", "일정", "분기"])
    has_traction_core = any(k in t for k in ["mou", "loi", "poc", "계약", "선정", "인증", "실매출", "mrr", "arr", "재계약", "파일럿", "베타"])
    has_product_core = any(k in t for k in ["ui", "ux", "화면", "스크린샷", "데모", "프로세스", "flow", "워크플로우", "아키텍처"])
    has_solution_core = any(k in t for k in ["해결", "솔루션", "개선", "제안", "대안", "효과", "as-is", "to-be"])
    has_team_core = any(k in t for k in ["ceo", "cto", "coo", "cmo", "founder", "팀", "멤버", "프로필", "경력", "학력"])
    has_cover_core = any(k in t for k in ["thank", "thanks", "q&a", "감사", "문의", "logo", "chapter", "section", "part", "overview", "agenda"])
    has_competition_core = any(k in t for k in ["경쟁", "경쟁사", "비교", "포지셔닝", "moat"])
    has_bm_core = any(k in t for k in ["business model", "비즈니스 모델", "수익", "수수료", "구독", "pricing"])
    has_gtm_core = any(k in t for k in ["go-to-market", "gtm", "초기 고객", "영업 전략", "제휴", "파트너십", "진입 전략", "확장 전략"])
    has_finance_core = any(k in t for k in ["투자", "투자금", "자금 사용", "손익", "bep", "runway", "재무"])

    # Cover/title slide heuristic
    if total_slides > 0:
        if slide_number == 1 and token_count <= 40:
            return "COVER", 0.82
        if slide_number == total_slides and token_count <= 60:
            return "COVER", 0.80
    if len(t) < 130 and any(k in t for k in ["ir", "pitch", "발표", "데크", "deck"]):
        return "COVER", 0.78
    if has_cover_core and token_count <= 20 and line_count <= 4 and num_cnt == 0:
        return "COVER", 0.70
    if token_count <= 10 and num_cnt == 0 and not (
        has_market_core
        or has_traction_core
        or has_product_core
        or has_solution_core
        or has_team_core
        or has_competition_core
        or has_bm_core
        or has_gtm_core
        or has_finance_core
    ):
        return "COVER", 0.66
    if total_slides > 0 and slide_number <= 2 and has_team_core:
        return "TEAM", 0.70

    scores: Dict[str, float] = {k: 0.0 for k in CATEGORY_KEYWORDS}
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                scores[category] += 1.0

    if num_cnt >= 8:
        scores["MARKET"] += 1.0
        scores["BUSINESS_MODEL"] += 0.9
        scores["TRACTION"] += 0.7
    if has_market_core:
        scores["MARKET"] += 1.0
    else:
        scores["MARKET"] -= 0.6
    if has_plan_core:
        scores["ASK"] += 1.1
        scores["TRACTION"] -= 0.4
    if has_traction_core:
        scores["TRACTION"] += 1.2
        scores["PROBLEM"] -= 0.3
        if has_bm_core:
            scores["BUSINESS_MODEL"] += 0.8
            scores["TRACTION"] -= 0.4
    if has_product_core:
        scores["PRODUCT"] += 1.2
        if has_plan_core:
            scores["PRODUCT"] += 0.4
            scores["ASK"] -= 0.3
    if has_solution_core:
        scores["SOLUTION"] += 1.2
        if not has_product_core:
            scores["PRODUCT"] -= 0.5
    if has_team_core:
        scores["TEAM"] += 1.4
        scores["TRACTION"] -= 0.4
        scores["SOLUTION"] -= 0.3
    if has_competition_core:
        scores["COMPETITION"] += 1.8
        scores["TEAM"] -= 0.7
    if has_bm_core:
        scores["BUSINESS_MODEL"] += 1.5
        scores["TRACTION"] -= 0.5
        scores["FINANCE"] -= 0.2
    if has_gtm_core:
        scores["GTM"] += 1.7
        scores["MARKET"] -= 0.4
        scores["ASK"] -= 0.2
    if has_finance_core:
        scores["FINANCE"] += 1.5
        scores["ASK"] += 0.5

    # Prefer solution for problem->solution storytelling slides.
    if has_solution_core and any(k in t for k in ["문제", "pain", "불편", "한계", "차별"]):
        scores["SOLUTION"] += 0.6
        scores["PROBLEM"] += 0.3

    if has_team_core and has_competition_core:
        scores["COMPETITION"] += 0.6
        scores["TEAM"] -= 0.8
    if has_gtm_core and has_market_core:
        scores["GTM"] += 0.4
        scores["MARKET"] -= 0.2

    best_score = max(scores.values()) if scores else 0.0
    if best_score < 1.0:
        return "OTHER", 0.35

    top = [c for c, s in scores.items() if s == best_score]
    conf = _clamp01(0.45 + min(0.40, best_score / 9.0))
    for c in CATEGORY_PRIORITY:
        if c in top:
            return c, conf
    return (top[0], conf) if top else ("OTHER", 0.35)


def _extract_claims(text: str) -> List[str]:
    chunks = re.split(r"[.!?\n]", text)
    claims = [_sanitize_claim(c.strip()) for c in chunks if len(c.strip()) >= 15]
    claims = [c for c in claims if c]
    return claims[:5]


def _init_embedding_client() -> Optional[GeminiEmbeddingClient]:
    # Prefer Gemini REST embedding (no Vertex AI dependency)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            client = GeminiEmbeddingClient(model_name="gemini-embedding-001")
            # Quick validation
            test = client.embed(["test"], task_type="RETRIEVAL_DOCUMENT")
            if test and len(test[0]) > 100:
                print("   - Gemini Embedding API 연결 성공")
                return client
        except Exception as e:
            print(f"   - Gemini Embedding API 실패: {e}")

    # Fallback: Vertex AI
    if os.getenv("ENABLE_VERTEX_EMBEDDING") == "1":
        try:
            project_id = os.getenv("PROJECT_ID")
            if project_id:
                vertex_client = EmbeddingClient(model_name="gemini-embedding-001")
                vertex_client.init_vertex(project_id=project_id, location=os.getenv("LOCATION", "us-central1"))
                return vertex_client
        except Exception:
            pass

    print("   - ⚠️ 임베딩 클라이언트 없음 → 폴백 임베딩 사용")
    return None


def _embed_slides(slides: List[Dict[str, Any]], embed_client: Optional[EmbeddingClient]) -> None:
    texts = [f"{s['clean_text']}\n{s['short_summary']}" for s in slides]
    vectors = _embed_texts(texts, embed_client)
    for slide, vec in zip(slides, vectors):
        slide["embedding"] = vec


def _load_rubric(pitch_type: str) -> Dict[str, Any]:
    path = os.getenv("PITCHCOACH_RUBRIC_PATH")
    if path:
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                rubric = data.get("rubrics", {}).get(pitch_type)
                if rubric:
                    return rubric
            except Exception:
                pass
    return _default_rubric(pitch_type)


def _default_rubric(pitch_type: str) -> Dict[str, Any]:
    common = [
        {
            "group_id": "PROBLEM",
            "group_name": "문제정의",
            "group_weight": 0.2,
            "max_score": 20,
            "items": [
                {"item_id": "PR_01", "item_name": "구체적 문제 정의", "description": "문제를 구체적으로 정의", "max_score": 10, "fail_if_missing": True},
                {"item_id": "PR_02", "item_name": "검증 근거", "description": "고객 니즈/데이터 근거", "max_score": 10, "fail_if_missing": False},
            ],
        },
        {
            "group_id": "SOLUTION",
            "group_name": "솔루션",
            "group_weight": 0.2,
            "max_score": 20,
            "items": [
                {"item_id": "SO_01", "item_name": "해결책 명확성", "description": "해결책의 구체성", "max_score": 10, "fail_if_missing": True},
                {"item_id": "SO_02", "item_name": "차별화", "description": "경쟁 대비 차별 포인트", "max_score": 10, "fail_if_missing": False},
            ],
        },
        {
            "group_id": "MARKET_BM",
            "group_name": "시장/비즈니스",
            "group_weight": 0.25,
            "max_score": 25,
            "items": [
                {"item_id": "MK_01", "item_name": "시장규모", "description": "TAM/SAM/SOM 등 시장 규모", "max_score": 10, "fail_if_missing": False},
                {"item_id": "MK_02", "item_name": "수익모델", "description": "BM/가격/수익식", "max_score": 15, "fail_if_missing": True},
            ],
        },
        {
            "group_id": "TRACTION",
            "group_name": "실적",
            "group_weight": 0.15,
            "max_score": 15,
            "items": [
                {"item_id": "TR_01", "item_name": "검증지표", "description": "베타/사용자/매출 지표", "max_score": 15, "fail_if_missing": False},
            ],
        },
        {
            "group_id": "TEAM",
            "group_name": "팀",
            "group_weight": 0.1,
            "max_score": 10,
            "items": [
                {"item_id": "TE_01", "item_name": "팀 역량", "description": "대표/핵심팀 역량", "max_score": 10, "fail_if_missing": False},
            ],
        },
        {
            "group_id": "FINANCE",
            "group_name": "자금 계획",
            "group_weight": 0.1,
            "max_score": 10,
            "items": [
                {"item_id": "FI_01", "item_name": "자금 활용 계획", "description": "자금 배분과 계획", "max_score": 10, "fail_if_missing": False},
            ],
        },
    ]
    return {"pitch_type": pitch_type, "total_points": 100, "groups": common}


def _apply_strategy_to_rubric(
    rubric: Dict[str, Any],
    strategy: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not strategy:
        return rubric

    criteria = strategy.get("evaluation_criteria", [])
    if not isinstance(criteria, list) or not criteria:
        return rubric

    parsed: List[Tuple[str, float]] = []
    for item in criteria:
        parsed_item = _parse_strategy_criterion(item)
        if parsed_item:
            parsed.append(parsed_item)

    total_points = sum(points for _, points in parsed)
    if total_points <= 0:
        return rubric

    base_groups = {
        str(group.get("group_id", "")).strip(): dict(group)
        for group in (rubric.get("groups", []) or [])
        if str(group.get("group_id", "")).strip()
    }

    updated_groups = []
    used_group_ids: set[str] = set()
    for idx, (name, points) in enumerate(parsed, start=1):
        mapped = _map_strategy_criterion_to_group(name)
        base = base_groups.get(mapped) if mapped else None
        group_id = mapped or f"CUSTOM_{idx}"

        # 동일 그룹으로 여러 기준이 매핑되면 첫 항목만 기준 그룹으로 쓰고 나머지는 커스텀으로 분리
        if group_id in used_group_ids:
            base = None
            group_id = f"CUSTOM_{idx}"
        used_group_ids.add(group_id)

        if base:
            updated = dict(base)
            updated["group_id"] = group_id
            updated["group_name"] = name
            max_score = float(points)
            updated["max_score"] = max_score
            updated["group_weight"] = max_score / total_points
            raw_items = base.get("items", []) or []
            items = _rescale_group_items(raw_items, max_score)
            updated["items"] = items or raw_items
            updated_groups.append(updated)
            continue

        max_score = float(points)
        updated_groups.append(
            {
                "group_id": group_id,
                "group_name": name,
                "group_weight": max_score / total_points,
                "max_score": max_score,
                "items": [
                    {
                        "item_id": f"{group_id}_01",
                        "item_name": name,
                        "description": f"{name} 관련 근거와 실행 계획",
                        "max_score": max_score,
                        "fail_if_missing": False,
                    }
                ],
            }
        )

    updated_rubric = dict(rubric)
    updated_rubric["groups"] = updated_groups
    updated_rubric["total_points"] = total_points
    return updated_rubric


def _map_strategy_criterion_to_group(name: str) -> Optional[str]:
    normalized = name.replace(" ", "").lower()
    mapping = {
        "문제정의": "PROBLEM",
        "문제인식": "PROBLEM",
        "솔루션": "SOLUTION",
        "혁신성": "SOLUTION",
        "기술혁신성": "SOLUTION",
        "시장/비즈니스": "MARKET_BM",
        "시장비즈니스": "MARKET_BM",
        "시장": "MARKET_BM",
        "시장성": "MARKET_BM",
        "비즈니스": "MARKET_BM",
        "실적": "TRACTION",
        "성장성": "TRACTION",
        "팀": "TEAM",
        "팀역량": "TEAM",
        "창업가(팀)역량": "TEAM",
        "창업가팀역량": "TEAM",
        "창업가역량": "TEAM",
        "자금계획": "FINANCE",
        "자금": "FINANCE",
    }
    return mapping.get(normalized)


def _parse_strategy_criterion(item: Any) -> Optional[Tuple[str, float]]:
    if isinstance(item, dict):
        name = str(item.get("criteria_name") or item.get("name") or "").strip()
        points = item.get("points")
        if name and isinstance(points, (int, float)):
            return name, float(points)
        return None

    if isinstance(item, str):
        raw = item.strip()
        if not raw:
            return None
        m = re.match(r"^(?P<name>.+?)\s*[\(\[]\s*(?P<points>\d+(?:\.\d+)?)\s*(?:점|%)?\s*[\)\]]\s*$", raw)
        if m:
            return m.group("name").strip(), float(m.group("points"))
        m = re.match(r"^(?P<name>.+?)\s*[:：-]\s*(?P<points>\d+(?:\.\d+)?)\s*(?:점|%)?\s*$", raw)
        if m:
            return m.group("name").strip(), float(m.group("points"))
    return None


def _rescale_group_items(raw_items: List[Dict[str, Any]], max_score: float) -> List[Dict[str, Any]]:
    if not raw_items:
        return []
    item_portion = max_score / max(1, len(raw_items))
    items: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        new_item = dict(item)
        if idx == len(raw_items) - 1:
            assigned = round(max_score - item_portion * (len(raw_items) - 1), 2)
        else:
            assigned = round(item_portion, 2)
        new_item["max_score"] = assigned
        items.append(new_item)
    return items


def _embed_rubric_items(rubric: Dict[str, Any], embed_client: Optional[EmbeddingClient]) -> None:
    items = []
    refs: List[Dict[str, Any]] = []
    for group in rubric.get("groups", []):
        for item in group.get("items", []):
            text = f"{item.get('item_name', '')}. {item.get('description', '')}"
            items.append(text)
            refs.append(item)
    vectors = _embed_texts(items, embed_client)
    for item, vec in zip(refs, vectors):
        item["embedding"] = vec


def _embed_texts(texts: List[str], embed_client) -> List[List[float]]:
    if embed_client is not None:
        try:
            return embed_client.embed(texts, task_type="RETRIEVAL_DOCUMENT")
        except Exception as e:
            print(f"   - 임베딩 API 호출 실패: {e}")
    return [_fallback_embed(t) for t in texts]


def _fallback_embed(text: str) -> List[float]:
    # Lightweight deterministic fallback embedding.
    vec = [0.0] * 64
    for token in re.findall(r"[a-zA-Z0-9가-힣_]+", text.lower()):
        idx = hash(token) % len(vec)
        vec[idx] += 1.0
    norm = sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _score_criteria_with_rag(
    slides: List[Dict[str, Any]],
    rubric: Dict[str, Any],
    gemini: GeminiJSONClient,
) -> List[Dict[str, Any]]:
    criteria_scores: List[Dict[str, Any]] = []
    slide_usage_counter: Dict[int, int] = {}

    for group in rubric.get("groups", []):
        group_items = group.get("items", [])
        raw_group_score = 0.0
        raw_group_max = float(group.get("max_score", 0))
        all_related: List[int] = []
        missing_items: List[Dict[str, str]] = []
        coverage_values: List[str] = []
        coverage_weights: List[float] = []
        evidence_for_group: List[Dict[str, Any]] = []

        for item in group_items:
            evidences = _retrieve_top_k(
                item,
                slides,
                top_k=_top_k(),
                group_id=str(group.get("group_id", "")),
                slide_usage_counter=slide_usage_counter,
            )
            max_sim = evidences[0]["similarity"] if evidences else 0.0
            coverage = _decide_coverage(item, evidences, gemini)
            item_max = float(item.get("max_score", 0))
            item_score = _score_item(item_max, coverage, max_sim)

            raw_group_score += item_score
            coverage_values.append(coverage)
            coverage_weights.append(item_max)
            # Bind evidence to score: for covered/partial items, keep at least top evidence.
            if coverage in {"COVERED", "PARTIALLY_COVERED"} and evidences:
                all_related.append(int(evidences[0]["slide_number"]))
                all_related.extend([e["slide_number"] for e in evidences[1:] if e["similarity"] >= (_sim_mid() - 0.1)])
                for evidence in evidences[:2]:
                    sn = int(evidence["slide_number"])
                    slide_usage_counter[sn] = slide_usage_counter.get(sn, 0) + 1
            evidence_for_group.append(
                {
                    "item_id": item.get("item_id"),
                    "item_name": item.get("item_name"),
                    "coverage": coverage,
                    "max_similarity": max_sim,
                    "related_slides": [e["slide_number"] for e in evidences],
                    "top_summary": (evidences[0]["summary"] if evidences else ""),
                    "top_quote": _extract_evidence_quote((evidences[0].get("clean_text", "") if evidences else "")),
                }
            )

            if coverage in {"NOT_COVERED", "PARTIALLY_COVERED"}:
                missing_items.append(
                    {
                        "item_id": item.get("item_id", ""),
                        "item_name": item.get("item_name", ""),
                        "suggestion": _build_missing_suggestion(item, coverage),
                    }
                )

        related_unique = sorted(set(all_related))
        if raw_group_score > 0 and not related_unique:
            fallback_related: List[int] = []
            for ev in evidence_for_group:
                rel = ev.get("related_slides", [])
                if rel:
                    fallback_related.append(int(rel[0]))
            related_unique = sorted(set(fallback_related))
        group_coverage = _reduce_group_coverage(coverage_values, coverage_weights)
        avg_max_similarity = _average(
            [float(ev.get("max_similarity", 0.0)) for ev in evidence_for_group]
        )
        covered_count = sum(1 for v in coverage_values if v == "COVERED")
        partial_count = sum(1 for v in coverage_values if v == "PARTIALLY_COVERED")
        item_count = max(1, len(group_items))
        coverage_ratio = (covered_count + (0.5 * partial_count)) / item_count
        score_100 = int(round((raw_group_score / raw_group_max) * 100)) if raw_group_max > 0 else 0
        score_100, related_unique, feedback_boost = _apply_group_score_floor(
            group_id=str(group.get("group_id", "")),
            score_100=score_100,
            slides=slides,
            related_slides=related_unique,
        )
        feedback, confidence = _build_group_feedback(
            group=group,
            evidence_for_group=evidence_for_group,
            missing_items=missing_items,
            gemini=gemini,
        )
        reliability = _compute_group_reliability(
            llm_confidence=confidence,
            avg_max_similarity=avg_max_similarity,
            related_count=len(related_unique),
            item_count=item_count,
            missing_count=len(missing_items),
            coverage=group_coverage,
        )
        score_100 = _calibrate_group_score(
            score_100=score_100,
            coverage=group_coverage,
            confidence=confidence,
            related_count=len(related_unique),
            missing_count=len(missing_items),
            avg_max_similarity=avg_max_similarity,
            coverage_ratio=coverage_ratio,
            reliability=reliability,
        )
        if feedback_boost:
            feedback = f"{feedback} {feedback_boost}".strip()
        blended_confidence = _clamp01((confidence * 0.4) + (reliability * 0.6))

        criteria_scores.append(
            {
                "criteria_score_id": f"cs-{group.get('group_id', '').lower()}",
                "criteria_id": group.get("group_id"),
                "criteria_name": group.get("group_name"),
                "pitchcoach_interpretation": _group_interpretation(group),
                "raw_score": round(raw_group_score, 2),
                "raw_max_score": raw_group_max,
                "score": max(0, min(100, score_100)),
                "max_score": 100,
                "is_covered": group_coverage != "NOT_COVERED",
                "coverage_status": group_coverage,
                "feedback": feedback,
                "related_slides": related_unique,
                "missing_items": missing_items,
                "confidence": round(blended_confidence, 3),
                "evidence_reliability": round(reliability, 3),
            }
        )
    return _validate_and_repair_criteria(criteria_scores)


def _build_missing_suggestion(item: Dict[str, Any], coverage: str) -> str:
    item_name = str(item.get("item_name", "")).strip()
    desc = str(item.get("description", "")).strip()
    if coverage == "PARTIALLY_COVERED":
        return f"'{item_name}' 관련 내용은 보이지만 근거가 약합니다. {desc}를 수치/사례와 함께 보강하세요."
    return f"'{item_name}' 항목을 명시적으로 추가하세요. {desc}"


def _validate_and_repair_criteria(criteria_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # SEB_01/SEB_02 style guardrail: score>0 must have related slides.
    for c in criteria_scores:
        score = int(c.get("score", 0))
        related = c.get("related_slides", []) or []
        if score > 0 and not related:
            c["score"] = 0
            c["raw_score"] = 0.0
            c["coverage_status"] = "NOT_COVERED"
            c["is_covered"] = False
            c["feedback"] = "근거 슬라이드가 확인되지 않아 점수를 0점으로 보정했습니다."
            if not c.get("missing_items"):
                c["missing_items"] = [
                    {
                        "item_id": f"{c.get('criteria_id', 'unknown')}_MISSING",
                        "item_name": c.get("criteria_name", "기준"),
                        "suggestion": "해당 기준을 다루는 근거 슬라이드를 명시적으로 추가하세요.",
                    }
                ]
    return criteria_scores


def _retrieve_top_k(
    item: Dict[str, Any],
    slides: List[Dict[str, Any]],
    top_k: int,
    group_id: str = "",
    slide_usage_counter: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    item_vec = item.get("embedding", [])
    item_text = f"{item.get('item_name', '')} {item.get('description', '')}".strip()
    prior_categories = GROUP_CATEGORY_PRIORS.get(group_id, set())
    scored = []
    min_retrieval_sim = float(os.getenv("IR_RETR_MIN_SIM", "0.02"))
    for slide in slides:
        if slide.get("text_deficiency_flag"):
            continue
        vec_sim = _cosine(item_vec, slide.get("embedding", []))
        slide_text = f"{slide.get('clean_text', '')} {slide.get('short_summary', '')}"
        lex_sim = _lexical_similarity(item_text, slide_text)
        ngram_sim = _ngram_similarity(item_text, slide_text)
        kw_sim = _keyword_overlap_score(item_text, slide_text)
        blend_sim = (0.40 * vec_sim) + (0.25 * lex_sim) + (0.20 * ngram_sim) + (0.15 * kw_sim)
        robust_sim = max(lex_sim, ngram_sim, (0.85 * vec_sim) + (0.15 * kw_sim))
        sim = max(blend_sim, robust_sim)
        if prior_categories and slide.get("category") in prior_categories:
            sim = min(1.0, sim + 0.12)
            if float(slide.get("category_confidence", 0.0)) >= 0.7:
                sim = min(1.0, sim + 0.04)
        if _item_prefers_numeric(item_text):
            digit_cnt = len(re.findall(r"\d", slide_text))
            if digit_cnt >= 6:
                sim = min(1.0, sim + 0.06)
            elif digit_cnt >= 3:
                sim = min(1.0, sim + 0.03)
        if slide_usage_counter:
            reuse_count = int(slide_usage_counter.get(int(slide["slide_number"]), 0))
            if reuse_count > 0:
                sim = max(0.0, sim - min(0.18, reuse_count * 0.07))
        if sim < min_retrieval_sim:
            continue
        scored.append(
            {
                "slide_number": slide["slide_number"],
                "similarity": sim,
                "summary": slide["short_summary"],
                "clean_text": slide["clean_text"][:1000],
            }
        )
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def _keyword_overlap_score(item_text: str, slide_text: str) -> float:
    item_tokens = set(re.findall(r"[a-zA-Z0-9가-힣_]+", (item_text or "").lower()))
    slide_tokens = set(re.findall(r"[a-zA-Z0-9가-힣_]+", (slide_text or "").lower()))
    if not item_tokens or not slide_tokens:
        return 0.0
    overlap = len(item_tokens & slide_tokens)
    denom = max(1, min(len(item_tokens), 10))
    return max(0.0, min(1.0, overlap / denom))


def _item_prefers_numeric(item_text: str) -> bool:
    t = (item_text or "").lower()
    numeric_hints = [
        "tam",
        "sam",
        "som",
        "시장",
        "매출",
        "수익",
        "가격",
        "arpu",
        "ltv",
        "mau",
        "dau",
        "bep",
        "재무",
        "성장",
        "추세",
    ]
    return any(k in t for k in numeric_hints)


def _lexical_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-zA-Z0-9가-힣_]+", (a or "").lower()))
    tb = set(re.findall(r"[a-zA-Z0-9가-힣_]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return inter / union


def _ngram_similarity(a: str, b: str, n: int = 3) -> float:
    aa = re.sub(r"\s+", "", (a or "").lower())
    bb = re.sub(r"\s+", "", (b or "").lower())
    if len(aa) < n or len(bb) < n:
        return 0.0
    ga = {aa[i : i + n] for i in range(len(aa) - n + 1)}
    gb = {bb[i : i + n] for i in range(len(bb) - n + 1)}
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _decide_coverage(item: Dict[str, Any], evidences: List[Dict[str, Any]], gemini: GeminiJSONClient) -> str:
    max_sim = evidences[0]["similarity"] if evidences else 0.0
    fail_if_missing = bool(item.get("fail_if_missing", False))
    sim_high = _sim_high()
    sim_mid = _sim_mid()
    sim_low = _sim_low()

    # Local/offline fallback: keep partial coverage signal when semantic model is unavailable.
    if not gemini.model and sim_low <= max_sim < sim_mid:
        return "PARTIALLY_COVERED"

    if max_sim >= sim_high:
        if evidences and len((evidences[0].get("clean_text") or "").strip()) < 20:
            return _llm_review(item, evidences[:2], gemini)
        return "COVERED"

    if sim_mid <= max_sim < sim_high:
        reviewed = _llm_review(item, evidences[:2], gemini)
        return "PARTIALLY_COVERED" if reviewed == "NOT_COVERED" else reviewed

    if sim_low <= max_sim < sim_mid and evidences:
        if fail_if_missing:
            reviewed = _llm_review(item, evidences[:2], gemini)
            return "PARTIALLY_COVERED" if reviewed == "NOT_COVERED" else reviewed
        return "PARTIALLY_COVERED"

    if fail_if_missing:
        return _llm_review(item, evidences[:2], gemini)
    return "NOT_COVERED"


def _llm_review(item: Dict[str, Any], evidences: List[Dict[str, Any]], gemini: GeminiJSONClient) -> str:
    if os.getenv("IR_FAST_MODE") == "1":
        top = evidences[0]["similarity"] if evidences else 0.0
        if top >= _sim_high():
            return "COVERED"
        if top >= _sim_mid():
            return "PARTIALLY_COVERED"
        if top >= _sim_low():
            return "PARTIALLY_COVERED"
        return "NOT_COVERED"

    if not gemini.model or not evidences:
        top = evidences[0]["similarity"] if evidences else 0.0
        if top >= _sim_high():
            return "COVERED"
        if top >= _sim_mid():
            return "PARTIALLY_COVERED"
        if top >= _sim_low():
            return "PARTIALLY_COVERED"
        return "NOT_COVERED"
    try:
        prompt = {
            "item_name": item.get("item_name"),
            "item_description": item.get("description"),
            "evidence_slides": [
                {"slide_number": e["slide_number"], "summary": e["summary"], "similarity": e["similarity"]} for e in evidences
            ],
            "question": "증거 슬라이드가 항목을 충족하는가? JSON만 반환",
            "output_format": {"is_relevant": True, "confidence": 0.0},
        }
        out = gemini.generate_json(json.dumps(prompt, ensure_ascii=False), temperature=0.0)
        is_rel = bool(out.get("is_relevant", False))
        conf = _clamp01(float(out.get("confidence", 0.0)))
        if is_rel and conf >= 0.6:
            return "COVERED"
        if is_rel:
            return "PARTIALLY_COVERED"
        return "NOT_COVERED"
    except Exception:
        top = evidences[0]["similarity"] if evidences else 0.0
        if top >= _sim_high():
            return "COVERED"
        if top >= _sim_mid():
            return "PARTIALLY_COVERED"
        if top >= _sim_low():
            return "PARTIALLY_COVERED"
        return "NOT_COVERED"


def _score_item(item_max: float, coverage: str, similarity: float) -> float:
    if coverage == "NOT_COVERED":
        return 0.0
    if coverage == "PARTIALLY_COVERED":
        sim_high = max(0.01, min(0.99, _sim_high()))
        ratio = max(0.30, min(0.75, (similarity / sim_high) * 0.75))
        return round(item_max * ratio, 2)
    # COVERED
    sim_high = max(0.01, min(0.99, _sim_high()))
    ratio = 0.55 + max(0.0, min(1.0, (similarity - sim_high) / (1.0 - sim_high))) * 0.45
    return round(item_max * ratio, 2)


def _reduce_group_coverage(values: List[str], weights: Optional[List[float]] = None) -> str:
    if not values:
        return "NOT_COVERED"
    score_map = {"COVERED": 1.0, "PARTIALLY_COVERED": 0.5, "NOT_COVERED": 0.0}
    if not weights or len(weights) != len(values):
        weighted = sum(score_map.get(v, 0.0) for v in values) / max(1, len(values))
    else:
        denom = sum(max(0.0, w) for w in weights) or 1.0
        weighted = sum(score_map.get(v, 0.0) * max(0.0, w) for v, w in zip(values, weights)) / denom
    if weighted >= 0.60:
        return "COVERED"
    if weighted >= 0.25:
        return "PARTIALLY_COVERED"
    return "NOT_COVERED"


def _group_interpretation(group: Dict[str, Any]) -> str:
    item_names = [str(i.get("item_name", "")).strip() for i in group.get("items", []) if i.get("item_name")]
    return f"{group.get('group_name', '')} 평가는 다음 항목을 중심으로 판단합니다: {', '.join(item_names)}"


def _build_group_feedback(
    group: Dict[str, Any],
    evidence_for_group: List[Dict[str, Any]],
    missing_items: List[Dict[str, str]],
    gemini: GeminiJSONClient,
) -> Tuple[str, float]:
    if os.getenv("IR_FAST_MODE") == "1":
        if missing_items:
            return (
                f"{group.get('group_name')} 항목은 일부 근거가 확인되지만 "
                f"{', '.join(m['item_name'] for m in missing_items[:2])} 보강이 필요합니다.",
                0.68,
            )
        return f"{group.get('group_name')} 항목은 주요 근거가 확인되었습니다.", 0.74

    if gemini.model:
        try:
            prompt = {
                "group_name": group.get("group_name"),
                "group_items": [i.get("item_name") for i in group.get("items", [])],
                "evidence": evidence_for_group,
                "missing_items": missing_items,
                "instruction": (
                    "근거 기반으로 2문장 피드백 작성. 과장 금지. JSON만 반환. "
                    "가능하면 근거 문구(숫자/표현)를 문장 안에 자연스럽게 녹여 쓰되, "
                    "별도 태그([근거 인용 권장] 등)는 출력하지 마세요."
                ),
                "output_format": {"feedback": "...", "confidence": 0.0},
            }
            out = gemini.generate_json(json.dumps(prompt, ensure_ascii=False), temperature=0.2)
            feedback = str(out.get("feedback", "")).strip()
            confidence = _clamp01(float(out.get("confidence", 0.75)))
            if feedback:
                return _append_optional_evidence_quote(feedback, evidence_for_group), confidence
        except Exception:
            pass

    if missing_items:
        top_ev = [ev for ev in evidence_for_group if ev.get("top_summary")]
        ev_summary = _compress_evidence_summary(top_ev[0]["top_summary"]) if top_ev else ""
        ev_hint = f" 근거 예시: {ev_summary}" if ev_summary else ""
        return (
            f"{group.get('group_name')} 항목은 관련 슬라이드가 일부 확인되지만 "
            f"{', '.join(m['item_name'] for m in missing_items[:2])} 근거가 더 필요합니다.{ev_hint}",
            0.65,
        )
    top_ev = [ev for ev in evidence_for_group if ev.get("top_summary")]
    ev_summary = _compress_evidence_summary(top_ev[0]["top_summary"]) if top_ev else ""
    ev_hint = f" 주요 근거: {ev_summary}" if ev_summary else ""
    base = f"{group.get('group_name')} 항목은 근거 슬라이드가 확인되어 비교적 안정적으로 커버되었습니다.{ev_hint}"
    return _append_optional_evidence_quote(base, evidence_for_group), 0.72


def _append_optional_evidence_quote(
    feedback: str,
    evidence_for_group: List[Dict[str, Any]],
) -> str:
    # External tag is intentionally suppressed.
    # Evidence quoting should be handled by prompt-level guidance only.
    _ = evidence_for_group
    return feedback


def _compress_evidence_summary(text: str) -> str:
    summary = _sanitize_summary(text or "")
    if not summary:
        return ""
    tokens = summary.split()
    keyword_heavy = len([t for t in tokens if re.search(r"[A-Za-z가-힣0-9]", t)]) >= 4
    if not keyword_heavy:
        return ""
    return summary[:150] + ("..." if len(summary) > 150 else "")


def _extract_evidence_quote(text: str) -> str:
    raw = re.sub(r"\s+", " ", (text or "")).strip()
    if not raw:
        return ""
    candidates = re.split(r"(?<=[\.\!\?다])\s+", raw)
    candidates = [c.strip(" \"'“”[]()") for c in candidates if c and c.strip()]
    if not candidates:
        candidates = [raw]
    keyword_re = re.compile(r"(시장|매출|수익|성장|고객|검증|실적|투자|경쟁|팀|자금|\d)")
    ranked = sorted(
        candidates,
        key=lambda s: (
            0 if 30 <= len(s) <= 160 else 1,
            0 if keyword_re.search(s) else 1,
            abs(len(s) - 90),
        ),
    )
    quote = ranked[0][:180]
    return quote.strip()


def _average(values: List[float]) -> float:
    clean = [float(v) for v in values]
    if not clean:
        return 0.0
    return sum(clean) / max(1, len(clean))


def _compute_group_reliability(
    llm_confidence: float,
    avg_max_similarity: float,
    related_count: int,
    item_count: int,
    missing_count: int,
    coverage: str,
) -> float:
    related_ratio = min(1.0, related_count / max(1, item_count))
    missing_ratio = min(1.0, missing_count / max(1, item_count))
    coverage_base = 0.0
    if coverage == "COVERED":
        coverage_base = 1.0
    elif coverage == "PARTIALLY_COVERED":
        coverage_base = 0.65

    reliability = (
        (0.35 * _clamp01(llm_confidence))
        + (0.30 * _clamp01(avg_max_similarity))
        + (0.20 * related_ratio)
        + (0.15 * coverage_base)
        - (0.25 * missing_ratio)
    )
    if coverage == "PARTIALLY_COVERED":
        reliability = min(reliability, 0.78)
    elif coverage == "NOT_COVERED":
        reliability = min(reliability, 0.25)
    return _clamp01(reliability)


def _calibrate_group_score(
    score_100: int,
    coverage: str,
    confidence: float,
    related_count: int,
    missing_count: int,
    avg_max_similarity: float,
    coverage_ratio: float,
    reliability: float,
) -> int:
    if coverage == "NOT_COVERED":
        return 0

    raw_score_100 = int(score_100)
    adjusted = float(score_100)

    # 1) Reliability-aware correction (strong evidence keeps score, weak evidence pulls down).
    reliability_center = float(os.getenv("IR_RELIABILITY_CENTER", "0.66"))
    reliability_gain = float(os.getenv("IR_RELIABILITY_GAIN", "35"))
    adjusted += (reliability - reliability_center) * reliability_gain

    # 2) Similarity/coverage correction.
    if avg_max_similarity < _sim_mid():
        adjusted -= 6.0
    elif avg_max_similarity >= (_sim_high() + 0.08):
        adjusted += 2.0
    if coverage_ratio < 0.45:
        adjusted -= 6.0
    elif coverage_ratio < 0.60:
        adjusted -= 2.0
    elif coverage_ratio >= 0.85:
        adjusted += 1.0

    # 3) Coverage-aware cap/floor.
    if coverage == "PARTIALLY_COVERED":
        partial_cap = int(os.getenv("IR_PARTIAL_SCORE_CAP", "82"))
        partial_floor = int(os.getenv("IR_PARTIAL_SCORE_FLOOR_WITH_EVIDENCE", "35"))
        adjusted = min(adjusted, partial_cap)
        if related_count >= 1:
            adjusted = max(adjusted, partial_floor)
    else:  # COVERED
        if reliability < 0.55:
            covered_cap = 90
        elif reliability < 0.72:
            covered_cap = 93
        else:
            covered_cap = 95
        covered_floor = int(os.getenv("IR_COVERED_SCORE_FLOOR_WITH_EVIDENCE", "50"))
        if confidence < 0.55:
            covered_cap = min(covered_cap, 88)
        adjusted = min(adjusted, covered_cap)
        if related_count >= 2 and coverage_ratio >= 0.5:
            adjusted = max(adjusted, covered_floor)
        elif related_count >= 1:
            adjusted = max(adjusted, 45)

    # 4) Missing requirements penalty (stronger for discrimination).
    adjusted -= min(15, missing_count * 5)

    # 4.5) Low-reliability additional pull-down.
    if reliability < 0.45:
        adjusted -= 6.0
    elif reliability < 0.55:
        adjusted -= 3.0

    # 5) Keep calibration bounded around raw evidence score.
    max_upshift = int(os.getenv("IR_MAX_CALIBRATION_UPSHIFT", "4"))
    max_downshift = int(os.getenv("IR_MAX_CALIBRATION_DOWNSHIFT", "24"))
    adjusted = min(adjusted, raw_score_100 + max_upshift)
    adjusted = max(adjusted, raw_score_100 - max_downshift)

    return max(0, min(100, int(round(adjusted))))


def _calculate_total_score(
    criteria_scores: List[Dict[str, Any]],
    rubric: Dict[str, Any],
) -> int:
    # Use calibrated criterion score to keep total score consistent with displayed criteria.
    group_by_id = {g.get("group_id"): g for g in rubric.get("groups", [])}
    weighted_sum = 0.0
    for c in criteria_scores:
        gid = c.get("criteria_id")
        weight = float(group_by_id.get(gid, {}).get("group_weight", 0.0))
        weighted_sum += (float(c.get("score", 0.0)) / 100.0) * weight
    raw_total = int(round(weighted_sum * 100))
    # Discrimination-friendly anchored scaling:
    # keep center near anchor while expanding/reducing spread with gain.
    anchor = float(os.getenv("IR_TOTAL_SCORE_ANCHOR", "65"))
    gain = float(os.getenv("IR_TOTAL_SCORE_GAIN", "1.0"))
    bias = float(os.getenv("IR_TOTAL_SCORE_BIAS", "0"))
    floor = int(os.getenv("IR_TOTAL_SCORE_FLOOR", "0"))
    ceiling = int(os.getenv("IR_TOTAL_SCORE_CEILING", "100"))
    calibrated = anchor + ((raw_total - anchor) * gain) + bias
    return max(floor, min(ceiling, int(round(calibrated))))


def _build_deck_summary_bundle_llm(
    slides: List[Dict[str, Any]],
    criteria_scores: List[Dict[str, Any]],
    total_score: int,
    strategy: Optional[Dict[str, Any]],
    gemini: GeminiJSONClient,
) -> Dict[str, Any]:
    if (
        not gemini
        or not getattr(gemini, "model", None)
        or os.getenv("IR_FAST_MODE") == "1"
    ):
        return {}

    try:
        print("   - deck_bundle: 덱 전체 요약/가이드 생성")
        criteria_info = [
            {
                "name": c["criteria_name"],
                "score": c["score"],
                "coverage": c.get("coverage_status"),
                "feedback": (c.get("feedback", "") or "")[:150],
                "related_slides": c.get("related_slides", []),
                "missing": [m.get("item_name", "") for m in c.get("missing_items", [])],
            }
            for c in criteria_scores
        ]
        slide_overview = [
            {
                "num": s["slide_number"],
                "cat": s.get("category", "OTHER"),
                "summary": (s.get("short_summary", "") or "")[:80],
            }
            for s in slides
        ]
        prompt = json.dumps(
            {
                "instruction": (
                    "IR 피치덱 전체 요약과 발표 가이드를 JSON으로 작성하세요.\n"
                    "- strengths: 덱 전체 강점 2~3개\n"
                    "- improvements: 덱 전체 개선점 2~3개\n"
                    "- top_actions: 즉시 실행 액션 2~3개\n"
                    "- structure_summary: 3~4문장 구조 총평\n"
                    "- guide: 발표 전략 조언 3~5개\n"
                    "- time_allocation: 발표 시간 배분 문자열 3개 이상\n"
                    "- 모든 내용은 실제 슬라이드/기준 점수에 근거해야 하며 일반론 금지"
                ),
                "pitch_type": (strategy or {}).get("type", "VC_DEMO"),
                "presentation_minutes": (strategy or {}).get("presentation_minutes", 8),
                "total_score": total_score,
                "criteria_scores": criteria_info,
                "slides": slide_overview,
                "output_format": {
                    "strengths": ["강점1", "강점2"],
                    "improvements": ["개선점1", "개선점2"],
                    "top_actions": ["액션1", "액션2"],
                    "structure_summary": "총평",
                    "guide": ["가이드1", "가이드2"],
                    "time_allocation": [
                        "오프닝 (1분): 설명",
                        "본론 (6분): 설명",
                        "클로징 (1분): 설명",
                    ],
                },
            },
            ensure_ascii=False,
        )
        out = gemini.generate_json(prompt, temperature=0.2)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _build_deck_score(
    criteria_scores: List[Dict[str, Any]],
    rubric: Dict[str, Any],
    total_score: int,
    llm_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    llm_bundle = llm_bundle or {}

    sorted_low = sorted(criteria_scores, key=lambda x: x.get("score", 0))
    sorted_high = sorted(criteria_scores, key=lambda x: x.get("score", 0), reverse=True)
    strengths = [
        str(s).strip() for s in llm_bundle.get("strengths", []) if str(s).strip()
    ][:3]
    improvements = [
        str(s).strip() for s in llm_bundle.get("improvements", []) if str(s).strip()
    ][:3]
    top_actions = [
        str(s).strip() for s in llm_bundle.get("top_actions", []) if str(s).strip()
    ][:3]
    if not strengths:
        strengths = _build_deck_strengths(sorted_high)
    if not improvements:
        improvements = _build_deck_improvements(sorted_low)
    if not top_actions:
        top_actions = []
        for c in sorted_low[:3]:
            missing = c.get("missing_items", [])
            if missing:
                top_actions.append(f"{c['criteria_name']}: {missing[0].get('suggestion', '')}")
            else:
                top_actions.append(f"{c['criteria_name']}: 핵심 근거 슬라이드 수치를 강화하세요.")

    structure_summary = str(llm_bundle.get("structure_summary", "")).strip()
    if not structure_summary:
        structure_summary = "문제-해결-시장-실행계획의 기본 흐름은 유지되었지만, 낮은 점수 항목의 근거를 보강하면 설득력이 개선됩니다."
    return {
        "total_score": max(0, min(100, total_score)),
        "max_score": 100,
        "scoring_method": "weighted_average",
        "criteria_weights": {
            str(g.get("group_name", "")): float(g.get("group_weight", 0.0))
            for g in rubric.get("groups", [])
            if str(g.get("group_name", "")).strip()
        },
        "structure_summary": structure_summary,
        "strengths": strengths,
        "improvements": improvements,
        "top_actions": top_actions,
    }


def _apply_group_score_floor(
    group_id: str,
    score_100: int,
    slides: List[Dict[str, Any]],
    related_slides: List[int],
) -> Tuple[int, List[int], str]:
    floor_rules = {
        "TEAM": {"categories": {"TEAM"}, "floor": 15, "message": "팀 슬라이드는 존재하지만 역할 적합성과 실행 성과 설명은 더 보강할 필요가 있습니다."},
        "FINANCE": {"categories": {"FINANCE", "ASK", "BUSINESS_MODEL"}, "floor": 15, "message": "자금 또는 실행 계획 슬라이드는 있으나 예산 배분과 마일스톤 연결이 더 필요합니다."},
    }
    rule = floor_rules.get(group_id)
    if not rule or score_100 >= rule["floor"]:
        return score_100, related_slides, ""

    candidate_slides = [
        int(slide["slide_number"])
        for slide in slides
        if slide.get("category") in rule["categories"]
    ]
    if not candidate_slides:
        return score_100, related_slides, ""

    merged = sorted(set((related_slides or []) + candidate_slides[:2]))
    return rule["floor"], merged, rule["message"]


def _build_deck_strengths(criteria_scores: List[Dict[str, Any]]) -> List[str]:
    strengths: List[str] = []
    for criterion in criteria_scores:
        if int(criterion.get("score", 0)) < 55:
            continue
        message = _criterion_strength_message(criterion)
        if message:
            strengths.append(message)
        if len(strengths) == 3:
            break
    if not strengths and criteria_scores:
        fallback = max(criteria_scores, key=lambda x: int(x.get("score", 0)))
        strengths.append(
            f"{fallback.get('criteria_name', '핵심 항목')} 관련 슬라이드가 확인되어 기본 발표 흐름은 갖춰져 있습니다."
        )
    return strengths[:3]


def _build_deck_improvements(criteria_scores: List[Dict[str, Any]]) -> List[str]:
    improvements: List[str] = []
    for criterion in criteria_scores:
        message = _criterion_improvement_message(criterion)
        if message:
            improvements.append(message)
        if len(improvements) == 3:
            break
    if not improvements and criteria_scores:
        improvements.append("낮은 점수 항목의 정량 근거와 비교 기준을 보강해 전체 설득력을 높이세요.")
    return improvements[:3]


def _criterion_strength_message(criterion: Dict[str, Any]) -> str:
    name = str(criterion.get("criteria_name", "")).strip()
    related = criterion.get("related_slides", []) or []
    score = int(criterion.get("score", 0))
    if not name:
        return ""

    templates = {
        "문제정의": "문제와 고객 맥락을 다루는 슬라이드가 포함되어 발표 도입 흐름이 잡혀 있습니다.",
        "솔루션": "해결 방향을 설명하는 슬라이드가 확보되어 문제-해결 연결 구조는 보입니다.",
        "시장/비즈니스": "시장 또는 수익화 관점의 슬라이드가 포함되어 사업화 그림은 제시되고 있습니다.",
        "실적": "실제 검증이나 고객 반응을 시사하는 슬라이드가 있어 실행 흔적이 보입니다.",
        "팀": "팀 또는 역할 소개 슬라이드가 있어 실행 주체를 설명할 기반은 있습니다.",
        "자금 계획": "자금 계획이나 실행 일정 관련 슬라이드가 있어 후속 계획을 연결할 수 있습니다.",
    }
    base = templates.get(name)
    if base:
        if related:
            return f"{base} 근거 슬라이드: {', '.join(str(x) for x in related[:2])}번."
        return base
    if score >= 70:
        return f"{name} 항목은 관련 근거 슬라이드가 확인되어 기본 설득력이 확보되었습니다."
    return ""


def _criterion_improvement_message(criterion: Dict[str, Any]) -> str:
    name = str(criterion.get("criteria_name", "")).strip()
    missing = criterion.get("missing_items", []) or []
    if missing:
        first = missing[0]
        return f"{name} 보강: {str(first.get('suggestion', '')).strip()}"
    feedback = str(criterion.get("feedback", "")).strip()
    if int(criterion.get("score", 0)) < 60 and feedback:
        return f"{name} 보강: {feedback}"
    return ""


def _build_presentation_guide(
    slides: List[Dict[str, Any]],
    criteria_scores: List[Dict[str, Any]],
    strategy: Optional[Dict[str, Any]],
    llm_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    llm_bundle = llm_bundle or {}
    low_criteria = sorted(criteria_scores, key=lambda x: x.get("score", 0))[:2]
    emphasized = []
    seen = set()
    for c in low_criteria:
        for sn in c.get("related_slides", [])[:2]:
            if sn in seen:
                continue
            seen.add(sn)
            emphasized.append(
                {
                    "slide_number": sn,
                    "reason": f"{c.get('criteria_name')} 보완을 위해 해당 슬라이드의 핵심 수치/근거를 먼저 강조하세요.",
                }
            )
    if not emphasized and slides:
        emphasized = [{"slide_number": 1, "reason": "오프닝 메시지를 명확히 제시하세요."}]

    pitch_hint = str((strategy or {}).get("type", "VC_DEMO"))
    guide = [
        str(x).strip() for x in llm_bundle.get("guide", []) if str(x).strip()
    ][:5]
    time_allocation = [
        str(x).strip()
        for x in llm_bundle.get("time_allocation", [])
        if str(x).strip()
    ]

    if not guide:
        guide = [
            "오프닝에서 문제의 크기와 대상 고객을 한 문장으로 먼저 제시하세요.",
            "중간에는 수치 근거가 있는 슬라이드를 중심으로 설명 순서를 유지하세요.",
            "클로징에서는 실행 계획과 요청사항(투자/선정/지원 필요성)을 명확히 정리하세요.",
            f"현재 피칭 맥락({pitch_hint})에 맞춰 심사 포인트를 반복 강조하세요.",
        ]
    if not time_allocation:
        time_allocation = _build_dynamic_time_allocation(slides, strategy)

    return {
        "emphasized_slides": emphasized,
        "guide": guide,
        "time_allocation": time_allocation,
    }


def _build_dynamic_time_allocation(
    slides: List[Dict[str, Any]],
    strategy: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_minutes = (strategy or {}).get("presentation_minutes", 8)
    try:
        total_seconds = max(180, int(float(raw_minutes) * 60))
    except Exception:
        total_seconds = 480

    categories = [str(slide.get("category", "OTHER")) for slide in slides]
    story_slides = sum(1 for category in categories if category in {"PROBLEM", "SOLUTION", "MARKET", "BUSINESS_MODEL", "TRACTION"})
    setup_slides = sum(1 for category in categories if category in {"COVER", "PROBLEM"})
    closing_slides = sum(1 for category in categories if category in {"TEAM", "FINANCE", "ASK", "GTM"})
    total_slides = max(1, len(slides))

    opening_ratio = 0.14 + min(0.05, setup_slides / total_slides * 0.06)
    closing_ratio = 0.12 + min(0.06, closing_slides / total_slides * 0.08)
    if total_seconds <= 300:
        opening_ratio += 0.03
        closing_ratio -= 0.01
    elif total_seconds >= 720:
        closing_ratio += 0.02

    opening_seconds = int(round(total_seconds * opening_ratio))
    closing_seconds = int(round(total_seconds * closing_ratio))
    body_seconds = total_seconds - opening_seconds - closing_seconds

    if story_slides >= max(5, total_slides // 2):
        body_seconds += 20
        opening_seconds = max(40, opening_seconds - 10)
        closing_seconds = max(35, closing_seconds - 10)

    min_opening = 35 if total_seconds < 360 else 45
    min_closing = 30 if total_seconds < 360 else 40
    opening_seconds = max(min_opening, opening_seconds)
    closing_seconds = max(min_closing, closing_seconds)
    body_seconds = max(90, total_seconds - opening_seconds - closing_seconds)

    allocated = opening_seconds + body_seconds + closing_seconds
    if allocated != total_seconds:
        body_seconds += total_seconds - allocated

    return [
        {"section": "오프닝", "seconds": opening_seconds},
        {"section": "본론", "seconds": body_seconds},
        {"section": "클로징", "seconds": closing_seconds},
    ]


def _build_slide_thumbnail_urls(
    source_pdf_path: Optional[str],
    deck_id: str,
    slide_numbers: List[int],
) -> Dict[int, str]:
    if not source_pdf_path:
        return {}

    adapter = S3Adapter()
    if not adapter.enabled:
        return {}

    try:
        dpi = max(72, min(600, int(os.getenv("THUMBNAIL_DPI", "220"))))
        max_width = max(640, min(4096, int(os.getenv("THUMBNAIL_MAX_WIDTH", "1600"))))
        image_format = os.getenv("THUMBNAIL_FORMAT", "PNG").strip().upper()
        if image_format not in {"PNG", "JPEG", "WEBP"}:
            image_format = "PNG"
        jpeg_quality = max(85, min(100, int(os.getenv("THUMBNAIL_JPEG_QUALITY", "95"))))
        webp_quality = max(85, min(100, int(os.getenv("THUMBNAIL_WEBP_QUALITY", "95"))))
        pages = convert_from_path(source_pdf_path, dpi=dpi)
    except Exception:
        return {}

    urls: Dict[int, str] = {}
    unique_slide_numbers = sorted({n for n in slide_numbers if n > 0})
    for slide_number in unique_slide_numbers:
        page_index = slide_number - 1
        if page_index >= len(pages):
            continue
        try:
            image = pages[page_index].convert("RGB")
            if image.width > max_width:
                resize_ratio = max_width / float(image.width)
                resized_height = max(1, int(image.height * resize_ratio))
                image = image.resize(
                    (max_width, resized_height),
                    resample=THUMBNAIL_RESAMPLE_FILTER,
                )
            buffer = BytesIO()
            save_kwargs: Dict[str, Any] = {"format": image_format}
            content_type = "image/png"
            if image_format == "JPEG":
                content_type = "image/jpeg"
                save_kwargs.update(
                    {
                        "quality": jpeg_quality,
                        "optimize": True,
                        # Keep text edges sharp in slide-heavy documents.
                        "subsampling": 0,
                    }
                )
            elif image_format == "WEBP":
                content_type = "image/webp"
                save_kwargs.update({"quality": webp_quality, "method": 6})
            else:
                save_kwargs.update({"optimize": True})

            image.save(buffer, **save_kwargs)
            url = adapter.upload_slide_thumbnail(
                deck_id=deck_id,
                slide_number=slide_number,
                image_bytes=buffer.getvalue(),
                content_type=content_type,
            )
            if url:
                urls[slide_number] = url
        except Exception:
            continue
    return urls


def _llm_score_slides(
    slides: List[Dict[str, Any]],
    criteria_scores: List[Dict[str, Any]],
    score_by_slide: Dict[int, List[int]],
    gemini: GeminiJSONClient,
) -> Dict[int, int]:
    """LLM에게 슬라이드별 점수(0~100)를 요청하여 변별력 있는 점수 산출."""
    try:
        print("   - slide_cards: LLM 슬라이드 점수 생성")
        slide_info = []
        for s in slides:
            sn = s["slide_number"]
            linked = score_by_slide.get(sn, [])
            slide_info.append({
                "num": sn,
                "cat": s.get("category", "OTHER"),
                "summary": (s.get("short_summary", "") or "")[:120],
                "claims_count": len(s.get("key_claims", [])),
                "text_len": len(s.get("clean_text", "")),
                "linked_criteria_avg": int(round(sum(linked) / len(linked))) if linked else 0,
            })
        criteria_info = [
            {"name": c["criteria_name"], "score": c["score"], "coverage": c.get("coverage_status")}
            for c in criteria_scores
        ]
        prompt = json.dumps({
            "instruction": (
                "IR 피치덱의 각 슬라이드를 0~100점으로 채점하세요. JSON만 반환.\n"
                "채점 기준:\n"
                "- 핵심 메시지의 명확성 (투자자가 한 눈에 이해할 수 있는가)\n"
                "- 정량적 근거의 충분성 (수치, 데이터, 출처가 있는가)\n"
                "- 해당 카테고리에 맞는 핵심 요소 포함 여부\n"
                "- 논리적 설득력 (주장과 근거가 연결되는가)\n"
                "점수 분포 가이드:\n"
                "- 90~100: 수치/사례 근거 충분 + 메시지 명확 + 설득력 높음\n"
                "- 70~89: 메시지 명확하지만 근거가 부족하거나 일부 보완 필요\n"
                "- 50~69: 내용은 있지만 구체성 부족, 설득력 약함\n"
                "- 30~49: 핵심 내용 부재 또는 불명확\n"
                "- 0~29: 거의 의미 없는 슬라이드\n"
                "COVER 슬라이드는 40~60 범위로 채점하세요."
            ),
            "slides": slide_info,
            "criteria_scores": criteria_info,
            "output_format": {"scores": [{"num": 1, "score": 75}]},
        }, ensure_ascii=False)
        out = gemini.generate_json(prompt, temperature=0.1)
        scores_list = out.get("scores", [])
        result: Dict[int, int] = {}
        for item in scores_list:
            num = int(item.get("num", 0))
            score = int(item.get("score", 50))
            result[num] = max(0, min(100, score))
        if len(result) >= len(slides) * 0.5:
            return result
    except Exception:
        pass
    return {}


def _build_slide_cards(
    slides: List[Dict[str, Any]],
    criteria_scores: List[Dict[str, Any]],
    gemini: Optional[Any] = None,
    thumbnail_urls: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    score_by_slide: Dict[int, List[int]] = {}
    criteria_by_slide: Dict[int, List[str]] = {}
    for c in criteria_scores:
        for sn in c.get("related_slides", []):
            score_by_slide.setdefault(sn, []).append(int(c.get("score", 0)))
            criteria_by_slide.setdefault(sn, []).append(str(c.get("criteria_name", "")))

    # LLM 기반 슬라이드 점수 산출
    if gemini and getattr(gemini, "model", None) and os.getenv("IR_FAST_MODE") != "1":
        slide_scores = _llm_score_slides(slides, criteria_scores, score_by_slide, gemini)
    else:
        slide_scores = None

    slide_metrics: List[Dict[str, Any]] = []
    for slide in slides:
        sn = slide["slide_number"]
        linked_scores = score_by_slide.get(sn, [])
        text_len = len(slide.get("clean_text", ""))
        numbers = len(re.findall(r"\d", slide.get("clean_text", "")))
        cat_conf = float(slide.get("category_confidence", 0.5))
        claims_count = len(slide.get("key_claims", []))

        # LLM 점수가 있으면 우선 사용
        if slide_scores and sn in slide_scores:
            base = slide_scores[sn]
        else:
            # 규칙 기반 점수 (변별력 강화)
            base = 35  # 기본 베이스 낮춤

            # 연결된 기준 점수 반영 (비중 높임)
            if linked_scores:
                avg_linked = sum(linked_scores) / len(linked_scores)
                base += int(round(avg_linked * 0.30))
            # 카테고리 분류 신뢰도 반영 (비중 줄임)
            base += int(round(cat_conf * 10))

            # 내용 풍부도 보너스/패널티
            if numbers >= 8:
                base += 6
            elif numbers >= 3:
                base += 3
            if claims_count >= 3:
                base += 5
            elif claims_count >= 1:
                base += 2

            # 카테고리별 차등
            if slide.get("category") == "OTHER":
                base -= 12
            elif slide.get("category") == "COVER":
                base -= 8

            # 텍스트 밀도 패널티
            if text_len > 1200:
                base -= 8
            elif text_len > 800:
                base -= 4
            if text_len < 40:
                base -= 15
            elif text_len < 80:
                base -= 8

            if slide["text_deficiency_flag"]:
                base = min(base, 40)

        slide_metrics.append(
            {
                "slide": slide,
                "score": max(0, min(100, base)),
                "matched_criteria": sorted(set(criteria_by_slide.get(sn, []))),
                "numeric_count": numbers,
            }
        )

    feedback_by_slide = _build_slide_feedback_batch(slide_metrics, gemini)

    print("   - slide_cards: 카드 조립")
    thumbnail_urls = thumbnail_urls or {}
    out = []
    for item in slide_metrics:
        slide = item["slide"]
        sn = slide["slide_number"]
        detail = feedback_by_slide.get(sn) or _slide_feedback(
            slide=slide,
            score=item["score"],
            matched_criteria=item["matched_criteria"],
            numeric_count=item["numeric_count"],
        )
        out.append(
            {
                "slide_id": f"slide-{sn}",
                "slide_number": sn,
                "category": slide["category"],
                "score": item["score"],
                "thumbnail_url": thumbnail_urls.get(sn),
                "content": slide["short_summary"],
                "display_order": sn,
                "feedback": detail,
            }
        )
    return out


def _build_slide_feedback_batch(
    slide_metrics: List[Dict[str, Any]],
    gemini: Optional[Any] = None,
) -> Dict[int, Dict[str, Any]]:
    if (
        not gemini
        or not getattr(gemini, "model", None)
        or os.getenv("IR_FAST_MODE") == "1"
        or not slide_metrics
    ):
        return {}

    batch_size = max(1, int(os.getenv("IR_SLIDE_FEEDBACK_BATCH_SIZE", "6")))
    feedback_by_slide: Dict[int, Dict[str, Any]] = {}

    def _to_payload(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for item in items:
            slide = item["slide"]
            text_len = len(slide.get("clean_text", "") or "")
            if text_len > 1000:
                density_hint = f"텍스트 매우 많음({text_len}자) - 정보 과부하 가능성"
            elif text_len > 600:
                density_hint = f"텍스트 다소 많음({text_len}자)"
            elif text_len < 50:
                density_hint = f"텍스트 매우 적음({text_len}자) - 이미지/도표 위주이거나 내용 부족"
            elif text_len < 150:
                density_hint = f"텍스트 적음({text_len}자) - 비주얼 중심 슬라이드"
            else:
                density_hint = f"텍스트 적정({text_len}자)"
            payload.append(
                {
                    "slide_number": slide["slide_number"],
                    "category": slide.get("category", "OTHER"),
                    "score": item["score"],
                    "summary": (slide.get("short_summary", "") or "")[:280],
                    "text_density": density_hint,
                    "numeric_count": item["numeric_count"],
                    "num_claims": len(slide.get("key_claims", [])),
                    "matched_criteria": item["matched_criteria"][:3],
                }
            )
        return payload

    def _request(payload: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        prompt = json.dumps(
            {
                "instruction": (
                    "IR 피치덱 슬라이드 피드백을 배치로 작성하세요. JSON만 반환.\n"
                    "- slides 배열의 각 항목에 대해 detailed_feedback, strengths, improvements를 작성\n"
                    "- detailed_feedback은 3~4문장\n"
                    "- strengths/improvements는 각각 2~3개\n"
                    "- 투자자 관점, 슬라이드 내용, 구성/디자인 관점을 모두 반영\n"
                    "- 일반론 금지, 각 슬라이드에 특화된 내용만 작성"
                ),
                "slides": payload,
                "output_format": {
                    "slides": [
                        {
                            "slide_number": 1,
                            "detailed_feedback": "피드백",
                            "strengths": ["강점1", "강점2"],
                            "improvements": ["개선1", "개선2"],
                        }
                    ]
                },
            },
            ensure_ascii=False,
        )
        out = gemini.generate_json(prompt, temperature=0.2)
        rows = out.get("slides", []) if isinstance(out, dict) else []
        parsed: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            slide_number = int(row.get("slide_number", 0) or 0)
            detailed = str(row.get("detailed_feedback", "")).strip()
            strengths = row.get("strengths", [])
            improvements = row.get("improvements", [])
            if (
                slide_number > 0
                and detailed
                and isinstance(strengths, list)
                and isinstance(improvements, list)
            ):
                strength_list, improvement_list = _sanitize_feedback_lists(
                    [str(x).strip() for x in strengths if str(x).strip()],
                    [str(x).strip() for x in improvements if str(x).strip()],
                )
                parsed[slide_number] = {
                    "detailed_feedback": detailed,
                    "strengths": strength_list[:3],
                    "improvements": improvement_list[:3],
                }
        return parsed

    # 1) Single-call first: 호출 수를 줄여 병목 완화.
    use_single = os.getenv("IR_SLIDE_FEEDBACK_SINGLE_CALL", "1") == "1"
    if use_single:
        try:
            print(f"   - slide_cards: LLM 피드백 단일 호출 생성 1-{len(slide_metrics)}/{len(slide_metrics)}")
            feedback_by_slide.update(_request(_to_payload(slide_metrics)))
        except Exception:
            pass

    # 2) 누락된 슬라이드만 배치 재시도.
    missing_numbers = {
        int(item["slide"]["slide_number"])
        for item in slide_metrics
        if int(item["slide"]["slide_number"]) not in feedback_by_slide
    }
    if not missing_numbers:
        return feedback_by_slide

    for start in range(0, len(slide_metrics), batch_size):
        batch = slide_metrics[start : start + batch_size]
        batch = [item for item in batch if int(item["slide"]["slide_number"]) in missing_numbers]
        if not batch:
            continue
        try:
            print(
                f"   - slide_cards: LLM 피드백 배치 생성 {start + 1}-{start + len(batch)}/{len(slide_metrics)}"
            )
            feedback_by_slide.update(_request(_to_payload(batch)))
        except Exception:
            continue

    return feedback_by_slide


def _slide_feedback(
    slide: Dict[str, Any],
    score: int,
    matched_criteria: List[str],
    numeric_count: int,
) -> Dict[str, Any]:
    strengths = []
    improvements = []
    text_len = len(slide.get("clean_text", "") or "")
    num_claims = len(slide.get("key_claims", []))

    # 내용 강점
    if slide["category"] != "OTHER":
        strengths.append(f"{DISPLAY_CATEGORY_LABELS.get(slide['category'], slide['category'])} 목적의 메시지가 보입니다.")
    if num_claims >= 2:
        strengths.append("핵심 주장 문장이 2개 이상 있어 전달 포인트가 분명합니다.")
    if numeric_count >= 3:
        strengths.append("수치 정보가 포함되어 객관적 설명에 유리합니다.")
    # 구성 강점
    if 150 <= text_len <= 600 and num_claims >= 1:
        strengths.append("텍스트 분량이 적절하여 투자자가 빠르게 핵심을 파악할 수 있습니다.")
    if matched_criteria:
        strengths.append(f"관련 기준: {', '.join(matched_criteria[:2])}")

    # 내용 개선점
    if slide.get("text_deficiency_flag"):
        improvements.append("텍스트 근거가 부족하므로 핵심 문장/수치를 1~2개 추가하세요.")
    if numeric_count == 0:
        improvements.append("정량 근거(시장/사용자/매출 등) 수치를 최소 1개 이상 넣어주세요.")

    # 디자인/구성 개선점
    if text_len > 900:
        improvements.append(f"텍스트가 {text_len}자로 과도합니다. 투자자는 슬라이드당 3초를 봅니다. 제목 1줄 + 본문 3~5줄 + 시각자료로 압축하세요.")
    elif text_len > 600:
        improvements.append("텍스트가 다소 많습니다. '한 슬라이드 = 한 메시지' 원칙에 맞게 핵심만 남기세요.")
    if text_len < 50 and slide.get("category") not in {"COVER"}:
        improvements.append("텍스트가 너무 적습니다. 시각 자료와 함께 핵심 메시지 1~2문장을 반드시 포함하세요.")

    if slide.get("category") in {"MARKET", "BUSINESS_MODEL", "TRACTION", "FINANCE"} and numeric_count >= 3:
        improvements.append("수치가 여러 개 나열되어 있으므로 차트나 그래프로 시각화하면 전달력이 훨씬 높아집니다.")
    elif slide.get("category") in {"MARKET", "BUSINESS_MODEL"} and numeric_count < 2:
        improvements.append("시장/수익 슬라이드는 계산식 또는 기준년/출처를 함께 제시하세요.")

    # 카테고리별 구성 피드백
    if slide.get("category") == "COVER":
        improvements.append("커버 슬라이드에 회사명, 한 줄 태그라인, 산업 카테고리가 즉시 파악되도록 구성하세요.")
    elif slide.get("category") == "TEAM":
        improvements.append("팀 슬라이드는 사진/이름/역할/핵심경력을 카드형으로 배치하면 가독성이 높아집니다.")
    elif slide.get("category") == "COMPETITION":
        improvements.append("경쟁 비교는 표 형태로 기준 항목과 자사 우위를 한눈에 볼 수 있게 구성하세요.")
    elif slide.get("category") == "TRACTION":
        improvements.append("실적은 시계열 차트로 성장 추세를 시각화하면 투자자의 신뢰도가 크게 올라갑니다.")
    elif slide.get("category") == "BUSINESS_MODEL":
        improvements.append("수익 모델은 플로우차트나 도식으로 돈의 흐름을 시각적으로 보여주세요.")
    elif slide.get("category") == "GTM":
        improvements.append("고객 확보 채널/기간/목표 수치를 타임라인 형태로 정리하면 실행력이 선명해집니다.")
    elif slide.get("category") == "FINANCE":
        improvements.append("자금 사용 항목을 마일스톤과 연결한 파이차트 또는 워터폴 차트로 제시하세요.")

    if not improvements:
        improvements.append("핵심 주장 1개를 제목으로 끌어올리고, 본문은 근거 2개로 압축하세요.")

    preview = (slide.get("short_summary", "") or "").strip()
    preview = preview[:90] + ("..." if len(preview) > 90 else "")
    detailed = (
        f"슬라이드 {slide['slide_number']}은 {DISPLAY_CATEGORY_LABELS.get(slide['category'], slide['category'])} 성격이며 점수는 {score}점입니다. "
        f"핵심 메시지: {preview} "
    )
    # 구성/디자인 관련 한마디 추가
    if text_len > 900:
        detailed += "텍스트 과다로 정보 전달력이 떨어질 수 있으니 시각 자료 활용을 권장합니다."
    elif text_len < 50 and slide.get("category") not in {"COVER"}:
        detailed += "텍스트가 부족하여 핵심 메시지 전달이 어려울 수 있습니다."
    else:
        detailed += "슬라이드 구성과 정보 밀도를 점검하여 투자자의 시선 흐름을 최적화하세요."
    return {
        "detailed_feedback": detailed,
        "strengths": strengths[:3],
        "improvements": improvements[:3],
    }


def _sanitize_feedback_lists(strengths: List[str], improvements: List[str]) -> Tuple[List[str], List[str]]:
    negative_markers = ["부족", "누락", "미흡", "불명확", "약함", "보강", "추가", "필요", "없습니다", "낮습니다"]
    positive_markers = ["명확", "구체", "확인", "보입니다", "드러납니다", "강점", "장점", "유리", "탄탄"]

    clean_strengths: List[str] = []
    clean_improvements: List[str] = list(improvements)

    for item in strengths:
        if any(marker in item for marker in negative_markers):
            clean_improvements.append(item)
            continue
        clean_strengths.append(item)

    filtered_improvements = []
    for item in clean_improvements:
        if any(marker in item for marker in positive_markers) and not any(marker in item for marker in negative_markers):
            continue
        filtered_improvements.append(item)

    return clean_strengths[:3], filtered_improvements[:3]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
