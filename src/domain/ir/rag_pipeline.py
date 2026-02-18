import json
import os
import re
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.embedding.client import EmbeddingClient
from src.infrastructure.gemini.client import GeminiJSONClient


DEFAULT_TOP_K = 3
SIM_HIGH = 0.72
SIM_MID = 0.60
DEFAULT_LLM_SLIDE_LIMIT = 12
GROUP_CATEGORY_PRIORS = {
    "PROBLEM": {"PROBLEM", "MARKET"},
    "SOLUTION": {"SOLUTION", "PRODUCT"},
    "MARKET_BM": {"MARKET", "BUSINESS_MODEL", "COMPETITION"},
    "TRACTION": {"TRACTION", "MARKET"},
    "TEAM": {"TEAM"},
    "FINANCE": {"FINANCE", "BUSINESS_MODEL"},
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
}

CATEGORY_PRIORITY = [
    "TRACTION",
    "FINANCE",
    "BUSINESS_MODEL",
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
) -> Dict[str, Any]:
    if not docai_result:
        raise RuntimeError("OCR 결과가 비어 있습니다.")

    print("🧠 [RAG] 분석 엔진 시작")
    gemini = GeminiJSONClient()
    slides = _build_slides(docai_result)
    pitch_type = _resolve_pitch_type(strategy, pitch_type, slides)
    rubric = _load_rubric(pitch_type)
    print(f"🧾 [RAG] 슬라이드 로드 완료: {len(slides)}장")

    print("🏷️ [RAG] 슬라이드 분류/요약 진행")
    _classify_and_summarize_slides(slides, gemini)

    print("🔢 [RAG] 임베딩 생성 진행")
    embed_client = _init_embedding_client()
    _embed_slides(slides, embed_client)
    _embed_rubric_items(rubric, embed_client)

    print("📚 [RAG] 루브릭 매칭 및 기준별 점수 계산")
    criteria_scores = _score_criteria_with_rag(
        slides=slides,
        rubric=rubric,
        gemini=gemini,
    )

    print("🧩 [RAG] 종합 점수/가이드 생성")
    deck_score = _build_deck_score(criteria_scores, rubric, strategy, gemini)
    presentation_guide = _build_presentation_guide(slides, criteria_scores, strategy)
    slide_cards = _build_slide_cards(slides, criteria_scores)

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
            "total_slides": len(slides),
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
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


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
                    "다음 IR 슬라이드를 분석해서 JSON만 반환하세요.\n"
                    "category는 COVER|PROBLEM|SOLUTION|PRODUCT|MARKET|BUSINESS_MODEL|TRACTION|"
                    "COMPETITION|TEAM|FINANCE|ASK|OTHER 중 하나.\n"
                    "출력: {\"category\":\"...\",\"category_confidence\":0.0~1.0,"
                    "\"short_summary\":\"...\",\"key_claims\":[\"...\", \"...\"]}\n\n"
                    f"[슬라이드 텍스트]\n{slide['clean_text'][:4000]}"
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
                    "OTHER",
                }:
                    category = "OTHER"
                slide["category"] = category
                slide["category_confidence"] = _clamp01(float(out.get("category_confidence", 0.7)))
                slide["short_summary"] = str(out.get("short_summary", ""))[:280] or slide["clean_text"][:180]
                claims = out.get("key_claims", [])
                if isinstance(claims, list):
                    slide["key_claims"] = [str(c).strip() for c in claims if str(c).strip()][:5]
                else:
                    slide["key_claims"] = []
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
        slide["short_summary"] = slide["clean_text"][:180]
        slide["key_claims"] = _extract_claims(slide["clean_text"])


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
    if token_count <= 10 and num_cnt == 0 and not (has_market_core or has_traction_core or has_product_core or has_solution_core or has_team_core):
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

    # Prefer solution for problem->solution storytelling slides.
    if has_solution_core and any(k in t for k in ["문제", "pain", "불편", "한계", "차별"]):
        scores["SOLUTION"] += 0.6
        scores["PROBLEM"] += 0.3

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
    claims = [c.strip() for c in chunks if len(c.strip()) >= 15]
    return claims[:5]


def _init_embedding_client() -> Optional[EmbeddingClient]:
    if os.getenv("ENABLE_VERTEX_EMBEDDING") != "1":
        return None
    try:
        project_id = os.getenv("PROJECT_ID")
        if not project_id:
            return None
        client = EmbeddingClient(model_name="gemini-embedding-001")
        client.init_vertex(project_id=project_id, location=os.getenv("LOCATION", "us-central1"))
        return client
    except Exception:
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


def _embed_texts(texts: List[str], embed_client: Optional[EmbeddingClient]) -> List[List[float]]:
    if embed_client is not None:
        try:
            return embed_client.embed(texts, task_type="RETRIEVAL_DOCUMENT")
        except Exception:
            pass
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
            evidence_for_group.append(
                {
                    "item_id": item.get("item_id"),
                    "item_name": item.get("item_name"),
                    "coverage": coverage,
                    "max_similarity": max_sim,
                    "related_slides": [e["slide_number"] for e in evidences],
                    "top_summary": (evidences[0]["summary"] if evidences else ""),
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
        score_100 = int(round((raw_group_score / raw_group_max) * 100)) if raw_group_max > 0 else 0
        feedback, confidence = _build_group_feedback(
            group=group,
            evidence_for_group=evidence_for_group,
            missing_items=missing_items,
            gemini=gemini,
        )

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
                "confidence": confidence,
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
        ratio = max(0.35, min(0.70, (similarity / sim_high) * 0.70))
        return round(item_max * ratio, 2)
    # COVERED
    sim_high = max(0.01, min(0.99, _sim_high()))
    ratio = 0.65 + max(0.0, min(1.0, (similarity - sim_high) / (1.0 - sim_high))) * 0.35
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
                f"{group.get('group_name')} 항목에서 누락 요소가 감지되었습니다. "
                f"누락: {', '.join(m['item_name'] for m in missing_items[:2])}",
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
                "instruction": "근거 기반으로 2문장 피드백 작성. 과장 금지. JSON만 반환.",
                "output_format": {"feedback": "...", "confidence": 0.0},
            }
            out = gemini.generate_json(json.dumps(prompt, ensure_ascii=False), temperature=0.2)
            feedback = str(out.get("feedback", "")).strip()
            confidence = _clamp01(float(out.get("confidence", 0.75)))
            if feedback:
                return feedback, confidence
        except Exception:
            pass

    if missing_items:
        top_ev = [ev for ev in evidence_for_group if ev.get("top_summary")]
        ev_hint = f" 근거 예시: {top_ev[0]['top_summary'][:70]}..." if top_ev else ""
        return (
            f"{group.get('group_name')} 항목에서 일부 필수 근거가 부족합니다. "
            f"누락: {', '.join(m['item_name'] for m in missing_items[:2])}.{ev_hint}",
            0.65,
        )
    top_ev = [ev for ev in evidence_for_group if ev.get("top_summary")]
    ev_hint = f" 주요 근거: {top_ev[0]['top_summary'][:80]}..." if top_ev else ""
    return f"{group.get('group_name')} 항목은 근거 슬라이드가 확인되어 비교적 안정적으로 커버되었습니다.{ev_hint}", 0.72


def _build_deck_score(
    criteria_scores: List[Dict[str, Any]],
    rubric: Dict[str, Any],
    strategy: Optional[Dict[str, Any]],
    gemini: GeminiJSONClient,
) -> Dict[str, Any]:
    group_by_id = {g.get("group_id"): g for g in rubric.get("groups", [])}
    weighted_sum = 0.0
    for c in criteria_scores:
        gid = c.get("criteria_id")
        weight = float(group_by_id.get(gid, {}).get("group_weight", 0.0))
        weighted_sum += (float(c.get("raw_score", 0.0)) / max(float(c.get("raw_max_score", 1.0)), 1.0)) * weight
    total_score = int(round(weighted_sum * 100))

    sorted_low = sorted(criteria_scores, key=lambda x: x.get("score", 0))
    improvements = [f"{c['criteria_name']} 보강: {c['feedback']}" for c in sorted_low[:3]]
    strengths = [f"{c['criteria_name']} 강점: {c['feedback']}" for c in sorted(criteria_scores, key=lambda x: x.get("score", 0), reverse=True)[:3]]
    top_actions = []
    for c in sorted_low[:3]:
        missing = c.get("missing_items", [])
        if missing:
            top_actions.append(f"{c['criteria_name']}: {missing[0].get('suggestion', '')}")
        else:
            top_actions.append(f"{c['criteria_name']}: 핵심 근거 슬라이드 수치를 강화하세요.")

    structure_summary = _build_structure_summary(criteria_scores, strategy, gemini)
    return {
        "total_score": max(0, min(100, total_score)),
        "structure_summary": structure_summary,
        "strengths": strengths,
        "improvements": improvements,
        "top_actions": top_actions,
    }


def _build_structure_summary(
    criteria_scores: List[Dict[str, Any]],
    strategy: Optional[Dict[str, Any]],
    gemini: GeminiJSONClient,
) -> str:
    if gemini.model:
        try:
            payload = {
                "strategy": strategy or {},
                "criteria_scores": [
                    {"name": c["criteria_name"], "score": c["score"], "feedback": c["feedback"]} for c in criteria_scores
                ],
                "instruction": "IR 덱 구조 총평을 3~4문장으로 작성. JSON만 반환.",
                "output_format": {"summary": "..."},
            }
            out = gemini.generate_json(json.dumps(payload, ensure_ascii=False), temperature=0.2)
            summary = str(out.get("summary", "")).strip()
            if summary:
                return summary
        except Exception:
            pass
    return "문제-해결-시장-실행계획의 기본 흐름은 유지되었지만, 낮은 점수 항목의 근거를 보강하면 설득력이 개선됩니다."


def _build_presentation_guide(
    slides: List[Dict[str, Any]],
    criteria_scores: List[Dict[str, Any]],
    strategy: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
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
    return {
        "emphasized_slides": emphasized,
        "guide": [
            "오프닝에서 문제의 크기와 대상 고객을 한 문장으로 먼저 제시하세요.",
            "중간에는 수치 근거가 있는 슬라이드를 중심으로 설명 순서를 유지하세요.",
            "클로징에서는 실행 계획과 요청사항(투자/선정/지원 필요성)을 명확히 정리하세요.",
            f"현재 피칭 맥락({pitch_hint})에 맞춰 심사 포인트를 반복 강조하세요.",
        ],
        "time_allocation": [
            {"section": "오프닝", "seconds": 60},
            {"section": "본론", "seconds": 360},
            {"section": "클로징", "seconds": 60},
        ],
    }


def _build_slide_cards(slides: List[Dict[str, Any]], criteria_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    score_by_slide: Dict[int, List[int]] = {}
    criteria_by_slide: Dict[int, List[str]] = {}
    for c in criteria_scores:
        for sn in c.get("related_slides", []):
            score_by_slide.setdefault(sn, []).append(int(c.get("score", 0)))
            criteria_by_slide.setdefault(sn, []).append(str(c.get("criteria_name", "")))

    out = []
    for slide in slides:
        sn = slide["slide_number"]
        linked_scores = score_by_slide.get(sn, [])
        text_len = len(slide.get("clean_text", ""))
        numbers = len(re.findall(r"\d", slide.get("clean_text", "")))
        cat_conf = float(slide.get("category_confidence", 0.5))

        base = 45
        if linked_scores:
            base += int(round(sum(linked_scores) / len(linked_scores) * 0.35))
        base += int(round(cat_conf * 20))
        if numbers >= 8:
            base += 8
        elif numbers >= 3:
            base += 4
        if slide.get("category") == "OTHER":
            base -= 8
        if text_len > 1200:
            base -= 6
        if text_len < 40:
            base -= 10
        if slide["text_deficiency_flag"]:
            base = min(base, 50)
        detail = _slide_feedback(
            slide=slide,
            score=max(0, min(100, base)),
            matched_criteria=sorted(set(criteria_by_slide.get(sn, []))),
            numeric_count=numbers,
        )
        out.append(
            {
                "slide_id": f"slide-{sn}",
                "slide_number": sn,
                "category": slide["category"],
                "score": max(0, min(100, base)),
                "thumbnail_url": None,
                "content": slide["short_summary"],
                "display_order": sn,
                "feedback": detail,
            }
        )
    return out


def _slide_feedback(
    slide: Dict[str, Any],
    score: int,
    matched_criteria: List[str],
    numeric_count: int,
) -> Dict[str, Any]:
    strengths = []
    improvements = []
    if slide["category"] != "OTHER":
        strengths.append(f"{slide['category']} 목적의 메시지가 확인됩니다.")
    if len(slide.get("key_claims", [])) >= 2:
        strengths.append("핵심 주장 문장이 2개 이상 있어 전달 포인트가 분명합니다.")
    if numeric_count >= 3:
        strengths.append("수치 정보가 포함되어 객관적 설명에 유리합니다.")
    if matched_criteria:
        strengths.append(f"관련 기준: {', '.join(matched_criteria[:2])}")

    if slide.get("text_deficiency_flag"):
        improvements.append("텍스트 근거가 부족하므로 핵심 문장/수치를 1~2개 추가하세요.")
    if len(slide.get("clean_text", "")) > 900:
        improvements.append("텍스트 밀도가 높아 핵심 문장 중심으로 압축하는 것이 좋습니다.")
    if numeric_count == 0:
        improvements.append("정량 근거(시장/사용자/매출 등) 수치를 최소 1개 이상 넣어주세요.")
    if slide.get("category") in {"MARKET", "BUSINESS_MODEL"} and numeric_count < 2:
        improvements.append("시장/수익 슬라이드는 계산식 또는 기준년/출처를 함께 제시하세요.")
    if slide.get("category") == "TEAM":
        improvements.append("팀 슬라이드는 역할/경력/실행성과를 한 줄씩 분리해 가독성을 높이세요.")
    if not improvements:
        improvements.append("핵심 주장 1개를 제목으로 끌어올리고, 본문은 근거 2개로 압축하세요.")

    preview = (slide.get("short_summary", "") or "").strip()
    preview = preview[:90] + ("..." if len(preview) > 90 else "")
    detailed = (
        f"슬라이드 {slide['slide_number']}({slide['category']}) 점수는 {score}점입니다. "
        f"요약: {preview}"
    )
    return {
        "detailed_feedback": detailed,
        "strengths": strengths[:3],
        "improvements": improvements[:3],
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
