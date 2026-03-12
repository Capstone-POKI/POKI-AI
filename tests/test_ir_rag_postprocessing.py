from src.domain.ir.rag_pipeline import (
    _apply_strategy_to_rubric,
    _apply_group_score_floor,
    _keyword_classify_with_confidence,
    _mark_slide_validity,
    _summarize_slide_text,
)


def test_keyword_classifier_prefers_competition_over_team():
    category, confidence = _keyword_classify_with_confidence(
        "경쟁사 비교표 포지셔닝 차별점 CEO CTO 언급 포함",
        slide_number=9,
        total_slides=18,
    )
    assert category == "COMPETITION"
    assert confidence >= 0.45


def test_keyword_classifier_prefers_business_model_over_traction():
    category, _ = _keyword_classify_with_confidence(
        "Business Model 서비스 수수료 구독 pricing 고객당 월 과금 구조",
        slide_number=11,
        total_slides=18,
    )
    assert category == "BUSINESS_MODEL"


def test_keyword_classifier_prefers_gtm_over_market():
    category, _ = _keyword_classify_with_confidence(
        "초기 고객 확보 전략 제휴 파트너십 영업 전략 go-to-market 확장 전략",
        slide_number=13,
        total_slides=18,
    )
    assert category == "GTM"


def test_mark_slide_validity_flags_duplicate_and_mixed_slides():
    slides = [
        {
            "slide_number": 1,
            "clean_text": "문제 정의 고객 불편 시장 현황 데이터 근거",
            "category": "PROBLEM",
            "ocr_noise_ratio": 0.05,
            "is_valid": True,
            "invalid_reason": "",
        },
        {
            "slide_number": 2,
            "clean_text": "문제 정의 고객 불편 시장 현황 데이터 근거",
            "category": "PROBLEM",
            "ocr_noise_ratio": 0.05,
            "is_valid": True,
            "invalid_reason": "",
        },
        {
            "slide_number": 3,
            "clean_text": "Problem Solution Team Market 내용이 한 페이지에 함께 반복 표기됨",
            "category": "OTHER",
            "ocr_noise_ratio": 0.08,
            "is_valid": True,
            "invalid_reason": "",
        },
    ]

    _mark_slide_validity(slides)

    assert slides[0]["is_valid"] is True
    assert slides[1]["is_valid"] is False
    assert "중복" in slides[1]["invalid_reason"]
    assert slides[2]["is_valid"] is False
    assert "혼합" in slides[2]["invalid_reason"]


def test_summarize_slide_text_removes_ocr_like_noise():
    summary = _summarize_slide_text(
        "Business Model\n워시편 위시민 울산점\n서비스 수수료 12%\n월 구독 39000원",
        "BUSINESS_MODEL",
    )
    assert "서비스 수수료 12%" in summary
    assert "월 구독 39000원" in summary


def test_team_group_score_floor_uses_existing_team_slide():
    score, related, message = _apply_group_score_floor(
        group_id="TEAM",
        score_100=0,
        slides=[
            {"slide_number": 17, "category": "TEAM"},
            {"slide_number": 14, "category": "FINANCE"},
        ],
        related_slides=[],
    )

    assert score == 15
    assert related == [17]
    assert "팀 슬라이드" in message


def test_apply_strategy_to_rubric_uses_notice_points():
    rubric = {
        "pitch_type": "VC_DEMO",
        "total_points": 100,
        "groups": [
            {"group_id": "PROBLEM", "group_name": "문제정의", "group_weight": 0.2, "max_score": 20, "items": [{"max_score": 10}, {"max_score": 10}]},
            {"group_id": "SOLUTION", "group_name": "솔루션", "group_weight": 0.2, "max_score": 20, "items": [{"max_score": 10}, {"max_score": 10}]},
            {"group_id": "MARKET_BM", "group_name": "시장/비즈니스", "group_weight": 0.25, "max_score": 25, "items": [{"max_score": 10}, {"max_score": 15}]},
            {"group_id": "TRACTION", "group_name": "실적", "group_weight": 0.15, "max_score": 15, "items": [{"max_score": 15}]},
            {"group_id": "TEAM", "group_name": "팀", "group_weight": 0.1, "max_score": 10, "items": [{"max_score": 10}]},
            {"group_id": "FINANCE", "group_name": "자금 계획", "group_weight": 0.1, "max_score": 10, "items": [{"max_score": 10}]},
        ],
    }
    strategy = {
        "evaluation_criteria": [
            {"criteria_name": "문제정의", "points": 25},
            {"criteria_name": "솔루션", "points": 25},
            {"criteria_name": "시장/비즈니스", "points": 20},
            {"criteria_name": "실적", "points": 10},
            {"criteria_name": "팀", "points": 10},
            {"criteria_name": "자금 계획", "points": 10},
        ]
    }

    updated = _apply_strategy_to_rubric(rubric, strategy)
    groups = {group["group_id"]: group for group in updated["groups"]}

    assert updated["total_points"] == 100
    assert groups["PROBLEM"]["max_score"] == 25
    assert groups["TEAM"]["group_weight"] == 0.1
