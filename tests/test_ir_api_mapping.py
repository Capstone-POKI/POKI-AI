from app.api.decks import _map_deck_payload_to_result
from src.domain.ir.rag_pipeline import (
    _build_deck_improvements,
    _build_deck_strengths,
    _sanitize_feedback_lists,
)


def test_map_ir_payload_keeps_criteria_scores_without_notice():
    payload = {
        "deck_score": {
            "total_score": 32,
            "max_score": 100,
            "scoring_method": "weighted_average",
            "criteria_weights": {"문제정의": 0.2},
            "structure_summary": "요약",
            "strengths": ["강점 1"],
            "improvements": ["개선 1"],
        },
        "criteria_scores": [
            {
                "criteria_name": "문제정의",
                "pitchcoach_interpretation": "문제 평가",
                "ir_guide": "문제를 구체화",
                "score": 41,
                "max_score": 100,
                "raw_score": 6.0,
                "raw_max_score": 15.0,
                "coverage_status": "PARTIALLY_COVERED",
                "evidence_slides": [2, 3],
                "related_slides": [2, 3],
                "feedback": "문제는 보이지만 검증 근거가 부족합니다.",
            }
        ],
        "presentation_guide": {},
        "slides": [],
    }

    result = _map_deck_payload_to_result(payload, pitch_id="pitch-without-notice")

    assert result.deck_score["max_score"] == 100
    assert result.deck_score["scoring_method"] == "weighted_average"
    assert result.deck_score["criteria_weights"] == {"문제정의": 0.2}
    assert len(result.criteria_scores) == 1
    assert result.criteria_scores[0]["criteria_name"] == "문제정의"
    assert result.criteria_scores[0]["evidence_slides"] == [2, 3]
    assert result.criteria_scores[0]["related_slides"] == [2, 3]


def test_deck_strengths_and_improvements_split_polarity():
    criteria_scores = [
        {
            "criteria_name": "시장/비즈니스",
            "score": 72,
            "related_slides": [11, 13],
            "missing_items": [],
            "feedback": "시장 관련 슬라이드는 있으나 일부 보강이 필요합니다.",
        },
        {
            "criteria_name": "문제정의",
            "score": 35,
            "related_slides": [2],
            "missing_items": [
                {"suggestion": "구체적 문제 정의와 고객 검증 근거를 수치로 보강하세요."}
            ],
            "feedback": "문제는 보이지만 근거가 약합니다.",
        },
    ]

    strengths = _build_deck_strengths(criteria_scores)
    improvements = _build_deck_improvements(criteria_scores)

    assert strengths
    assert all("부족" not in item and "누락" not in item for item in strengths)
    assert improvements
    assert any("보강" in item or "근거" in item for item in improvements)


def test_feedback_sanitizer_moves_negative_strengths_to_improvements():
    strengths, improvements = _sanitize_feedback_lists(
        ["시장 규모가 부족합니다.", "시장 진입 전략이 명확합니다."],
        ["TAM 근거를 추가하세요."],
    )

    assert "시장 진입 전략이 명확합니다." in strengths
    assert "시장 규모가 부족합니다." in improvements
