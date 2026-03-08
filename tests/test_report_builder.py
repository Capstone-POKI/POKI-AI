from src.domain.report.feature_builder import build_analysis_context
from src.domain.report.qa_generator import generate_qa
from src.domain.report.report_builder import build_final_report


def _sample_deck_raw():
    return {
        "meta": {"doc_type": "IR", "pitch_strategy": {}},
        "diagnosis": {"missing_sections": ["finance"], "logic_flow_issues": ["market-traction weak link"]},
        "slides": [
            {
                "page_number": 1,
                "section_type": "problem",
                "contents": {"full_text": "problem text", "summary": "s", "char_count": 10, "image_count": 0},
                "voice_guide": {"estimated_duration_sec": 20.0, "pacing_advice": "steady"},
                "design_feedback": ["keep contrast"],
            },
            {
                "page_number": 2,
                "section_type": "solution",
                "contents": {"full_text": "solution text", "summary": "s", "char_count": 12, "image_count": 1},
                "voice_guide": {"estimated_duration_sec": 22.0, "pacing_advice": "steady"},
                "design_feedback": ["show metric"],
            },
        ],
    }


def _sample_speech_raw():
    return {
        "발표_상황": "정부지원사업 발표",
        "상황_적합성_점수": {
            "총점": 78,
            "세부_기준": {
                "문제_정의": 82,
                "솔루션_명확성": 79,
                "시장성": 68,
                "사업성_BM": 70,
                "경쟁력_차별성": 75,
                "전달력": 73,
                "톤_일관성": 74,
            },
        },
        "음성_전달력_분석": {
            "말하기_속도_WPM": 118.0,
            "억양_강조_안정성": "보통",
            "감정_톤": "차분",
            "문장_명료성": "양호",
            "불필요한_말버릇": "적음",
            "강점": ["핵심 전달 명확"],
            "개선점": ["속도 완급"],
        },
        "1분_요약": "요약 텍스트",
    }


def test_context_pitch_type_and_qa_categories():
    context = build_analysis_context(_sample_deck_raw(), _sample_speech_raw())
    assert context["pitch_type"] == "government_support"
    qa = generate_qa(context)
    categories = {item["category"] for item in qa}
    assert "problem_definition" in categories
    assert "policy_fit" in categories


def test_build_final_report_v1_shape():
    report = build_final_report(_sample_deck_raw(), _sample_speech_raw())
    assert "basic_info" in report
    assert "deck_evaluation" in report
    assert "speech_evaluation" in report
    assert "alignment_evaluation" in report
    assert "score_summary" in report
    assert "qa_suggestions_by_category" in report
    assert "final_opinion" in report
