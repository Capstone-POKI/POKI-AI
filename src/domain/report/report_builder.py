from typing import Dict, Any, List
from .feature_builder import build_analysis_context
from .qa_generator import generate_qa
from .score_engine import compute_feature_impacts
from .schemas import DeckAnalysisResult, SpeechAnalysisResult


def build_final_report(deck_raw: Dict, speech_raw: Dict) -> Dict:
    """
    PitchCoach 최종 분석 리포트 JSON 생성
    """
    context = build_analysis_context(deck_raw, speech_raw)

    qa = generate_qa(context)
    scores = compute_feature_impacts(context)

    speech = SpeechAnalysisResult(**speech_raw)
    deck = DeckAnalysisResult(**deck_raw)

    strengths = speech.음성_전달력_분석.강점
    improvements = speech.음성_전달력_분석.개선점

    slide_tips: List[Dict[str, Any]] = []
    for s in deck.slides:
        slide_tips.append({
            "page_number": s.page_number,
            "section_type": s.section_type,
            "design_feedback": s.design_feedback,
            "voice_guide": {
                "estimated_duration_sec": s.voice_guide.estimated_duration_sec,
                "pacing_advice": s.voice_guide.pacing_advice,
            }
        })

    final = {
        "summary": {
            "headline": "핵심 IR 섹션 누락과 전달력 개선 필요",
            "one_line": "문제 정의는 명확하지만 재무/팀 섹션이 없어 설득력이 낮아지고 있습니다.",
            "one_min_summary": context["summary_1min"],
        },
        "scores": {
            "total": scores["total_score"],
            "axes": context["evaluation_axes"],
            "feature_impacts": scores["feature_impacts"]
        },
        "strengths": strengths,
        "improvements": improvements,
        "qa_suggestions": qa,
        "slide_tips": slide_tips,
    }

    return final