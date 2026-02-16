from typing import Dict
from .schemas import DeckAnalysisResult, SpeechAnalysisResult


def build_analysis_context(deck_raw: Dict, speech_raw: Dict) -> Dict:
    """
    현서/예린 JSON(dict)을 요약한 context 생성
    """
    deck = DeckAnalysisResult(**deck_raw)
    speech = SpeechAnalysisResult(**speech_raw)

    detail = speech.상황_적합성_점수.세부_기준

    evaluation_axes = [
        {"name": "문제 정의", "score": detail.문제_정의},
        {"name": "솔루션 명확성", "score": detail.솔루션_명확성},
        {"name": "시장성", "score": detail.시장성},
        {"name": "사업성/비즈니스 모델", "score": detail.사업성_BM},
        {"name": "경쟁력/차별성", "score": detail.경쟁력_차별성},
        {"name": "전달력", "score": detail.전달력},
        {"name": "톤 일관성", "score": detail.톤_일관성},
    ]

    problem_texts = [
        s.contents.full_text for s in deck.slides if s.section_type.lower() == "problem"
    ]
    solution_texts = [
        s.contents.full_text for s in deck.slides if s.section_type.lower() == "solution"
    ]

    context = {
        "pitch_situation": speech.발표_상황,
        "total_score": speech.상황_적합성_점수.총점,
        "evaluation_axes": evaluation_axes,
        "missing_sections": deck.diagnosis.missing_sections,
        "logic_issues": deck.diagnosis.logic_flow_issues,
        "problem_slide_texts": problem_texts,
        "solution_slide_texts": solution_texts,
        "voice_speed_wpm": speech.음성_전달력_분석.말하기_속도_WPM,
        "voice_strengths": speech.음성_전달력_분석.강점,
        "voice_weaknesses": speech.음성_전달력_분석.개선점,
        "summary_1min": speech.summary_1min,
    }

    return context