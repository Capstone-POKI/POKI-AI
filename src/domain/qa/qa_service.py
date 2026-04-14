"""Q&A 메인 파이프라인 및 서비스 관리"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from uuid import uuid4

from app.models.qa_schema import QuestionResponse, AnswerResponse, AnswerFeedback
from src.domain.qa.question_generator import run_question_generation, QuestionItem
from src.domain.qa.answer_evaluator import (
    run_answer_evaluation,
    calculate_weighted_score,
)
from src.domain.qa.voice_transcriber import transcribe_audio_to_text, validate_audio_file


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class QASessionData:
    """QA 세션 데이터"""
    session_id: str
    pitch_id: str
    notice_id: Optional[str] = None
    ir_deck_id: Optional[str] = None
    voice_report_id: Optional[str] = None
    questions: List[Dict] = field(default_factory=list)
    answers: Dict[str, Dict] = field(default_factory=dict)  # question_id -> answer
    status: str = "IN_PROGRESS"
    created_at: datetime = field(default_factory=_now)
    completed_at: Optional[datetime] = None


def run_qa_question_generation(
    pitch_id: str,
    notice_content: str,
    irdecksummary: str,
    presentation_content: Optional[str] = None,
) -> Optional[List[QuestionItem]]:
    """
    주어진 공고, IR Deck, 발표 내용을 기반으로 Q&A 질문을 생성합니다.
    
    Args:
        pitch_id: 피치 ID
        notice_content: 공고문 전체 내용
        irdecksummary: IR Deck 요약
        presentation_content: 발표 내용 (선택)
    
    Returns:
        생성된 질문 리스트
    """
    print(f"\n🎤 [QA 질문 생성] pitch_id={pitch_id}")
    
    try:
        questions = run_question_generation(
            notice_content=notice_content,
            irdecksummary=irdecksummary,
            presentation_content=presentation_content,
        )
        
        print(f"✅ 총 {len(questions)}개의 질문이 생성되었습니다")
        
        # 질문 타입별 통계
        type_counts = {}
        for q in questions:
            type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
        print(f"📊 질문 타입: {type_counts}")
        
        return questions
    
    except Exception as e:
        print(f"❌ 질문 생성 중 오류: {e}")
        return None


def prepare_qa_session(
    session_id: str,
    pitch_id: str,
    questions: List[QuestionItem],
    notice_id: Optional[str] = None,
    ir_deck_id: Optional[str] = None,
) -> QASessionData:
    """
    QA 세션을 준비합니다.
    
    Args:
        session_id: 세션 ID
        pitch_id: 피치 ID
        questions: 생성된 질문들
        notice_id: 공고 ID (선택)
        ir_deck_id: IR Deck ID (선택)
    
    Returns:
        준비된 QA 세션 데이터
    """
    session = QASessionData(
        session_id=session_id,
        pitch_id=pitch_id,
        notice_id=notice_id,
        ir_deck_id=ir_deck_id,
    )
    
    # 질문을 세션에 추가
    for idx, q in enumerate(questions, 1):
        question_id = str(uuid4())
        session.questions.append({
            "question_id": question_id,
            "order": idx,
            "question_type": q.question_type,
            "content": q.content,
            "guidance": q.guidance,
            "rationale": q.rationale,
        })
    
    return session


def process_answer(
    session: QASessionData,
    question_id: str,
    audio_path: Optional[str] = None,
    text_transcript: Optional[str] = None,
) -> Optional[AnswerResponse]:
    """
    사용자의 답변을 처리합니다.
    
    Args:
        session: QA 세션
        question_id: 질문 ID
        audio_path: 음성 파일 경로 (선택)
        text_transcript: 직접 입력한 텍스트 (선택)
    
    Returns:
        처리된 답변 또는 None (실패 시)
    """
    # 해당 질문 찾기
    question = None
    for q in session.questions:
        if q["question_id"] == question_id:
            question = q
            break
    
    if not question:
        print(f"❌ 질문을 찾을 수 없습니다: {question_id}")
        return None
    
    # STT 처리
    if audio_path and not text_transcript:
        if not validate_audio_file(audio_path):
            return None
        print(f"🎙️ 음성 변환 중... ({audio_path})")
        text_transcript = transcribe_audio_to_text(audio_path)
        if not text_transcript:
            print("❌ 음성 변환 실패")
            return None
    
    if not text_transcript:
        print("❌ 답변 텍스트가 없습니다")
        return None
    
    # 답변 평가
    print(f"📝 답변 평가 중... (질문: {question['content']})")
    evaluation = run_answer_evaluation(
        question_type=question["question_type"],
        question_content=question["content"],
        guidance=question.get("guidance", ""),
        answer_transcript=text_transcript,
    )
    
    if not evaluation:
        print("❌ 답변 평가 실패")
        return None
    
    # 최종 점수 계산
    final_score = calculate_weighted_score(evaluation)
    
    # 답변 객체 생성
    answer_id = str(uuid4())
    answer_response = AnswerResponse(
        answer_id=answer_id,
        question_id=question_id,
        pitch_id=session.pitch_id,
        text_transcript=text_transcript,
        feedback=AnswerFeedback(
            score=final_score,
            relevance=evaluation.relevance,
            clarity=evaluation.clarity,
            structure=evaluation.structure,
            strengths=evaluation.strengths,
            improvements=evaluation.improvements,
            reasoning=evaluation.reasoning,
        ),
        evaluation_status="COMPLETED",
        created_at=_now(),
        evaluated_at=_now(),
    )
    
    # 세션에 답변 저장
    session.answers[question_id] = asdict(answer_response)
    
    print(f"✅ 답변 평가 완료 (점수: {final_score}/100)")
    
    return answer_response


def export_qa_results(
    session: QASessionData,
    output_dir: Path,
) -> Path:
    """
    QA 세션 결과를 JSON으로 저장합니다.
    
    Args:
        session: QA 세션
        output_dir: 출력 디렉토리
    
    Returns:
        저장된 파일 경로
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{session.session_id}_qa_results.json"
    
    result = {
        "session_id": session.session_id,
        "pitch_id": session.pitch_id,
        "total_questions": len(session.questions),
        "answered_count": len(session.answers),
        "status": session.status,
        "questions": session.questions,
        "answers": list(session.answers.values()),
        "created_at": session.created_at.isoformat(),
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ QA 결과 저장: {output_file}")
    
    return Path(output_file)
