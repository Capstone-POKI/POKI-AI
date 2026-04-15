from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Path as FPath, UploadFile
from fastapi import Form

from app.models.qa_schema import (
    AnalysisStatus,
    QuestionResponse,
    AnswerResponse,
    AnswerFeedback,
    AnswerRequest,
    QASessionRequest,
    QASessionResponse,
    ErrorResponse,
    QuestionType,
)
from src.domain.qa.qa_service import (
    run_qa_question_generation,
    prepare_qa_session,
    process_answer,
    export_qa_results,
    QASessionData,
)

router = APIRouter(prefix="/api", tags=["qa"])

QA_SESSION_DIR = Path("data/output/qa_sessions")
QA_RESULTS_DIR = Path("data/output/qa_results")

# 메모리 내 세션 스토리지 (개발용)
_LOCK = threading.Lock()
_QA_SESSIONS: dict[str, QASessionData] = {}
_SESSION_IDS_BY_PITCH: dict[str, list[str]] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _raise_error(status_code: int, error: str, message: str | None = None) -> None:
    payload = {"error": error}
    if message:
        payload["message"] = message
    raise HTTPException(status_code=status_code, detail=payload)


# ==================== 질문 생성 엔드포인트 ====================

@router.post("/pitches/{pitch_id}/qa/questions/generate")
async def generate_qa_questions(
    pitch_id: str,
    notice_content: str = Form(...),
    irdecksummary: str = Form(...),
    presentation_content: str = Form(default=None),
    background_tasks: BackgroundTasks = None,
) -> QASessionResponse:
    """
    Q&A 질문을 생성하고 새로운 세션을 시작합니다.
    
    공고문, IR Deck, 발표 내용을 바탕으로 다양한 관점의 질문을 생성합니다.
    """
    session_id = str(uuid4())
    
    # 백그라운드에서 질문 생성
    def generate_questions():
        try:
            # 질문 생성
            questions = run_qa_question_generation(
                pitch_id=pitch_id,
                notice_content=notice_content,
                irdecksummary=irdecksummary,
                presentation_content=presentation_content,
            )
            
            if not questions:
                with _LOCK:
                    if session_id in _QA_SESSIONS:
                        session = _QA_SESSIONS[session_id]
                        session.status = "FAILED"
                return
            
            # 세션 준비
            session = prepare_qa_session(
                session_id=session_id,
                pitch_id=pitch_id,
                questions=questions,
            )
            
            with _LOCK:
                _QA_SESSIONS[session_id] = session
            
            print(f"✅ QA 세션 생성 완료: {session_id}")
        
        except Exception as e:
            print(f"❌ 질문 생성 중 오류: {e}")
            with _LOCK:
                if session_id in _QA_SESSIONS:
                    _QA_SESSIONS[session_id].status = "FAILED"
    
    # 초기 세션 생성 (빈 상태)
    session = QASessionData(
        session_id=session_id,
        pitch_id=pitch_id,
    )
    
    with _LOCK:
        _QA_SESSIONS[session_id] = session
        if pitch_id not in _SESSION_IDS_BY_PITCH:
            _SESSION_IDS_BY_PITCH[pitch_id] = []
        _SESSION_IDS_BY_PITCH[pitch_id].append(session_id)
    
    # 백그라운드 작업 추가
    if background_tasks:
        background_tasks.add_task(generate_questions)
    else:
        # 동기 실행
        generate_questions()
    
    return _to_qa_session_response(session)


@router.get("/pitches/{pitch_id}/qa/sessions/{session_id}")
async def get_qa_session(
    pitch_id: str,
    session_id: str,
) -> QASessionResponse:
    """특정 QA 세션의 질문 리스트를 조회합니다."""
    with _LOCK:
        session = _QA_SESSIONS.get(session_id)
    
    if not session or session.pitch_id != pitch_id:
        _raise_error(404, "SESSION_NOT_FOUND", "해당 세션을 찾을 수 없습니다")
    
    return _to_qa_session_response(session)


@router.get("/pitches/{pitch_id}/questions")
async def get_all_qa_questions(pitch_id: str) -> dict:
    """
    해당 피치의 모든 Q&A 질문을 조회합니다.
    (Backend와 호환되는 엔드포인트)
    """
    with _LOCK:
        session_ids = _SESSION_IDS_BY_PITCH.get(pitch_id, [])
        if not session_ids:
            # 최신 세션 찾기
            latest_session = None
            for sess in _QA_SESSIONS.values():
                if sess.pitch_id == pitch_id:
                    if not latest_session or sess.created_at > latest_session.created_at:
                        latest_session = sess
            
            if not latest_session:
                return {"questions": [], "total": 0}
            
            session = latest_session
        else:
            # 가장 최신 세션 사용
            session = _QA_SESSIONS.get(session_ids[-1])
            if not session:
                return {"questions": [], "total": 0}
    
    questions = []
    for q in session.questions:
        questions.append({
            "question_id": q["question_id"],
            "order": q["order"],
            "type": q["question_type"],
            "content": q["content"],
            "guidance": q.get("guidance"),
        })
    
    return {"questions": questions, "total": len(questions)}


# ==================== 답변 제출 엔드포인트 ====================

@router.post("/pitches/{pitch_id}/qa/sessions/{session_id}/answers")
async def submit_answer(
    pitch_id: str,
    session_id: str,
    question_id: str = Form(...),
    audio: UploadFile = File(None),
    text_transcript: str = Form(None),
) -> AnswerResponse:
    """
    답변을 제출하고 평가합니다.
    
    음성 파일 또는 텍스트 중 하나는 반드시 제공되어야 합니다.
    """
    with _LOCK:
        session = _QA_SESSIONS.get(session_id)
    
    if not session or session.pitch_id != pitch_id:
        _raise_error(404, "SESSION_NOT_FOUND", "해당 세션을 찾을 수 없습니다")
    
    # 음성 파일 저장 (필요한 경우)
    audio_path = None
    if audio:
        audio_dir = Path("data/output/qa_audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{session_id}_{question_id}.wav"
        
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
    
    # 백그라운드에서 답변 처리
    def evaluate_answer():
        try:
            with _LOCK:
                answer = process_answer(
                    session=session,
                    question_id=question_id,
                    audio_path=str(audio_path) if audio_path else None,
                    text_transcript=text_transcript,
                )
            
            if not answer:
                print(f"❌ 답변 처리 실패: {question_id}")
            else:
                print(f"✅ 답변 처리 완료: {question_id}")
        
        except Exception as e:
            print(f"❌ 답변 평가 중 오류: {e}")
    
    # 백그라운드 작업으로 처리
    evaluate_answer()
    
    # 임시 응답 반환 (평가는 비동기로 진행)
    answer_id = str(uuid4())
    return AnswerResponse(
        answer_id=answer_id,
        question_id=question_id,
        pitch_id=pitch_id,
        audio_url=str(audio_path) if audio_path else None,
        text_transcript=text_transcript or "",
        evaluation_status=AnalysisStatus.IN_PROGRESS,
        created_at=_now(),
    )


@router.get("/pitches/{pitch_id}/qa/sessions/{session_id}/answers/{answer_id}")
async def get_answer_evaluation(
    pitch_id: str,
    session_id: str,
    answer_id: str,
) -> AnswerResponse:
    """답변 평가 결과를 조회합니다."""
    with _LOCK:
        session = _QA_SESSIONS.get(session_id)
    
    if not session or session.pitch_id != pitch_id:
        _raise_error(404, "SESSION_NOT_FOUND", "해당 세션을 찾을 수 없습니다")
    
    # 답변 찾기
    for q_id, answer_data in session.answers.items():
        if answer_data.get("answer_id") == answer_id:
            # 데이터를 AnswerResponse로 변환
            feedback_data = answer_data.get("feedback")
            answer = AnswerResponse(
                answer_id=answer_data["answer_id"],
                question_id=answer_data["question_id"],
                pitch_id=answer_data["pitch_id"],
                audio_url=answer_data.get("audio_url"),
                text_transcript=answer_data["text_transcript"],
                feedback=AnswerFeedback(**feedback_data) if feedback_data else None,
                evaluation_status=answer_data.get("evaluation_status"),
                created_at=datetime.fromisoformat(answer_data["created_at"]),
                evaluated_at=datetime.fromisoformat(answer_data["evaluated_at"]) if answer_data.get("evaluated_at") else None,
            )
            return answer
    
    _raise_error(404, "ANSWER_NOT_FOUND", "해당 답변을 찾을 수 없습니다")


# ==================== 헬퍼 함수 ====================

def _to_qa_session_response(session: QASessionData) -> QASessionResponse:
    """QASessionData를 QASessionResponse로 변환"""
    questions = []
    for q in session.questions:
        questions.append(
            QuestionResponse(
                question_id=q["question_id"],
                pitch_id=session.pitch_id,
                question_type=QuestionType(q["question_type"]),
                content=q["content"],
                guidance=q.get("guidance"),
                order=q["order"],
                created_at=datetime.fromisoformat(q.get("created_at", _now().isoformat())),
            )
        )
    
    return QASessionResponse(
        session_id=session.session_id,
        pitch_id=session.pitch_id,
        status=AnalysisStatus(session.status),
        questions=questions,
        total_questions=len(session.questions),
        answered_count=len(session.answers),
        created_at=session.created_at,
    )


# ==================== 건강 체크 ====================

@router.get("/health/qa")
async def health_check() -> dict:
    """QA 서비스 건강 상태 확인"""
    return {
        "status": "healthy",
        "service": "qa",
        "sessions": len(_QA_SESSIONS),
        "timestamp": _now().isoformat(),
    }
