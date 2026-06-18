from __future__ import annotations

import asyncio
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import Form

from app.models.qa_schema import (
    AnalysisStatus,
    QuestionResponse,
    AnswerResponse,
    AnswerFeedback,
    QASessionResponse,
    QuestionType,
)
from app.state_store import load_state, save_state
from app.upload import read_upload_limited, validate_audio_payload
from src.domain.qa.qa_service import (
    run_qa_question_generation,
    prepare_qa_session,
    process_answer,
    QASessionData,
)

router = APIRouter(prefix="/api", tags=["qa"])

QA_SESSION_DIR = Path("data/output/qa_sessions")
QA_RESULTS_DIR = Path("data/output/qa_results")
MAX_QA_AUDIO_FILE_SIZE = 25 * 1024 * 1024

# 메모리 내 세션 스토리지
_LOCK = threading.Lock()
_QA_SESSIONS: dict[str, QASessionData] = {}
_SESSION_IDS_BY_PITCH: dict[str, list[str]] = {}
_ANALYZED_ANSWERS: dict[str, dict] = {}  # answer_id → spec-format answer dict
_STATE_NAME = "qa"


def _serialize_session(session: QASessionData) -> dict:
    return {
        "session_id": session.session_id,
        "pitch_id": session.pitch_id,
        "notice_id": session.notice_id,
        "ir_deck_id": session.ir_deck_id,
        "voice_report_id": session.voice_report_id,
        "questions": session.questions,
        "answers": session.answers,
        "status": session.status,
        "created_at": session.created_at.isoformat(),
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
    }


def _persist_state_locked() -> None:
    save_state(
        _STATE_NAME,
        {
            "sessions": {
                sid: _serialize_session(sess)
                for sid, sess in _QA_SESSIONS.items()
            },
            "session_ids_by_pitch": _SESSION_IDS_BY_PITCH,
            "analyzed_answers": _ANALYZED_ANSWERS,
        },
    )


def _restore_state() -> None:
    payload = load_state(_STATE_NAME)
    sessions = payload.get("sessions", {})
    by_pitch = payload.get("session_ids_by_pitch", {})

    if isinstance(sessions, dict):
        for sid, raw in sessions.items():
            if not isinstance(raw, dict):
                continue
            try:
                completed_at = raw.get("completed_at")
                session = QASessionData(
                    session_id=raw["session_id"],
                    pitch_id=raw["pitch_id"],
                    notice_id=raw.get("notice_id"),
                    ir_deck_id=raw.get("ir_deck_id"),
                    voice_report_id=raw.get("voice_report_id"),
                    questions=raw.get("questions", []),
                    answers=raw.get("answers", {}),
                    status=raw.get("status", "COMPLETED"),
                    created_at=datetime.fromisoformat(raw["created_at"]),
                    completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
                )
                _QA_SESSIONS[str(sid)] = session
            except (KeyError, TypeError, ValueError):
                continue

    if isinstance(by_pitch, dict):
        _SESSION_IDS_BY_PITCH.update({
            str(pid): [str(item) for item in ids]
            for pid, ids in by_pitch.items()
            if isinstance(ids, list)
        })

    analyzed = payload.get("analyzed_answers", {})
    if isinstance(analyzed, dict):
        _ANALYZED_ANSWERS.update({
            str(k): v for k, v in analyzed.items() if isinstance(v, dict)
        })


_restore_state()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _raise_error(status_code: int, error: str, message: str | None = None) -> None:
    payload = {"error": error}
    if message:
        payload["message"] = message
    raise HTTPException(status_code=status_code, detail=payload)


# ==================== 파일 캐시 헬퍼 ====================

def _cache_path(pitch_id: str) -> Path:
    """pitch_id별 질문 캐시 파일 경로"""
    return QA_SESSION_DIR / f"{pitch_id}_questions.json"


def _save_questions_to_cache(pitch_id: str, session: QASessionData) -> None:
    """세션의 질문 목록을 파일로 캐시 저장 (Docker 재시작 대비)"""
    try:
        QA_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        cache = {
            "session_id": session.session_id,
            "pitch_id": session.pitch_id,
            "status": session.status,
            "questions": session.questions,
            "saved_at": _now().isoformat(),
        }
        with open(_cache_path(pitch_id), "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"[캐시] 질문 파일 저장 완료: {_cache_path(pitch_id)}")
    except Exception as e:
        print(f"[캐시] 저장 실패 (무시): {e}")


def _load_questions_from_cache(pitch_id: str) -> list[dict] | None:
    """파일 캐시에서 질문 목록을 복원합니다."""
    try:
        path = _cache_path(pitch_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        questions = cache.get("questions", [])
        if questions:
            print(f"[캐시] 파일에서 질문 {len(questions)}개 복원: {path}")
            return questions
        return None
    except Exception as e:
        print(f"[캐시] 로드 실패 (무시): {e}")
        return None


# ==================== 질문 생성 엔드포인트 ====================

@router.post(
    "/pitches/{pitch_id}/qa/questions/generate",
    summary="Q&A 질문 생성",
    tags=["qa"],
)
async def generate_qa_questions(
    pitch_id: str,
    notice_content: str = Form(..., description="공고문 전체 텍스트"),
    irdecksummary: str = Form(..., description="IR 덱 분석 요약"),
    presentation_content: str = Form(default=None, description="발표 내용 (선택)"),
    qa_mode: str = Form(default="REALTIME", description="REALTIME | GUIDE_ONLY"),
) -> dict:
    if qa_mode not in ("REALTIME", "GUIDE_ONLY"):
        _raise_error(400, "INVALID_QA_MODE", "qa_mode는 REALTIME 또는 GUIDE_ONLY 이어야 합니다.")

    session_id = str(uuid4())
    now_str = _now().isoformat()

    questions = await asyncio.to_thread(
        run_qa_question_generation,
        pitch_id=pitch_id,
        notice_content=notice_content,
        irdecksummary=irdecksummary,
        presentation_content=presentation_content,
    )

    if not questions:
        session = QASessionData(session_id=session_id, pitch_id=pitch_id, status="FAILED")
        with _LOCK:
            _QA_SESSIONS[session_id] = session
        _raise_error(500, "QA_GENERATION_FAILED", "질문 생성에 실패했습니다.")

    session = prepare_qa_session(session_id=session_id, pitch_id=pitch_id, questions=questions)
    session.status = "COMPLETED"

    with _LOCK:
        _QA_SESSIONS[session_id] = session
        _SESSION_IDS_BY_PITCH.setdefault(pitch_id, []).append(session_id)
        _persist_state_locked()

    print(f"[QA] 세션 생성 완료: {session_id} ({len(questions)}개 질문)")
    return {
        "pitch_id": pitch_id,
        "qa_training": {
            "qa_training_id": session_id,
            "notice_id": None,
            "ir_deck_id": None,
            "voice_analysis_id": None,
            "mode": qa_mode,
            "total_questions": len(session.questions),
            "version": 1,
            "is_latest": True,
            "created_at": now_str,
            "updated_at": now_str,
        },
        "questions": _format_questions(session.questions, session.answers),
    }


@router.get(
    "/pitches/{pitch_id}/qa/sessions/{session_id}",
    response_model=QASessionResponse,
    summary="QA 세션 조회",
    description="특정 QA 세션의 질문 리스트를 조회합니다.",
    tags=["qa"],
)
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


@router.get(
    "/pitches/{pitch_id}/questions",
    summary="Q&A 질문 목록 조회 (BACK 호환)",
    description=(
        "BACK 서버가 폴링할 때 사용하는 엔드포인트.\n\n"
        "메모리 세션이 없으면 파일 캐시에서 자동 복원합니다 (Docker 재시작 대비).\n\n"
        "**응답 필드:**\n"
        "- `questions[].type`: NOTICE | PITCHBOOK | PRESENTER | EVALUATOR\n"
        "- `questions[].content`: 생성된 질문 텍스트\n"
        "- `questions[].guidance`: 답변 방향 가이드"
    ),
    tags=["qa"],
)
async def get_all_qa_questions(pitch_id: str) -> dict:
    """Q&A 질문 목록 조회 (BACK 호환)"""
    # 1) 메모리에서 조회
    with _LOCK:
        session_ids = _SESSION_IDS_BY_PITCH.get(pitch_id, [])
        if session_ids:
            session = _QA_SESSIONS.get(session_ids[-1])
        else:
            # 메모리 전체에서 해당 pitch_id 최신 세션 탐색
            latest_session = None
            for sess in _QA_SESSIONS.values():
                if sess.pitch_id == pitch_id:
                    if not latest_session or sess.created_at > latest_session.created_at:
                        latest_session = sess
            session = latest_session

    if session and session.questions:
        now_str = _now().isoformat()
        questions = _format_questions(session.questions, session.answers)
        return {
            "pitch_id": pitch_id,
            "qa_training": {
                "qa_training_id": session.session_id,
                "mode": "REALTIME",
                "total_questions": len(session.questions),
                "version": 1,
                "is_latest": True,
            },
            "questions": questions,
            "updated_at": now_str,
        }

    # 2) 메모리 미스 → 파일 캐시 복원
    cached_questions = _load_questions_from_cache(pitch_id)
    if cached_questions:
        now_str = _now().isoformat()
        questions = _format_questions(cached_questions)
        return {
            "pitch_id": pitch_id,
            "qa_training": {
                "qa_training_id": None,
                "mode": "REALTIME",
                "total_questions": len(cached_questions),
                "version": 1,
                "is_latest": True,
            },
            "questions": questions,
            "updated_at": now_str,
        }

    return {
        "pitch_id": pitch_id,
        "qa_training": None,
        "questions": [],
        "updated_at": _now().isoformat(),
    }


def _format_questions(raw_questions: list[dict], session_answers: dict | None = None) -> list[dict]:
    result = []
    for q in raw_questions:
        qid = q["question_id"]
        result.append({
            "question_id": qid,
            "category": q.get("category", "PROBLEM"),
            "display_order": q["order"],
            "question": q["content"],
            "answer_guide": q.get("guidance"),
            "has_answer": bool(session_answers and qid in session_answers),
        })
    return result


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

    # 음성 파일 저장 (크기 제한 + 실제 확장자 보존 + question_id sanitize)
    safe_qid = re.sub(r"[^a-zA-Z0-9\-]", "_", question_id)
    audio_path = None
    if audio and audio.filename:
        audio_dir = Path("data/output/qa_audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(audio.filename).suffix.lower() or ".wav"
        audio_path = audio_dir / f"{session_id}_{safe_qid}{suffix}"
        payload = await read_upload_limited(
            audio,
            MAX_QA_AUDIO_FILE_SIZE,
            too_large_message="파일 크기는 25MB 이하여야 합니다",
        )
        audio_path.write_bytes(payload)

    # 답변 처리 — _LOCK을 잡지 않고 Gemini/STT 호출
    try:
        answer = await asyncio.to_thread(
            process_answer,
            session,
            question_id,
            str(audio_path) if audio_path else None,
            text_transcript,
        )
        if answer:
            with _LOCK:
                _persist_state_locked()
            print(f"[QA] 답변 처리 완료: {question_id}")
            return answer
    except Exception as e:
        print(f"[QA] 답변 평가 중 오류: {e}")

    # 평가 실패 시 FAILED 상태로 저장해 GET 폴링이 끝날 수 있도록 함
    answer_id = str(uuid4())
    failed_stub: dict = {
        "answer_id": answer_id,
        "question_id": question_id,
        "pitch_id": pitch_id,
        "audio_url": str(audio_path) if audio_path else None,
        "text_transcript": text_transcript or "",
        "evaluation_status": AnalysisStatus.FAILED.value,
        "created_at": _now().isoformat(),
        "evaluated_at": None,
        "feedback": None,
    }
    with _LOCK:
        live_session = _QA_SESSIONS.get(session_id)
        if live_session:
            live_session.answers[question_id] = failed_stub
        _persist_state_locked()
    return AnswerResponse(
        answer_id=answer_id,
        question_id=question_id,
        pitch_id=pitch_id,
        audio_url=str(audio_path) if audio_path else None,
        text_transcript=text_transcript or "",
        evaluation_status=AnalysisStatus.FAILED,
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

    # 답변 찾기 — session.answers 순회는 _LOCK 안에서 수행해 동시 쓰기와의 race 방지
    with _LOCK:
        found = None
        for _q_id, answer_data in session.answers.items():
            if answer_data.get("answer_id") == answer_id:
                found = dict(answer_data)
                break

    if found:
        feedback_data = found.get("feedback")
        return AnswerResponse(
            answer_id=found["answer_id"],
            question_id=found["question_id"],
            pitch_id=found["pitch_id"],
            audio_url=found.get("audio_url"),
            text_transcript=found["text_transcript"],
            feedback=AnswerFeedback(**feedback_data) if feedback_data else None,
            evaluation_status=found.get("evaluation_status"),
            created_at=datetime.fromisoformat(found["created_at"]),
            evaluated_at=datetime.fromisoformat(found["evaluated_at"])
            if found.get("evaluated_at")
            else None,
        )

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


# ==================== 답변 분석 엔드포인트 (BACK 호환) ====================

@router.post(
    "/questions/{question_id}/answers/analyze",
    summary="Q&A 답변 분석",
    description="음성 또는 텍스트 답변을 받아 Gemini로 평가합니다. BACK의 analyzeQaAnswer()가 호출하는 엔드포인트.",
    tags=["qa"],
)
async def analyze_qa_answer(
    question_id: str,
    file: UploadFile = File(None),
    question: str = Form(...),
    answer_guide: str = Form(default=""),
    question_type: str = Form(default="EVALUATOR"),
) -> dict:
    """
    POST /api/questions/{question_id}/answers/analyze

    음성 파일을 STT 처리 후 답변을 평가합니다.
    음성 없이 텍스트만 있을 경우도 평가합니다.
    """
    from src.domain.qa.answer_evaluator import run_answer_evaluation, calculate_weighted_score

    transcript = ""
    audio_file_url = None

    # 음성 파일 처리
    if file and file.filename:
        try:
            from src.domain.voice.whisper_adapter import transcribe_audio
            audio_dir = Path("data/output/qa_audio")
            audio_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(file.filename).suffix or ".webm"
            audio_path = audio_dir / f"{question_id}_answer{suffix}"
            payload = await read_upload_limited(
                file,
                MAX_QA_AUDIO_FILE_SIZE,
                too_large_message="파일 크기는 25MB 이하여야 합니다",
            )
            validate_audio_payload(payload)
            audio_path.write_bytes(payload)
            audio_file_url = str(audio_path)
            text, _ = transcribe_audio(audio_path)
            transcript = text
        except Exception as e:
            print(f"[QA 답변] STT 실패: {e}")

    if not transcript:
        transcript = "(음성 변환 실패 또는 텍스트 미제공)"

    # 답변 평가
    answer_id = str(uuid4())
    now_str = _now().isoformat()

    try:
        evaluation = await asyncio.to_thread(
            lambda: run_answer_evaluation(
                question_type=question_type,
                question_content=question,
                guidance=answer_guide,
                answer_transcript=transcript,
            )
        )
        if evaluation is None:
            raise ValueError("Gemini 평가 결과 파싱 실패")
        final_score = calculate_weighted_score(evaluation)
        answer_dict = {
            "answer_id": answer_id,
            "audio_file_url": audio_file_url,
            "transcription": transcript,
            "briefness_score": evaluation.clarity,
            "evidence_score": evaluation.relevance,
            "structure_score": evaluation.structure,
            "strengths": ", ".join(evaluation.strengths) if evaluation.strengths else None,
            "weaknesses": ", ".join(evaluation.improvements) if evaluation.improvements else None,
            "answered_at": now_str,
            "created_at": now_str,
            "updated_at": now_str,
        }
    except Exception as e:
        print(f"[QA 답변] 평가 실패: {e}")
        answer_dict = {
            "answer_id": answer_id,
            "audio_file_url": audio_file_url,
            "transcription": transcript,
            "briefness_score": None,
            "evidence_score": None,
            "structure_score": None,
            "strengths": None,
            "weaknesses": None,
            "answered_at": now_str,
            "created_at": now_str,
            "updated_at": now_str,
        }

    with _LOCK:
        _ANALYZED_ANSWERS[answer_id] = {"question_id": question_id, **answer_dict}
        _persist_state_locked()

    return {"question_id": question_id, "answer": answer_dict}


# ==================== 답변 단건 조회 ====================

@router.get(
    "/answers/{answer_id}",
    summary="Q&A 답변 조회",
    tags=["qa"],
)
async def get_answer_by_id(answer_id: str) -> dict:
    """GET /api/answers/{answer_id} — analyze_qa_answer가 저장한 답변을 반환합니다."""
    with _LOCK:
        record = _ANALYZED_ANSWERS.get(answer_id)

    if not record:
        _raise_error(404, "ANSWER_NOT_FOUND", "해당 답변을 찾을 수 없습니다")

    question_id = record.get("question_id")
    answer_dict = {k: v for k, v in record.items() if k != "question_id"}
    return {"question_id": question_id, "answer": answer_dict}


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
