from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class QuestionType(str, Enum):
    """질문의 관점에 따른 분류"""
    NOTICE = "NOTICE"  # 공고 관점
    PITCHBOOK = "PITCHBOOK"  # IR Deck 관점
    PRESENTER = "PRESENTER"  # 발표자 관점
    EVALUATOR = "EVALUATOR"  # 심사위원 관점


class QuestionResponse(BaseModel):
    """QA 질문 객체"""
    question_id: str
    pitch_id: str
    question_type: QuestionType
    content: str
    guidance: Optional[str] = None  # 답변 가이드
    context: Optional[dict] = None  # 컨텍스트 (공고 내용, 발표 내용 등)
    order: int = Field(ge=1)
    created_at: datetime


class AnswerRequest(BaseModel):
    """답변 제출 요청"""
    question_id: str
    audio_url: Optional[str] = None  # 음성 URL (사전 수록)
    text_transcript: Optional[str] = None  # STT 변환 결과


class AnswerFeedback(BaseModel):
    """답변에 대한 피드백"""
    score: int = Field(ge=0, le=100)
    max_score: int = Field(default=100, ge=0)
    relevance: int = Field(ge=0, le=100)  # 적합도
    clarity: int = Field(ge=0, le=100)  # 명료도
    structure: int = Field(ge=0, le=100)  # 구성도
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    reasoning: str  # 평가 근거


class AnswerResponse(BaseModel):
    """답변 객체"""
    answer_id: str
    question_id: str
    pitch_id: str
    audio_url: Optional[str] = None
    text_transcript: str  # STT 변환 결과 또는 직접 입력
    feedback: Optional[AnswerFeedback] = None
    evaluation_status: AnalysisStatus = AnalysisStatus.IN_PROGRESS
    created_at: datetime
    evaluated_at: Optional[datetime] = None


class QASessionRequest(BaseModel):
    """QA 세션 시작 요청"""
    pitch_id: str
    notice_id: Optional[str] = None
    ir_deck_id: Optional[str] = None
    voice_report_id: Optional[str] = None


class QASessionResponse(BaseModel):
    """QA 세션 응답"""
    session_id: str
    pitch_id: str
    status: AnalysisStatus = AnalysisStatus.IN_PROGRESS
    questions: list[QuestionResponse] = Field(default_factory=list)
    total_questions: int = Field(ge=0)
    answered_count: int = Field(ge=0)
    created_at: datetime


class ErrorResponse(BaseModel):
    error: str
    message: Optional[str] = None
