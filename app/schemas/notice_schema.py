from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class NoticeAnalysisStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None


class NoticeUploadResponse(BaseModel):
    notice_id: str
    pitch_id: str
    analysis_status: NoticeAnalysisStatus = NoticeAnalysisStatus.IN_PROGRESS
    message: str


class EvaluationCriteriaItem(BaseModel):
    criteria_name: str
    points: int = Field(ge=0, le=100)
    pitchcoach_interpretation: str
    ir_guide: str


class EvaluationCriteriaUpdateItem(BaseModel):
    criteria_name: str
    points: int = Field(ge=0, le=100)


class NoticeResultInProgressResponse(BaseModel):
    notice_id: str
    pitch_id: str
    analysis_status: NoticeAnalysisStatus = NoticeAnalysisStatus.IN_PROGRESS
    updated_at: datetime


class NoticeResultFailedResponse(BaseModel):
    notice_id: str
    pitch_id: str
    analysis_status: NoticeAnalysisStatus = NoticeAnalysisStatus.FAILED
    error_message: str
    updated_at: datetime


class NoticeResultCompletedResponse(BaseModel):
    notice_id: str
    pitch_id: str
    analysis_status: NoticeAnalysisStatus = NoticeAnalysisStatus.COMPLETED
    notice_name: str | None = None
    host_organization: str | None = None
    recruitment_type: str | None = None
    target_audience: str | None = None
    application_period: str | None = None
    evaluation_criteria: list[EvaluationCriteriaItem] = Field(default_factory=list)
    additional_criteria: str | None = None
    ir_deck_guide: str | None = None
    created_at: datetime
    updated_at: datetime


class NoticeUpdateRequest(BaseModel):
    notice_name: str | None = None
    host_organization: str | None = None
    recruitment_type: str | None = None
    target_audience: str | None = None
    application_period: str | None = None
    evaluation_criteria: list[EvaluationCriteriaUpdateItem] | None = None
    additional_criteria: str | None = None

