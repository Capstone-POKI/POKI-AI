from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None


class IRUploadResponse(BaseModel):
    ir_deck_id: str
    pitch_id: str
    analysis_status: AnalysisStatus = AnalysisStatus.IN_PROGRESS
    version: int = Field(ge=1)
    message: str


class DeckScoreResponse(BaseModel):
    total_score: int = Field(ge=0, le=100)
    structure_summary: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


class CriteriaScoreResponse(BaseModel):
    criteria_name: str
    pitchcoach_interpretation: str
    ir_guide: str
    score: int = Field(ge=0, le=100)
    feedback: str


class EmphasizedSlideResponse(BaseModel):
    slide_number: int = Field(ge=1)
    reason: str


class PresentationGuideResponse(BaseModel):
    emphasized_slides: list[EmphasizedSlideResponse] = Field(default_factory=list)
    guide: list[str] = Field(default_factory=list)
    time_allocation: list[str] = Field(default_factory=list)


class IRDeckSummaryInProgressResponse(BaseModel):
    ir_deck_id: str
    pitch_id: str
    analysis_status: AnalysisStatus = AnalysisStatus.IN_PROGRESS
    version: int = Field(ge=1)


class IRDeckSummaryFailedResponse(BaseModel):
    ir_deck_id: str
    pitch_id: str
    analysis_status: AnalysisStatus = AnalysisStatus.FAILED
    error_message: str
    version: int = Field(ge=1)


class IRDeckSummaryCompletedResponse(BaseModel):
    ir_deck_id: str
    pitch_id: str
    analysis_status: AnalysisStatus = AnalysisStatus.COMPLETED
    version: int = Field(ge=1)
    deck_score: DeckScoreResponse
    criteria_scores: list[CriteriaScoreResponse] = Field(default_factory=list)
    presentation_guide: PresentationGuideResponse
    analyzed_at: datetime | None = None


class IRDeckSlideItemResponse(BaseModel):
    slide_number: int = Field(ge=1)
    category: str
    score: int = Field(ge=0, le=100)
    thumbnail_url: str | None = None
    content_summary: str
    detailed_feedback: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


class IRDeckSlidesInProgressResponse(BaseModel):
    ir_deck_id: str
    analysis_status: AnalysisStatus = AnalysisStatus.IN_PROGRESS


class IRDeckSlidesCompletedResponse(BaseModel):
    ir_deck_id: str
    analysis_status: AnalysisStatus = AnalysisStatus.COMPLETED
    total_slides: int = Field(ge=0)
    slides: list[IRDeckSlideItemResponse] = Field(default_factory=list)
