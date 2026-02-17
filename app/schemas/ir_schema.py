from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class IRAnalyzeRequest(BaseModel):
    ir_pdf: str = "data/input/sample_irdeck.pdf"
    output_dir: str = "data/output/ir_analysis"
    pitch_type: "PitchType | None" = None
    strategy_json: str | None = None
    notice_output_dir: str = "data/output/notice_analysis"
    no_auto_strategy: bool = False
    no_chunking: bool = False


class AnalysisStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class CoverageStatus(str, Enum):
    COVERED = "COVERED"
    PARTIALLY_COVERED = "PARTIALLY_COVERED"
    NOT_COVERED = "NOT_COVERED"


class PitchType(str, Enum):
    ELEVATOR = "ELEVATOR"
    VC_DEMO = "VC_DEMO"
    GOVERNMENT = "GOVERNMENT"
    COMPETITION = "COMPETITION"
    # Backward compatibility
    GOV_SUPPORT = "GOV_SUPPORT"
    STARTUP_CONTEST = "STARTUP_CONTEST"


class DeckScoreModel(BaseModel):
    total_score: int = Field(ge=0, le=100)
    structure_summary: str
    strengths: list[str] | str = Field(default_factory=list)
    improvements: list[str] | str = Field(default_factory=list)
    top_actions: list[str] | str = Field(default_factory=list)


class MissingItemModel(BaseModel):
    item_id: str
    item_name: str
    suggestion: str


class CriteriaScoreModel(BaseModel):
    criteria_score_id: str
    criteria_id: str
    criteria_name: str
    pitchcoach_interpretation: str | None = None
    score: int = Field(ge=0, le=100)
    max_score: int = 100
    raw_score: float = Field(ge=0)
    raw_max_score: float = Field(gt=0)
    is_covered: bool
    coverage_status: CoverageStatus
    feedback: str
    related_slides: list[int] | str = Field(default_factory=list)
    missing_items: list[MissingItemModel] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)


class EmphasizedSlideModel(BaseModel):
    slide_number: int = Field(ge=1)
    reason: str


class TimeAllocationModel(BaseModel):
    section: str
    seconds: int = Field(gt=0)


class PresentationGuideModel(BaseModel):
    emphasized_slides: list[EmphasizedSlideModel] = Field(default_factory=list)
    guide: list[str] = Field(default_factory=list)
    time_allocation: list[TimeAllocationModel] | list[str] = Field(default_factory=list)


class IRDeckAnalysisResponse(BaseModel):
    ir_deck_id: str
    pitch_id: str
    analysis_status: AnalysisStatus
    version: int = Field(ge=1)
    pitch_type: PitchType | None = None
    deck_score: DeckScoreModel
    criteria_scores: list[CriteriaScoreModel] = Field(default_factory=list)
    presentation_guide: PresentationGuideModel
    analyzed_at: datetime | None = None


class SlideFeedbackModel(BaseModel):
    detailed_feedback: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


class IRSlideItemModel(BaseModel):
    slide_id: str
    slide_number: int = Field(ge=1)
    category: str
    score: int = Field(ge=0, le=100)
    thumbnail_url: str | None = None
    content: str
    display_order: int = Field(ge=1)
    feedback: SlideFeedbackModel


class IRDeckSlidesResponse(BaseModel):
    ir_deck_id: str
    analysis_status: AnalysisStatus
    total_slides: int = Field(ge=0)
    slides: list[IRSlideItemModel] = Field(default_factory=list)
