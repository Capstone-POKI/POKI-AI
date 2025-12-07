from typing import List, Dict, Any
from pydantic import BaseModel, Field


# =======================
# Deck / Document Analysis
# =======================

class VoiceGuide(BaseModel):
    estimated_duration_sec: float
    pacing_advice: str


class SlideContents(BaseModel):
    full_text: str
    summary: str
    char_count: int
    image_count: int


class Slide(BaseModel):
    page_number: int
    section_type: str
    contents: SlideContents
    voice_guide: VoiceGuide
    design_feedback: List[str]


class Diagnosis(BaseModel):
    missing_sections: List[str]
    logic_flow_issues: List[str]


class DeckMeta(BaseModel):
    doc_type: str
    pitch_strategy: Dict[str, Any]


class DeckAnalysisResult(BaseModel):
    meta: DeckMeta
    diagnosis: Diagnosis
    slides: List[Slide]


# =======================
# Speech / Voice Analysis
# =======================

class DetailScore(BaseModel):
    문제_정의: int
    솔루션_명확성: int
    시장성: int
    사업성_BM: int
    경쟁력_차별성: int
    전달력: int
    톤_일관성: int


class SituationScore(BaseModel):
    총점: int
    세부_기준: DetailScore


class VoiceDeliveryAnalysis(BaseModel):
    말하기_속도_WPM: float
    억양_강조_안정성: str
    감정_톤: str
    문장_명료성: str
    불필요한_말버릇: str
    강점: List[str]
    개선점: List[str]


class SpeechAnalysisResult(BaseModel):
    발표_상황: str
    상황_적합성_점수: SituationScore
    음성_전달력_분석: VoiceDeliveryAnalysis
    summary_1min: str = Field(..., alias="1분_요약")

    class Config:
        allow_population_by_field_name = True