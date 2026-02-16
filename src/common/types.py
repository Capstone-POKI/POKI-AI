from typing import List, TypedDict, Union


Number = Union[int, float]


class EvaluationCriterion(TypedDict):
    criteria_name: str
    points: Number
    sub_requirements: List[str]
    pitchcoach_interpretation: str


class NoticeAnalysisResult(TypedDict):
    notice_name: str
    host_organization: str
    recruitment_type: str
    target_audience: str
    application_period: str
    summary: str
    core_requirements: str
    source_reference: str
    evaluation_structure_type: str
    extraction_confidence: float
    evaluation_criteria: List[EvaluationCriterion]
    ir_deck_guide: str
