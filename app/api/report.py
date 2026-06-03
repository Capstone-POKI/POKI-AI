from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path as FPath
from pydantic import BaseModel

from src.infrastructure.gemini.client import GeminiJSONClient

router = APIRouter(prefix="/api", tags=["report"])

_gemini = GeminiJSONClient()


# ──────────────────────────────────────────
# Request / Response 스키마
# ──────────────────────────────────────────

class GenerateReportRequest(BaseModel):
    ir_deck_summary: str = ""
    voice_summary: str = ""
    notice_summary: str = ""
    qa_summary: str = ""
    ir_deck_score: float = 0
    voice_score: float = 0
    notice_score: float = 0
    qa_score: float = 0


class RadarChart(BaseModel):
    labels: List[str]
    scores: List[float]


class BarChartItem(BaseModel):
    label: str
    score: float


class BarChart(BaseModel):
    items: List[BarChartItem]


class DetailScore(BaseModel):
    title: str
    score: float
    description: str
    strengths: List[str]
    improvements: List[str]


class GenerateReportResponse(BaseModel):
    pitch_id: str
    final_score: float
    radar_chart: RadarChart
    bar_chart: BarChart
    detail_scores: List[DetailScore]
    improvement_points: List[str]
    summary: str
    generated_at: str


# ──────────────────────────────────────────
# 폴백 데이터 생성 (Gemini 실패 시)
# ──────────────────────────────────────────

def _build_fallback(pitch_id: str, req: GenerateReportRequest) -> GenerateReportResponse:
    """Gemini 호출 실패 시 rule-based로 응답 구성."""
    final_score = round(
        (req.ir_deck_score + req.voice_score + req.notice_score + req.qa_score) / 4, 1
    )
    return GenerateReportResponse(
        pitch_id=pitch_id,
        final_score=final_score,
        radar_chart=RadarChart(
            labels=["문제정의", "솔루션", "시장성", "전달력", "Q&A대응력"],
            scores=[
                round(req.ir_deck_score * 0.9, 1),
                round(req.ir_deck_score * 0.95, 1),
                round(req.ir_deck_score * 1.0, 1),
                round(req.voice_score * 1.0, 1),
                round(req.qa_score * 1.0, 1),
            ],
        ),
        bar_chart=BarChart(
            items=[
                BarChartItem(label="AI 플랫폼", score=min(100, round(req.ir_deck_score * 1.1, 1))),
                BarChartItem(label="시장 창출", score=round(req.ir_deck_score, 1)),
                BarChartItem(label="비즈니스 모델", score=round(req.notice_score, 1)),
                BarChartItem(label="투자 유치", score=round(req.qa_score, 1)),
                BarChartItem(label="재무 계획", score=round(req.voice_score * 0.85, 1)),
                BarChartItem(label="경쟁 분석", score=round(req.ir_deck_score * 0.8, 1)),
            ]
        ),
        detail_scores=[
            DetailScore(
                title="발표 완성도",
                score=round((req.notice_score + req.voice_score) / 2, 1),
                description="전반적인 발표 구성과 내용의 완성도를 평가한 결과입니다.",
                strengths=["발표 구조가 체계적으로 구성됨", "핵심 메시지 전달이 명확함"],
                improvements=["세부 근거 데이터 보완 필요", "공고 요구사항과의 연계 강화 필요"],
            ),
            DetailScore(
                title="Deck 구성 정합도",
                score=round(req.ir_deck_score, 1),
                description="IR 덱의 슬라이드 구성과 논리적 흐름을 평가한 결과입니다.",
                strengths=["비즈니스 모델 설명이 명확함", "시장 분석 자료가 충실함"],
                improvements=["재무 계획 구체화 필요", "팀 역량 페이지 보완 필요"],
            ),
            DetailScore(
                title="피칭 전달력",
                score=round(req.voice_score, 1),
                description="발표 음성 및 전달 방식을 분석한 결과입니다.",
                strengths=["발화 속도와 리듬이 안정적임", "자신감 있는 발표 태도"],
                improvements=["강조 구간에서 목소리 톤 변화 활용", "핵심 수치 발표 시 명확한 강조 필요"],
            ),
            DetailScore(
                title="Q&A 대응력",
                score=round(req.qa_score, 1),
                description="질문에 대한 답변 품질과 대응 역량을 평가한 결과입니다.",
                strengths=["논리적 답변 구조 유지", "피칭 내용과 일관된 답변"],
                improvements=["두괄식 답변 방식 적용 필요", "예상 질문에 대한 사전 준비 강화"],
            ),
        ],
        improvement_points=[
            "공고 요구사항과 발표 내용의 일치도를 높이기 위해 핵심 키워드를 강조하세요.",
            "투자자가 관심 있어 하는 재무 지표와 성장 계획을 구체적으로 제시하세요.",
            "경쟁사 대비 차별화된 기술력이나 시장 포지셔닝을 명확히 표현하세요.",
        ],
        summary=(
            f"발표 완성도 {round((req.notice_score + req.voice_score) / 2, 1)}점, "
            f"IR Deck {round(req.ir_deck_score, 1)}점, "
            f"Q&A 대응력 {round(req.qa_score, 1)}점으로 "
            f"종합 {final_score}점입니다. "
            "전반적으로 준비가 잘 되어 있으며 세부 보완을 통해 완성도를 높일 수 있습니다."
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ──────────────────────────────────────────
# Gemini 프롬프트 구성 및 호출
# ──────────────────────────────────────────

_REPORT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "final_score": {"type": "NUMBER"},
        "summary": {"type": "STRING"},
        "radar_chart": {
            "type": "OBJECT",
            "properties": {
                "labels": {"type": "ARRAY", "items": {"type": "STRING"}},
                "scores": {"type": "ARRAY", "items": {"type": "NUMBER"}},
            },
        },
        "bar_chart": {
            "type": "OBJECT",
            "properties": {
                "items": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "label": {"type": "STRING"},
                            "score": {"type": "NUMBER"},
                        },
                    },
                }
            },
        },
        "detail_scores": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "score": {"type": "NUMBER"},
                    "description": {"type": "STRING"},
                    "strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "improvements": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
            },
        },
        "improvement_points": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
}


def _build_prompt(pitch_id: str, req: GenerateReportRequest) -> str:
    return f"""당신은 스타트업 IR 피칭 전문 코치입니다.
아래 분석 데이터를 바탕으로 발표자를 위한 종합 리포트를 JSON 형식으로 생성하세요.

=== 분석 데이터 ===
[공고문 분석] (점수: {req.notice_score}/100)
{req.notice_summary or '(데이터 없음)'}

[IR Deck 분석] (점수: {req.ir_deck_score}/100)
{req.ir_deck_summary or '(데이터 없음)'}

[음성 분석] (점수: {req.voice_score}/100)
{req.voice_summary or '(데이터 없음)'}

[Q&A 분석] (점수: {req.qa_score}/100)
{req.qa_summary or '(데이터 없음)'}

=== 출력 지침 ===
반드시 아래 구조의 JSON만 출력하세요. 설명 텍스트 없이 JSON만 출력합니다.

1. final_score: 네 점수의 가중 평균 (공고문 20%, IR Deck 35%, 음성 25%, Q&A 20%), 소수점 1자리
2. summary: 전체 발표를 2~3문장으로 요약하는 한국어 총평
3. radar_chart: 5개 축 레이더 차트
   - labels: ["문제정의", "솔루션", "시장성", "전달력", "Q&A대응력"]
   - scores: 각 축 점수 (0~100, 소수점 1자리)
4. bar_chart: 핵심 주제별 커버리지 점수
   - items: 6개 항목 [{{"label": "AI 플랫폼", "score": ...}}, {{"label": "시장 창출", "score": ...}}, {{"label": "비즈니스 모델", "score": ...}}, {{"label": "투자 유치", "score": ...}}, {{"label": "재무 계획", "score": ...}}, {{"label": "경쟁 분석", "score": ...}}]
5. detail_scores: 4개 세부 항목
   - 발표 완성도: 공고 적합성과 발표 전체 완성도
   - Deck 구성 정합도: IR 덱 슬라이드 구성과 논리
   - 피칭 전달력: 음성 및 발표 스타일
   - Q&A 대응력: 질문 대응 역량
   각 항목: title(제목), score(0~100), description(2~3문장 한국어 설명), strengths(2개 강점), improvements(2개 개선점)
6. improvement_points: 가장 중요한 개선 포인트 3개 (한국어, 각 1문장)

pitch_id: {pitch_id}
"""


def _parse_gemini_response(
    pitch_id: str, raw: dict, req: GenerateReportRequest
) -> GenerateReportResponse:
    """Gemini JSON 응답을 GenerateReportResponse로 변환."""
    def _safe_float(val, default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # radar_chart
    rc_raw = raw.get("radar_chart", {})
    radar = RadarChart(
        labels=rc_raw.get("labels", ["문제정의", "솔루션", "시장성", "전달력", "Q&A대응력"]),
        scores=[_safe_float(s) for s in rc_raw.get("scores", [0, 0, 0, 0, 0])],
    )

    # bar_chart
    bc_raw = raw.get("bar_chart", {})
    bar_items = [
        BarChartItem(label=item.get("label", ""), score=_safe_float(item.get("score", 0)))
        for item in bc_raw.get("items", [])
    ]
    bar = BarChart(items=bar_items)

    # detail_scores
    detail_list = []
    for ds in raw.get("detail_scores", []):
        detail_list.append(
            DetailScore(
                title=str(ds.get("title", "")),
                score=_safe_float(ds.get("score", 0)),
                description=str(ds.get("description", "")),
                strengths=list(ds.get("strengths", [])),
                improvements=list(ds.get("improvements", [])),
            )
        )

    return GenerateReportResponse(
        pitch_id=pitch_id,
        final_score=_safe_float(raw.get("final_score", 0)),
        radar_chart=radar,
        bar_chart=bar,
        detail_scores=detail_list,
        improvement_points=list(raw.get("improvement_points", [])),
        summary=str(raw.get("summary", "")),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ──────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────

@router.post(
    "/pitches/{pitch_id}/report/generate",
    response_model=GenerateReportResponse,
    summary="AI 종합 리포트 생성",
    description=(
        "IR Deck 분석, 음성 분석, 공고 분석, Q&A 분석 결과를 바탕으로 "
        "Gemini API를 호출해 프론트 UI용 종합 리포트 JSON을 생성합니다. "
        "Gemini 호출 실패 시 rule-based 폴백으로 응답합니다."
    ),
)
async def generate_report(
    pitch_id: str = FPath(..., description="Pitch ID"),
    body: GenerateReportRequest = ...,
) -> GenerateReportResponse:
    """
    POST /api/pitches/{pitch_id}/report/generate

    Request body:
    - ir_deck_summary: IR 덱 분석 요약 텍스트
    - voice_summary: 음성 분석 요약 텍스트
    - notice_summary: 공고 분석 요약 텍스트
    - qa_summary: Q&A 분석 요약 텍스트
    - ir_deck_score: IR 덱 점수 (0~100)
    - voice_score: 음성 점수 (0~100)
    - notice_score: 공고 점수 (0~100)
    - qa_score: Q&A 점수 (0~100)
    """
    try:
        prompt = _build_prompt(pitch_id, body)
        raw = _gemini.generate_json(prompt, temperature=0.3, response_schema=_REPORT_SCHEMA)
        return _parse_gemini_response(pitch_id, raw, body)
    except Exception as exc:
        print(f"[report] Gemini 호출 실패 (pitch_id={pitch_id}): {exc}. 폴백 사용.")
        return _build_fallback(pitch_id, body)
