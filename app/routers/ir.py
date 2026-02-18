from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Path as FPath, UploadFile

from app.schemas.ir_schema import (
    AnalysisStatus,
    CriteriaScoreResponse,
    DeckScoreResponse,
    ErrorResponse,
    IRDeckSlideItemResponse,
    IRDeckSlidesCompletedResponse,
    IRDeckSlidesInProgressResponse,
    IRDeckSummaryCompletedResponse,
    IRDeckSummaryFailedResponse,
    IRDeckSummaryInProgressResponse,
    IRUploadResponse,
    PresentationGuideResponse,
)
from src.domain.ir.pipeline import run_ir_analysis

try:
    from app.routers import notice as notice_router_module
except Exception:  # pragma: no cover
    notice_router_module = None

router = APIRouter(prefix="/api", tags=["ir-deck"])

MAX_IR_FILE_SIZE = 30 * 1024 * 1024  # 30MB
IR_UPLOAD_DIR = Path("data/output/ir_uploads")
IR_ANALYSIS_DIR = Path("data/output/ir_analysis")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _raise_error(status_code: int, error: str, message: str | None = None) -> None:
    payload = {"error": error}
    if message:
        payload["message"] = message
    raise HTTPException(status_code=status_code, detail=payload)


@dataclass
class IRDeckRow:
    id: str
    pitch_id: str
    notice_id: str | None = None
    pdf_url: str | None = None
    pdf_size_bytes: int | None = None
    pdf_upload_status: str | None = None
    version: int = 1
    is_latest: bool = True
    analysis_status: AnalysisStatus = AnalysisStatus.IN_PROGRESS
    error_message: str | None = None
    analyzed_at: datetime | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@dataclass
class IRDeckResultRow:
    # API projection-ready cache
    deck_score: dict = field(default_factory=dict)
    criteria_scores: list[dict] = field(default_factory=list)
    presentation_guide: dict = field(default_factory=dict)
    slides: list[dict] = field(default_factory=list)


_LOCK = threading.Lock()
_IR_BY_ID: dict[str, IRDeckRow] = {}
_IR_IDS_BY_PITCH: dict[str, list[str]] = {}
_RESULT_BY_IR_ID: dict[str, IRDeckResultRow] = {}


def _next_ir_version(pitch_id: str) -> int:
    ids = _IR_IDS_BY_PITCH.get(pitch_id, [])
    if not ids:
        return 1
    return max((_IR_BY_ID[i].version for i in ids if i in _IR_BY_ID), default=0) + 1


def _latest_notice_id_for_pitch(pitch_id: str) -> str | None:
    if notice_router_module is None:
        return None
    try:
        lock = getattr(notice_router_module, "_LOCK", None)
        by_pitch = getattr(notice_router_module, "_NOTICE_IDS_BY_PITCH", None)
        by_id = getattr(notice_router_module, "_NOTICE_BY_ID", None)
        if lock is None or by_pitch is None or by_id is None:
            return None
        with lock:
            ids = by_pitch.get(pitch_id, [])
            latest_id = None
            latest_ver = -1
            for nid in ids:
                row = by_id.get(nid)
                if row is None:
                    continue
                if getattr(row, "is_latest", False) and int(getattr(row, "version", 0)) >= latest_ver:
                    latest_id = nid
                    latest_ver = int(getattr(row, "version", 0))
            return latest_id
    except Exception:
        return None


def _to_score_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [line.strip() for line in value.split("\n") if line.strip()]
    return []


def _to_display_category(raw: str) -> str:
    mapping = {
        "COVER": "표지",
        "PROBLEM": "문제 정의",
        "SOLUTION": "솔루션",
        "PRODUCT": "제품",
        "MARKET": "시장 분석",
        "BUSINESS_MODEL": "비즈니스 모델",
        "COMPETITION": "경쟁 분석",
        "TRACTION": "실적",
        "TEAM": "팀 소개",
        "FINANCE": "자금 계획",
        "ASK": "요청사항",
        "OTHER": "기타",
    }
    key = (raw or "").strip().upper()
    return mapping.get(key, raw or "기타")


def _normalize_time_allocation(value: object) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                section = str(item.get("section", "")).strip()
                seconds = int(item.get("seconds", 0) or 0)
                if section and seconds > 0:
                    mins = seconds // 60
                    out.append(f"{section} ({mins}분)")
        return out
    return []


def _load_notice_criteria_for_ir(pitch_id: str) -> list[dict] | None:
    if notice_router_module is None:
        return None
    try:
        lock = getattr(notice_router_module, "_LOCK", None)
        by_pitch = getattr(notice_router_module, "_NOTICE_IDS_BY_PITCH", None)
        by_id = getattr(notice_router_module, "_NOTICE_BY_ID", None)
        c_by_notice = getattr(notice_router_module, "_CRITERIA_BY_NOTICE_ID", None)
        if lock is None or by_pitch is None or by_id is None or c_by_notice is None:
            return None
        with lock:
            ids = by_pitch.get(pitch_id, [])
            if not ids:
                return None
            latest = None
            latest_ver = -1
            for nid in ids:
                row = by_id.get(nid)
                if row is None:
                    continue
                if getattr(row, "is_latest", False) and int(getattr(row, "version", 0)) >= latest_ver:
                    latest = nid
                    latest_ver = int(getattr(row, "version", 0))
            if latest is None:
                return None
            rows = c_by_notice.get(latest, [])
            mapped = []
            for r in sorted(rows, key=lambda x: int(getattr(x, "display_order", 9999))):
                mapped.append(
                    {
                        "criteria_name": str(getattr(r, "criteria_name", "")),
                        "pitchcoach_interpretation": str(getattr(r, "pitchcoach_interpretation", "")),
                        "ir_guide": str(getattr(r, "ir_guide", "")),
                    }
                )
            return mapped or None
    except Exception:
        return None


def _map_ir_payload_to_result(payload: dict, pitch_id: str) -> IRDeckResultRow:
    deck_raw = payload.get("deck_score", {}) if isinstance(payload, dict) else {}
    criteria_raw = payload.get("criteria_scores", []) if isinstance(payload, dict) else []
    guide_raw = payload.get("presentation_guide", {}) if isinstance(payload, dict) else {}
    slides_raw = payload.get("slides", []) if isinstance(payload, dict) else []

    deck_score = {
        "total_score": int(deck_raw.get("total_score", 0) or 0),
        "structure_summary": str(deck_raw.get("structure_summary", "") or ""),
        "strengths": _to_score_list(deck_raw.get("strengths")),
        "improvements": _to_score_list(deck_raw.get("improvements")),
    }

    criteria_scores: list[dict] = []
    notice_criteria = _load_notice_criteria_for_ir(pitch_id)
    notice_by_name = {c["criteria_name"]: c for c in (notice_criteria or [])}
    for item in criteria_raw:
        name = str(item.get("criteria_name", "")).strip()
        if not name:
            continue
        # UI 정책: 공고 없는 pitch는 criteria_scores 숨김
        if notice_criteria is None:
            continue
        n = notice_by_name.get(name, {})
        criteria_scores.append(
            {
                "criteria_name": name,
                "pitchcoach_interpretation": n.get("pitchcoach_interpretation")
                or str(item.get("pitchcoach_interpretation", "") or f"{name} 항목을 평가합니다."),
                "ir_guide": n.get("ir_guide")
                or str(item.get("ir_guide", "") or f"{name} 관련 근거와 실행 계획을 포함하세요."),
                "score": int(item.get("score", 0) or 0),
                "feedback": str(item.get("feedback", "") or ""),
            }
        )
    # Fallback: notice 기준은 있는데 IR criteria raw가 비어있는 경우에도 UI 섹션을 유지한다.
    if notice_criteria is not None and not criteria_scores:
        for c in notice_criteria:
            name = str(c.get("criteria_name", "")).strip()
            if not name:
                continue
            criteria_scores.append(
                {
                    "criteria_name": name,
                    "pitchcoach_interpretation": str(c.get("pitchcoach_interpretation", "") or f"{name} 항목을 평가합니다."),
                    "ir_guide": str(c.get("ir_guide", "") or f"{name} 관련 근거와 실행 계획을 포함하세요."),
                    "score": 0,
                    "feedback": "공고 기준은 확인되었지만 해당 기준의 세부 점수 산출 근거가 부족합니다.",
                }
            )

    presentation_guide = {
        "emphasized_slides": [
            {
                "slide_number": int(x.get("slide_number", 1) or 1),
                "reason": str(x.get("reason", "") or ""),
            }
            for x in (guide_raw.get("emphasized_slides", []) if isinstance(guide_raw, dict) else [])
            if isinstance(x, dict)
        ],
        "guide": [str(x) for x in (guide_raw.get("guide", []) if isinstance(guide_raw, dict) else [])],
        "time_allocation": _normalize_time_allocation(guide_raw.get("time_allocation", []) if isinstance(guide_raw, dict) else []),
    }

    slides = []
    for s in slides_raw:
        feedback = s.get("feedback", {}) if isinstance(s.get("feedback"), dict) else {}
        slides.append(
            {
                "slide_number": int(s.get("slide_number", 0) or 0),
                "category": _to_display_category(str(s.get("category", "") or "")),
                "score": int(s.get("score", 0) or 0),
                "thumbnail_url": s.get("thumbnail_url"),
                "content_summary": str(s.get("content_summary") or s.get("content") or ""),
                "detailed_feedback": str(feedback.get("detailed_feedback", "") or ""),
                "strengths": _to_score_list(feedback.get("strengths")),
                "improvements": _to_score_list(feedback.get("improvements")),
            }
        )

    return IRDeckResultRow(
        deck_score=deck_score,
        criteria_scores=criteria_scores,
        presentation_guide=presentation_guide,
        slides=slides,
    )


def _run_ir_analysis_background(ir_deck_id: str, pdf_path: Path) -> None:
    with _LOCK:
        row = _IR_BY_ID.get(ir_deck_id)
        if row is None:
            return
        pitch_id = row.pitch_id

    try:
        out_dir = IR_ANALYSIS_DIR / ir_deck_id
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_ir_analysis(
            ir_pdf=pdf_path,
            output_dir=out_dir,
            strategy=None,
            use_chunking=True,
            pitch_type=None,
        )
        final_path = Path(str(result.get("final_path", "")))
        if not final_path.exists():
            raise RuntimeError("최종 분석 JSON이 생성되지 않았습니다.")
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        mapped = _map_ir_payload_to_result(payload, pitch_id=pitch_id)

        with _LOCK:
            row = _IR_BY_ID.get(ir_deck_id)
            if row is None:
                return
            _RESULT_BY_IR_ID[ir_deck_id] = mapped
            row.analysis_status = AnalysisStatus.COMPLETED
            row.pdf_upload_status = "COMPLETED"
            row.error_message = None
            row.analyzed_at = _now()
            row.updated_at = _now()
    except Exception as exc:  # pragma: no cover
        with _LOCK:
            row = _IR_BY_ID.get(ir_deck_id)
            if row is None:
                return
            row.analysis_status = AnalysisStatus.FAILED
            row.pdf_upload_status = "FAILED"
            row.error_message = str(exc)
            row.updated_at = _now()


@router.post(
    "/pitches/{pitch_id}/ir-decks/analyze",
    response_model=IRUploadResponse,
    status_code=202,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def upload_ir_and_analyze(
    background_tasks: BackgroundTasks,
    pitch_id: str = FPath(..., description="Pitch ID"),
    file: UploadFile = File(...),
):
    if not pitch_id.strip():
        _raise_error(404, "PITCH_NOT_FOUND", "존재하지 않는 피칭입니다")

    filename = file.filename or ""
    content_type = (file.content_type or "").lower()
    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        _raise_error(400, "INVALID_FILE", "PDF 파일만 업로드 가능합니다")

    payload = await file.read()
    if len(payload) > MAX_IR_FILE_SIZE:
        _raise_error(400, "FILE_TOO_LARGE", "파일 크기는 30MB 이하여야 합니다")

    IR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        existing_ids = _IR_IDS_BY_PITCH.get(pitch_id, [])
        for eid in existing_ids:
            if eid in _IR_BY_ID:
                _IR_BY_ID[eid].is_latest = False
                _IR_BY_ID[eid].updated_at = _now()

        ir_deck_id = f"ir-{uuid4()}"
        version = _next_ir_version(pitch_id)
        notice_id = _latest_notice_id_for_pitch(pitch_id)
        pdf_path = IR_UPLOAD_DIR / f"{ir_deck_id}.pdf"
        row = IRDeckRow(
            id=ir_deck_id,
            pitch_id=pitch_id,
            notice_id=notice_id,
            pdf_url=str(pdf_path.as_posix()),
            pdf_size_bytes=len(payload),
            pdf_upload_status="PROCESSING",
            version=version,
            is_latest=True,
            analysis_status=AnalysisStatus.IN_PROGRESS,
            created_at=_now(),
            updated_at=_now(),
        )
        _IR_BY_ID[ir_deck_id] = row
        _IR_IDS_BY_PITCH.setdefault(pitch_id, []).append(ir_deck_id)
        _RESULT_BY_IR_ID[ir_deck_id] = IRDeckResultRow()

    pdf_path.write_bytes(payload)
    background_tasks.add_task(_run_ir_analysis_background, ir_deck_id, pdf_path)

    return IRUploadResponse(
        ir_deck_id=ir_deck_id,
        pitch_id=pitch_id,
        analysis_status=AnalysisStatus.IN_PROGRESS,
        version=version,
        message="IR Deck 분석이 시작되었습니다.",
    )


@router.get(
    "/ir-decks/{deck_id}",
    response_model=IRDeckSummaryInProgressResponse | IRDeckSummaryCompletedResponse | IRDeckSummaryFailedResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_ir_summary(deck_id: str = FPath(..., description="IR Deck ID")):
    with _LOCK:
        row = _IR_BY_ID.get(deck_id)
        result = _RESULT_BY_IR_ID.get(deck_id)
    if row is None:
        _raise_error(404, "IR_DECK_NOT_FOUND")

    if row.analysis_status == AnalysisStatus.IN_PROGRESS:
        return IRDeckSummaryInProgressResponse(
            ir_deck_id=row.id,
            pitch_id=row.pitch_id,
            analysis_status=AnalysisStatus.IN_PROGRESS,
            version=row.version,
        )
    if row.analysis_status == AnalysisStatus.FAILED:
        return IRDeckSummaryFailedResponse(
            ir_deck_id=row.id,
            pitch_id=row.pitch_id,
            analysis_status=AnalysisStatus.FAILED,
            error_message=row.error_message or "IR Deck 분석 중 오류가 발생했습니다.",
            version=row.version,
        )

    result = result or IRDeckResultRow()
    return IRDeckSummaryCompletedResponse(
        ir_deck_id=row.id,
        pitch_id=row.pitch_id,
        analysis_status=AnalysisStatus.COMPLETED,
        version=row.version,
        deck_score=DeckScoreResponse(**(result.deck_score or {})),
        criteria_scores=[CriteriaScoreResponse(**x) for x in (result.criteria_scores or [])],
        presentation_guide=PresentationGuideResponse(**(result.presentation_guide or {})),
        analyzed_at=row.analyzed_at,
    )


@router.get(
    "/ir-decks/{deck_id}/slides",
    response_model=IRDeckSlidesInProgressResponse | IRDeckSlidesCompletedResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_ir_slides(deck_id: str = FPath(..., description="IR Deck ID")):
    with _LOCK:
        row = _IR_BY_ID.get(deck_id)
        result = _RESULT_BY_IR_ID.get(deck_id)
    if row is None:
        _raise_error(404, "IR_DECK_NOT_FOUND", "존재하지 않는 IR Deck입니다")

    if row.analysis_status == AnalysisStatus.IN_PROGRESS:
        return IRDeckSlidesInProgressResponse(
            ir_deck_id=row.id,
            analysis_status=AnalysisStatus.IN_PROGRESS,
        )
    if row.analysis_status == AnalysisStatus.FAILED:
        _raise_error(404, "IR_DECK_NOT_FOUND", "존재하지 않는 IR Deck입니다")

    result = result or IRDeckResultRow()
    slides = [IRDeckSlideItemResponse(**x) for x in (result.slides or []) if int(x.get("slide_number", 0) or 0) > 0]
    return IRDeckSlidesCompletedResponse(
        ir_deck_id=row.id,
        analysis_status=AnalysisStatus.COMPLETED,
        total_slides=len(slides),
        slides=slides,
    )
