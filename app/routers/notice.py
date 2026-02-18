from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Path as FPath, UploadFile

from app.schemas.notice_schema import (
    ErrorResponse,
    EvaluationCriteriaItem,
    NoticeAnalysisStatus,
    NoticeResultCompletedResponse,
    NoticeResultFailedResponse,
    NoticeResultInProgressResponse,
    NoticeUpdateRequest,
    NoticeUploadResponse,
)
from src.domain.notice.pipeline import init_gemini, run_notice_analysis

router = APIRouter(tags=["notice"])

MAX_NOTICE_FILE_SIZE = 10 * 1024 * 1024  # 10MB
NOTICE_UPLOAD_DIR = Path("data/output/notice_uploads")
NOTICE_ANALYSIS_DIR = Path("data/output/notice_analysis")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _raise_error(status_code: int, error: str, message: str | None = None) -> None:
    payload = {"error": error}
    if message:
        payload["message"] = message
    raise HTTPException(status_code=status_code, detail=payload)


@dataclass
class NoticeCriteriaRow:
    # ERD v4: NoticeEvaluationCriteria
    id: str
    notice_id: str
    criteria_name: str
    points: int
    importance: str | None
    display_order: int
    parent_id: str | None = None
    pitchcoach_interpretation: str | None = None
    ir_guide: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@dataclass
class NoticeRow:
    # ERD v4: Notice
    id: str
    pitch_id: str
    pdf_url: str | None = None
    pdf_size_bytes: int | None = None
    pdf_upload_status: str | None = None
    notice_name: str | None = None
    host_organization: str | None = None
    recruitment_type: str | None = None
    target_audience: str | None = None
    application_period: str | None = None
    summary: str | None = None
    core_requirements: str | None = None
    source_reference: str | None = None
    additional_criteria: str | None = None
    ir_deck_guide: str | None = None
    analysis_status: NoticeAnalysisStatus = NoticeAnalysisStatus.IN_PROGRESS
    error_message: str | None = None
    version: int = 1
    is_latest: bool = True
    pitch_type: str = "VC_DEMO"
    # pseudo pitch status storage (for transition simulation)
    pitch_status: str = "NOTICE_ANALYSIS"
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


_LOCK = threading.Lock()
_NOTICE_BY_ID: dict[str, NoticeRow] = {}
_NOTICE_IDS_BY_PITCH: dict[str, list[str]] = {}
_CRITERIA_BY_NOTICE_ID: dict[str, list[NoticeCriteriaRow]] = {}


def _infer_pitch_type(recruitment_type: str | None, fallback: str = "VC_DEMO") -> str:
    text = (recruitment_type or "").lower()
    if "정부" in text or "support" in text or "government" in text:
        return "GOV_SUPPORT"
    if "경진" in text or "contest" in text or "competition" in text or "comp" in text:
        return "STARTUP_CONTEST"
    return fallback


def _criterion_templates(pitch_type: str) -> list[tuple[str, int, str, str]]:
    if pitch_type == "GOV_SUPPORT":
        return [
            ("문제정의", 20, "사회·산업 문제의 구체성과 검증 근거를 평가합니다.", "문제 규모와 수혜대상, 검증 데이터(인터뷰/설문)를 포함하세요."),
            ("솔루션", 15, "해결책의 실현 가능성과 차별성을 평가합니다.", "핵심 기능과 구현 단계(MVP/실증), 차별점 3가지를 제시하세요."),
            ("시장/비즈니스", 15, "지속 가능 운영 구조와 시장 타당성을 평가합니다.", "시장 추정 근거와 운영 모델(지원 종료 후 자립 방안)을 명시하세요."),
            ("실적", 15, "현장 검증, 실증 결과, 성장 추세를 평가합니다.", "파일럿 결과와 핵심 지표를 시계열로 제시하세요."),
            ("팀", 15, "공공사업 수행 역량과 팀 구성 적합성을 평가합니다.", "역할 분담, 산업 경험, 외부 자문 체계를 보여주세요."),
            ("자금 계획", 20, "예산 배분과 로드맵 정합성을 평가합니다.", "항목별 예산, 분기별 마일스톤, 지원 종료 후 계획을 포함하세요."),
        ]
    if pitch_type == "STARTUP_CONTEST":
        return [
            ("문제정의", 25, "문제의 현실성과 임팩트를 평가합니다.", "타겟 고객의 불편과 문제 규모를 수치/사례로 제시하세요."),
            ("솔루션", 25, "아이디어의 창의성과 해결력, 데모 완성도를 평가합니다.", "문제-해결 매핑, 핵심 기능 3~5개, 사용 시나리오를 보여주세요."),
            ("시장/비즈니스", 20, "시장성 및 수익화 가능성을 평가합니다.", "시장 크기와 BM(가격/수익 구조)을 명확히 제시하세요."),
            ("실적", 10, "초기 검증 및 사용자 반응을 평가합니다.", "파일럿/설문/사용자 반응 등 검증 근거를 포함하세요."),
            ("팀", 10, "팀 실행력을 평가합니다.", "핵심 인력 역할과 프로젝트 실행 이력을 보여주세요."),
            ("자금 계획", 10, "자금 사용 계획과 성장 로드맵을 평가합니다.", "필요 자금, 사용 우선순위, 1~3년 목표를 제시하세요."),
        ]
    return [
        ("문제정의", 15, "문제의 구체성과 고객 검증을 평가합니다.", "고객 페르소나, 문제 규모, 기존 한계를 포함하세요."),
        ("솔루션", 20, "해결책의 명확성과 차별화를 평가합니다.", "차별점 3가지, 핵심 기능, MVP 단계를 제시하세요."),
        ("시장/비즈니스", 25, "시장 규모와 수익 모델 타당성을 평가합니다.", "TAM/SAM/SOM, 가격 정책, 수익 계산식을 포함하세요."),
        ("실적", 20, "트랙션과 성장 속도를 평가합니다.", "MAU/매출/전환율 등 핵심 지표를 제시하세요."),
        ("팀", 10, "팀의 실행 역량을 평가합니다.", "핵심 멤버 경력과 역할 분담을 명확히 하세요."),
        ("자금 계획", 10, "자금 사용 계획과 마일스톤 정합성을 평가합니다.", "투자/지원금 배분과 BEP/로드맵을 제시하세요."),
    ]


def _compute_importance(points: int | None) -> str | None:
    if points is None:
        return None
    if points >= 30:
        return "HIGH"
    if points >= 15:
        return "MEDIUM"
    return "LOW"


def _normalize_criteria_rows(criteria: list[dict], pitch_type: str, notice_id: str) -> list[NoticeCriteriaRow]:
    normalized: list[NoticeCriteriaRow] = []
    for row in criteria:
        name = str(row.get("criteria_name", "")).strip()
        points = int(row.get("points", 0))
        interp = str(row.get("pitchcoach_interpretation", "")).strip()
        guide = str(row.get("ir_guide", "")).strip()
        if not name:
            continue
        if not interp or not guide:
            for t_name, _p, t_interp, t_guide in _criterion_templates(pitch_type):
                if t_name == name:
                    interp = interp or t_interp
                    guide = guide or t_guide
                    break
        safe_points = max(0, min(100, points))
        normalized.append(
            NoticeCriteriaRow(
                id=f"nec-{uuid4()}",
                notice_id=notice_id,
                criteria_name=name,
                points=safe_points,
                importance=_compute_importance(safe_points),
                display_order=len(normalized) + 1,
                pitchcoach_interpretation=interp or f"{name} 항목을 중심으로 평가합니다.",
                ir_guide=guide or f"{name} 관련 근거와 실행 계획을 슬라이드에 포함하세요.",
            )
        )
    return normalized


def _default_criteria_rows(pitch_type: str, notice_id: str) -> list[NoticeCriteriaRow]:
    rows: list[NoticeCriteriaRow] = []
    for name, points, interp, guide in _criterion_templates(pitch_type):
        rows.append(
            NoticeCriteriaRow(
                id=f"nec-{uuid4()}",
                notice_id=notice_id,
                criteria_name=name,
                points=points,
                importance=_compute_importance(points),
                display_order=len(rows) + 1,
                pitchcoach_interpretation=interp,
                ir_guide=guide,
            )
        )
    return rows


def _criteria_rows_to_api_items(rows: list[NoticeCriteriaRow]) -> list[EvaluationCriteriaItem]:
    ordered = sorted(rows, key=lambda x: x.display_order)
    return [
        EvaluationCriteriaItem(
            criteria_name=row.criteria_name,
            points=row.points,
            pitchcoach_interpretation=row.pitchcoach_interpretation or f"{row.criteria_name} 항목을 평가합니다.",
            ir_guide=row.ir_guide or f"{row.criteria_name} 관련 근거를 제시하세요.",
        )
        for row in ordered
    ]


def _next_notice_version(pitch_id: str) -> int:
    ids = _NOTICE_IDS_BY_PITCH.get(pitch_id, [])
    if not ids:
        return 1
    return max((_NOTICE_BY_ID[i].version for i in ids if i in _NOTICE_BY_ID), default=0) + 1


def _run_notice_analysis_background(notice_id: str, pdf_path: Path) -> None:
    try:
        gemini = init_gemini()
        result = run_notice_analysis(
            notice_pdf=pdf_path,
            output_dir=NOTICE_ANALYSIS_DIR,
            gemini=gemini,
        )
        analysis = result.get("analysis", {}) if isinstance(result, dict) else {}
        with _LOCK:
            row = _NOTICE_BY_ID.get(notice_id)
            if row is None:
                return
            row.notice_name = (analysis.get("notice_name") or None) if isinstance(analysis, dict) else None
            row.host_organization = (analysis.get("host_organization") or None) if isinstance(analysis, dict) else None
            row.recruitment_type = (analysis.get("recruitment_type") or None) if isinstance(analysis, dict) else None
            row.target_audience = (analysis.get("target_audience") or None) if isinstance(analysis, dict) else None
            row.application_period = (analysis.get("application_period") or None) if isinstance(analysis, dict) else None
            row.summary = (analysis.get("summary") or None) if isinstance(analysis, dict) else None
            row.core_requirements = (analysis.get("core_requirements") or None) if isinstance(analysis, dict) else None
            row.source_reference = (analysis.get("source_reference") or None) if isinstance(analysis, dict) else None
            row.ir_deck_guide = (analysis.get("ir_deck_guide") or None) if isinstance(analysis, dict) else None
            criteria = analysis.get("evaluation_criteria") if isinstance(analysis, dict) else None
            if isinstance(criteria, list):
                pitch_type = _infer_pitch_type(row.recruitment_type, row.pitch_type)
                _CRITERIA_BY_NOTICE_ID[notice_id] = _normalize_criteria_rows(criteria, pitch_type, notice_id)
                row.pitch_type = pitch_type
            elif not _CRITERIA_BY_NOTICE_ID.get(notice_id):
                _CRITERIA_BY_NOTICE_ID[notice_id] = _default_criteria_rows(row.pitch_type, notice_id)
            row.analysis_status = NoticeAnalysisStatus.COMPLETED
            row.pitch_status = "IRDECK_ANALYSIS"
            row.pdf_upload_status = "COMPLETED"
            row.error_message = None
            row.updated_at = _now()
    except Exception as exc:  # pragma: no cover - defensive path
        with _LOCK:
            row = _NOTICE_BY_ID.get(notice_id)
            if row is None:
                return
            row.analysis_status = NoticeAnalysisStatus.FAILED
            row.pdf_upload_status = "FAILED"
            row.error_message = str(exc)
            row.updated_at = _now()


@router.post(
    "/pitches/{pitch_id}/notice",
    response_model=NoticeUploadResponse,
    status_code=202,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def upload_notice_and_analyze(
    background_tasks: BackgroundTasks,
    pitch_id: str = FPath(..., description="Pitch ID"),
    file: UploadFile = File(...),
):
    if not pitch_id.strip():
        _raise_error(404, "PITCH_NOT_FOUND")

    filename = file.filename or ""
    content_type = (file.content_type or "").lower()
    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        _raise_error(400, "INVALID_FILE", "PDF 파일만 업로드 가능합니다")

    payload = await file.read()
    if len(payload) > MAX_NOTICE_FILE_SIZE:
        _raise_error(400, "FILE_TOO_LARGE", "파일 크기는 10MB 이하여야 합니다")

    with _LOCK:
        # overwrite semantics for UI + DB-friendly version history
        existing_ids = _NOTICE_IDS_BY_PITCH.get(pitch_id, [])
        for eid in existing_ids:
            if eid in _NOTICE_BY_ID:
                _NOTICE_BY_ID[eid].is_latest = False
                _NOTICE_BY_ID[eid].updated_at = _now()

        notice_id = f"notice-{uuid4()}"
        now = _now()
        row = NoticeRow(
            id=notice_id,
            pitch_id=pitch_id,
            pdf_url=str((NOTICE_UPLOAD_DIR / f"{notice_id}.pdf").as_posix()),
            pdf_size_bytes=len(payload),
            pdf_upload_status="PROCESSING",
            analysis_status=NoticeAnalysisStatus.IN_PROGRESS,
            version=_next_notice_version(pitch_id),
            is_latest=True,
            created_at=now,
            updated_at=now,
        )
        _NOTICE_BY_ID[notice_id] = row
        _NOTICE_IDS_BY_PITCH.setdefault(pitch_id, []).append(notice_id)
        _CRITERIA_BY_NOTICE_ID[notice_id] = _default_criteria_rows(row.pitch_type, notice_id)

    NOTICE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = NOTICE_UPLOAD_DIR / f"{notice_id}.pdf"
    pdf_path.write_bytes(payload)

    background_tasks.add_task(_run_notice_analysis_background, notice_id, pdf_path)

    return NoticeUploadResponse(
        notice_id=notice_id,
        pitch_id=pitch_id,
        analysis_status=NoticeAnalysisStatus.IN_PROGRESS,
        message="공고문 분석이 시작되었습니다.",
    )


@router.get(
    "/notices/{notice_id}",
    response_model=NoticeResultInProgressResponse | NoticeResultCompletedResponse | NoticeResultFailedResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def get_notice_result(notice_id: str = FPath(..., description="Notice ID")):
    with _LOCK:
        row = _NOTICE_BY_ID.get(notice_id)
    if row is None:
        _raise_error(404, "NOTICE_NOT_FOUND")

    if row.analysis_status == NoticeAnalysisStatus.IN_PROGRESS:
        return NoticeResultInProgressResponse(
            notice_id=row.id,
            pitch_id=row.pitch_id,
            analysis_status=NoticeAnalysisStatus.IN_PROGRESS,
            updated_at=row.updated_at,
        )

    if row.analysis_status == NoticeAnalysisStatus.FAILED:
        return NoticeResultFailedResponse(
            notice_id=row.id,
            pitch_id=row.pitch_id,
            analysis_status=NoticeAnalysisStatus.FAILED,
            error_message=row.error_message or "분석 중 오류가 발생했습니다.",
            updated_at=row.updated_at,
        )

    criteria_rows = _CRITERIA_BY_NOTICE_ID.get(notice_id, [])
    if not criteria_rows:
        criteria_rows = _default_criteria_rows(row.pitch_type, notice_id)
        with _LOCK:
            _CRITERIA_BY_NOTICE_ID[notice_id] = criteria_rows
    criteria = _criteria_rows_to_api_items(criteria_rows)
    return NoticeResultCompletedResponse(
        notice_id=row.id,
        pitch_id=row.pitch_id,
        analysis_status=NoticeAnalysisStatus.COMPLETED,
        notice_name=row.notice_name,
        host_organization=row.host_organization,
        recruitment_type=row.recruitment_type,
        target_audience=row.target_audience,
        application_period=row.application_period,
        evaluation_criteria=criteria,
        additional_criteria=row.additional_criteria,
        ir_deck_guide=row.ir_deck_guide
        or f"{row.pitch_type} 기반 IR Deck 가이드 템플릿...",
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.patch(
    "/notices/{notice_id}",
    response_model=NoticeResultCompletedResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def patch_notice(notice_id: str, payload: NoticeUpdateRequest):
    with _LOCK:
        row = _NOTICE_BY_ID.get(notice_id)
    if row is None:
        _raise_error(404, "NOTICE_NOT_FOUND")

    if payload.evaluation_criteria is not None:
        if len(payload.evaluation_criteria) == 0:
            _raise_error(400, "INVALID_REQUEST", "심사 기준은 1개 이상 필수")
        points_sum = sum(item.points for item in payload.evaluation_criteria)
        if points_sum != 100:
            _raise_error(400, "POINTS_SUM_INVALID", "배점 합계는 100이어야 합니다")

    with _LOCK:
        row = _NOTICE_BY_ID[notice_id]
        if payload.notice_name is not None:
            row.notice_name = payload.notice_name
        if payload.host_organization is not None:
            row.host_organization = payload.host_organization
        if payload.recruitment_type is not None:
            row.recruitment_type = payload.recruitment_type
            row.pitch_type = _infer_pitch_type(payload.recruitment_type, row.pitch_type)
        if payload.target_audience is not None:
            row.target_audience = payload.target_audience
        if payload.application_period is not None:
            row.application_period = payload.application_period
        if payload.additional_criteria is not None:
            row.additional_criteria = payload.additional_criteria

        if payload.evaluation_criteria is not None:
            criteria_payload = [
                {"criteria_name": item.criteria_name, "points": item.points}
                for item in payload.evaluation_criteria
            ]
            _CRITERIA_BY_NOTICE_ID[notice_id] = _normalize_criteria_rows(
                criteria_payload,
                row.pitch_type,
                notice_id,
            )
            row.ir_deck_guide = "수정된 기준 반영 IR Deck 가이드..."

        row.analysis_status = NoticeAnalysisStatus.COMPLETED
        row.updated_at = _now()

        criteria_rows = _CRITERIA_BY_NOTICE_ID.get(notice_id, []) or _default_criteria_rows(row.pitch_type, notice_id)
        criteria = _criteria_rows_to_api_items(criteria_rows)
        return NoticeResultCompletedResponse(
            notice_id=row.id,
            pitch_id=row.pitch_id,
            analysis_status=NoticeAnalysisStatus.COMPLETED,
            notice_name=row.notice_name,
            host_organization=row.host_organization,
            recruitment_type=row.recruitment_type,
            target_audience=row.target_audience,
            application_period=row.application_period,
            evaluation_criteria=criteria,
            additional_criteria=row.additional_criteria,
            ir_deck_guide=row.ir_deck_guide
            or f"{row.pitch_type} 기반 IR Deck 가이드 템플릿...",
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
