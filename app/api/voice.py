from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Path as FPath, UploadFile

from src.domain.voice.whisper_adapter import (
    PITCH_TYPE_TO_SCENARIO,
    SCENARIO_CONFIG,
    analyze,
)

router = APIRouter(prefix="/api", tags=["voice"])

MAX_VOICE_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".webm", ".mp3", ".m4a", ".wav", ".ogg", ".mp4"}
VOICE_UPLOAD_DIR = Path("data/output/voice_uploads")

_LOCK = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _raise_error(status_code: int, error: str, message: str | None = None) -> None:
    payload: dict = {"error": error}
    if message:
        payload["message"] = message
    raise HTTPException(status_code=status_code, detail=payload)


@dataclass
class VoiceRow:
    id: str
    pitch_id: str
    version: int = 1
    status: str = "IN_PROGRESS"
    result: dict = field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime = field(default_factory=_now)
    completed_at: datetime | None = None


_VOICE_BY_ID: dict[str, VoiceRow] = {}
_VOICE_IDS_BY_PITCH: dict[str, list[str]] = {}


def _get_latest_deck_json(pitch_id: str) -> dict:
    try:
        from app.api import decks as deck_module
        with deck_module._LOCK:
            ids = deck_module._DECK_IDS_BY_PITCH.get(pitch_id, [])
            latest_id = None
            latest_ver = -1
            for did in ids:
                row = deck_module._DECK_BY_ID.get(did)
                if row and row.is_latest and row.version >= latest_ver:
                    latest_id = did
                    latest_ver = row.version
            if latest_id:
                r = deck_module._RESULT_BY_DECK_ID.get(latest_id)
                if r and getattr(r, "raw_payload", None):
                    return r.raw_payload
    except Exception:
        pass
    return {}


def _parse_json_object(value: str | None, error_code: str, message: str) -> dict | None:
    if not value:
        return None
    try:
        loaded = json.loads(value)
    except Exception:
        _raise_error(400, error_code, message)
    if not isinstance(loaded, dict):
        _raise_error(400, error_code, message)
    return loaded


def _run_voice_background(
    voice_id: str,
    audio_path: Path,
    deck_json: dict,
    scenario: str,
    slide_timestamps: list[dict] | None,
) -> None:
    try:
        result = analyze(audio_path, deck_json, scenario, slide_timestamps)
        with _LOCK:
            row = _VOICE_BY_ID.get(voice_id)
            if row:
                row.status = "COMPLETED"
                row.result = result
                row.completed_at = _now()
    except Exception as exc:
        with _LOCK:
            row = _VOICE_BY_ID.get(voice_id)
            if row:
                row.status = "FAILED"
                row.error_message = str(exc)
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


def _next_voice_version(pitch_id: str) -> int:
    ids = _VOICE_IDS_BY_PITCH.get(pitch_id, [])
    if not ids:
        return 1
    return max((_VOICE_BY_ID[i].version for i in ids if i in _VOICE_BY_ID), default=0) + 1


@router.post("/pitches/{pitch_id}/voice/analyze", status_code=202)
async def upload_voice_and_analyze(
    background_tasks: BackgroundTasks,
    pitch_id: str = FPath(..., description="Pitch ID"),
    file: UploadFile = File(...),
    pitch_type: str = Form(default="VC_DEMO"),
    slide_timestamps: str | None = Form(default=None),
    context_json: str | None = Form(default=None),
    options_json: str | None = Form(default=None),
):
    if not pitch_id.strip():
        _raise_error(404, "PITCH_NOT_FOUND")

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        _raise_error(400, "INVALID_FILE", "지원하지 않는 음성 파일 형식입니다")

    payload = await file.read()
    if len(payload) > MAX_VOICE_FILE_SIZE:
        _raise_error(400, "FILE_TOO_LARGE", "파일 크기는 50MB 이하여야 합니다")

    explicit_context = _parse_json_object(
        context_json,
        "INVALID_CONTEXT",
        "음성 분석 context가 유효하지 않습니다",
    )
    options = _parse_json_object(
        options_json,
        "INVALID_OPTIONS",
        "음성 분석 options가 유효하지 않습니다",
    ) or {}

    # Form slide_timestamps 우선, 없으면 options fallback
    timestamps: list[dict] | None = None
    if not slide_timestamps:
        option_timestamps = options.get("slide_timestamps")
        if isinstance(option_timestamps, list):
            slide_timestamps = json.dumps(option_timestamps, ensure_ascii=False)

    if slide_timestamps:
        try:
            loaded = json.loads(slide_timestamps)
            if not isinstance(loaded, list):
                _raise_error(400, "INVALID_TIMESTAMPS", "슬라이드 타임스탬프가 유효하지 않습니다")
            for ts in loaded:
                if not all(k in ts for k in ("slide_number", "start_timestamp", "end_timestamp")):
                    _raise_error(400, "INVALID_TIMESTAMPS", "슬라이드 타임스탬프가 유효하지 않습니다")
            timestamps = loaded
        except HTTPException:
            raise
        except Exception:
            _raise_error(400, "INVALID_TIMESTAMPS", "슬라이드 타임스탬프가 유효하지 않습니다")

    scenario = str(options.get("scenario") or PITCH_TYPE_TO_SCENARIO.get(pitch_type, "VC 데모데이"))
    voice_context = explicit_context or {}
    if not voice_context.get("ir_deck"):
        voice_context = {"ir_deck": _get_latest_deck_json(pitch_id)}

    VOICE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        version = _next_voice_version(pitch_id)
        voice_id = f"voice-{uuid4()}"
        audio_path = VOICE_UPLOAD_DIR / f"{voice_id}{suffix}"
        row = VoiceRow(id=voice_id, pitch_id=pitch_id, version=version)
        _VOICE_BY_ID[voice_id] = row
        _VOICE_IDS_BY_PITCH.setdefault(pitch_id, []).append(voice_id)

    audio_path.write_bytes(payload)

    background_tasks.add_task(
        _run_voice_background,
        voice_id,
        audio_path,
        voice_context,
        scenario,
        timestamps,
    )

    return {
        "voice_id": voice_id,
        "pitch_id": pitch_id,
        "analysis_status": "IN_PROGRESS",
        "version": version,
        "message": "음성 분석이 시작되었습니다.",
    }


@router.get("/voice/{voice_id}")
def get_voice_result(voice_id: str = FPath(..., description="Voice ID")):
    with _LOCK:
        row = _VOICE_BY_ID.get(voice_id)
    if row is None:
        _raise_error(404, "VOICE_ANALYSIS_NOT_FOUND")

    if row.status == "IN_PROGRESS":
        return {
            "voice_id": row.id,
            "pitch_id": row.pitch_id,
            "analysis_status": "IN_PROGRESS",
            "version": row.version,
        }

    if row.status == "FAILED":
        return {
            "voice_id": row.id,
            "pitch_id": row.pitch_id,
            "analysis_status": "FAILED",
            "error_message": row.error_message,
            "version": row.version,
        }

    result = row.result
    overall = result.get("overall", {})
    wpm = float(result.get("wpm", 0))
    duration_sec = float(result.get("duration_seconds", 0))

    total_sec = int(duration_sec)
    if total_sec >= 60:
        m, s = divmod(total_sec, 60)
        duration_display = f"{m}:{s:02d}"
    else:
        duration_display = f"0:{total_sec:02d}"

    delivery = overall.get("음성_전달력_분석", {})
    situation_score = overall.get("상황_적합성_점수", {})
    sub_criteria = situation_score.get("세부_기준", {})

    scenario_name = overall.get("발표_상황", "VC 데모데이")
    cfg = SCENARIO_CONFIG.get(scenario_name, SCENARIO_CONFIG["VC 데모데이"])
    low, high = cfg["target_wpm"]
    if wpm < low:
        wpm_label = "느림"
    elif wpm > high:
        wpm_label = "빠름"
    else:
        wpm_label = "적정"

    detail_scores = [
        {"category": "문제 정의", "score": sub_criteria.get("문제_정의", 0)},
        {"category": "솔루션 명확성", "score": sub_criteria.get("솔루션_명확성", 0)},
        {"category": "시장성", "score": sub_criteria.get("시장성", 0)},
        {"category": "사업성 BM", "score": sub_criteria.get("사업성_BM", 0)},
        {"category": "경쟁력 차별성", "score": sub_criteria.get("경쟁력_차별성", 0)},
        {"category": "전달력", "score": sub_criteria.get("전달력", 0)},
        {"category": "톤 일관성", "score": sub_criteria.get("톤_일관성", 0)},
        {"category": "시간 적합성", "score": overall.get("시간_적합성_점수", 0)},
    ]

    delivery_items = [
        {"category": "억양 강조 안정성", "feedback": delivery.get("억양_강조_안정성", "")},
        {"category": "감정 톤", "feedback": delivery.get("감정_톤", "")},
        {"category": "문장 명료성", "feedback": delivery.get("문장_명료성", "")},
        {"category": "불필요한 말버릇", "feedback": delivery.get("불필요한_말버릇", "")},
    ]

    return {
        "voice_id": row.id,
        "pitch_id": row.pitch_id,
        "analysis_status": "COMPLETED",
        "version": row.version,
        "audio_duration_display": duration_display,
        "duration_seconds": total_sec,
        "wpm": int(wpm),
        "total_score": situation_score.get("총점", 0),
        "structure_summary": overall.get("1분_요약", ""),
        "overall_strengths": delivery.get("강점", []),
        "overall_improvements": delivery.get("개선점", []),
        "detail_scores": detail_scores,
        "delivery_analysis": {
            "speaking_speed": {
                "metric_value": f"{int(wpm)} WPM",
                "metric_label": wpm_label,
            },
            "items": delivery_items,
        },
        "analyzed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


@router.get("/voice/{voice_id}/slides")
def get_voice_slides(voice_id: str = FPath(..., description="Voice ID")):
    with _LOCK:
        row = _VOICE_BY_ID.get(voice_id)
    if row is None:
        _raise_error(404, "VOICE_ANALYSIS_NOT_FOUND")

    if row.status == "IN_PROGRESS":
        return {"voice_id": row.id, "analysis_status": "IN_PROGRESS"}

    if row.status == "FAILED":
        _raise_error(404, "VOICE_ANALYSIS_NOT_FOUND")

    slides = row.result.get("slides", [])
    return {
        "voice_id": row.id,
        "analysis_status": "COMPLETED",
        "total_slides": len(slides),
        "slides": [
            {
                "slide_number": s["slide_number"],
                "category": s["category"],
                "score": s["score"],
                "thumbnail_url": None,
                "start_timestamp": s.get("start_timestamp", 0.0),
                "end_timestamp": s.get("end_timestamp", 0.0),
                "duration_display": s.get("duration_display", ""),
                "content_summary": s["content_summary"],
                "detailed_feedback": s["detailed_feedback"],
                "strengths": s["strengths"],
                "improvements": s["improvements"],
            }
            for s in slides
        ],
    }
