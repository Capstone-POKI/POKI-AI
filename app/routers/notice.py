from pathlib import Path

from fastapi import APIRouter

from app.schemas.notice_schema import NoticeAnalyzeRequest
from src.domain.notice.pipeline import init_gemini, run_notice_analysis

router = APIRouter(prefix="/notice", tags=["notice"])


@router.post("/analyze")
def analyze_notice(payload: NoticeAnalyzeRequest):
    gemini = None if payload.no_gemini else init_gemini()
    return run_notice_analysis(
        notice_pdf=Path(payload.notice_pdf),
        output_dir=Path(payload.output_dir),
        gemini=gemini,
    )
