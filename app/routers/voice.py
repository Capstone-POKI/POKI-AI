from fastapi import APIRouter

from src.domain.voice.pipeline import run_voice_analysis

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/analyze")
def analyze_voice():
    run_voice_analysis()
    return {"status": "ok"}
