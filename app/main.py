import hmac
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import env as _env  # noqa: F401
from app.api.health import router as health_router
from app.api.decks import router as ir_router
from app.api.notices import router as notice_router
from app.api.voice import router as voice_router
from app.api.qa import router as qa_router
from app.api.report import router as report_router

app = FastAPI(title="POKI-AI Service", version="0.1.0")


@app.middleware("http")
async def require_internal_api_key(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    expected_key = os.getenv("AI_INTERNAL_API_KEY", "").strip()
    if not expected_key:
        return JSONResponse(
            status_code=503,
            content={
                "error": "AI_AUTH_NOT_CONFIGURED",
                "message": "AI service authentication is not configured.",
            },
        )

    provided_key = request.headers.get("x-internal-api-key", "")
    if not hmac.compare_digest(provided_key, expected_key):
        return JSONResponse(
            status_code=401,
            content={
                "error": "UNAUTHORIZED",
                "message": "A valid internal API key is required.",
            },
        )

    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://pitchcoach.duckdns.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(notice_router)
app.include_router(ir_router)
app.include_router(voice_router)
app.include_router(qa_router)
app.include_router(report_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        payload = {"error": exc.detail.get("error")}
        if exc.detail.get("message") is not None:
            payload["message"] = exc.detail.get("message")
        return JSONResponse(status_code=exc.status_code, content=payload)
    return JSONResponse(status_code=exc.status_code, content={"error": "HTTP_ERROR", "message": str(exc.detail)})
