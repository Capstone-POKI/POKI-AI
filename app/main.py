from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.decks import router as ir_router
from app.api.notices import router as notice_router

app = FastAPI(title="POKI-AI Service", version="0.1.0")

app.include_router(health_router)
app.include_router(notice_router)
app.include_router(ir_router)
from app.api.voice import router as voice_router
app.include_router(voice_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        payload = {"error": exc.detail.get("error")}
        if exc.detail.get("message") is not None:
            payload["message"] = exc.detail.get("message")
        return JSONResponse(status_code=exc.status_code, content=payload)
    return JSONResponse(status_code=exc.status_code, content={"error": "HTTP_ERROR", "message": str(exc.detail)})
