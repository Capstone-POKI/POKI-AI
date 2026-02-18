from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.routers.ir import router as ir_router
from app.routers.notice import router as notice_router

app = FastAPI(title="POKI-AI Service", version="0.1.0")

app.include_router(notice_router)
app.include_router(ir_router)
try:
    from app.routers.voice import router as voice_router

    app.include_router(voice_router)
except Exception:  # pragma: no cover
    # Voice dependencies (e.g. pydub) may be optional in IR/Notice-only environments.
    pass


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    # API contract: {"error":"...", "message":"..."} flat payload (not {"detail": {...}})
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        payload = {"error": exc.detail.get("error")}
        if exc.detail.get("message") is not None:
            payload["message"] = exc.detail.get("message")
        return JSONResponse(status_code=exc.status_code, content=payload)
    return JSONResponse(status_code=exc.status_code, content={"error": "HTTP_ERROR", "message": str(exc.detail)})


@app.get("/health")
def health():
    return {"status": "ok"}
