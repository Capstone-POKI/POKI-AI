from fastapi import FastAPI

from app.routers.ir import router as ir_router
from app.routers.notice import router as notice_router
from app.routers.voice import router as voice_router

app = FastAPI(title="POKI-AI Service", version="0.1.0")

app.include_router(notice_router)
app.include_router(ir_router)
app.include_router(voice_router)


@app.get("/health")
def health():
    return {"status": "ok"}
