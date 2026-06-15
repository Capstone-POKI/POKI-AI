from __future__ import annotations

from fastapi import HTTPException, UploadFile

READ_CHUNK_SIZE = 1024 * 1024


def _raise_upload_error(error: str, message: str) -> None:
    raise HTTPException(
        status_code=400,
        detail={"error": error, "message": message},
    )


async def read_upload_limited(
    file: UploadFile,
    max_bytes: int,
    *,
    too_large_message: str,
) -> bytes:
    chunks: list[bytes] = []
    total = 0

    while chunk := await file.read(READ_CHUNK_SIZE):
        total += len(chunk)
        if total > max_bytes:
            _raise_upload_error("FILE_TOO_LARGE", too_large_message)
        chunks.append(chunk)

    if total == 0:
        _raise_upload_error("INVALID_FILE", "빈 파일은 업로드할 수 없습니다")
    return b"".join(chunks)


def validate_pdf_payload(payload: bytes) -> None:
    if not payload.startswith(b"%PDF-"):
        _raise_upload_error("INVALID_FILE", "유효한 PDF 파일만 업로드 가능합니다")


def validate_audio_payload(payload: bytes) -> None:
    is_webm = payload.startswith(b"\x1aE\xdf\xa3")
    is_ogg = payload.startswith(b"OggS")
    is_wav = payload.startswith(b"RIFF") and payload[8:12] == b"WAVE"
    is_id3 = payload.startswith(b"ID3")
    is_mp3 = len(payload) >= 2 and payload[0] == 0xFF and payload[1] & 0xE0 == 0xE0
    is_mp4 = len(payload) >= 12 and payload[4:8] == b"ftyp"

    if not any((is_webm, is_ogg, is_wav, is_id3, is_mp3, is_mp4)):
        _raise_upload_error(
            "INVALID_FILE",
            "유효한 음성 파일만 업로드 가능합니다",
        )
