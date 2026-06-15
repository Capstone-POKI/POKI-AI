import pytest
from fastapi import HTTPException

from app.upload import validate_audio_payload, validate_pdf_payload


def test_pdf_signature_is_required():
    validate_pdf_payload(b"%PDF-1.7\n")

    with pytest.raises(HTTPException):
        validate_pdf_payload(b"not-a-pdf")


@pytest.mark.parametrize(
    "payload",
    [
        b"\x1aE\xdf\xa3webm",
        b"OggSdata",
        b"RIFF\x00\x00\x00\x00WAVE",
        b"ID3data",
        b"\xff\xfbdata",
        b"\x00\x00\x00\x18ftypisom",
    ],
)
def test_supported_audio_signatures(payload):
    validate_audio_payload(payload)


def test_unknown_audio_signature_is_rejected():
    with pytest.raises(HTTPException):
        validate_audio_payload(b"not-audio")
