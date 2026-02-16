import json
import os
from pathlib import Path
from typing import Any, Dict

from src.infrastructure.document_ai.client import DocumentAIClient


def run_notice_document_ai(notice_pdf: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{notice_pdf.stem}_docai.json"

    if output_path.exists():
        return _read_json(output_path)

    project_id = os.getenv("PROJECT_ID", "pitchcoachai")
    location = os.getenv("LOCATION", "us")
    processor_id = os.getenv("OCR_PROCESSOR_ID", "e41bb5d1cae96184")

    client = DocumentAIClient(
        project_id=project_id,
        location=location,
        ocr_processor_id=processor_id,
    )
    doc_dict = client.process_ocr_pdf(notice_pdf)
    _write_json(output_path, doc_dict)
    return doc_dict


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
