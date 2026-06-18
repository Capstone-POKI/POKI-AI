import json
from pathlib import Path
from typing import Any, Dict

from src.infrastructure.document_ai.pipeline import run_document_ai_pipeline


def run_notice_document_ai(notice_pdf: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{notice_pdf.stem}_docai.json"

    if output_path.exists():
        return _read_json(output_path)

    return run_document_ai_pipeline(
        pdf_path=notice_pdf,
        output_dir=output_dir,
        use_chunking=True,
        pages_per_chunk=15,
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
