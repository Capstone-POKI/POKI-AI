import json
from pathlib import Path
from typing import Any, Dict

from google.cloud import documentai_v1beta3 as documentai


class DocumentAIClient:
    def __init__(
        self,
        project_id: str,
        location: str,
        ocr_processor_id: str,
    ):
        self.project_id = project_id
        self.location = location
        self.ocr_processor_id = ocr_processor_id
        self.client = documentai.DocumentProcessorServiceClient()

    def process_ocr_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        name = self.client.processor_path(self.project_id, self.location, self.ocr_processor_id)
        raw_document = documentai.RawDocument(
            content=self._read_bytes(pdf_path),
            mime_type="application/pdf",
        )
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                compute_style_info=True,
                enable_native_pdf_parsing=True,
                enable_image_quality_scores=True,
                enable_symbol=True,
            )
        )
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document,
            process_options=process_options,
        )
        result = self.client.process_document(request=request)
        return json.loads(documentai.Document.to_json(result.document))

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        with open(path, "rb") as f:
            return f.read()
