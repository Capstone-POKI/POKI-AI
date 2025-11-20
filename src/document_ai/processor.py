import json
from google.cloud import documentai
from src.utils.io_utils import save_json, read_bytes
from src.document_ai.config import PROJECT_ID, LOCATION, PROCESSORS

def process_document(file_path: str, processor_type: str, output_path: str):
    processor_id = PROCESSORS[processor_type]

    client = documentai.DocumentProcessorServiceClient()

    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{processor_id}"

    print(f"📄 [{processor_type}] {file_path} 분석 시작...")

    document = documentai.Document.from_json(
        json.dumps({
            "mimeType": "application/pdf",
            "content": read_bytes(file_path).decode("latin1")  # PDF는 base64가 아님 → bytes 처리 필요
        })
    )

    request = documentai.ProcessRequest(
        name=name,
        inline_document=document
    )

    result = client.process_document(request=request)
    doc = result.document

    output = {
        "processor": processor_type,
        "text": doc.text[:1000],
        "entities": [
            {"type": e.type_, "mention_text": e.mention_text, "confidence": e.confidence}
            for e in doc.entities
        ],
        "pages": [
            {
                "pageNumber": p.page_number,
                "tables": len(p.tables),
                "paragraphs": len(p.paragraphs),
                "blocks": len(p.blocks),
            }
            for p in doc.pages
        ],
    }

    save_json(output, output_path)
    print(f"[{processor_type}] 결과 저장 완료\n")
    return output
