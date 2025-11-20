from src.document_ai.processor import process_document

# 입력/출력 경로
notice_input = "data/input/sample_notice.pdf"
irdeck_input = "data/input/sample_irdeck.pdf"
notice_output = "data/output/docai_notice.json"
irdeck_output = "data/output/docai_irdeck.json"

# OCR + LAYOUT + FORM 파이프라인 실행
for processor in ["OCR", "LAYOUT", "FORM"]:
    process_document(notice_input, processor, f"data/output/notice_{processor.lower()}.json")
    process_document(irdeck_input, processor, f"data/output/irdeck_{processor.lower()}.json")

from src.layoutlm.config import processor, model, LABELS
from src.layoutlm.preprocess import load_docai_json, prepare_layoutlm_input
from src.layoutlm.inference import run_inference

if __name__ == "__main__":
    json_path = "data/output/docai_notice.json"
    image_path = "data/input/sample_notice.png"  # PDF 1페이지 캡처본

    json_data = load_docai_json(json_path)
    encoding = prepare_layoutlm_input(json_data, image_path, processor)
    results = run_inference(model, encoding, LABELS)

    print("📊 LayoutLMv3 분석 결과:")
    for r in results[:30]:
        print(r)