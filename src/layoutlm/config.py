from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

MODEL_NAME = "microsoft/layoutlmv3-base"
LABELS = ["O", "HEADER", "QUESTION", "ANSWER", "TABLE", "FOOTER"]

processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))