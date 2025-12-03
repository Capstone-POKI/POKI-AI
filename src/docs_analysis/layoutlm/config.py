# src/layoutlm/config.py

import os

LAYOUTLM_MODEL_PATH = "microsoft/layoutlmv3-base"

# 전역 변수
_MODEL = None
_PROCESSOR = None

# ✅ inference.py가 찾고 있는 그 함수!
def load_model():
    global _MODEL
    if _MODEL is None:
        print("⏳ LayoutLM 모델 로딩 중...")
        from transformers import LayoutLMv3ForTokenClassification
        _MODEL = LayoutLMv3ForTokenClassification.from_pretrained(LAYOUTLM_MODEL_PATH)
        _MODEL.eval()
    return _MODEL

def load_processor():
    global _PROCESSOR
    if _PROCESSOR is None:
        from transformers import LayoutLMv3Processor
        _PROCESSOR = LayoutLMv3Processor.from_pretrained(LAYOUTLM_MODEL_PATH)
    return _PROCESSOR