import os

# main.py에서 이 변수명을 찾고 있습니다.
LAYOUTLM_MODEL_PATH = "microsoft/layoutlmv3-base"

# 호환성을 위해 MODEL_NAME도 남겨둠
MODEL_NAME = LAYOUTLM_MODEL_PATH

# 학습 및 전처리 설정
MAX_LENGTH = 512
DOC_STRIDE = 128

# 학습 파라미터 (추후 학습 시 사용)
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 20

# 주의: LABELS 리스트는 여기서 고정하지 않고, 
# src/layoutlm/preprocess.py의 get_labels() 함수를 통해 동적으로 가져옵니다.