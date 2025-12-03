# src/layoutlm/inference.py
import torch
from typing import List, Dict, Any
from src.docs_analysis.layoutlm.config import load_model

def run_inference(inputs: Dict[str, torch.Tensor], label_list: List[str], tokenizer=None) -> List[List[Dict[str, Any]]]:
    """
    LayoutLM 모델로 추론 실행
    
    Args:
        inputs: 입력 텐서 딕셔너리 (input_ids, attention_mask, bbox 등)
        label_list: 라벨 이름 리스트 (예: ["O", "B-제목", "I-제목", ...])
        tokenizer: 토큰 ID를 텍스트로 변환하는 토크나이저 (선택사항이지만 권장)
    
    Returns:
        배치의 각 문서별 결과 리스트. 각 결과는:
        - token_id: 토큰 ID
        - token_text: 디코딩된 토큰 텍스트 (tokenizer 제공시)
        - label: 예측된 라벨
        - position: 시퀀스 내 위치
    """
    model = load_model()
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (batch_size, seq_len, num_labels)
        predictions = logits.argmax(dim=-1)  # (batch_size, seq_len)

    # attention mask로 유효한(패딩 아닌) 토큰 식별
    attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
    input_ids = inputs["input_ids"]
    
    batch_size, seq_len = input_ids.shape
    batch_results = []

    # 배치의 각 문서를 개별적으로 처리
    for batch_idx in range(batch_size):
        doc_results = []
        
        for seq_idx in range(seq_len):
            # 패딩 토큰은 건너뛰기
            if attention_mask[batch_idx, seq_idx] == 0:
                continue
            
            token_id = input_ids[batch_idx, seq_idx].item()
            pred_idx = predictions[batch_idx, seq_idx].item()
            
            # 특수 토큰 건너뛰기 (선택사항 - 용도에 따라 다름)
            # if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            #     continue
            
            # 예측 인덱스가 유효한지 확인
            if pred_idx >= len(label_list):
                continue
                
            label = label_list[pred_idx]
            
            # "O" (외부) 라벨 건너뛰기 (엔티티만 원할 경우)
            if label != "O":
                result = {
                    "token_id": token_id,
                    "label": label,
                    "position": seq_idx
                }
                
                # tokenizer가 있으면 디코딩된 텍스트 추가
                if tokenizer is not None:
                    result["token_text"] = tokenizer.decode([token_id])
                
                doc_results.append(result)
        
        batch_results.append(doc_results)
    
    return batch_results


def aggregate_entities(results: List[Dict[str, Any]], tokenizer=None) -> List[Dict[str, Any]]:
    """
    BIO 태깅을 사용해 서브워드 토큰들을 완전한 엔티티로 집계
    
    Args:
        results: run_inference의 토큰 예측 결과 리스트
        tokenizer: 텍스트 재구성을 위한 토크나이저
    
    Returns:
        결합된 텍스트를 가진 집계된 엔티티 리스트
    """
    entities = []
    current_entity = None
    
    for result in results:
        label = result["label"]
        
        if label.startswith("B-"):
            # 새 엔티티 시작
            if current_entity is not None:
                entities.append(current_entity)
            
            entity_type = label[2:]  # "B-" 접두사 제거
            current_entity = {
                "entity_type": entity_type,
                "tokens": [result["token_text"]] if "token_text" in result else [],
                "token_ids": [result["token_id"]],
                "start_position": result["position"]
            }
            
        elif label.startswith("I-") and current_entity is not None:
            # 현재 엔티티 계속
            entity_type = label[2:]
            if current_entity["entity_type"] == entity_type:
                if "token_text" in result:
                    current_entity["tokens"].append(result["token_text"])
                current_entity["token_ids"].append(result["token_id"])
    
    # 마지막 엔티티 잊지 않기
    if current_entity is not None:
        entities.append(current_entity)
    
    # 각 엔티티의 전체 텍스트 재구성
    for entity in entities:
        if tokenizer and entity["tokens"]:
            # 서브워드 토큰을 올바르게 처리하여 재구성
            entity["text"] = tokenizer.convert_tokens_to_string(entity["tokens"])
        else:
            entity["text"] = " ".join(entity["tokens"])
    
    return entities