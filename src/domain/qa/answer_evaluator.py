"""답변 평가 파이프라인"""

import json
from typing import Dict, Optional
from dataclasses import dataclass

import google.generativeai as genai

from src.domain.qa.prompts import (
    ANSWER_EVALUATION_SYSTEM_PROMPT,
    ANSWER_EVALUATION_USER_PROMPT,
)


@dataclass
class AnswerEvaluation:
    """답변 평가 결과"""
    score: int
    relevance: int  # 적합도
    clarity: int    # 명료도
    structure: int  # 구성도
    strengths: list
    improvements: list
    reasoning: str


def run_answer_evaluation(
    question_type: str,
    question_content: str,
    guidance: str,
    answer_transcript: str,
    model_name: str = "gemini-2.0-flash",
) -> Optional[AnswerEvaluation]:
    """
    Gemini를 사용하여 답변을 평가합니다.
    
    Args:
        question_type: 질문 유형 (NOTICE, PITCHBOOK, PRESENTER, EVALUATOR)
        question_content: 질문 내용
        guidance: 예상 답변 방향/평가 포인트
        answer_transcript: 답변 transcript (음성 변환 결과)
        model_name: 사용할 Gemini 모델
    
    Returns:
        평가 결과 또는 None (실패 시)
    """
    client = genai.Client()
    
    # 사용자 프롬프트 구성
    user_prompt = ANSWER_EVALUATION_USER_PROMPT.format(
        question_type=question_type,
        question_content=question_content,
        guidance=guidance,
        answer_transcript=answer_transcript,
    )
    
    # Gemini API 호출
    response = client.models.generate_content(
        model=model_name,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": ANSWER_EVALUATION_SYSTEM_PROMPT},
                    {"text": user_prompt},
                ]
            }
        ],
        generation_config={
            "temperature": 0.3,  # 평가는 일관성이 중요하므로 낮은 온도
            "top_p": 0.95,
            "top_k": 40,
        }
    )
    
    # 응답 파싱
    response_text = response.text
    
    # JSON 추출
    try:
        # JSON 코드블록 제거
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text
        
        data = json.loads(json_str)
        
        return AnswerEvaluation(
            score=int(data.get("score", 50)),
            relevance=int(data.get("relevance", 50)),
            clarity=int(data.get("clarity", 50)),
            structure=int(data.get("structure", 50)),
            strengths=data.get("strengths", []),
            improvements=data.get("improvements", []),
            reasoning=data.get("reasoning", ""),
        )
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ 답변 평가 JSON 파싱 실패: {e}")
        print(f"응답: {response_text}")
        return None


def calculate_weighted_score(evaluation: AnswerEvaluation) -> int:
    """
    평가 항목의 가중 평균으로 최종 점수를 계산합니다.
    
    가중치:
    - 적합도 (relevance): 40%
    - 명료도 (clarity): 35%
    - 구성도 (structure): 25%
    """
    weights = {
        "relevance": 0.40,
        "clarity": 0.35,
        "structure": 0.25,
    }
    
    weighted_sum = (
        evaluation.relevance * weights["relevance"] +
        evaluation.clarity * weights["clarity"] +
        evaluation.structure * weights["structure"]
    )
    
    return int(weighted_sum)
