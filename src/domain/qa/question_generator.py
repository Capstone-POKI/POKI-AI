"""질문 생성 파이프라인"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass

from google import genai

from src.domain.qa.prompts import (
    QUESTION_GENERATION_SYSTEM_PROMPT,
    QUESTION_GENERATION_USER_PROMPT,
)


@dataclass
class QuestionItem:
    """생성된 질문"""
    question_type: str  # NOTICE, PITCHBOOK, PRESENTER, EVALUATOR
    content: str
    guidance: Optional[str] = None
    rationale: Optional[str] = None


def run_question_generation(
    notice_content: str,
    irdecksummary: str,
    presentation_content: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> List[QuestionItem]:
    """
    Gemini를 사용하여 Q&A 질문을 생성합니다.
    
    Args:
        notice_content: 공고문 전체 내용
        irdecksummary: IR Deck 요약 (OCR 결과)
        presentation_content: 발표 내용 (선택)
        model_name: 사용할 Gemini 모델
    
    Returns:
        생성된 질문 리스트
    """
    client = genai.Client()
    
    # 발표 내용이 없을 경우 기본값 설정
    if not presentation_content:
        presentation_content = "[발표 내용 미제공]"
    
    # 사용자 프롬프트 구성
    user_prompt = QUESTION_GENERATION_USER_PROMPT.format(
        notice_content=notice_content,
        irdecksummary=irdecksummary,
        presentation_content=presentation_content,
    )
    
    # Gemini API 호출
    response = client.models.generate_content(
        model=model_name,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": QUESTION_GENERATION_SYSTEM_PROMPT},
                    {"text": user_prompt},
                ]
            }
        ],
        generation_config={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
        }
    )
    
    # 응답 파싱
    response_text = response.text
    
    # JSON 추출
    try:
        # JSON 코드블록 제거 (```json ... ```)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text
        
        data = json.loads(json_str)
        questions = []
        
        for q in data.get("questions", []):
            questions.append(
                QuestionItem(
                    question_type=q.get("question_type", "NOTICE"),
                    content=q.get("content", ""),
                    guidance=q.get("guidance"),
                    rationale=q.get("rationale"),
                )
            )
        
        return questions
    
    except json.JSONDecodeError as e:
        print(f"❌ 질문 생성 JSON 파싱 실패: {e}")
        print(f"응답: {response_text}")
        return []
