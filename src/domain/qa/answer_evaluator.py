"""답변 평가 파이프라인"""

import json
import re
from dataclasses import dataclass
from typing import Optional

import google.genai as genai

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


def _keyword_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", text)
        if len(token) >= 2
    }


def _fallback_evaluation(
    question_content: str,
    guidance: str,
    answer_transcript: str,
) -> AnswerEvaluation:
    answer = answer_transcript.strip()
    expected_keywords = _keyword_set(f"{question_content} {guidance}")
    answer_keywords = _keyword_set(answer)
    overlap = len(expected_keywords & answer_keywords) / max(len(expected_keywords), 1)

    relevance = min(85, 40 + round(overlap * 60))
    clarity = min(85, 45 + min(len(answer), 240) // 8)
    sentence_count = len([part for part in re.split(r"[.!?。]+", answer) if part.strip()])
    structure = min(85, 45 + min(sentence_count, 4) * 8)
    score = round(relevance * 0.4 + clarity * 0.35 + structure * 0.25)

    strengths = []
    improvements = []
    if overlap >= 0.2:
        strengths.append("질문과 관련된 핵심 용어를 포함했습니다.")
    else:
        improvements.append("질문의 핵심 용어와 직접 연결되는 답변이 필요합니다.")
    if len(answer) >= 80:
        strengths.append("답변에 일정 수준의 구체성이 있습니다.")
    else:
        improvements.append("수치나 검증 사례를 추가해 답변을 구체화하세요.")
    if sentence_count < 2:
        improvements.append("결론과 근거를 분리해 구조적으로 답변하세요.")

    return AnswerEvaluation(
        score=score,
        relevance=relevance,
        clarity=clarity,
        structure=structure,
        strengths=strengths,
        improvements=improvements,
        reasoning="외부 모델을 사용할 수 없어 길이, 구조, 키워드 중첩 기반 fallback 평가를 적용했습니다.",
    )


def run_answer_evaluation(
    question_type: str,
    question_content: str,
    guidance: str,
    answer_transcript: str,
    model_name: str = "gemini-2.5-flash",
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
    try:
        client = genai.Client()
        user_prompt = ANSWER_EVALUATION_USER_PROMPT.format(
            question_type=question_type,
            question_content=question_content,
            guidance=guidance,
            answer_transcript=answer_transcript,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=ANSWER_EVALUATION_SYSTEM_PROMPT + "\n\n" + user_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.95,
            ),
        )
        response_text = response.text

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

    except Exception as e:
        print(f"[QA 답변 평가] 모델 평가 실패: {e}")
        return _fallback_evaluation(
            question_content,
            guidance,
            answer_transcript,
        )


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
