"""멀티에이전트 질문 생성 파이프라인"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass

import google.genai as genai

from src.domain.qa.prompts import (
    NOTICE_ANALYSIS_SYSTEM_PROMPT,
    NOTICE_ANALYSIS_USER_PROMPT,
    IR_DECK_ANALYSIS_SYSTEM_PROMPT,
    IR_DECK_ANALYSIS_USER_PROMPT,
    QUESTION_GENERATION_SYSTEM_PROMPT,
    QUESTION_GENERATION_USER_PROMPT,
    QUALITY_REVIEW_SYSTEM_PROMPT,
    QUALITY_REVIEW_USER_PROMPT,
    QUESTION_GENERATION_SINGLE_SYSTEM_PROMPT,
    QUESTION_GENERATION_SINGLE_USER_PROMPT,
)


@dataclass
class QuestionItem:
    """생성된 질문"""
    question_type: str  # NOTICE, PITCHBOOK, EVALUATOR, PRESENTER
    content: str
    guidance: Optional[str] = None
    rationale: Optional[str] = None


def _call_gemini(
    client: genai.Client,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.7,
) -> str:
    """Gemini API 단일 호출 헬퍼"""
    response = client.models.generate_content(
        model=model_name,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": system_prompt},
                    {"text": user_prompt},
                ],
            }
        ],
        config={
            "temperature": temperature,
            "top_p": 0.9,
        },
    )
    return response.text


def _extract_json(text: str) -> dict:
    """응답 텍스트에서 JSON을 추출합니다."""
    if "```json" in text:
        json_str = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        json_str = text.split("```")[1].split("```")[0].strip()
    else:
        json_str = text.strip()
    return json.loads(json_str)


# ==================== Agent 1: 공고 분석 ====================

def _agent_analyze_notice(
    client: genai.Client,
    notice_content: str,
    model_name: str,
) -> str:
    """
    Agent 1: 공고문에서 심사위원 핵심 평가 포인트를 추출합니다.

    Returns:
        분석 결과 JSON 문자열 (실패 시 빈 문자열)
    """
    print("  [Agent 1] 공고 분석 중...")
    try:
        user_prompt = NOTICE_ANALYSIS_USER_PROMPT.format(
            notice_content=notice_content,
        )
        raw = _call_gemini(
            client=client,
            system_prompt=NOTICE_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_name=model_name,
            temperature=0.3,
        )
        data = _extract_json(raw)
        points = data.get("evaluation_points", [])
        print(f"  [Agent 1] 완료 — 평가 포인트 {len(points)}개 추출")
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        print(f"  [Agent 1] 실패: {e}")
        return ""


# ==================== Agent 2: IR 덱 분석 ====================

def _agent_analyze_ir_deck(
    client: genai.Client,
    irdecksummary: str,
    model_name: str,
) -> str:
    """
    Agent 2: IR 덱 요약에서 약점·취약 클레임을 추출합니다.

    Returns:
        분석 결과 JSON 문자열 (실패 시 빈 문자열)
    """
    print("  [Agent 2] IR 덱 분석 중...")
    try:
        user_prompt = IR_DECK_ANALYSIS_USER_PROMPT.format(
            irdecksummary=irdecksummary,
        )
        raw = _call_gemini(
            client=client,
            system_prompt=IR_DECK_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_name=model_name,
            temperature=0.3,
        )
        data = _extract_json(raw)
        weak_points = data.get("weak_points", [])
        print(f"  [Agent 2] 완료 — 취약 포인트 {len(weak_points)}개 추출")
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        print(f"  [Agent 2] 실패: {e}")
        return ""


# ==================== Agent 3: 질문 생성 ====================

def _agent_generate_questions(
    client: genai.Client,
    notice_analysis: str,
    ir_analysis: str,
    irdecksummary: str,
    model_name: str,
) -> List[Dict]:
    """
    Agent 3: Agent 1·2 결과를 바탕으로 날카로운 질문 7개를 생성합니다.

    Returns:
        질문 딕셔너리 리스트 (실패 시 빈 리스트)
    """
    print("  [Agent 3] 질문 생성 중...")
    try:
        # 분석 결과가 없을 경우 요약 텍스트로 대체
        if not notice_analysis:
            notice_analysis = '{"evaluation_points": []}'
        if not ir_analysis:
            ir_analysis = '{"weak_points": []}'

        user_prompt = QUESTION_GENERATION_USER_PROMPT.format(
            notice_analysis=notice_analysis,
            ir_analysis=ir_analysis,
            irdecksummary=irdecksummary,
        )
        raw = _call_gemini(
            client=client,
            system_prompt=QUESTION_GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_name=model_name,
            temperature=0.8,
        )
        data = _extract_json(raw)
        questions = data.get("questions", [])
        print(f"  [Agent 3] 완료 — {len(questions)}개 질문 생성")
        return questions
    except Exception as e:
        print(f"  [Agent 3] 실패: {e}")
        return []


# ==================== Agent 4: 품질 검증 ====================

def _agent_quality_review(
    client: genai.Client,
    questions: List[Dict],
    model_name: str,
) -> List[Dict]:
    """
    Agent 4: 생성된 질문에서 뻔하거나 중복된 것을 걸러내고 최종 5개를 선별합니다.

    Returns:
        최종 선별된 질문 리스트 (실패 시 입력 그대로 최대 5개)
    """
    print("  [Agent 4] 품질 검증 중...")
    try:
        questions_json = json.dumps(questions, ensure_ascii=False, indent=2)
        user_prompt = QUALITY_REVIEW_USER_PROMPT.format(
            questions_json=questions_json,
        )
        raw = _call_gemini(
            client=client,
            system_prompt=QUALITY_REVIEW_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_name=model_name,
            temperature=0.2,
        )
        data = _extract_json(raw)
        selected = data.get("selected_questions", [])
        print(f"  [Agent 4] 완료 — 최종 {len(selected)}개 선별")
        return selected
    except Exception as e:
        print(f"  [Agent 4] 실패 (원본 질문 최대 5개 사용): {e}")
        return questions[:5]


# ==================== fallback: 단일 Gemini 호출 ====================

def _run_single_call_generation(
    client: genai.Client,
    notice_content: str,
    irdecksummary: str,
    model_name: str,
) -> List[Dict]:
    """
    멀티에이전트 파이프라인이 전부 실패했을 때 단일 호출로 질문을 생성합니다.
    """
    print("  [Fallback] 단일 호출 질문 생성 시도...")
    try:
        user_prompt = QUESTION_GENERATION_SINGLE_USER_PROMPT.format(
            notice_content=notice_content,
            irdecksummary=irdecksummary,
        )
        raw = _call_gemini(
            client=client,
            system_prompt=QUESTION_GENERATION_SINGLE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_name=model_name,
            temperature=0.7,
        )
        data = _extract_json(raw)
        questions = data.get("questions", [])
        print(f"  [Fallback] 완료 — {len(questions)}개 질문 생성")
        return questions
    except Exception as e:
        print(f"  [Fallback] 실패: {e}")
        return []


# ==================== 메인 파이프라인 ====================

def run_question_generation(
    notice_content: str,
    irdecksummary: str,
    presentation_content: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> List[QuestionItem]:
    """
    멀티에이전트 방식으로 Q&A 질문을 생성합니다.

    파이프라인:
      Agent 1 (공고 분석) → Agent 2 (IR 덱 분석)
        → Agent 3 (질문 생성, Agent 1·2 결과 활용)
          → Agent 4 (품질 검증, 최종 5개 선별)

    Agent가 실패하면 단계를 건너뛰고 다음 단계에서 가용 정보로 계속 진행합니다.
    전체가 실패하면 단일 호출 fallback을 사용합니다.

    Args:
        notice_content: 공고문 전체 내용
        irdecksummary: IR Deck 요약 (OCR 결과)
        presentation_content: 발표 내용 (현재 미사용, 확장 예정)
        model_name: 사용할 Gemini 모델

    Returns:
        생성된 질문 리스트 (QuestionItem)
    """
    client = genai.Client()
    print(f"\n[멀티에이전트 QA] 모델={model_name}")

    # Agent 1: 공고 분석
    notice_analysis = _agent_analyze_notice(
        client=client,
        notice_content=notice_content,
        model_name=model_name,
    )

    # Agent 2: IR 덱 분석
    ir_analysis = _agent_analyze_ir_deck(
        client=client,
        irdecksummary=irdecksummary,
        model_name=model_name,
    )

    # Agent 3: 질문 생성
    raw_questions = _agent_generate_questions(
        client=client,
        notice_analysis=notice_analysis,
        ir_analysis=ir_analysis,
        irdecksummary=irdecksummary,
        model_name=model_name,
    )

    # Agent 3 완전 실패 시 단일 호출 fallback
    if not raw_questions:
        raw_questions = _run_single_call_generation(
            client=client,
            notice_content=notice_content,
            irdecksummary=irdecksummary,
            model_name=model_name,
        )

    if not raw_questions:
        print("[멀티에이전트 QA] 질문 생성 실패 — 빈 리스트 반환")
        return []

    # Agent 4: 품질 검증 및 최종 선별
    final_questions = _agent_quality_review(
        client=client,
        questions=raw_questions,
        model_name=model_name,
    )

    # QuestionItem으로 변환
    result = []
    for q in final_questions:
        q_type = q.get("question_type", "EVALUATOR")
        # 허용된 타입으로 정규화
        if q_type not in {"NOTICE", "PITCHBOOK", "PRESENTER", "EVALUATOR"}:
            q_type = "EVALUATOR"
        result.append(
            QuestionItem(
                question_type=q_type,
                content=q.get("content", ""),
                guidance=q.get("guidance"),
                rationale=q.get("rationale"),
            )
        )

    print(f"[멀티에이전트 QA] 최종 {len(result)}개 질문 완료")
    return result
