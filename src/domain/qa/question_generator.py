"""Q&A 질문 생성 파이프라인 (단일 Gemini 호출)"""

import json
from dataclasses import dataclass
from typing import List, Optional

import google.genai as genai


@dataclass
class QuestionItem:
    """생성된 질문"""

    question_type: str  # NOTICE, PITCHBOOK, EVALUATOR, PRESENTER
    content: str
    guidance: Optional[str] = None
    rationale: Optional[str] = None


_SYSTEM_PROMPT = """당신은 스타트업 IR 피칭 심사위원이자 투자자입니다.
창업자의 공고문과 IR 덱을 분석하여 날카롭고 구체적인 질문 5개를 생성합니다.

절대 금지:
- "~에 대해 설명해주세요" 같은 열린 질문
- "[카테고리] 항목에서 핵심 근거를 어떻게 설명하시겠습니까?" 같은 템플릿 질문
- 모호하거나 뻔한 질문

반드시:
- IR 덱의 구체적 수치·클레임을 인용할 것
- 창업자가 미처 검증하지 못한 가정에 도전할 것
- 공고 요구사항과 IR 덱의 간극을 짚어낼 것

좋은 질문 예시:
- "베트남 시장 $60mn 규모가 TAM인지 SAM인지 불분명한데, 실제 접근 가능한 시장과 초기 3년 점유율 목표를 구체적으로 제시해주세요."
- "경쟁사 대비 40% 비용 절감을 주장하셨는데, 어느 원가 항목에서 발생하며 파일럿에서 실제 확인된 수치인지 말씀해주세요."
- "2028년 영업이익 500억 목표의 핵심 가정이 '글로벌 체외진단 시장 3% 점유'인데, 이 점유율 달성을 위한 구체적인 영업 전략과 현재까지의 레퍼런스 고객이 있나요?"
"""

_USER_PROMPT = """다음 자료를 분석하여 투자자/심사위원 관점의 날카로운 질문 5개를 생성하세요.

【공고문】
{notice_content}

【IR 덱 요약】
{irdecksummary}

{presentation_section}

질문 구성: NOTICE 1개, PITCHBOOK 2개, EVALUATOR 1개, PRESENTER 1개

JSON 형식으로만 반환:
{{
  "questions": [
    {{
      "question_type": "NOTICE|PITCHBOOK|EVALUATOR|PRESENTER",
      "content": "구체적이고 날카로운 질문 (IR 덱 수치 인용)",
      "guidance": "창업자가 반드시 다뤄야 할 핵심 포인트 1~2문장",
      "rationale": "이 질문이 중요한 이유 1문장"
    }}
  ]
}}"""


def _extract_json(text: str) -> dict:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text.strip())


def _fallback_questions(
    notice_content: str,
    irdecksummary: str,
    presentation_content: Optional[str],
) -> List[QuestionItem]:
    deck_excerpt = " ".join(irdecksummary.split())[:120] or "IR 덱의 핵심 주장"
    notice_excerpt = " ".join(notice_content.split())[:100] or "공고 요구사항"
    presentation_excerpt = (
        " ".join((presentation_content or "").split())[:100] or "발표에서 강조한 내용"
    )

    return [
        QuestionItem(
            question_type="NOTICE",
            content=f"공고의 핵심 요구사항인 '{notice_excerpt}'을 충족했다는 정량 근거는 무엇입니까?",
            guidance="공고 평가 항목과 연결되는 수치, 검증 자료, 달성 시점을 제시하세요.",
            rationale="공고 적합성을 확인하기 위한 오프라인 기본 질문입니다.",
        ),
        QuestionItem(
            question_type="PITCHBOOK",
            content=f"IR 덱의 주장인 '{deck_excerpt}'에서 가장 불확실한 가정과 검증 결과는 무엇입니까?",
            guidance="가정, 검증 방법, 표본, 결과와 다음 검증 계획을 구분해 답하세요.",
            rationale="핵심 사업 가정의 검증 수준을 확인합니다.",
        ),
        QuestionItem(
            question_type="PITCHBOOK",
            content="제시한 시장 규모 중 실제 3년 내 접근 가능한 시장과 목표 점유율의 산출 근거는 무엇입니까?",
            guidance="TAM, SAM, SOM을 구분하고 고객 수와 객단가로 역산하세요.",
            rationale="시장 수치의 실행 가능성을 확인합니다.",
        ),
        QuestionItem(
            question_type="EVALUATOR",
            content="경쟁사가 같은 기능을 제공할 때 고객이 귀사를 선택할 이유를 검증된 지표로 설명해 주세요.",
            guidance="비용, 성능, 전환 비용, 재구매율 등 비교 가능한 지표를 제시하세요.",
            rationale="차별성이 주장에 그치지 않는지 확인합니다.",
        ),
        QuestionItem(
            question_type="PRESENTER",
            content=f"발표에서 강조한 '{presentation_excerpt}'을 한 문장 결론과 두 개의 근거로 다시 설명해 주세요.",
            guidance="결론, 근거 수치, 사업적 의미 순서로 간결하게 답하세요.",
            rationale="발표 전달력과 답변 구조를 확인합니다.",
        ),
    ]


def run_question_generation(
    notice_content: str,
    irdecksummary: str,
    presentation_content: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
) -> List[QuestionItem]:
    """
    단일 Gemini 호출로 Q&A 질문 5개를 생성합니다.

    Args:
        notice_content: 공고문 전체 내용
        irdecksummary: IR Deck 요약
        presentation_content: 발표 내용 (선택)
        model_name: 사용할 Gemini 모델

    Returns:
        생성된 질문 리스트 (QuestionItem)
    """
    print(f"\n[QA 질문 생성] 모델={model_name}")

    presentation_section = ""
    if presentation_content:
        presentation_section = f"\n【발표 내용】\n{presentation_content}\n"

    user_prompt = _USER_PROMPT.format(
        notice_content=notice_content[:3000],
        irdecksummary=irdecksummary[:3000],
        presentation_section=presentation_section,
    )

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model=model_name,
            contents=_SYSTEM_PROMPT + "\n\n" + user_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
            ),
        )
        data = _extract_json(response.text)
        questions = data.get("questions", [])
        print(f"[QA 질문 생성] 완료 — {len(questions)}개 질문 생성")

        return [
            QuestionItem(
                question_type=q.get("question_type", "EVALUATOR"),
                content=q.get("content", ""),
                guidance=q.get("guidance"),
                rationale=q.get("rationale"),
            )
            for q in questions
            if q.get("content")
        ]

    except Exception as e:
        print(f"[QA 질문 생성] 실패: {e}")
        return _fallback_questions(
            notice_content,
            irdecksummary,
            presentation_content,
        )
