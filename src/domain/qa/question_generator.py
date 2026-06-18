"""Q&A 질문 생성 파이프라인 (단일 Gemini 호출)"""

import json
from dataclasses import dataclass
from typing import List, Optional

import google.genai as genai


@dataclass
class QuestionItem:
    """생성된 질문"""

    question_type: str  # NOTICE, PITCHBOOK, EVALUATOR, PRESENTER (내부 분류)
    content: str
    guidance: Optional[str] = None
    rationale: Optional[str] = None
    category: str = "PROBLEM"  # PROBLEM|SOLUTION|MARKET_BIZ|PERFORMANCE|TEAM|FUNDING|JUDGE_TYPE


_SYSTEM_PROMPT = """당신은 IR 피칭 현장에 앉아 있는 VC 심사위원입니다.
방금 창업자의 발표를 들었고, 이제 질문할 차례입니다.

[절대 금지 — 이런 질문은 투자자가 절대 하지 않음]
- "공고문에서 ~", "공고의 ~", "공고 요구사항에 따르면 ~"
- "슬라이드 N에서 ~", "IR 덱 요약에서 ~", "제출된 자료에서 ~"
- "(XX점)", "XX점을 받았으며", "분석 결과에 따르면 ~"
- "~에 대해 설명해주세요", "~를 어떻게 생각하시나요" 같은 열린 질문
- "왜 IR 덱에 넣지 않으셨나요?", "왜 자료에 포함하지 않으셨나요?"

[반드시 지킬 것]
- 발표에서 창업자가 직접 언급한 수치·클레임에서 출발할 것
- 투자자가 돈을 잃을 수 있는 리스크 포인트를 파고들 것
- 한 질문에 하나의 핵심만 물을 것 (복합 질문 금지)
- 질문은 1~2문장으로 끝낼 것

[좋은 질문 예시]
- "베트남 시장 $60mn이 TAM인가요, SAM인가요? 초기 3년 목표 점유율은 몇 퍼센트입니까?"
- "경쟁사 대비 40% 비용 절감이라고 하셨는데, 그 수치가 파일럿에서 실제로 확인된 건가요, 아니면 이론적 추정인가요?"
- "500억 영업이익 목표의 핵심 가정은 무엇이고, 그 가정이 틀렸을 때 플랜 B는 무엇인가요?"
- "현재 파일럿 고객사가 있다면 몇 곳이고, 계약 형태는 무엇인가요?"
- "진입 장벽으로 특허를 언급하셨는데, 등록 완료된 건가요, 출원 중인가요?"
"""

_USER_PROMPT = """당신은 아래 공고에 지원한 스타트업의 IR 발표를 방금 들은 심사위원입니다.
공고의 평가 기준을 머릿속에 가지고 있으며, 발표 내용을 바탕으로 날카로운 질문을 합니다.

[심사위원이 알고 있는 공고 정보]
{notice_content}

[방금 들은 IR 발표 내용]
{irdecksummary}

{presentation_section}

위 내용을 바탕으로, 실제 IR 현장에서 심사위원이 창업자에게 할 법한 질문 5개를 만드세요.

규칙:
- 질문은 1~2문장. 길면 안 됨.
- "공고문에서~", "슬라이드 N에서~", "IR 덱에서~" 같은 문서 인용 표현 금지
- 발표에서 나온 구체적 수치(금액, 기간, 비율 등)를 활용해 파고드는 질문
- 창업자가 준비 없이는 대답하기 어려운 것
- 질문 타입 구성: NOTICE 1개, PITCHBOOK 2개, EVALUATOR 1개, PRESENTER 1개
- 각 질문에 아래 category 중 하나 선택:
  PROBLEM | SOLUTION | MARKET_BIZ | PERFORMANCE | TEAM | FUNDING | JUDGE_TYPE

JSON 형식으로만 반환:
{{
  "questions": [
    {{
      "question_type": "NOTICE|PITCHBOOK|EVALUATOR|PRESENTER",
      "category": "PROBLEM|SOLUTION|MARKET_BIZ|PERFORMANCE|TEAM|FUNDING|JUDGE_TYPE",
      "content": "질문 (1~2문장, 구체적 수치 활용, 문서 인용 표현 없이)",
      "guidance": "창업자가 반드시 다뤄야 할 핵심 포인트 1문장",
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
            category="PROBLEM",
            content=f"공고의 핵심 요구사항인 '{notice_excerpt}'을 충족했다는 정량 근거는 무엇입니까?",
            guidance="공고 평가 항목과 연결되는 수치, 검증 자료, 달성 시점을 제시하세요.",
            rationale="공고 적합성을 확인하기 위한 오프라인 기본 질문입니다.",
        ),
        QuestionItem(
            question_type="PITCHBOOK",
            category="SOLUTION",
            content=f"IR 덱의 주장인 '{deck_excerpt}'에서 가장 불확실한 가정과 검증 결과는 무엇입니까?",
            guidance="가정, 검증 방법, 표본, 결과와 다음 검증 계획을 구분해 답하세요.",
            rationale="핵심 사업 가정의 검증 수준을 확인합니다.",
        ),
        QuestionItem(
            question_type="PITCHBOOK",
            category="MARKET_BIZ",
            content="제시한 시장 규모 중 실제 3년 내 접근 가능한 시장과 목표 점유율의 산출 근거는 무엇입니까?",
            guidance="TAM, SAM, SOM을 구분하고 고객 수와 객단가로 역산하세요.",
            rationale="시장 수치의 실행 가능성을 확인합니다.",
        ),
        QuestionItem(
            question_type="EVALUATOR",
            category="JUDGE_TYPE",
            content="경쟁사가 같은 기능을 제공할 때 고객이 귀사를 선택할 이유를 검증된 지표로 설명해 주세요.",
            guidance="비용, 성능, 전환 비용, 재구매율 등 비교 가능한 지표를 제시하세요.",
            rationale="차별성이 주장에 그치지 않는지 확인합니다.",
        ),
        QuestionItem(
            question_type="PRESENTER",
            category="TEAM",
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

        _valid_categories = {"PROBLEM", "SOLUTION", "MARKET_BIZ", "PERFORMANCE", "TEAM", "FUNDING", "JUDGE_TYPE"}
        return [
            QuestionItem(
                question_type=q.get("question_type", "EVALUATOR"),
                category=q.get("category", "PROBLEM") if q.get("category") in _valid_categories else "PROBLEM",
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
