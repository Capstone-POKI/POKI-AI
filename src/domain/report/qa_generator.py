from typing import List, Dict, Any


def _fallback_qa(context: Dict[str, Any]) -> List[Dict[str, str]]:
    questions: List[Dict[str, str]] = []
    missing_sections = set(context.get("missing_sections", []))
    raw_axes = context.get("evaluation_axes", {}) or {}
    if isinstance(raw_axes, list):
        evaluation_axes = {
            str(item.get("name", "")).replace(" ", "").replace("/", "_"): item.get("score", 100)
            for item in raw_axes
            if isinstance(item, dict)
        }
    else:
        evaluation_axes = raw_axes

    if "finance" in missing_sections:
        questions.append(
            {
                "category": "finance",
                "question": "재무 계획과 자금 사용 우선순위는 어떻게 설계되어 있나요?",
                "why_important": "누락된 재무 섹션은 사업의 실행 가능성과 지속가능성 판단에 직접 연결됩니다.",
            }
        )
    if "team" in missing_sections:
        questions.append(
            {
                "category": "team",
                "question": "현재 팀 구성으로 이 사업을 실제로 실행할 수 있는 근거는 무엇인가요?",
                "why_important": "팀 섹션 누락은 실행 역량에 대한 가장 직접적인 의문으로 이어집니다.",
            }
        )
    if (
        evaluation_axes.get("문제정의", evaluation_axes.get("문제_정의", 100)) <= 70
        or context.get("problem_slide_texts")
    ):
        questions.append(
            {
                "category": "problem_definition",
                "question": "고객이 겪는 문제의 빈도와 심각성을 어떤 데이터로 검증했나요?",
                "why_important": "문제 정의 점수가 낮으면 솔루션의 필요성 자체가 약해집니다.",
            }
        )
    if context.get("pitch_type") == "government_support":
        questions.append(
            {
                "category": "policy_fit",
                "question": "이 아이템이 정부지원사업의 목적과 어떤 방식으로 맞닿아 있나요?",
                "why_important": "정부지원형 발표에서는 정책 적합성과 공공성이 주요 심사 포인트가 됩니다.",
            }
        )
    if evaluation_axes.get("시장성", 100) <= 70:
        questions.append(
            {
                "category": "market",
                "question": "초기 타깃 시장 규모와 진입 전략을 어떤 근거로 설정했나요?",
                "why_important": "시장성 설명이 약하면 성장 가능성과 사업화 설득력이 떨어집니다.",
            }
        )

    while len(questions) < 5:
        questions.append(
            {
                "category": "execution",
                "question": "향후 3개월 내 가장 먼저 검증할 핵심 가설은 무엇인가요?",
                "why_important": "초기 발표에서는 실행 우선순위가 팀의 집중도를 보여줍니다.",
            }
        )

    return questions[:5]


def generate_qa(context: Dict) -> List[Dict]:
    """
    심사위원 Q&A 5개 생성
    """
    prompt = f"""
너는 창업경진대회 심사위원 전문가야.
다음 발표 정보를 참고해 날카로운 예상 질문 5개를 JSON 배열로 만들어라.

[발표 상황]  
{context['pitch_situation']}

[1분 요약]  
{context['summary_1min']}

[평가 점수]  
{context['evaluation_axes']}

[Deck Missing Sections]  
{context['missing_sections']}

[논리 흐름 문제]  
{context['logic_issues']}

[문제 정의 슬라이드 예시]  
{context['problem_slide_texts'][:1]}

[솔루션 슬라이드 예시]  
{context['solution_slide_texts'][:1]}

규칙:
- 출력은 JSON 배열만.
- 각 항목은 {{
    "category": "...",
    "question": "...",
    "why_important": "..."
  }}
- 누락 섹션(finance, team 등)과 점수 낮은 항목(<= 70)을 반드시 포함.
"""

    try:
        from .llm_client import call_llm, safe_json_parse

        raw = call_llm(prompt)
        return safe_json_parse(raw)
    except Exception:
        return _fallback_qa(context)
