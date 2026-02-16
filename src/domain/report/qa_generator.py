from typing import List, Dict
from .llm_client import call_llm, safe_json_parse


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

    raw = call_llm(prompt)
    return safe_json_parse(raw)