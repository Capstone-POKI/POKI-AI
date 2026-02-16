import json
from typing import Any, Dict, List


def build_notice_prompt(notice_text: str, tables: List[Dict[str, Any]]) -> str:
    truncated_text = notice_text[:40000]
    tables_json = json.dumps(tables[:15], ensure_ascii=False, indent=2)

    return f"""
당신은 공고문 분석 엔진입니다.
아래 텍스트/표를 한 번에 분석해, 반드시 지정된 JSON 스키마만 반환하세요.

[입력 텍스트]
{truncated_text}

[입력 표(JSON, 일부)]
{tables_json}

[규칙]
- 사실 기반으로만 추출
- 정보가 없으면 문자열은 "", 리스트는 []
- evaluation_criteria[].points는 숫자만 반환 (점수/비율 모두 숫자만)
- 출력은 JSON 객체 하나만 반환
- 아래 스키마의 키 이름을 절대 변경하지 말 것

[엄격한 추출 규칙]
- evaluation_criteria는 반드시 공고문에서 "평가 항목", "심사 기준", "배점", "점수", "%" 등과 함께 명시된 항목만 포함
- 단순 안내 문구, 자격 요건, 제출 서류, 일정 정보는 절대 포함하지 말 것
- 공고문에 없는 일반적인 평가 항목(예: 시장성, 기술성 등)을 추정 생성하지 말 것
- 평가 항목은 공고문에 제시된 순서를 그대로 유지할 것
- %로만 제시된 경우 숫자만 반환 (예: 40% -> 40)
- 점수와 비율이 동시에 존재하면 점수를 우선 사용
- 총점 정보는 evaluation_criteria에 포함하지 말 것
- 불확실한 경우 해당 항목을 생성하지 말 것
- extraction_confidence는 0~1 사이 숫자로 반환
- evaluation_structure_type은 반드시 아래 4개 중 하나로만 반환:
  POINT_BASED | PERCENT_BASED | MIXED | NOT_EXPLICIT

[출력 JSON 스키마]
{{
  "notice_name": "공고문에 명시된 공식 사업명",
  "host_organization": "주관/주최 기관명",
  "recruitment_type": "정부지원사업|공모전|액셀러레이팅|투자연계|기타",
  "target_audience": "모집 대상 요약",
  "application_period": "접수 기간 원문",
  "summary": "공고문 핵심 요약 (1~3문단 분량)",
  "core_requirements": "평가에 직접적인 핵심 요구사항 요약",
  "source_reference": "평가 관련 원문 발췌",
  "evaluation_structure_type": "POINT_BASED|PERCENT_BASED|MIXED|NOT_EXPLICIT",
  "extraction_confidence": 0.0,
  "evaluation_criteria": [
    {{
      "criteria_name": "평가 항목명",
      "points": 0,
      "sub_requirements": ["세부 요구사항1", "세부 요구사항2"],
      "pitchcoach_interpretation": "사용자에게 제공할 준비 가이드"
    }}
  ],
  "ir_deck_guide": "공고문 기반 IR Deck 작성 가이드"
}}
""".strip()
