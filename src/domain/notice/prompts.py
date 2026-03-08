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

[recruitment_type 분류 규칙 — 매우 중요]
- 아래 목록에서 가장 적합한 값을 선택:
  정부지원사업 | 공모전 | 액셀러레이팅 | IR 데모데이 | IR 피칭대회 | 투자연계 | 기타
- "데모데이", "피칭대회", "IR 발표", "발표 심사", "피칭 발표" 등 키워드가 있으면 "IR 데모데이" 또는 "IR 피칭대회" 우선 선택
- **"투자연계"는 실제로 투자 유치가 핵심 목적인 경우에만 사용**
- 공고문에 "투자 확정 과정이 아님", "투자와 무관", "투자를 보장하지 않음" 등의 문구가 있으면 절대 "투자연계"를 선택하지 말 것 → 대신 "IR 데모데이" 또는 "IR 피칭대회" 선택
- 정부/지자체 기관이 주관하고 지원금/보조금을 지급하면 "정부지원사업"
- 경진대회, 경쟁 형태의 공모이면 "공모전"
- 판단이 어려우면 "기타"

[엄격한 추출 규칙]
- evaluation_criteria는 반드시 공고문에서 "평가 항목", "심사 기준", "배점", "점수", "%" 등과 함께 명시된 항목만 포함
- 단순 안내 문구, 자격 요건, 제출 서류, 일정 정보는 절대 포함하지 말 것
- 공고문에 없는 일반적인 평가 항목(예: 시장성, 기술성 등)을 추정 생성하지 말 것
- 평가 항목은 공고문에 제시된 순서를 그대로 유지할 것
- %로만 제시된 경우 숫자만 반환 (예: 40% -> 40)
- 점수와 비율이 동시에 존재하면 점수를 우선 사용
- 총점 정보는 evaluation_criteria에 포함하지 말 것
- 불확실한 경우 해당 항목을 생성하지 말 것
- evaluation_criteria 각 항목마다 source_reference(원문 근거)를 반드시 포함할 것
- source_reference는 해당 항목명/배점이 들어있는 원문 일부를 그대로 발췌할 것
- extraction_confidence는 0~1 사이 숫자로 반환
- evaluation_structure_type은 반드시 아래 4개 중 하나로만 반환:
  POINT_BASED | PERCENT_BASED | MIXED | NOT_EXPLICIT

[additional_criteria (우대사항/가산점) 추출 — 반드시 수행]
- 공고문의 평가표/심사표에서 "우대사항", "가산점", "우대 가점", "가점 항목", "추가 배점", "우대점수" 행을 찾아 추출
- 표의 마지막 행이나 별도 섹션에 우대/가산 항목이 있는 경우가 매우 흔함 — 반드시 확인할 것
- 각 우대 항목을 개별 객체로 분리하여 배열로 반환 (item: 항목명, points: 점수)
- 우대사항이 총점과 별도의 가산점인 경우에도 반드시 추출
- 해당 정보가 정말 없는 경우에만 빈 배열 []을 반환
- **evaluation_criteria 배열에는 우대사항을 넣지 말 것** — additional_criteria 필드에만 넣을 것

[pitchcoach_interpretation 작성 규칙]
- 해당 평가 항목이 구체적으로 무엇을 요구하는지 분석하여 작성
- 공고문의 세부 평가 내용, sub_requirements를 반영하여 구체적으로 작성
- "~를 평가합니다" 같은 일반적 문구 대신, 해당 항목에서 높은 점수를 받으려면 어떤 내용이 필요한지 구체적으로 서술
- 예시: "기술성(20점)" → "보유 기술의 독자성과 특허/IP 현황, 기술 구현 단계(TRL), 경쟁 기술 대비 우위를 구체적으로 제시해야 합니다."

[ir_guide 작성 규칙 — 반드시 항목별로 다르게 작성]
- 이 필드는 반드시 비어있지 않아야 하며, 각 평가 항목마다 서로 다른 내용이어야 함
- "[항목명] 관련 근거와 실행 계획을 슬라이드에 포함하세요" 같은 일반적 문구는 금지
- 해당 평가 항목의 세부 요구사항에 맞춰, IR 덱 슬라이드에 구체적으로 넣어야 할 콘텐츠를 안내:
  * 어떤 종류의 시각 자료(차트/표/다이어그램/스크린샷)를 넣을지
  * 어떤 구체적 데이터나 수치를 보여줄지
  * 슬라이드 구성 팁
- 예시들:
  * "사업화 역량(25점)" → "매출 추이 그래프, 고객사 로고 월, 주요 계약 실적 표, 시장 점유율 비교 차트를 포함하세요."
  * "기술성(20점)" → "기술 아키텍처 다이어그램, 특허 출원/등록 현황 표, TRL 단계 진행도, 경쟁사 기술 비교 매트릭스를 넣으세요."
  * "팀 구성(15점)" → "핵심 팀원 프로필(사진+경력), 조직도, 자문단/멘토 현황, 팀 성장 타임라인을 보여주세요."
- 각 항목의 pitchcoach_interpretation 내용과 연계하되, 시각적/구조적 관점에서 작성

[출력 JSON 스키마]
{{
  "notice_name": "공고문에 명시된 공식 사업명",
  "host_organization": "주관/주최 기관명",
  "recruitment_type": "정부지원사업|공모전|액셀러레이팅|IR 데모데이|IR 피칭대회|투자연계|기타",
  "target_audience": "모집 대상 요약",
  "application_period": "접수 기간 원문",
  "summary": "공고문 핵심 요약 (1~3문단 분량)",
  "core_requirements": "평가에 직접적인 핵심 요구사항 요약",
  "source_reference": "평가 관련 원문 발췌",
  "evaluation_structure_type": "POINT_BASED|PERCENT_BASED|MIXED|NOT_EXPLICIT",
  "extraction_confidence": 0.0,
  "additional_criteria": [
    {{"item": "우대 항목명", "points": 0}}
  ],
  "evaluation_criteria": [
    {{
      "criteria_name": "평가 항목명",
      "points": 0,
      "source_reference": "원문 발췌",
      "sub_requirements": ["세부1", "세부2"],
      "pitchcoach_interpretation": "높은 점수를 받기 위해 구체적으로 준비할 내용",
      "ir_guide": "IR 덱 슬라이드에 넣어야 할 구체적 시각자료/데이터/구성 안내 (항목별로 다르게)"
    }}
  ],
  "ir_deck_guide": "공고문 기반 IR Deck 전체 구성 가이드"
}}
""".strip()
