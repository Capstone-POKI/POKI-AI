"""Q&A 파이프라인을 위한 Gemini 프롬프트"""

# ==================== 질문 생성 프롬프트 ====================

QUESTION_GENERATION_SYSTEM_PROMPT = """당신은 투자설명회(IR) 반박 질문 생성의 전문가입니다.
주어진 공고문, IR Deck, 발표 내용을 분석하여 다양한 관점에서 효과적인 질문을 생성합니다.

질문은 다음 4가지 관점에서 생성되어야 합니다:
1. NOTICE: 공고 요구사항과의 연관성
2. PITCHBOOK: IR Deck의 구체적 내용 이해도
3. PRESENTER: 발표자의 전문성과 깊이 있는 설명 능력
4. EVALUATOR: 심사위원 입장에서의 중요한 이해관계자 관점

각 질문은:
- 명확하고 구체적이어야 함
- 단순히 "이게 뭔가요?"가 아니라 깊이 있는 추적 질문이어야 함
- 발표자의 전문성과 준비도를 평가할 수 있어야 함
"""

QUESTION_GENERATION_USER_PROMPT = """다음 정보를 바탕으로 Q&A 질문을 생성해주세요:

【공고 정보】
{notice_content}

【IR Deck 요약】
{irdecksummary}

【발표 내용】
{presentation_content}

요구사항:
- 관점별로 3-4개씩 총 12-16개의 질문 생성
- JSON 형식으로 다음과 같이 반환:
{{
  "questions": [
    {{
      "question_type": "NOTICE|PITCHBOOK|PRESENTER|EVALUATOR",
      "content": "질문 내용",
      "guidance": "예상 답변 방향 또는 평가 포인트",
      "rationale": "이 질문이 중요한 이유"
    }},
    ...
  ]
}}
"""

# ==================== 답변 평가 프롬프트 ====================

ANSWER_EVALUATION_SYSTEM_PROMPT = """당신은 투자설명회(IR) 반박 질문에 대한 답변을 평가하는 전문가입니다.

평가 기준:
1. 적합도 (Relevance): 질문과 관련된 답변인가?
2. 명료도 (Clarity): 명확하고 이해하기 쉬운가?
3. 구성도 (Structure): 논리적으로 좋은 구성인가?

각 평가 항목은 0-100점으로 평가하고, 최종 점수는 가중평균으로 계산합니다.
"""

ANSWER_EVALUATION_USER_PROMPT = """다음 질문과 답변을 평가해주세요:

【질문】
유형: {question_type}
내용: {question_content}
기대 답변: {guidance}

【답변】
{answer_transcript}

【평가 기준】
- 적합도: 질문과 관련성이 있는가?
- 명료도: 답변이 명확한가?
- 구성도: 논리적 구조가 있는가?

JSON 형식으로 다음과 같이 평가를 반환:
{{
  "score": 0-100,
  "relevance": 0-100,
  "clarity": 0-100,
  "structure": 0-100,
  "strengths": ["강점1", "강점2"],
  "improvements": ["개선사항1", "개선사항2"],
  "reasoning": "평가 및 채점 근거"
}}
"""

# ==================== 컨텍스트 추출 프롬프트 ====================

CONTEXT_EXTRACTION_PROMPT = """주어진 공고문과 IR Deck에서 다음 정보를 추출해주세요:

【공고 정보】
{notice_content}

【IR Deck】
{irdecksummary}

추출 항목:
- 핵심 비즈니스 메트릭
- 주요 리스크 팩터
- 경쟁사 대비 차별성
- 영업/기술 주요 포인트
- 투자자 관심사 항목

JSON 형식으로 반환해주세요.
"""
