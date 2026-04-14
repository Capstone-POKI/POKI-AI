# Q&A AI 빠른 시작 가이드

## 설치 및 설정

### 1. 환경 설정

```bash
# .env 파일 생성/수정
cp .env.example .env

# 다음 항목 추가/수정:
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash
OPENAI_API_KEY=your-openai-api-key (음성 변환 사용 시)
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt

# Whisper (음성 변환) 지원 확인
pip list | grep openai
```

### 3. 서버 시작

```bash
# 단일 터미널에서
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 또는 Docker 사용
docker-compose up -d
```

### 4. 건강 체크

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/qa
```

## API 사용 예제

### 예제 1: 질문 생성 (텍스트 기반)

```bash
#!/bin/bash

PITCH_ID="06385611-df38-4543-bf39-7447a6d53f70"

# 간단한 테스트 데이터
NOTICE_CONTENT="
회사는 AI 기술 기반의 투자설명회 분석 플랫폼을 개발합니다.
- 핵심 상품: 자동 발표자료 분석, 실시간 Q&A 지원
- 시장 규모: 약 100억 원 규모의 B2B 시장
- 경쟁력: 한국어 특화 LLM 기술
"

IR_DECK_SUMMARY="
주요 슬라이드:
- 슬라이드 1: 회사 소개 및 비전
- 슬라이드 2: 시장 현황 및 크기 (100억 원)
- 슬라이드 3: 제품 기능 및 특징
- 슬라이드 4: 경영진 소개 및 경험
- 슬라이드 5: 재무 전망 (매출 성장 50% 예상)
"

PRESENTATION="
발표자가 강조한 주요 내용:
- 우리의 AI 기술은 업계 최고 수準
- 고객 만족도 95% 이상
- 시장 진입 전략은 확실함
"

curl -X POST "http://localhost:8000/api/pitches/$PITCH_ID/qa/questions/generate" \
  -F "notice_content=$NOTICE_CONTENT" \
  -F "irdecksummary=$IR_DECK_SUMMARY" \
  -F "presentation_content=$PRESENTATION" | jq .
```

**응답 예시:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "pitch_id": "06385611-df38-4543-bf39-7447a6d53f70",
  "status": "IN_PROGRESS",
  "questions": [
    {
      "question_id": "question-1",
      "pitch_id": "06385611-df38-4543-bf39-7447a6d53f70",
      "question_type": "NOTICE",
      "content": "공고에서 요구하는 B2B 시장 시뮬레이션을 어떻게 검증하셨나요?",
      "guidance": "공고의 시장 규모 검증 방식 설명 필요",
      "order": 1,
      "created_at": "2024-01-01T00:00:00Z"
    },
    {
      "question_id": "question-2",
      "pitch_id": "06385611-df38-4543-bf39-7447a6d53f70",
      "question_type": "PITCHBOOK",
      "content": "IR Deck 5쪽에 제시된 매출 성장 50% 근거는 무엇인가요?",
      "guidance": "매출 성장 예측의 근거 자료 및 배경 설명",
      "order": 2,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total_questions": 14,
  "answered_count": 0,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 예제 2: 질문 조회

```bash
SESSION_ID="550e8400-e29b-41d4-a716-446655440000"
PITCH_ID="06385611-df38-4543-bf39-7447a6d53f70"

curl -X GET "http://localhost:8000/api/pitches/$PITCH_ID/qa/sessions/$SESSION_ID" | jq .
```

### 예제 3: 텍스트 답변 제출

```bash
SESSION_ID="550e8400-e29b-41d4-a716-446655440000"
QUESTION_ID="question-1"
PITCH_ID="06385611-df38-4543-bf39-7447a6d53f70"

curl -X POST "http://localhost:8000/api/pitches/$PITCH_ID/qa/sessions/$SESSION_ID/answers" \
  -F "question_id=$QUESTION_ID" \
  -F "text_transcript=저희는 B2B 시장을 3개월간 조사했습니다. 
  LG, 삼성 등 10개 대기업에 인터뷰를 진행했고, 
  평균 시장 규모를 100억 원으로 추정했습니다." | jq .
```

**응답 예시:**
```json
{
  "answer_id": "answer-uuid",
  "question_id": "question-1",
  "pitch_id": "06385611-df38-4543-bf39-7447a6d53f70",
  "text_transcript": "저희는 B2B 시장을 ...",
  "evaluation_status": "IN_PROGRESS",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 예제 4: 평가 결과 조회

```bash
# 평가가 완료될 때까지 폴링
ANSWER_ID="answer-uuid"

for i in {1..10}; do
  echo "평가 상태 확인... ($i/10)"
  
  RESULT=$(curl -s -X GET \
    "http://localhost:8000/api/pitches/$PITCH_ID/qa/sessions/$SESSION_ID/answers/$ANSWER_ID")
  
  STATUS=$(echo $RESULT | jq -r '.evaluation_status')
  
  if [ "$STATUS" = "COMPLETED" ]; then
    echo "✅ 평가 완료!"
    echo $RESULT | jq .feedback
    break
  fi
  
  sleep 2
done
```

**응답 예시:**
```json
{
  "answer_id": "answer-uuid",
  "question_id": "question-1",
  "pitch_id": "06385611-df38-4543-bf39-7447a6d53f70",
  "text_transcript": "저희는 B2B 시장을 ...",
  "feedback": {
    "score": 82,
    "relevance": 85,
    "clarity": 80,
    "structure": 80,
    "strengths": [
      "구체적인 조사 방식 (인터뷰) 설명",
      "명확한 수치 제시",
      "논리적인 근거 제시"
    ],
    "improvements": [
      "더 많은 기업 샘플 추가 가능",
      "시장 조사 비용 및 기간 설명 필요",
      "경쟁 모니터링 과정 추가 설명"
    ],
    "reasoning": "답변이 질문과 매우 관련성 높고, 구체적인 조사 방식과 수치를 제시했습니다. 다만, 표본 크기의 대표성과 조사 방법론에 대한 추가 설명이 있으면 더욱 강력해질 것입니다."
  },
  "evaluation_status": "COMPLETED",
  "created_at": "2024-01-01T00:00:00Z",
  "evaluated_at": "2024-01-01T00:01:30Z"
}
```

## Python에서 직접 사용

```python
from pathlib import Path
from src.domain.qa.qa_service import (
    run_qa_question_generation,
    prepare_qa_session,
    process_answer,
    export_qa_results,
)
from uuid import uuid4

# 1. 질문 생성
pitch_id = "06385611-df38-4543-bf39-7447a6d53f70"
notice_content = "공고문 내용..."
irdecksummary = "IR Deck 요약..."
presentation_content = "발표 내용..."

questions = run_qa_question_generation(
    pitch_id=pitch_id,
    notice_content=notice_content,
    irdecksummary=irdecksummary,
    presentation_content=presentation_content,
)

print(f"생성된 질문: {len(questions)}개")
for q in questions:
    print(f"  [{q.question_type}] {q.content}")

# 2. 세션 준비
session_id = str(uuid4())
session = prepare_qa_session(
    session_id=session_id,
    pitch_id=pitch_id,
    questions=questions,
)

# 3. 첫 번째 질문에 답변
question_id = session.questions[0]["question_id"]
answer = process_answer(
    session=session,
    question_id=question_id,
    text_transcript="발표자의 답변 내용...",
)

# 4. 평가 결과 확인
if answer.feedback:
    print(f"\n📊 평가 결과")
    print(f"점수: {answer.feedback.score}/100")
    print(f"적합도: {answer.feedback.relevance} | 명료도: {answer.feedback.clarity} | 구성: {answer.feedback.structure}")
    print(f"강점: {', '.join(answer.feedback.strengths)}")
    print(f"개선사항: {', '.join(answer.feedback.improvements)}")

# 5. 결과 저장
output_dir = Path("data/output/qa_results")
result_file = export_qa_results(session, output_dir)
print(f"\n✅ 결과 저장: {result_file}")
```

## 테스트 실행

```bash
# 테스트 코드 실행
python -m pytest tests/test_qa_pipeline.py -v

# 또는 직접 실행
python tests/test_qa_pipeline.py
```

## 문제 해결

### Gemini API 오류
- API 키 확인
- 인터넷 연결 확인
- 할당량 확인: https://console.cloud.google.com

### OpenAI Whisper 오류
- API 키 확인
- 음성 파일 형식 확인 (mp3, wav, m4a 등)
- 파일 크기 확인 (최대 25MB)

### 메모리 부족
- 현재 메모리 기반 저장소 사용
- 대규모 운영 시 데이터베이스 영속화 필요

## 다음 단계

1. 📱 **Frontend 통합**: React UI 개발
2. 💾 **Database**: PostgreSQL 영속화
3. 🔄 **실시간 업데이트**: WebSocket 지원
4. 📊 **Analytics**: 평가 결과 분석 대시보드
5. 🔐 **권한 관리**: 역할별 접근 제어
