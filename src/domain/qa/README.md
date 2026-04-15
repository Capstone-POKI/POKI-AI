# Q&A (반박 질문) AI 파이프라인

PitchCoach의 Q&A 반박 질문 생성 및 평가 엔진입니다.

## 개요

Q&A 파이프라인은 사진의 4단계 구조를 따릅니다:

```
1. 입력데이터 통합 (Gemini)
   ↓
2. 의도 분석 & 질문 생성 (Gemini)
   ↓
3. 음성 답변 처리 (Whisper)
   ↓
4. 답변 평가 & 피드백 (Gemini)
```

## 주요 기능

### 1. 질문 생성 (Question Generation)
- **입력**: 공고문, IR Deck, 발표 내용
- **출력**: 다양한 관점의 Q&A 질문 12-16개
- **관점**:
  - `NOTICE`: 공고 요구사항 중심
  - `PITCHBOOK`: IR Deck 구체적 내용
  - `PRESENTER`: 발표자 전문성
  - `EVALUATOR`: 심사위원 관점

### 2. 답변 평가 (Answer Evaluation)
- **입력**: 질문, 답변 (음성 또는 텍스트)
- **출력**: 점수 및 상세 피드백
- **평가 항목**:
  - 적합도 (Relevance): 40%
  - 명료도 (Clarity): 35%
  - 구성도 (Structure): 25%

### 3. 음성 변환 (STT)
- OpenAI Whisper API를 사용하여 음성을 텍스트로 변환
- 지원 형식: mp3, mp4, mpeg, mpga, m4a, wav, webm (최대 25MB)

## API 엔드포인트

### 1. 질문 생성 및 세션 시작

```bash
POST /api/pitches/{pitch_id}/qa/questions/generate

요청:
- pitch_id: 피치 ID (URL)
- notice_content: 공고문 전체 내용 (FormData)
- irdecksummary: IR Deck 요약/OCR 결과 (FormData)
- presentation_content: 발표 내용 (FormData, 선택)

응답:
{
  "session_id": "uuid",
  "pitch_id": "uuid",
  "status": "IN_PROGRESS",
  "questions": [
    {
      "question_id": "uuid",
      "question_type": "NOTICE|PITCHBOOK|PRESENTER|EVALUATOR",
      "content": "질문 내용",
      "guidance": "예상 답변 방향",
      "order": 1,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total_questions": 14,
  "answered_count": 0,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 2. 세션 조회

```bash
GET /api/pitches/{pitch_id}/qa/sessions/{session_id}

응답: 위의 응답과 동일
```

### 3. 질문 리스트 조회 (Backend 호환)

```bash
GET /api/pitches/{pitch_id}/questions

응답:
{
  "questions": [
    {
      "question_id": "uuid",
      "order": 1,
      "type": "NOTICE",
      "content": "질문 내용",
      "guidance": "예상 답변 방향"
    }
  ],
  "total": 14
}
```

### 4. 답변 제출

```bash
POST /api/pitches/{pitch_id}/qa/sessions/{session_id}/answers

요청 (FormData):
- question_id: 질문 ID (필수)
- audio: 음성 파일 (선택, mp3/wav 등)
- text_transcript: 텍스트 답변 (선택, audio가 없으면 필수)

응답:
{
  "answer_id": "uuid",
  "question_id": "uuid",
  "pitch_id": "uuid",
  "text_transcript": "답변 내용",
  "evaluation_status": "IN_PROGRESS",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 5. 평가 결과 조회

```bash
GET /api/pitches/{pitch_id}/qa/sessions/{session_id}/answers/{answer_id}

응답:
{
  "answer_id": "uuid",
  "question_id": "uuid",
  "pitch_id": "uuid",
  "text_transcript": "답변 내용",
  "feedback": {
    "score": 75,
    "relevance": 80,
    "clarity": 75,
    "structure": 70,
    "strengths": ["명확한 설명", "논리적 구조"],
    "improvements": ["구체적 수치 추가", "배경 설명 보충"],
    "reasoning": "평가 근거 설명..."
  },
  "evaluation_status": "COMPLETED",
  "created_at": "2024-01-01T00:00:00Z",
  "evaluated_at": "2024-01-01T00:01:00Z"
}
```

## 환경 설정

### 필수 환경변수

```bash
# .env 파일에 추가

# Gemini API (필수)
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash

# OpenAI Whisper (음성 변환 시 필수)
OPENAI_API_KEY=your-openai-api-key

# 기타 필요한 설정...
```

### 의존성 설치

```bash
# OpenAI 라이브러리 추가 (이미 requirements.txt에 포함되어 있는지 확인)
pip install openai
```

## 사용 예제

### Python에서 직접 사용

```python
from pathlib import Path
from src.domain.qa.qa_service import (
    run_qa_question_generation,
    prepare_qa_session,
    process_answer,
)

# 1. 질문 생성
notice_content = "공고문 전체 내용..."
irdecksummary = "IR Deck OCR 결과..."
presentation_content = "발표 내용..."

questions = run_qa_question_generation(
    pitch_id="pitch-123",
    notice_content=notice_content,
    irdecksummary=irdecksummary,
    presentation_content=presentation_content,
)

# 2. 세션 준비
from uuid import uuid4
session_id = str(uuid4())
session = prepare_qa_session(
    session_id=session_id,
    pitch_id="pitch-123",
    questions=questions,
)

# 3. 답변 제출 (텍스트)
question_id = session.questions[0]["question_id"]
answer = process_answer(
    session=session,
    question_id=question_id,
    text_transcript="사용자의 답변 내용...",
)

# 4. 답변 평가 결과 확인
print(f"점수: {answer.feedback.score}/100")
print(f"강점: {answer.feedback.strengths}")
print(f"개선사항: {answer.feedback.improvements}")
```

### cURL로 테스트

```bash
# 1. 질문 생성
curl -X POST "http://localhost:8000/api/pitches/pitch-123/qa/questions/generate" \
  -F "notice_content=공고문 내용..." \
  -F "irdecksummary=IR Deck 요약..." \
  -F "presentation_content=발표 내용..."

# 2. 답변 제출 (텍스트)
curl -X POST "http://localhost:8000/api/pitches/pitch-123/qa/sessions/session-id/answers" \
  -F "question_id=question-id" \
  -F "text_transcript=답변 내용..."

# 3. 평가 결과 조회
curl "http://localhost:8000/api/pitches/pitch-123/qa/sessions/session-id/answers/answer-id"
```

## 구현 상태

- [x] 질문 생성 파이프라인 (Gemini)
- [x] 답변 평가 파이프라인 (Gemini)
- [x] 음성 변환 파이프라인 (Whisper)
- [x] FastAPI 엔드포인트
- [x] 세션 관리
- [ ] 데이터베이스 영속성 (현재: 메모리 기반)
- [x] 테스트 코드
- [ ] Backend 통합 테스트

## Backend 통합

Backend의 `/api/pitches/{pitch_id}/questions` 엔드포인트는 이 AI 서비스의 `GET /api/pitches/{pitch_id}/questions` 과 연동됩니다.

현재 Backend PR #23 (feature/qa-questions-read)에서 질문 리스트 조회 기능이 구현되고 있습니다.

## 다음 단계

1. 데이터베이스 영속성 추가 (PostgreSQL)
2. 배치 평가 기능 (여러 답변 동시 처리)
3. 평가 결과 대시보드
4. 평가 기록 및 개선도 추적
