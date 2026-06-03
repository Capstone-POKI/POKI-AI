# POKI AI 서버 API 명세 (프론트엔드 전달용)

> Base URL: `https://pitchcoach.duckdns.org` (nginx → port 8000 프록시 필요 시 `/ai` prefix 확인)  
> 직접 호출: `http://pitchcoach.duckdns.org:8000`  
> Swagger UI: `http://pitchcoach.duckdns.org:8000/docs`

---

## 1. AI 종합 리포트 생성

### `POST /api/pitches/{pitch_id}/report/generate`

IR Deck · 음성 · 공고문 · Q&A 분석 결과를 바탕으로 Gemini가 프론트 UI용 종합 리포트 JSON을 생성합니다.  
Gemini 호출 실패 시 rule-based 폴백으로 응답합니다.

#### Request

| 위치 | 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|------|
| path | `pitch_id` | string | ✅ | Pitch ID |
| body | `ir_deck_summary` | string | | IR 덱 분석 요약 텍스트 |
| body | `voice_summary` | string | | 음성 분석 요약 텍스트 |
| body | `notice_summary` | string | | 공고 분석 요약 텍스트 |
| body | `qa_summary` | string | | Q&A 분석 요약 텍스트 |
| body | `ir_deck_score` | float | | IR 덱 점수 (0~100) |
| body | `voice_score` | float | | 음성 점수 (0~100) |
| body | `notice_score` | float | | 공고 점수 (0~100) |
| body | `qa_score` | float | | Q&A 점수 (0~100) |

**Content-Type:** `application/json`

#### Response `200 OK`

```json
{
  "pitch_id": "b3eafcf6-fd2b-42ee-aab0-17a3b35478bf",
  "final_score": 83.4,
  "radar_chart": {
    "labels": ["문제정의", "솔루션", "시장성", "전달력", "Q&A대응력"],
    "scores": [83.0, 85.0, 80.0, 78.0, 91.0]
  },
  "bar_chart": {
    "items": [
      { "label": "AI 플랫폼",      "score": 75.0 },
      { "label": "시장 창출",      "score": 88.0 },
      { "label": "비즈니스 모델",  "score": 90.0 },
      { "label": "투자 유치",      "score": 82.0 },
      { "label": "재무 계획",      "score": 70.0 },
      { "label": "경쟁 분석",      "score": 72.0 }
    ]
  },
  "detail_scores": [
    {
      "title": "발표 완성도",
      "score": 85.0,
      "description": "공고문 적합성이 높고 발표 구성이 체계적입니다.",
      "strengths": ["공고문과의 높은 연관성", "발표 흐름의 안정성"],
      "improvements": ["도입부 임팩트 강화 필요", "공고 특정 요구사항 심층 분석"]
    },
    {
      "title": "Deck 구성 정합도",
      "score": 82.0,
      "description": "IR 덱의 논리적 흐름이 우수합니다.",
      "strengths": ["혁신적 BM의 명확한 제시", "제품-시장 논리적 연결"],
      "improvements": ["재무 계획 구체화", "경쟁사 분석 심화"]
    },
    {
      "title": "피칭 전달력",
      "score": 78.0,
      "description": "안정적인 발화 속도와 명확한 발음을 유지했습니다.",
      "strengths": ["명확한 발음과 적절한 속도", "안정적인 음성 톤"],
      "improvements": ["침묵 비율 감소 필요", "핵심 메시지 강조 강화"]
    },
    {
      "title": "Q&A 대응력",
      "score": 91.0,
      "description": "질문의 핵심을 정확히 파악하고 논리적으로 답변했습니다.",
      "strengths": ["질문 의도 정확한 파악", "자신감 있는 논리적 설명"],
      "improvements": ["예상 질문 확장 준비", "답변 간결성 향상"]
    }
  ],
  "improvement_points": [
    "침묵 비율을 줄이고 목소리 톤 변화를 주어 전달력을 높이세요.",
    "IR 덱의 재무 계획 부분을 구체화하고 투자 유치 전략을 보완하세요.",
    "핵심 메시지 전달 시 더욱 간결하고 명확한 표현을 사용하세요."
  ],
  "summary": "Q&A 대응력과 IR 덱 구성이 우수하며, 피칭 전달력 개선을 통해 더 완성된 발표를 만들 수 있습니다.",
  "generated_at": "2026-06-03T02:06:29.385072+00:00"
}
```

#### BACK에서 응답할 때 `ai_report` 필드로 래핑됨

`GET /api/reports/:reportId` 응답:
```json
{
  "report_id": "...",
  "notice":   { "summary": "...", "score": 85 },
  "ir_deck":  { "summary": "...", "score": 82 },
  "speech":   { "summary": "...", "score": 78 },
  "qa":       { "summary": "...", "score": 91 },
  "final_score": 83,
  "chart_data": { "labels": [...], "scores": [...] },
  "ai_report": { /* 위 응답 구조 전체 */ },
  "updated_at": "2026-06-03T..."
}
```

`ai_report`가 `null`이면 AI 생성 실패 → `chart_data` 기반 rule-based 점수만 표시.

---

## 2. Q&A 질문 생성 (멀티에이전트)

### `POST /api/pitches/{pitch_id}/qa/questions/generate`

공고문·IR 덱·발표 내용 기반으로 4-Agent 파이프라인이 투자자 관점 질문 5개를 생성합니다.  
**동기 처리** — 요청 완료 시 질문이 이미 생성된 상태로 반환됩니다.

#### Request

**Content-Type:** `multipart/form-data`

| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `notice_content` | string (form) | ✅ | 공고문 전체 텍스트 |
| `irdecksummary` | string (form) | ✅ | IR 덱 분석 요약 |
| `presentation_content` | string (form) | | 발표 내용 (선택) |

#### Response `200 OK`

```json
{
  "session_id": "d7c3be03-801e-494e-b73d-4d9276c23808",
  "pitch_id": "b3eafcf6-fd2b-42ee-aab0-17a3b35478bf",
  "status": "COMPLETED",
  "questions": [
    {
      "question_id": "f795035e-f68b-4b24-98fe-58331cb9998e",
      "pitch_id": "b3eafcf6-...",
      "question_type": "EVALUATOR",
      "content": "베트남을 초기 진입 시장으로 선택한 근거로 제시한 $60mn 규모가 TAM인지 SAM인지 불분명한데, 실제 접근 가능한 시장 규모와 초기 3년 목표 점유율을 구체적으로 제시해주세요.",
      "guidance": "시장 규모 산정 방식과 초기 고객 확보 전략을 수치와 함께 제시하세요.",
      "order": 1,
      "created_at": "2026-06-03T02:29:12.706Z"
    }
  ],
  "total_questions": 5,
  "answered_count": 0,
  "created_at": "2026-06-03T02:29:12.706Z"
}
```

#### question_type 값

| 값 | 설명 |
|----|------|
| `NOTICE` | 공고 요구사항 연관 질문 |
| `PITCHBOOK` | IR 덱 내용 이해도 질문 |
| `PRESENTER` | 발표자 전문성 질문 |
| `EVALUATOR` | 심사위원 관점 질문 |

---

## 3. Q&A 질문 조회 (BACK 호환)

### `GET /api/pitches/{pitch_id}/questions`

BACK에서 폴링할 때 사용하는 엔드포인트. 메모리 세션 없으면 파일 캐시에서 자동 복원.

#### Response `200 OK`

```json
{
  "questions": [
    {
      "question_id": "f795035e-...",
      "order": 1,
      "type": "EVALUATOR",
      "content": "베트남 시장...",
      "guidance": "수치와 함께 제시하세요."
    }
  ],
  "total": 5
}
```

---

## 4. 답변 제출 및 평가

### `POST /api/pitches/{pitch_id}/qa/sessions/{session_id}/answers`

**Content-Type:** `multipart/form-data`

| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `question_id` | string (form) | ✅ | 질문 ID |
| `audio` | file | | 음성 파일 (audio/* 형식) |
| `text_transcript` | string (form) | | 텍스트 답변 (audio 없을 경우 필수) |

#### Response `200 OK`

```json
{
  "answer_id": "a1b2c3d4-...",
  "question_id": "f795035e-...",
  "pitch_id": "b3eafcf6-...",
  "audio_url": null,
  "text_transcript": "저희 솔루션은...",
  "evaluation_status": "IN_PROGRESS",
  "created_at": "2026-06-03T02:30:00Z"
}
```

---

## 5. 답변 평가 결과 조회

### `GET /api/pitches/{pitch_id}/qa/sessions/{session_id}/answers/{answer_id}`

#### Response `200 OK`

```json
{
  "answer_id": "a1b2c3d4-...",
  "question_id": "f795035e-...",
  "pitch_id": "b3eafcf6-...",
  "text_transcript": "저희 솔루션은...",
  "feedback": {
    "relevance_score": 95,
    "clarity_score": 90,
    "structure_score": 88,
    "overall_score": 91,
    "strengths": ["핵심 메시지 명확", "구체적 수치 제시"],
    "improvements": ["두괄식 구성 필요", "리스크 언급 부족"],
    "summary": "질문의 핵심을 잘 파악하고 구체적으로 답변했습니다."
  },
  "evaluation_status": "COMPLETED",
  "created_at": "2026-06-03T02:30:00Z",
  "evaluated_at": "2026-06-03T02:30:05Z"
}
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|---------|
| 2026-06-03 | AI 리포트 생성 API 신규 추가 (`/report/generate`) |
| 2026-06-03 | QA 질문 생성 멀티에이전트 전환 (4-Agent 파이프라인) |
| 2026-06-03 | QA 질문 생성 동기 처리 변경 + 파일 캐시 추가 |
