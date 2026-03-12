# IR Deck API 명세서

## 1. IR Deck 분석 (업로드 + 분석)

`POST /api/pitches/{pitchId}/ir-decks/analyze`

IR Deck PDF 업로드 후 자동으로 분석을 시작

### Request

```
Content-Type: multipart/form-data
Authorization: Bearer {token}

Path Parameter:
  - pitchId (string, 필수): Pitch ID

FormData:
  - file: [IR_Deck.pdf]  (PDF 파일, 최대 10MB)
```

### Response (success)

```json
// HTTP 202 Accepted
{
  "ir_deck_id": "uuid-deck-1",
  "pitch_id": "uuid-1234",
  "analysis_status": "IN_PROGRESS",
  "version": 1,
  "message": "IR Deck 분석이 시작되었습니다."
}
```

### Response (error)

```json
// 400
{ "error": "INVALID_FILE", "message": "PDF 파일만 업로드 가능합니다" }

// 400
{ "error": "FILE_TOO_LARGE", "message": "파일 크기는 10MB 이하여야 합니다" }

// 404
{ "error": "PITCH_NOT_FOUND", "message": "존재하지 않는 피칭입니다" }
```

### 기타 설명

- 업로드 + 분석 동시 실행 (비동기)
- Pitch에 공고(Notice)가 있으면 notice_id를 자동 참조하여 공고 기반 분석 수행
- Pitch.status → IRDECK_ANALYSIS로 변경
- 분석 완료 여부는 polling으로 확인
- 버전 관리: 이미 IR Deck이 존재하면 기존 버전의 is_latest → false, 새 버전 생성 (version +1)
- 응답에서 제거: notice_id, pdf_url, pdf_size_bytes, pdf_upload_status, is_latest (프론트 불필요)

---

## 2. IR Deck 종합 분석 결과 조회

`GET /api/ir-decks/{deckId}`

종합 점수, 강점/개선점, 공고문 기준 평가, 발표 가이드를 조회합니다. (polling 겸용)

### Request

```
Authorization: Bearer {token}

Path Parameter:
  - deckId (string, 필수): IR Deck ID
```

### Response (success / 분석 완료)

```json
// HTTP 200 OK
{
  "ir_deck_id": "uuid-deck-1",
  "pitch_id": "uuid-1234",
  "analysis_status": "COMPLETED",
  "version": 1,

  "deck_score": {
    "total_score": 78,
    "structure_summary": "VC 데모데이에서는 제한 시간 안에 문제의 크기, 솔루션의 차별성, 성장 가능성, 그리고 투자 포인트를 확실히 각인...",
    "strengths": [
      "일정한 리듬과 모던한 발음으로 안정적인 신뢰감 전달",
      "그래프 흐름을 잘 해석하여 결론을 끌어내 연결",
      "슬라이드에 없는 시장 해석을 더하여 이해도 향상"
    ],
    "improvements": [
      "문제 정의 구간에서 현재 사례 추가",
      "핵심 수치 투자 포인트 강조",
      "핵심 기능 정리 후 마무리"
    ]
  },

  "criteria_scores": [
    {
      "criteria_name": "시장성",
      "pitchcoach_interpretation": "시장 규모와 확장 전략이 명확하고, 수익 모델이 실제로 성립 가능한지 평가합니다.",
      "ir_guide": "TAM·SAM·SOM 구조, 수익 구조(누가 얼마를 내는지)를 IR에 반드시 포함하세요.",
      "score": 90,
      "feedback": "시장 규모 분석이 잘 되어있고..."
    },
    {
      "criteria_name": "문제정의",
      "pitchcoach_interpretation": "해결하려는 문제가 구체적이고 실재하는지, 고객 검증 근거가 있는지 평가합니다.",
      "ir_guide": "고객 페르소나, 문제의 규모·빈도, 기존 해결책의 한계를 명시하세요.",
      "score": 90,
      "feedback": "..."
    },
    {
      "criteria_name": "기술 혁신성",
      "pitchcoach_interpretation": "기술의 독창성과 경쟁 우위가 명확한지 평가합니다.",
      "ir_guide": "핵심 기술 차별점, MVP 단계, 특허·인증 보유 여부를 포함하세요.",
      "score": 90,
      "feedback": "..."
    },
    {
      "criteria_name": "팀 역량",
      "pitchcoach_interpretation": "대표자와 핵심 팀원의 역량이 사업 실행에 적합한지 평가합니다.",
      "ir_guide": "관련 산업 경험, 기술/사업 역할 분담, 팀 결속력을 보여주세요.",
      "score": 90,
      "feedback": "..."
    }
  ],

  "presentation_guide": {
    "emphasized_slides": [
      {
        "slide_number": 2,
        "reason": "공고에서 '사회문제 해결'을 중요하게 평가하므로 78% 통계를 강조하세요."
      },
      {
        "slide_number": 3,
        "reason": "..."
      }
    ],
    "guide": [
      "오프닝에서 핵심 문제를 짧게 제시하며 청중의 관심을 끌어들이세요",
      "공고의 평가 기준 흐름에 맞춰 자연스럽게 언급하세요",
      "클로징에서는 3년 비전과 투자금 활용 계획을 간결하게 정리해주세요"
    ],
    "time_allocation": [
      "오프닝 (1분): 핵심 문제 제시",
      "본론 (6분): 솔루션, 시장성, 비즈니스 모델 순서로 설명",
      "클로징 (1분): 비전과 투자 계획 정리"
    ]
  },

  "analyzed_at": "2026-02-06T10:05:00Z"
}
```

### Response (success / 분석 중)

```json
// HTTP 200 OK
{
  "ir_deck_id": "uuid-deck-1",
  "pitch_id": "uuid-1234",
  "analysis_status": "IN_PROGRESS",
  "version": 1
}
```

### Response (success / 분석 실패)

```json
// HTTP 200 OK
{
  "ir_deck_id": "uuid-deck-1",
  "pitch_id": "uuid-1234",
  "analysis_status": "FAILED",
  "error_message": "PDF 파일을 읽을 수 없습니다.",
  "version": 1
}
```

### Response (error)

```json
// 404
{ "error": "IR_DECK_NOT_FOUND" }
```

### 기타 설명

- 프론트에서 3~5초 간격으로 polling → COMPLETED 시 전체 렌더링
- criteria_scores: 공고가 없는 Pitch면 빈 배열 [] 반환 → 프론트에서 섹션 숨김
- presentation_guide: 발표 가이드 + 강조 슬라이드 + 시간 배분 통합
- deck_score 변경사항:
    - strengths, improvements → string 배열로 변경
    - top_actions 제거 (improvements에 통합)
- criteria_scores 변경사항:
    - pitchcoach_interpretation + ir_guide 분리 (공고문 API와 통일)
    - 제거: criteria_score_id, criteria_id, max_score, is_covered, related_slides

---

## 3. 슬라이드별 상세 조회

`GET /api/ir-decks/{deckId}/slides`

### Request

```
Authorization: Bearer {token}

Path Parameter:
  - deckId (string, 필수): IR Deck ID
```

### Response (success)

```json
// HTTP 200 OK
{
  "ir_deck_id": "uuid-deck-1",
  "analysis_status": "COMPLETED",
  "total_slides": 12,
  "slides": [
    {
      "slide_number": 1,
      "category": "표지",
      "score": 90,
      "thumbnail_url": "https://s3.../slide-1-thumb.png",
      "content_summary": "슬라이드 내용 요약 텍스트...",
      "detailed_feedback": "표지가 깔끔하고 전문적입니다...",
      "strengths": [
        "서비스명과 부제목이 명확하게 표현됨",
        "디자인이 깔끔하고 전문적"
      ],
      "improvements": [
        "로고 해상도 개선 필요",
        "발표 날짜 추가 권장"
      ]
    },
    {
      "slide_number": 2,
      "category": "문제 정의",
      "score": 85,
      "thumbnail_url": "https://s3.../slide-2-thumb.png",
      "content_summary": "슬라이드 내용 요약 텍스트...",
      "detailed_feedback": "문제 상황을 구체적인 수치로 잘 표현했습니다...",
      "strengths": [
        "구체적 통계 데이터 활용"
      ],
      "improvements": [
        "고객 페르소나 추가 필요",
        "기존 해결책 한계 명시 필요"
      ]
    }
  ]
}
```

### Response (success / 분석 중)

```json
// HTTP 200 OK
{
  "ir_deck_id": "uuid-deck-1",
  "analysis_status": "IN_PROGRESS"
}
```

### Response (error)

```json
// 404
{ "error": "IR_DECK_NOT_FOUND", "message": "존재하지 않는 IR Deck입니다" }
```

### 기타 설명

- 슬라이드와 피드백을 한번에 내려줌 (N+1 방지)
- category 값: COVER, PROBLEM, MARKET, SOLUTION, BUSINESS_MODEL, TEAM, TRACTION, FUNDING, APPENDIX 등
- 제거: slide_id, content, display_order (프론트 불필요, 배열 순서 = 표시 순서)
