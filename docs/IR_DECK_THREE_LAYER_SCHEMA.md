# IR Deck 3계층 스키마 (현재 코드 반영)

목표: API 계약(UI)과 저장 구조(ERD), 내부 계산 모델(RAG/스코어링)을 분리한다.

## 1. 저장 스키마 (DB 최적화 관점)

현재 코드에선 메모리 dataclass로 시뮬레이션 중이며, 실제 DB는 ERD v4로 이관 예정.

### 1-1) IRDeck 저장(예: `IRDeckRow`)
- `id`, `pitch_id`, `notice_id`
- `pdf_url`, `pdf_size_bytes`, `pdf_upload_status`
- `version`, `is_latest`
- `analysis_status`, `error_message`
- `analyzed_at`, `created_at`, `updated_at`

### 1-2) 분석 결과 저장(예: `IRDeckResultRow`)
- `deck_score` (JSON)
- `criteria_scores` (JSON 배열)
- `presentation_guide` (JSON)
- `slides` (JSON 배열)

## 2. 내부 도메인 스키마 (판정/계산용)

### 2-1) RAG 결과 원본 (`src/domain/ir/rag_pipeline.py`)
- `deck_score.total_score/structure_summary/strengths/improvements/top_actions`
- `criteria_scores[]`:
  - `criteria_id`, `criteria_name`, `score`, `max_score`
  - `coverage_status`, `related_slides`, `missing_items`, `confidence`
- `slides[]`:
  - `slide_id`, `slide_number`, `category`, `content`, `feedback`

### 2-2) 도메인 스키마 특징
- 설명가능성 필드(`related_slides`, `missing_items`) 유지
- UI 불필요 필드는 API 변환 단계에서 제거

## 3. API 스키마 (프론트 응답용)

`app/schemas/ir_schema.py` 기준

### 3-1) 업로드 응답
- `IRUploadResponse`

### 3-2) 종합 조회 응답
- `IRDeckSummaryInProgressResponse`
- `IRDeckSummaryFailedResponse`
- `IRDeckSummaryCompletedResponse`

### 3-3) 슬라이드 조회 응답
- `IRDeckSlidesInProgressResponse`
- `IRDeckSlidesCompletedResponse`
- `IRDeckSlideItemResponse`

## 4. 매핑 규칙

`app/routers/ir.py::_map_ir_payload_to_result`가 핵심 매퍼

### 4-1) deck_score
- 도메인 `strengths/improvements` 문자열/배열 -> API 배열로 정규화
- `top_actions`는 API에서 제거

### 4-2) criteria_scores
- Notice 기준이 있으면 Notice 기준 텍스트(`pitchcoach_interpretation`, `ir_guide`) 우선
- Notice 기준 있고 IR raw 비어있으면 fallback 0점 행 생성

### 4-3) slides
- `content` -> `content_summary`
- `feedback.detailed_feedback` -> `detailed_feedback`
- `category` 영문 -> 한글 매핑 (`_to_display_category`)

## 5. 왜 3계층이 필요한가

1. 저장 구조 변경(ERD 개편) 시 API 계약 영향 최소화  
2. 모델 출력 스키마 변경 시 API 방어 가능  
3. 프론트는 안정된 필드 계약만 사용 가능
