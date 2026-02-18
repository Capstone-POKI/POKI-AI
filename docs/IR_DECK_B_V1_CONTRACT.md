# IR Deck B안 계약서 (현재 코드 기준)

이 문서는 `app/routers/ir.py`, `app/schemas/ir_schema.py`, `src/domain/ir/*` 기준의 실제 동작 계약을 정리합니다.

## 1. API 계약

### 1-1) 업로드 + 분석 시작
- `POST /api/pitches/{pitch_id}/ir-decks/analyze`
- 입력: `multipart/form-data` (`file`)
- 제한:
  - PDF만 허용
  - 파일 크기 최대 30MB
- 응답(202):
```json
{
  "ir_deck_id": "ir-uuid",
  "pitch_id": "pitch-uuid",
  "analysis_status": "IN_PROGRESS",
  "version": 1,
  "message": "IR Deck 분석이 시작되었습니다."
}
```

### 1-2) 종합 결과 조회 (polling)
- `GET /api/ir-decks/{deck_id}`
- 상태별 응답:
  - `IN_PROGRESS`: 최소 필드
  - `FAILED`: `error_message` 포함
  - `COMPLETED`: `deck_score`, `criteria_scores`, `presentation_guide`, `analyzed_at`

### 1-3) 슬라이드 상세 조회
- `GET /api/ir-decks/{deck_id}/slides`
- 상태별 응답:
  - `IN_PROGRESS`
  - `COMPLETED` + `slides[]`

## 2. 현재 코드상의 응답 정책

### 2-1) `criteria_scores`
- Notice가 없는 pitch면 `[]`
- Notice가 있으면 Notice 기준(`criteria_name`, `pitchcoach_interpretation`, `ir_guide`)에 맞춰 반환
- IR 원본 기준 점수가 비어도 fallback으로 0점 행 생성(빈 배열 방지)

### 2-2) `deck_score`
- `top_actions`는 API에서 제거됨
- `strengths`, `improvements`는 문자열 배열

### 2-3) 슬라이드 `category`
- 내부 분류값(`COVER`, `PROBLEM`, ...)은 API 응답에서 한글로 매핑
  - 예: `COVER -> 표지`, `BUSINESS_MODEL -> 비즈니스 모델`

## 3. 내부 파이프라인 연결

엔트리:
- `app/routers/ir.py::_run_ir_analysis_background`

실행 체인:
1. `src/domain/ir/pipeline.py::run_ir_analysis`
2. `src/infrastructure/document_ai/pipeline.py::run_document_ai_pipeline`
3. `src/domain/ir/rag_pipeline.py::run_rag_ir_analysis`
4. 최종 JSON(`*_final.json`) 생성
5. API 전용 스키마로 재매핑 후 메모리 저장

## 4. 상태 전이

- 업로드 직후:
  - `analysis_status = IN_PROGRESS`
  - `pdf_upload_status = PROCESSING`
- 성공:
  - `analysis_status = COMPLETED`
  - `pdf_upload_status = COMPLETED`
  - `analyzed_at` 채움
- 실패:
  - `analysis_status = FAILED`
  - `pdf_upload_status = FAILED`
  - `error_message` 저장

## 5. 버전 정책

- `pitch_id` 단위 버전 증가
- 신규 업로드 시 기존 `is_latest=false`, 신규 `is_latest=true`

## 6. 현재 구현 범위/제약

- DB 미연결: 현재는 `app/routers/ir.py` 내부 메모리 저장소 사용
- 프로세스 재시작 시 분석 상태/결과 초기화
- 성능 품질(정확도)과 API 계약 준수는 분리해서 관리 중
