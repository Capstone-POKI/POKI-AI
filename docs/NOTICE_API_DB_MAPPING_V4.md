# Notice API ↔ ERD v4 매핑 (현재 코드 기준)

기준 파일:
- `app/routers/notice.py`
- `app/schemas/notice_schema.py`

## 1. API 계약 (UI)

### 1-1) 업로드 + 분석 시작
- `POST /pitches/{pitch_id}/notice`
- 응답 202:
  - `notice_id`, `pitch_id`, `analysis_status`, `message`

### 1-2) 결과 조회
- `GET /notices/{notice_id}`
- 상태별 응답:
  - `IN_PROGRESS`: 최소 상태 필드
  - `FAILED`: `error_message`
  - `COMPLETED`: `evaluation_criteria`, `ir_deck_guide` 포함

### 1-3) 수동 수정
- `PATCH /notices/{notice_id}`
- `evaluation_criteria` 전달 시 전체 교체
- 배점 합 100 검증

## 2. 내부 저장 스키마 (ERD v4 대응)

`NoticeRow` + `NoticeCriteriaRow`로 시뮬레이션

### 2-1) NoticeRow
- `id`, `pitch_id`
- `pdf_url`, `pdf_size_bytes`, `pdf_upload_status`
- `notice_name`, `host_organization`, `recruitment_type`, `target_audience`, `application_period`
- `summary`, `core_requirements`, `source_reference` (API 미노출)
- `additional_criteria`, `ir_deck_guide`
- `analysis_status`, `error_message`
- `version`, `is_latest`
- `created_at`, `updated_at`

### 2-2) NoticeCriteriaRow
- `id`, `notice_id`
- `criteria_name`, `points`, `importance`, `display_order`, `parent_id`
- `pitchcoach_interpretation`, `ir_guide`
- `created_at`, `updated_at`

## 3. 상태/버전 정책

### 3-1) 버전
- 같은 `pitch_id`로 재업로드 시 새 레코드 생성
- 이전 버전 `is_latest=false`, 최신만 `is_latest=true`

### 3-2) 상태
- 업로드 직후: `IN_PROGRESS`, `pdf_upload_status=PROCESSING`
- 성공: `COMPLETED`, `pdf_upload_status=COMPLETED`
- 실패: `FAILED`, `pdf_upload_status=FAILED`

## 4. fallback 정책

### 4-1) criteria fallback
- Notice 생성 시 기본 criteria 템플릿을 즉시 저장
- 분석 결과가 비어도 `evaluation_criteria`는 빈 배열이 되지 않도록 보장

### 4-2) pitch_type fallback
- `recruitment_type` 텍스트 기반:
  - 정부 성격 -> `GOV_SUPPORT`
  - 경진 성격 -> `STARTUP_CONTEST`
  - 기본 -> `VC_DEMO`

## 5. 에러 포맷

전체 앱 공통으로 평탄 포맷 사용:
```json
{ "error": "ERROR_CODE", "message": "..." }
```

`app/main.py`의 `HTTPException` 핸들러에서 `detail` 래핑을 제거해 일관성을 강제한다.
