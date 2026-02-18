# Backend Dev Plan (v4 반영 운영본)

파일명은 `ERD_V3_*`지만, 실제 내용은 현재 코드/요구사항 기준(v4)을 반영합니다.

## 1. 현재 상태

- API 계약:
  - Notice: UI 계약 반영
  - IR Deck: UI 계약 반영
- 저장 계층:
  - 실제 DB 미연결
  - 라우터 내부 메모리 저장소로 ERD v4 컬럼 구조 시뮬레이션
- 에러 포맷:
  - `{error, message}` 평탄 응답 통일

## 2. DB 이관 우선순위

1. Notice/NoticeEvaluationCriteria
2. IRDeck/DeckScore/CriteriaScore/Slide/SlideFeedback
3. Rehearsal 계열
4. Report

## 3. Notice 이관 체크리스트

- `NoticeRow` -> `Notice` 테이블 매핑
- `_CRITERIA_BY_NOTICE_ID` -> `NoticeEvaluationCriteria`
- 버전/is_latest 유지
- `display_order`, `importance`, `ir_guide` 유지
- PATCH 시 criteria 전체교체 트랜잭션 보장

## 4. IR Deck 이관 체크리스트

- `IRDeckRow` -> `IRDeck`
- `IRDeckResultRow.deck_score` -> `DeckScore`
- `IRDeckResultRow.criteria_scores` -> `CriteriaScore`
- `IRDeckResultRow.slides` -> `Slide`, `SlideFeedback`
- Notice 연동 시 Notice latest 버전 조회

## 5. API 안정성 체크리스트

1. 스키마 누락 필드 없음 (`ir_summary`, `ir_slides`, `notice_result`)
2. `criteria_scores` 공고 연동 정상
3. 에러 응답 평탄 구조 유지
4. category 한글 매핑 유지

## 6. 운영 이슈

- 메모리 저장이므로 프로세스 재시작 시 상태 유실
- 백그라운드 태스크 실패 재시도 큐 미구현
- 인증/권한(토큰 검증, 소유자 검증) 미구현

## 7. 다음 개발 권장 순서

1. DB/Repository 계층 도입
2. Auth + ownership guard
3. 비동기 작업 큐(Celery/RQ 등) 분리
4. 파일 스토리지(S3/GCS) 연동
5. 운영 모니터링/로그 표준화
