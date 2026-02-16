# POKI-AI

POKI-AI는 공고문/IR/음성 분석을 수행하는 AI 엔진 레포입니다.  
현재 코드는 FastAPI API 레이어와 도메인 파이프라인을 분리한 구조로 정리되어 있습니다.

## 1. 현재 아키텍처

- `app/`: FastAPI 엔트리포인트와 라우터
- `src/domain/`: 비즈니스 파이프라인
  - `notice`: 공고문 분석 (`DocAI -> Gemini 단일 호출 -> JSON`)
  - `ir`: IR deck 분석
  - `voice`: 음성 분석
  - `report`: 최종 리포트 생성
- `src/infrastructure/`: 외부 API/SDK 클라이언트
  - Document AI
  - Gemini
  - Embedding
- `src/common/`: 공통 타입/예외/유틸
- `tools/`: CLI 보조 실행 스크립트

## 2. 리팩토링 전후 차이

최근 구조 정리에서 핵심은 "코드를 버리는 것"이 아니라 "경로를 정렬하는 것"입니다.

### 2.1 Path Mapping

| 리팩토링 전 경로 | 현재 경로 | 상태 |
|---|---|---|
| `src/report_engine/*` | `src/domain/report/*` | 코드 유지 + 경로 정렬 |
| `src/voice_analysis/whisper/whisper.py` | `src/domain/voice/whisper_adapter.py` | 코드 유지 + 경로 정렬 |
| `src/voice_analysis/whisper/whisper_prompt.text` | `src/domain/voice/whisper_prompt.text` | 코드 유지 + 경로 정렬 |

### 2.2 Notice 파이프라인 변경점

- 기존: OCR/LLM 결과 정규화가 단순해 `evaluation_criteria.points`가 0으로 남는 케이스 발생
- 현재:
  - `points` 다단 fallback 보강(텍스트/표/문맥)
  - 엄격 추출 규칙 프롬프트 강화
  - `evaluation_structure_type`, `extraction_confidence` 정규화 강화
  - Gemini REST 호출 안정화(`v1/models/*:generateContent`, 모델 fallback)

## 3. 현재 프로젝트 트리 (핵심)

```text
app/
  main.py
  routers/
    notice.py
    ir.py
    voice.py
  schemas/
    notice_schema.py
    ir_schema.py

src/
  common/
    exceptions.py
    types.py
    utils.py
  domain/
    notice/
      __init__.py
      pipeline.py
      document_ai.py
      parser.py
      prompts.py
    ir/
      __init__.py
      pipeline.py
      scorer.py
      feature_extractor.py
      prompts.py
    voice/
      __init__.py
      pipeline.py
      whisper_adapter.py
      whisper_prompt.text
    report/
      __init__.py
      feature_builder.py
      llm_client.py
      qa_generator.py
      report_builder.py
      schemas.py
      score_engine.py
  infrastructure/
    document_ai/
      __init__.py
      client.py
      processor.py
      pipeline.py
    gemini/
      __init__.py
      client.py
    embedding/
      __init__.py
      client.py
    storage/
      __init__.py
      s3_adapter.py
  utils/
    io_utils.py
    pdf_split.py

tests/
  test_notice_parser.py
  test_gemini_live.py
  test_notice_e2e_live.py

tools/
  run_poki_analysis.py
main.py
pytest.ini
```

## 4. Notice 파이프라인 상세

엔트리:

- API: `POST /notice/analyze`
- CLI: `python main.py --mode notice`
- 내부 진입: `src/domain/notice/pipeline.py`

### 4.1 처리 단계

1. Stage 1 (DocAI)
   - 파일: `src/domain/notice/document_ai.py`
   - 입력 PDF를 OCR 처리하여 text/pages/tables 기반 구조화
   - 아티팩트: `*_docai.json`, `*_stage1_structured.json`

2. Stage 2 (Gemini 단일 호출)
   - 파일: `src/domain/notice/parser.py`
   - Gemini 1회 호출로 목표 JSON 스키마 생성
   - 아티팩트: `*_stage2_analysis.json`

3. Final 결과/전략 생성
   - 최종 분석: `*_notice_analysis.json`
   - 전략: `*_strategy.json`
   - 매니페스트: `*_manifest.json`

### 4.2 출력 스키마(상위)

- `notice_name`
- `host_organization`
- `recruitment_type`
- `target_audience`
- `application_period`
- `summary`
- `core_requirements`
- `source_reference`
- `evaluation_structure_type`
- `extraction_confidence`
- `evaluation_criteria`
- `ir_deck_guide`

### 4.3 `evaluation_criteria.points` 보강 로직

`points`는 아래 순서로 복원됩니다.

1. 원본 `points`
2. `raw_points_text`
3. `source_snippet`
4. `pitchcoach_interpretation`
5. DocAI 표(`tables`)에서 항목 매칭
6. 공고문 본문(`notice_text`) 문맥 매칭

추가 정책:

- 총점/합계는 배점 항목에서 제외
- 비평가 항목(안내/자격/접수/가산점 성격) 필터링
- `evaluation_structure_type` 허용값 강제
- `extraction_confidence` 0~1 clamp

## 5. 환경 설정

## 5.1 Python 설치

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 5.2 `.env` 예시

```dotenv
# Document AI
PROJECT_ID=your-gcp-project-id
LOCATION=us
OCR_PROCESSOR_ID=your-processor-id
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json

# Gemini (AI Studio key)
GEMINI_API_KEY=your_ai_studio_key
# optional
GEMINI_MODEL=gemini-2.5-flash
```

## 6. 실행 방법

## 6.1 API 서버

```bash
uvicorn app.main:app --reload
```

헬스체크:

```http
GET /health
```

## 6.2 CLI

```bash
python main.py --mode all
python main.py --mode notice
python main.py --mode ir
```

기본 경로:

- notice 입력: `data/input/sample_notice.pdf`
- ir 입력: `data/input/sample_irdeck.pdf`
- notice 출력: `data/output/notice_analysis`
- ir 출력: `data/output/ir_analysis`

## 6.3 리포트 빌더 실행

```bash
python tools/run_poki_analysis.py
```

## 7. 테스트

`pytest.ini`에 `integration` 마커가 정의되어 있습니다.

### 7.1 단위 테스트(오프라인)

```bash
python -m pytest -q tests/test_notice_parser.py
```

### 7.2 Gemini live smoke (opt-in)

```bash
set -a; source .env; set +a
RUN_LIVE_GEMINI=1 python -m pytest -m integration -q tests/test_gemini_live.py -s
```

### 7.3 Notice E2E live (DocAI + Gemini, opt-in)

```bash
set -a; source .env; set +a
RUN_LIVE_E2E=1 python -m pytest -m integration -q tests/test_notice_e2e_live.py -s
```

검증 포인트:

- final JSON 생성 여부
- top-level key 12개 존재
- `evaluation_criteria` 타입/항목 타입
- 빈 항목 난발(hallucination) 최소 조건 검사

## 8. 트러블슈팅 요약

### 8.1 Gemini 401

- 원인: key 타입/호출 API mismatch
- 조치: AI Studio 키 + `generativelanguage.googleapis.com/v1` 사용

### 8.2 Gemini 404

- 원인: 모델/버전 조합 불일치
- 조치: 모델 fallback 적용 (`gemini-2.5-flash` 우선)

### 8.3 Gemini 400

- 원인: payload 필드 불일치
- 조치: generation config 최소화

### 8.4 `ModuleNotFoundError` (`google.cloud`, `PyPDF2`, `pytest`)

- 원인: 현재 인터프리터에 패키지 누락
- 조치: venv 고정 + `pip install -r requirements.txt`

### 8.5 DNS `gaierror`

- 원인: 인터프리터/세션 DNS 상태 불일치
- 조치: 동일 세션에서 `.env` 로드 후 재실행, 필요시 DNS flush

## 9. 개발 규칙

- API key/서비스 계정 파일은 절대 커밋 금지
- `__pycache__`, `.pytest_cache`, 로컬 venv는 추적 금지
- 기능 변경 커밋과 저장소 위생 커밋을 분리 권장
  - 예: `feat(notice): ...`
  - 예: `chore: ignore caches ...`

