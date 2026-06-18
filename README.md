# POKI-AI

PitchCoach의 공고문(Notice) / IR Deck / Voice / Q&A / Report 분석 엔진입니다.

FastAPI 기반 AI 서버로, NestJS 백엔드(`Pitchcoach-BACK`)와 내부 API Key로 연동됩니다.

## 코드 구조

```
app/
  main.py              # FastAPI 앱 진입점, 미들웨어, 라우터 등록
  env.py               # 환경변수 로딩
  state_store.py       # 인메모리 상태 저장소
  upload.py            # 파일 업로드 유틸
  api/
    health.py          # GET /health
    notices.py         # 공고문 분석 라우터
    decks.py           # IR Deck 분석 라우터
    voice.py           # 음성 분석 라우터
    qa.py              # Q&A 질문 생성/답변 평가 라우터
    report.py          # 종합 리포트 생성 라우터
  models/
    notice_schema.py   # 공고문 Pydantic 스키마
    deck_schema.py     # IR Deck Pydantic 스키마
    qa_schema.py       # Q&A Pydantic 스키마

src/
  domain/
    notice/            # 공고문 파싱·분석 파이프라인
    ir/                # IR Deck 슬라이드 분석·채점 파이프라인
    voice/             # 음성 전사·분석 파이프라인
    qa/                # Q&A 질문 생성·답변 평가
    report/            # 종합 리포트 생성
  infrastructure/
    document_ai/       # Google Document AI 클라이언트
    gemini/            # Gemini API 클라이언트
    embedding/         # 임베딩 클라이언트
    storage/           # S3 파일 저장소 어댑터
  common/              # 공통 예외·타입·유틸
  utils/               # PDF 분할, I/O 유틸

data/
  input/               # 샘플 입력 파일 (sample_notice.pdf, sample_irdeck.pdf, sample_sound.m4a)
  config/              # 파이프라인 설정·루브릭·채점 규칙 JSON
  gt_labels/           # IR Deck 평가용 Ground Truth 라벨 JSON (8개 기업)
  chunks/              # PDF 청크 파일 (RAG용)

tools/                 # E2E 실행 스크립트
tests/                 # 오프라인·라이브 테스트
docs/                  # API 계약서, ERD, 배포 가이드
```

## How to Install

Python 3.11+ 환경에서 실행합니다.

```bash
pip install -r requirements.txt
```

개발 도구(pytest, ruff 등) 포함:

```bash
pip install -r requirements-dev.txt
```

## How to Build (Docker)

```bash
# 이미지 빌드
docker build -t poki-ai .

# 컨테이너 실행 (환경변수 주입)
docker run --env-file .env -p 8000:8000 poki-ai
```

또는 docker-compose 사용:

```bash
docker-compose up --build
```

## How to Run (로컬)

**1. 환경변수 설정 (`.env`)**

```dotenv
# GCP / Document AI
PROJECT_ID=your-gcp-project-id
LOCATION=us
OCR_PROCESSOR_ID=your-docai-processor-id

# Gemini
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

# 내부 인증 (BACK 서버와 동일한 값 사용)
AI_INTERNAL_API_KEY=replace-with-a-long-random-internal-key

# (선택) Voice용 OpenAI Whisper
OPENAI_API_KEY=your-openai-api-key
```

**2. GCP ADC 인증 (Document AI / Vertex AI 사용 시)**

```bash
gcloud auth application-default login
```

**3. 서버 실행**

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**4. 헬스체크**

```bash
curl http://127.0.0.1:8000/health
```

## API 엔드포인트

모든 `/api/` 경로는 `x-internal-api-key` 헤더 인증이 필요합니다.

### Notice (공고문 분석)

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/pitches/{pitch_id}/notices/analyze` | PDF 업로드 → 비동기 분석 시작 (202) |
| GET | `/api/notices/{notice_id}` | 분석 결과 조회 (진행중/완료/실패) |
| PATCH | `/api/notices/{notice_id}` | 분석 결과 수동 수정 |

### IR Deck 분석

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/pitches/{pitch_id}/ir-decks/analyze` | PDF 업로드 → 비동기 분석 시작 (202) |
| GET | `/api/ir-decks/{deck_id}` | 덱 전체 요약 조회 |
| GET | `/api/ir-decks/{deck_id}/slides` | 슬라이드별 분석 결과 조회 |

### Voice (음성 분석)

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/pitches/{pitch_id}/voice/analyze` | 음성 파일 업로드 → 비동기 분석 시작 (202) |
| GET | `/api/voice/{voice_id}` | 분석 결과 조회 |
| GET | `/api/voice/{voice_id}/slides` | 슬라이드별 음성 분석 조회 |

### Q&A

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/pitches/{pitch_id}/qa/questions/generate` | 4-Agent 파이프라인으로 투자자 질문 5개 생성 |
| GET | `/api/pitches/{pitch_id}/qa/sessions/{session_id}` | QA 세션 조회 |
| GET | `/api/pitches/{pitch_id}/questions` | 질문 목록 조회 (BACK 폴링용) |
| POST | `/api/pitches/{pitch_id}/qa/sessions/{session_id}/answers` | 답변 제출 |
| GET | `/api/pitches/{pitch_id}/qa/sessions/{session_id}/answers/{answer_id}` | 답변 결과 조회 |
| POST | `/api/questions/{question_id}/answers/analyze` | 음성/텍스트 답변 Gemini 평가 |

### Report (종합 리포트)

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/pitches/{pitch_id}/report/generate` | IR·음성·공고·Q&A 결과 통합 → 종합 리포트 생성 |

### 에러 응답 형식

```json
{ "error": "ERROR_CODE", "message": "..." }
```

## How to Test

**오프라인 단위 테스트:**

```bash
python -m pytest -q
```

**라이브 통합 테스트 (GCP 인증 및 실제 API Key 필요):**

```bash
# Notice 분석
python -m pytest tests/test_notice_e2e_live.py -v

# IR Deck 분석
python -m pytest tests/test_ir_e2e_live.py -v

# IR Deck 배치 평가
python -m pytest tests/test_ir_batch_live.py -v
```

**E2E 전체 플로우 실행 (서버 구동 후):**

```bash
./tools/run_local_notice_ir_e2e.sh
```

기본 입력: `data/input/sample_notice.pdf`, `data/input/sample_irdeck.pdf`

출력: `/tmp/poki_e2e/notice_result.json`, `/tmp/poki_e2e/ir_summary.json`, `/tmp/poki_e2e/ir_slides.json`

## Sample Data

`data/input/` 폴더에 테스트용 샘플 파일이 포함되어 있습니다:

- `sample_notice.pdf` — 샘플 공고문 PDF
- `sample_irdeck.pdf` — 샘플 IR Deck PDF
- `sample_sound.m4a` — 샘플 발표 음성

`data/gt_labels/` 폴더에는 IR Deck 평가 정확도 측정을 위한 Ground Truth 라벨(8개 기업)이 JSON 형식으로 포함되어 있습니다.

`data/config/` 폴더에는 파이프라인 설정, 루브릭, 채점 규칙 JSON이 포함되어 있습니다.

## 사용 오픈소스

| 분류 | 라이브러리 | 용도 |
|------|-----------|------|
| API 프레임워크 | FastAPI, Uvicorn | HTTP 서버 |
| AI / LLM | google-genai (Gemini 2.5) | 문서 분석·채점·리포트 생성 |
| AI / LLM | openai (Whisper) | 음성 전사 |
| ML | torch, transformers | 로컬 임베딩·모델 추론 |
| GCP | google-cloud-documentai | PDF OCR |
| GCP | google-cloud-storage, google-cloud-aiplatform | 파일 저장·Vertex AI |
| PDF | pdf2image, PyPDF2, pillow | PDF 처리 |
| 음성 | librosa, pydub | 음성 전처리 |
| 클라우드 | boto3 | S3 파일 저장 |
| 유틸 | python-dotenv, requests, PyYAML | 설정·HTTP |

## 주요 구현 사항

- DB 미연결: 상태는 인메모리 저장소(`state_store.py`)에 유지되며 서버 재시작 시 초기화됩니다. QA 세션은 파일 캐시에도 저장되어 Docker 재시작 후 복원됩니다.
- Voice 라우터는 `pydub`, `librosa`, `openai` 의존성이 없으면 import 실패할 수 있어 optional 로딩 처리되어 있습니다.
- 모든 `/api/` 경로는 `x-internal-api-key` 헤더를 통해 BACK 서버와 상호 인증합니다.

## docs 읽는 순서

1. `docs/README.md`
2. `docs/NOTICE_API_DB_MAPPING_V4.md`
3. `docs/IR_DECK_B_V1_CONTRACT.md`
4. `docs/IR_DECK_THREE_LAYER_SCHEMA.md`
5. `docs/ERD_V3_BACKEND_DEV_PLAN.md`
6. `docs/DEPLOY_GUIDE.md`
