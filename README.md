# POKI-AI

PitchCoach의 공고문(`Notice`) / IR Deck / Voice / Report 분석 엔진 코드베이스입니다.

이 레포는 현재:
- FastAPI API 레이어(`app/`)
- 도메인 파이프라인(`src/domain/`)
- 외부 연동(`src/infrastructure/`)
으로 분리되어 있습니다.

## 빠른 시작
1. 의존성 설치
```bash
pip install -r requirements.txt
```

2. 환경변수 설정 (`.env`)
```dotenv
# GCP / DocAI
PROJECT_ID=your-gcp-project-id
LOCATION=us
OCR_PROCESSOR_ID=your-docai-processor-id

# Gemini
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

# (선택) Voice용 OpenAI Whisper
OPENAI_API_KEY=your-openai-api-key
```

3. ADC 인증 (DocAI/Gemini Vertex 사용 시)
```bash
gcloud auth application-default login
```

4. 서버 실행
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

5. 헬스체크
```bash
curl http://127.0.0.1:8000/health
```

## API 엔드포인트 (현재 코드 기준)
- Notice
  - `POST /pitches/{pitch_id}/notice`
  - `GET /notices/{notice_id}`
  - `PATCH /notices/{notice_id}`
- IR Deck
  - `POST /api/pitches/{pitch_id}/ir-decks/analyze`
  - `GET /api/ir-decks/{deck_id}`
  - `GET /api/ir-decks/{deck_id}/slides`
- Voice
  - `POST /voice/analyze` (현재 입력 파라미터 없는 데모형 엔드포인트)

에러 응답은 공통적으로 평탄 포맷을 사용합니다.
```json
{ "error": "ERROR_CODE", "message": "..." }
```

## E2E 실행 (Notice + IR)
서버 실행 후:
```bash
./tools/run_local_notice_ir_e2e.sh
```

기본 입력:
- `data/input/sample_notice.pdf`
- `data/input/sample_irdeck.pdf`

출력:
- `/tmp/poki_e2e/notice_result.json`
- `/tmp/poki_e2e/ir_summary.json`
- `/tmp/poki_e2e/ir_slides.json`

## 코드 구조
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
  domain/
    notice/
    ir/
    voice/
    report/
  infrastructure/
    document_ai/
    gemini/
    embedding/
    storage/
  common/
  utils/

tools/
tests/
docs/
```

## docs 읽는 순서
문서가 많아 보이지만 아래 순서로 보면 빠릅니다.

1. `docs/README.md`  
2. `docs/NOTICE_API_DB_MAPPING_V4.md`  
3. `docs/IR_DECK_B_V1_CONTRACT.md`  
4. `docs/IR_DECK_THREE_LAYER_SCHEMA.md`  
5. `docs/VOICE_REPORT_REVIEW_GUIDE.md`  
6. `docs/ERD_V3_BACKEND_DEV_PLAN.md` (파일명은 v3지만 내용은 v4 기준 현황 반영)

## 테스트
기본 오프라인 테스트:
```bash
python -m pytest -q
```

선택 실행:
- Notice live: `tests/test_notice_e2e_live.py`
- IR live: `tests/test_ir_e2e_live.py`
- IR 배치: `tests/test_ir_batch_live.py`

## 현재 구현 주의사항
- DB는 아직 미연결이며, API 라우터 내부 메모리 저장소로 상태를 유지합니다.
- 라우터 재시작 시 상태가 초기화됩니다.
- Voice 라우터는 의존성(`pydub`, `librosa`, `openai`)이 없으면 import 실패할 수 있어 `app.main`에서 optional 로딩 처리되어 있습니다.
