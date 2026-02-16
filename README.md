# POKI-AI

POKI-AI는 공고문/IR/음성 분석을 수행하는 AI 서비스 레포입니다.  
현재 구조는 `app(API) / domain(비즈니스 로직) / infrastructure(외부 연동) / common(공통)` 레이어로 분리되어 있습니다.

## Architecture

- `app/`: FastAPI 진입점, 라우터, 요청 스키마
- `src/domain/`: 도메인 파이프라인 (notice, ir, voice, report)
- `src/infrastructure/`: 외부 API/SDK 연동 (Gemini, Document AI, Embedding)
- `src/common/`: 공통 타입, 예외, 유틸
- `tools/`: CLI/실행 보조 스크립트
- `data/`: 현재 기본 입출력 경로
- `data_dev/`: 개발/실험용 데이터 영역

## Project Tree

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
      pipeline.py
      document_ai.py
      parser.py
      prompts.py
    ir/
      pipeline.py
      scorer.py
      feature_extractor.py
      prompts.py
    voice/
      pipeline.py
      whisper_adapter.py
    report/
      report_builder.py
      qa_generator.py
      schemas.py
  infrastructure/
    gemini/client.py
    document_ai/
      client.py
      processor.py
      pipeline.py
    embedding/client.py
    storage/s3_adapter.py
  utils/
    io_utils.py
    pdf_split.py
tools/
  run_poki_analysis.py
main.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

필수 환경변수(`.env`):

- `PROJECT_ID`
- `LOCATION`
- `OCR_PROCESSOR_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`
- (옵션) `OPENAI_API_KEY` (음성 파이프라인/리포트 연계 시)

## Run

### 1) FastAPI 서버 실행

```bash
uvicorn app.main:app --reload
```

헬스체크:

```bash
GET /health
```

### 2) CLI 실행 (notice/ir/all)

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

### 3) 최종 리포트 엔진 실행

```bash
python tools/run_poki_analysis.py
```

## API

### POST `/notice/analyze`

요청(`NoticeAnalyzeRequest`):

```json
{
  "notice_pdf": "data/input/sample_notice.pdf",
  "output_dir": "data/output/notice_analysis",
  "no_gemini": false
}
```

동작:

- Document AI OCR
- Gemini 단일 호출 분석
- `*_notice_analysis.json`, `*_strategy.json`, `*_manifest.json` 생성

### POST `/ir/analyze`

요청(`IRAnalyzeRequest`):

```json
{
  "ir_pdf": "data/input/sample_irdeck.pdf",
  "output_dir": "data/output/ir_analysis",
  "strategy_json": null,
  "notice_output_dir": "data/output/notice_analysis",
  "no_auto_strategy": false,
  "no_chunking": false
}
```

동작:

- OCR (단일/청크)
- Gemini 기반 IR 분석 결과 생성(`*_final.json`)

### POST `/voice/analyze`

동작:

- `src/domain/voice/whisper_adapter.py` 실행
- 현재는 파일 기반 분석 스크립트를 API로 감싼 형태

## Domain Pipelines

### Notice

- 파일: `src/domain/notice/pipeline.py`
- 흐름: Stage1 OCR -> Stage2 Gemini parse -> strategy 생성/저장

### IR

- 파일: `src/domain/ir/pipeline.py`
- 흐름: OCR -> scorer(exporter) -> final JSON

### Report

- 파일: `src/domain/report/report_builder.py`
- `qa_generator.py`를 통해 심사 Q&A 제안 생성

### Voice

- 파일: `src/domain/voice/whisper_adapter.py`
- Whisper/오디오 특성 분석 + Gemini 평가

## Notes

- 현재 레포는 구조적으로 서비스 레이어 분리를 완료한 상태입니다.
- `data/`는 기존 실행 경로와 호환을 위해 유지되고 있습니다.
- 운영 환경 전환 시 `data_dev/` 기준 경로로 추가 정리 가능합니다.
# 📘 POKI-AI

## 1. 프로젝트 개요 (Overview)
- 프로젝트 한줄 소개
- 주요 기능 요약

---

## 2. 전체 폴더 구조 (Project Structure)
- <트리 삽입>

---

## 3. 기술 구성 요소 (Tech Stack)
- Document AI
- LayoutLM
- Gemini LLM
- Whisper
- 기타 라이브러리

---

## 4. 설치 방법 (Installation)
- 저장소 클론
- 패키지 설치
- 가상환경 설정 (옵션)

---

## 5. 환경 변수 설정 (.env)
- GOOGLE_APPLICATION_CREDENTIALS
- GEMINI_API_KEY
- 기타 필요한 값

---

## 6. 실행 방법 (How to Run)

### 6.1 문서 분석 파이프라인 실행
### 6.2 음성 분석 파이프라인 실행
### 6.3 전체 파이프라인 실행

---

## 7. 파이프라인 구조 (Pipeline Flow)

### 7.1 문서 분석 흐름
- Document AI OCR
- Chunk 처리 및 병합
- LayoutLM 구조 분석
- Gemini 평가/진단 생성
- 최종 JSON 출력

### 7.2 음성 분석 흐름
- Whisper 음성 변환
- (옵션) Gemini 분석

---

## 8. 입출력 구조 (Input / Output)

### 입력 폴더 (`data/input`)
- PDF 파일
- 음성 파일

### 출력 폴더 (`data/output`)
- OCR 결과
- LayoutLM 결과
- 최종 분석 JSON

---

## 9. 모듈 설명 (Modules)

### docs_analysis
- document_ai
- layoutlm
- llm
- post_processing
- __main__.py

### voice_analysis
- whisper 모듈
- 음성 파이프라인

### utils
- 공통 함수 모음

---

## 10. 향후 확장 계획 (Future Work)
- 고도화 기능
- 웹 UI 연결
- 추가 모델 도입

---

## 11. 라이선스 (License)
- MIT / Apache 2.0 등
