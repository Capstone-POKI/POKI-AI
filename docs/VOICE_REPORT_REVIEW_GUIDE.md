# Voice / Report 코드 리뷰 가이드

이 문서는 `voice`, `report` 담당자가 실제 코드 파일을 어떤 순서로 읽어야 하는지 정리합니다.

## 1. Voice

## 1-1) 엔트리
- API: `app/routers/voice.py`
  - `POST /voice/analyze`
  - 현재 파라미터 없는 데모형 엔드포인트
- 도메인 엔트리: `src/domain/voice/pipeline.py`
  - `run_voice_analysis = whisper_adapter.main`

## 1-2) 핵심 구현
- `src/domain/voice/whisper_adapter.py`
  - OpenAI Whisper STT (`openai`)
  - 음성 feature 추출 (`librosa`, `numpy`, `pydub`)
  - Gemini 기반 코칭 분석 (`google.genai`)
  - 현재는 파일 경로 상수(`sample_sound.m4a`, `asleep_irdeck.json`) 기반 실행

## 1-3) 현재 제약
- API 입력 스키마/응답 스키마가 미정형(고정 파일 기반)
- Voice 라우터는 optional 로딩(`app/main.py`) 처리됨
- production-ready API라기보다 실험 파이프라인 상태

## 1-4) 의존성
- `openai`
- `librosa`
- `pydub`

## 2. Report

## 2-1) 엔트리
- CLI: `tools/run_poki_analysis.py`
- 내부 엔트리: `src/domain/report/report_builder.py::build_final_report`

## 2-2) 구성 파일
- `feature_builder.py`: 덱/음성 입력 결합 컨텍스트 생성
- `score_engine.py`: 가중 점수 계산
- `qa_generator.py`: 예상 Q&A 생성
- `llm_client.py`: Vertex Gemini 호출
- `schemas.py`: Pydantic 데이터 구조

## 2-3) 주의사항
- `llm_client.py`는 import 시점에 `GOOGLE_APPLICATION_CREDENTIALS`를 강제 확인
- 서비스계정 JSON에서 project_id를 읽는 방식이라 로컬 환경변수 불일치 시 바로 실패

## 3. 리뷰 체크포인트

1. API 엔드포인트와 실제 입력/출력의 명세 일치 여부  
2. 고정 경로 상수 제거 가능성(voice)  
3. LLM 호출 실패 시 fallback/에러 처리 일관성  
4. report 스키마와 프론트 소비 스키마 분리 여부  
5. 인증/권한/개인정보 처리 경계 정의

## 4. 권장 리팩토링 순서

1. Voice API에 request/response schema 도입  
2. Whisper/Gemini 호출을 서비스 계층으로 분리  
3. Report 입력을 DB/스토리지 기반으로 일반화  
4. `llm_client.py`의 import-time side effect 제거(런타임 초기화로 이동)
