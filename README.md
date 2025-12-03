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
