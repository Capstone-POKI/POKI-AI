import json
import os
from typing import Any

from google import genai
from google.genai import types

# --- 1) 서비스 계정 경로 체크 ---
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not CREDENTIALS_PATH:
    raise RuntimeError(
        "환경변수 GOOGLE_APPLICATION_CREDENTIALS 가 설정되지 않았습니다.\n"
        "서비스 계정 JSON 파일 경로를 환경변수로 등록해주세요."
    )

# 서비스 계정 JSON에서 project_id 읽기
with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
    sa = json.load(f)

PROJECT_ID = sa["project_id"]
LOCATION = "us-central1"

# --- 2) Vertex AI 모드로 GenAI Client 생성 ---
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

MODEL_NAME = "gemini-2.0-flash"


def call_llm(prompt: str) -> str:
    """
    Vertex AI Gemini 호출 래퍼
    """
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=2048,
            temperature=0.4,
        ),
    )

    # google-genai 응답 객체는 .text에 최종 텍스트가 들어있음
    return response.text


def safe_json_parse(text: str) -> Any:
    """
    LLM이 준 JSON 형식 문자열을 안전하게 파싱
    ```json ...``` 감싸진 경우도 처리
    """
    cleaned = text.strip()

    # ```json ... ``` 형태 처리
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        lines = cleaned.splitlines()
        if lines and lines[0].lower().strip() in ("json", "javascript"):
            cleaned = "\n".join(lines[1:]).strip()

    try:
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"⚠ JSON 파싱 실패: {e}\n원본 출력:\n{text}")