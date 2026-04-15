"""음성 변환 (STT) 파이프라인"""

import os
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def transcribe_audio_to_text(
    audio_url_or_path: str,
    language: str = "ko",  # 한글 (ISO-639-1)
) -> Optional[str]:
    """
    OpenAI Whisper API를 사용하여 음성을 텍스트로 변환합니다.
    
    Args:
        audio_url_or_path: 음성 파일 URL 또는 로컬 경로
        language: 음성 언어 코드 (기본값: ko - 한글)
    
    Returns:
        변환된 텍스트 또는 None (실패 시)
    """
    if not OPENAI_AVAILABLE:
        print("⚠️ OpenAI 라이브러리가 설치되지 않았습니다. pip install openai를 실행하세요.")
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return None
    
    client = OpenAI(api_key=api_key)
    
    try:
        # 로컬 파일인 경우
        if not audio_url_or_path.startswith("http"):
            if not os.path.exists(audio_url_or_path):
                print(f"❌ 음성 파일을 찾을 수 없습니다: {audio_url_or_path}")
                return None
            
            with open(audio_url_or_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                )
            return transcript.text
        
        # URL인 경우 (직접 지원 안 함, 다운로드 필요)
        else:
            print("⚠️ URL 기반 음성은 현재 지원하지 않습니다. 로컬 파일을 사용해주세요.")
            return None
    
    except Exception as e:
        print(f"❌ 음성 변환 중 오류 발생: {e}")
        return None


def validate_audio_file(audio_path: str) -> bool:
    """
    음성 파일의 유효성을 검사합니다.
    
    Args:
        audio_path: 음성 파일 경로
    
    Returns:
        유효한 경우 True, 그렇지 않으면 False
    """
    if not os.path.exists(audio_path):
        print(f"❌ 파일이 없습니다: {audio_path}")
        return False
    
    # 지원하는 형식
    supported_formats = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")
    
    if not audio_path.lower().endswith(supported_formats):
        print(f"❌ 지원하지 않는 형식입니다. 지원 형식: {supported_formats}")
        return False
    
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > 25:
        print(f"❌ 파일 크기가 너무 큽니다 ({file_size_mb:.1f}MB). 최대 25MB까지 지원합니다.")
        return False
    
    return True
