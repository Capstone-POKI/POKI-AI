import os
import json
from typing import Dict, Any, Tuple
from pathlib import Path
import io
from pydub import AudioSegment
import numpy as np
import librosa
from openai import OpenAI
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parents[2]
AUDIO_FILE = BASE_DIR / "data" / "input" / "sample_sound.m4a"
DECK_JSON_PATH = BASE_DIR / "data" / "output" / "asleep_irdeck.json"
PROMPT_PATH = Path(__file__).resolve().with_name("whisper_prompt.text")
SCENARIO = "창업경진대회"

SCENARIO_CONFIG = {
    "VC 데모데이": {
        "target_wpm": (150, 190),
        "importance": {"speed": 0.4, "intonation": 0.3, "clarity": 0.3},
    },
    "창업경진대회": {
        "target_wpm": (130, 170),
        "importance": {"speed": 0.3, "intonation": 0.3, "clarity": 0.4},
    },
    "정부지원·정책 IR": {
        "target_wpm": (110, 150),
        "importance": {"speed": 0.25, "intonation": 0.25, "clarity": 0.5},
    },
    "1분 엘리베이터 피치": {
        "target_wpm": (160, 210),
        "importance": {"speed": 0.5, "intonation": 0.3, "clarity": 0.2},
    },
}

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    IR_PROMPT_TEMPLATE = f.read()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

gemini_client = genai.Client(
    vertexai=True,
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION"),
)


def load_deck_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_deck_context_text(deck_json: Dict[str, Any]) -> str:
    lines = []
    lines.append("[IR 덱 분석 요약]")

    diagnosis = deck_json.get("diagnosis", {})
    missing_sections = diagnosis.get("missing_sections", [])
    if missing_sections:
        lines.append(f"- 빠진 섹션: {', '.join(missing_sections)}")

    slides = deck_json.get("slides", [])
    if slides:
        lines.append("\n[슬라이드별 요약]")
        for slide in slides:
            page = slide.get("page_number")
            section = slide.get("section_type", "")
            contents = slide.get("contents", {})
            summary = contents.get("summary") or contents.get("full_text", "")[:80]
            voice_guide = slide.get("voice_guide", {})
            est_sec = voice_guide.get("estimated_duration_sec")

            line = f"- p.{page} ({section}): {summary}"
            if est_sec:
                line += f" / 권장 발화 시간: {est_sec}초"
            lines.append(line)

    return "\n".join(lines)


def transcribe_audio(path: Path) -> str:
    with path.open("rb") as audio_file:
        result = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return result.text


def extract_audio_features(path: Path) -> Tuple[float, Dict[str, float]]:
    audio = AudioSegment.from_file(str(path), format=path.suffix.replace('.', ''))
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    
    y, sr = librosa.load(wav_io, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    energy = y ** 2
    energy_std = float(np.std(energy))

    thresh = np.percentile(energy, 20)
    silence_ratio = float(np.mean(energy < thresh))

    pitch_mean = 0.0
    pitch_std = 0.0
    pitch_range = 0.0

    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=2048,
        )
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            pitch_mean = float(np.mean(f0_clean))
            pitch_std = float(np.std(f0_clean))
            pitch_range = float(np.max(f0_clean) - np.min(f0_clean))
    except Exception:
        pass

    return duration, {
        "duration": duration,
        "energy_std": energy_std,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_range": pitch_range,
        "silence_ratio": silence_ratio,
    }


def calc_wpm(transcript: str, duration_sec: float) -> float:
    words = transcript.strip().split()
    if duration_sec <= 0 or not words:
        return 0.0
    minutes = duration_sec / 60.0
    return round(len(words) / minutes, 1)


def analyze_with_gemini(
    transcript_text: str,
    scenario: str,
    wpm: float,
    features: Dict[str, float],
    deck_json: Dict[str, Any],
) -> str:
    deck_ctx = build_deck_context_text(deck_json)
    scenario_cfg = SCENARIO_CONFIG.get(scenario, SCENARIO_CONFIG["VC 데모데이"])
    target_low, target_high = scenario_cfg["target_wpm"]
    imp = scenario_cfg["importance"]

    audio_ctx = f"""
[발표 시나리오 설정]

- 현재 분석 대상 발표 상황: {scenario}
- 이 상황에서 권장 말하기 속도 범위: 약 {target_low} ~ {target_high} WPM
- 이 상황에서의 평가 중요도 비중:
  · 속도(speed): {int(imp["speed"] * 100)}%
  · 억양·강조(intonation): {int(imp["intonation"] * 100)}%
  · 명료성(clarity): {int(imp["clarity"] * 100)}%

[음성 분석 요약]

- 실제 측정 말하기 속도(WPM): {wpm}
- 전체 음성 길이(초): {features.get("duration", 0):.1f}
- 피치 평균(pitch_mean, Hz): {features.get("pitch_mean", 0):.2f}
- 피치 표준편차(pitch_std): {features.get("pitch_std", 0):.2f}
- 피치 범위(pitch_range): {features.get("pitch_range", 0):.2f}
- 에너지 표준편차(energy_std): {features.get("energy_std", 0):.4f}
- 침묵 비율(silence_ratio): {features.get("silence_ratio", 0):.3f}

위 정보를 참고하여
- '말하기_속도_WPM' 필드에는 실제 측정값인 {wpm}을 넣으세요.
- '억양_강조_안정성'은 주로 피치 평균/표준편차/범위와 에너지 변동성을 기반으로,
- '문장_명료성'과 '불필요한_말버릇'은 침묵 비율과 속도(WPM)를 참고하여,
- 해당 시나리오의 권장 속도 범위와 중요도(weight)를 고려해
구체적으로 평가하세요.

최종 출력 형식은 반드시 지정된 JSON 구조만 사용하세요.
"""

    prompt_prefix = deck_ctx + "\n\n" + audio_ctx + "\n\n"
    final_prompt = prompt_prefix + IR_PROMPT_TEMPLATE.replace('{{$json["text"]}}', transcript_text)

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=final_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return response.text

def main():
    deck_json = load_deck_json(DECK_JSON_PATH)

    print("🎧 Whisper로 음성 → 텍스트 변환 중...")
    transcript_text = transcribe_audio(AUDIO_FILE)

    print("\n🎼 librosa로 음성 특징 추출 중...")
    duration_sec, features = extract_audio_features(AUDIO_FILE)
    wpm = calc_wpm(transcript_text, duration_sec)

    print("\n Gemini로 IR 발표 분석 중...")
    json_result = analyze_with_gemini(
        transcript_text=transcript_text,
        scenario=SCENARIO,
        wpm=wpm,
        features=features,
        deck_json=deck_json,
    )

    print("\n--- Gemini JSON 결과 ---")
    print(json_result)

if __name__ == "__main__":
    main()