import os
import json
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import io
from pydub import AudioSegment
import numpy as np
import librosa
from openai import OpenAI

from src.infrastructure.gemini.client import GeminiJSONClient

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

# [신규] pitch_type → 시나리오 이름 매핑 (API 요청의 pitch_type 파라미터 변환용)
PITCH_TYPE_TO_SCENARIO = {
    "ELEVATOR": "1분 엘리베이터 피치",
    "VC_DEMO": "VC 데모데이",
    "GOVERNMENT": "정부지원·정책 IR",
    "COMPETITION": "창업경진대회",
}

# [신규] IR 덱 section_type(영문) → 한글 카테고리명 변환 (슬라이드별 분석 결과 표시용)
SECTION_TYPE_TO_KO = {
    "COVER": "표지",
    "PROBLEM": "문제 정의",
    "SOLUTION": "솔루션",
    "PRODUCT": "제품",
    "MARKET": "시장 분석",
    "BUSINESS_MODEL": "비즈니스 모델",
    "COMPETITION": "경쟁 분석",
    "TRACTION": "실적",
    "TEAM": "팀 소개",
    "FINANCE": "자금 계획",
    "ASK": "요청사항",
    "GTM": "시장 진입 전략",
    "OTHER": "기타",
}

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    IR_PROMPT_TEMPLATE = f.read()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = GeminiJSONClient()


def load_deck_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_voice_context_to_deck_json(voice_context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(voice_context, dict):
        return {}

    ir_deck = voice_context.get("ir_deck")
    if not isinstance(ir_deck, dict):
        return voice_context if isinstance(voice_context.get("slides"), list) else {}

    if any(key in ir_deck for key in ("deck_score", "criteria_scores", "presentation_guide")):
        normalized_slides: List[Dict[str, Any]] = []
        for slide in ir_deck.get("slides", []) or []:
            if not isinstance(slide, dict):
                continue
            slide_number = int(slide.get("slide_number", 0) or 0)
            category = str(slide.get("category", "") or "")
            content_summary = str(slide.get("content_summary", "") or "")
            normalized_slides.append(
                {
                    "page_number": slide_number,
                    "section_type": category,
                    "contents": {
                        "summary": content_summary,
                        "full_text": content_summary,
                    },
                    "voice_guide": {},
                }
            )

        return {
            "deck_score": ir_deck.get("deck_score", {}),
            "criteria_scores": ir_deck.get("criteria_scores", []),
            "presentation_guide": ir_deck.get("presentation_guide", {}),
            "slides": normalized_slides,
        }

    return ir_deck


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
            page = slide.get("page_number") or slide.get("slide_number")
            section = slide.get("section_type", "") or slide.get("category", "")
            contents = slide.get("contents", {})
            if not isinstance(contents, dict):
                contents = {}
            summary = (
                contents.get("summary")
                or slide.get("content_summary")
                or contents.get("full_text", "")[:80]
            )
            voice_guide = slide.get("voice_guide", {})
            est_sec = voice_guide.get("estimated_duration_sec")

            line = f"- p.{page} ({section}): {summary}"
            if est_sec:
                line += f" / 권장 발화 시간: {est_sec}초"
            lines.append(line)

    return "\n".join(lines)


# [변경] 기존: str 반환 / 변경 후: (text, words) 튜플 반환
# verbose_json + timestamp_granularities=["word"] 추가 → 슬라이드별 발화 분리에 필요
WHISPER_MAX_BYTES = 25 * 1024 * 1024  # OpenAI Whisper API hard limit


def transcribe_audio(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    file_size = path.stat().st_size
    if file_size > WHISPER_MAX_BYTES:
        raise ValueError(
            f"음성 파일이 너무 큽니다 ({file_size // 1024 // 1024}MB). "
            f"OpenAI Whisper API는 최대 25MB까지 지원합니다."
        )
    with path.open("rb") as audio_file:
        result = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    text = result.text
    words = [
        {"word": w.word, "start": float(w.start), "end": float(w.end)}
        for w in (result.words or [])
    ]
    return text, words


# [신규] 프론트에서 받은 slide_timestamps 기준으로 단어 리스트를 슬라이드별로 분할
def segment_by_slides(
    words: List[Dict[str, Any]],
    slide_timestamps: List[Dict[str, Any]],
) -> Dict[int, str]:
    """Segment word transcription by slide timestamps."""
    result: Dict[int, str] = {}
    for ts in slide_timestamps:
        num = int(ts["slide_number"])
        start = float(ts["start_timestamp"])
        end = float(ts["end_timestamp"])
        slide_words = [w["word"] for w in words if w["start"] >= start and w["start"] < end]
        result[num] = " ".join(slide_words).strip()
    return result


def extract_audio_features(path: Path) -> Tuple[float, Dict[str, float]]:
    audio = AudioSegment.from_file(str(path), format=path.suffix.replace(".", ""))
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


# [신규] WPM이 시나리오별 권장 범위 대비 얼마나 적합한지 점수화 (8번째 세부 기준 "시간 적합성")
def calc_time_suitability_score(wpm: float, scenario: str) -> int:
    cfg = SCENARIO_CONFIG.get(scenario, SCENARIO_CONFIG["VC 데모데이"])
    low, high = cfg["target_wpm"]
    if low <= wpm <= high:
        return 90
    diff = min(abs(wpm - low), abs(wpm - high))
    if diff <= 15:
        return 75
    if diff <= 30:
        return 60
    return 45


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

    deck_json_str = json.dumps(deck_json, ensure_ascii=False, indent=2)
    prompt_prefix = deck_ctx + "\n\n" + audio_ctx + "\n\n"
    final_prompt = (
        prompt_prefix
        + IR_PROMPT_TEMPLATE
        .replace("{{deck_json}}", deck_json_str)
        .replace('{{$json["text"]}}', transcript_text)
    )

    if not gemini_client.model:
        raise RuntimeError("Gemini model is not available")
    result = gemini_client.generate_json(final_prompt, temperature=0.2)
    return json.dumps(result, ensure_ascii=False)


# [신규] 슬라이드별 발화 텍스트를 Gemini에 한 번에 보내 슬라이드 단위 피드백 생성
def analyze_slides_with_gemini(
    slide_transcriptions: Dict[int, str],
    slide_timestamps: List[Dict[str, Any]],
    deck_json: Dict[str, Any],
) -> List[Dict[str, Any]]:
    deck_slides_map: Dict[int, Dict[str, Any]] = {}
    for s in deck_json.get("slides", []):
        pg = s.get("page_number")
        if pg:
            deck_slides_map[int(pg)] = s

    slides_input = []
    for ts in slide_timestamps:
        num = int(ts["slide_number"])
        duration = float(ts["end_timestamp"]) - float(ts["start_timestamp"])
        transcription = slide_transcriptions.get(num, "")
        deck_slide = deck_slides_map.get(num, {})
        section = deck_slide.get("section_type", "")
        section_ko = SECTION_TYPE_TO_KO.get(section, section or "기타")
        summary = (deck_slide.get("contents", {}) or {}).get("summary", "") or ""

        slides_input.append({
            "slide_number": num,
            "category": section_ko,
            "duration_seconds": round(duration, 1),
            "transcription": transcription,
            "deck_summary": summary[:200],
        })

    if not slides_input:
        return []

    prompt = f"""당신은 IR 피칭 코치입니다. 아래 각 슬라이드별 발표 데이터를 분석하여 피드백을 제공하세요.

[입력 데이터]
{json.dumps(slides_input, ensure_ascii=False, indent=2)}

[출력 형식]
반드시 아래 구조의 JSON 배열만 반환하세요. JSON 외의 텍스트는 절대 포함하지 마세요.
[
  {{
    "slide_number": 1,
    "score": 85,
    "content_summary": "발표자가 실제로 말한 내용 기반 1~2문장 요약",
    "detailed_feedback": "이 슬라이드에서의 발표 방식에 대한 구체적 피드백 (2~4문장, ~니다 톤)",
    "strengths": ["강점 1", "강점 2"],
    "improvements": ["개선점 1", "개선점 2"]
  }}
]

[작성 규칙]
- score: 0~100 정수. 발표 내용의 완성도, 전달력, 덱 내용 커버 여부 종합 평가
- transcription이 비어 있으면 score를 30 이하로 낮게 평가하고 피드백에 "해당 슬라이드에서 발표 내용이 없거나 매우 짧습니다"를 포함
- content_summary는 발표자가 실제 말한 내용 기반으로 작성 (덱 summary 그대로 복사 금지)
- strengths, improvements: 각 1~3개, 서술형, ~니다 톤
- 모든 문장은 ~니다 톤
"""

    if not gemini_client.model:
        return []
    parsed = gemini_client.generate_json(prompt, temperature=0.2)
    if isinstance(parsed, dict):
        for key in ("slides", "result", "data", "items"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        return []

    ts_map = {int(ts["slide_number"]): ts for ts in slide_timestamps}
    input_map = {s["slide_number"]: s for s in slides_input}

    result = []
    for item in parsed:
        num = int(item.get("slide_number", 0))
        ts = ts_map.get(num, {})
        duration = float(ts.get("end_timestamp", 0)) - float(ts.get("start_timestamp", 0))
        total_sec = int(duration)
        if total_sec >= 60:
            m, s = divmod(total_sec, 60)
            display = f"{m}분 {s}초" if s > 0 else f"{m}분"
        else:
            display = f"{total_sec}초"

        slide_input = input_map.get(num, {})
        result.append({
            "slide_number": num,
            "category": slide_input.get("category", "기타"),
            "score": int(item.get("score", 50)),
            "start_timestamp": float(ts.get("start_timestamp", 0)),
            "end_timestamp": float(ts.get("end_timestamp", 0)),
            "duration_seconds": round(duration, 1),
            "duration_display": display,
            "content_summary": str(item.get("content_summary", "")),
            "detailed_feedback": str(item.get("detailed_feedback", "")),
            "strengths": [str(s) for s in item.get("strengths", [])],
            "improvements": [str(s) for s in item.get("improvements", [])],
        })

    return result


# [신규] API 진입점 — 기존 함수들을 순서대로 호출하고 전체 결과를 dict로 반환
# slide_timestamps 있으면 슬라이드별 분석도 함께 실행
def analyze(
    audio_path: Path,
    voice_context: Dict[str, Any],
    scenario: str,
    slide_timestamps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    deck_json = _normalize_voice_context_to_deck_json(voice_context)
    transcript_text, words = transcribe_audio(audio_path)
    duration_sec, features = extract_audio_features(audio_path)
    wpm = calc_wpm(transcript_text, duration_sec)

    overall_json = analyze_with_gemini(
        transcript_text=transcript_text,
        scenario=scenario,
        wpm=wpm,
        features=features,
        deck_json=deck_json,
    )
    overall = json.loads(overall_json)
    overall["시간_적합성_점수"] = calc_time_suitability_score(wpm, scenario)

    slides: List[Dict[str, Any]] = []
    if slide_timestamps:
        slide_transcriptions = segment_by_slides(words, slide_timestamps)
        slides = analyze_slides_with_gemini(slide_transcriptions, slide_timestamps, deck_json)

    return {
        "transcript": transcript_text,
        "wpm": wpm,
        "duration_seconds": round(duration_sec, 1),
        "overall": overall,
        "slides": slides,
    }


def main():
    deck_json = load_deck_json(DECK_JSON_PATH)
    print("🎧 Whisper로 음성 → 텍스트 변환 중...")
    result = analyze(AUDIO_FILE, deck_json, SCENARIO)
    print(f"\n전사 텍스트: {result['transcript'][:200]}...")
    print(f"WPM: {result['wpm']}, 길이: {result['duration_seconds']}초")
    print("\n--- 전체 분석 결과 ---")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
