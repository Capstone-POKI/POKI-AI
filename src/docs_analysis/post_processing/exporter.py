import json
import re
from typing import Dict, List, Optional

from src.docs_analysis.llm.gemini_client import GeminiAnalyst
from src.docs_analysis.llm.prompts.ir_analysis_prompt import build_ir_analysis_prompt


DEFAULT_REQUIRED_SECTIONS = {
    "problem", "solution", "market", "business_model",
    "competition", "growth", "team", "finance"
}


def estimate_speech_duration(text: str) -> int:
    clean_text = re.sub(r'\s+', '', text)
    length = len(clean_text)
    if length == 0:
        return 0
    return int(length / 3.5) + 2


def analyze_visual_balance(text_len: int, image_count: int) -> Dict:
    score = 50
    status = "balanced"

    if image_count == 0:
        score -= 30
        status = "text_heavy"
    elif image_count > 3:
        score += 10

    if text_len > 600:
        score -= 20
        status = "text_heavy"
    elif text_len > 400:
        score -= 10
    elif text_len < 50 and image_count > 0:
        score += 20
        status = "image_centric"
    elif text_len < 100 and image_count > 2:
        status = "image_centric"

    return {"score": max(0, min(100, score)), "status": status}


def generate_voice_guide(text_len: int, duration: int) -> Dict:
    advice = "적절한 속도로 발표하세요."
    if duration > 100:
        advice = f"내용이 많습니다({duration}초 예상). 핵심 키워드 위주로 요약 발언이 필요합니다."
    elif duration > 60:
        advice = f"내용이 많은 편입니다({duration}초 예상). 핵심 포인트를 강조하며 진행하세요."
    elif duration < 10:
        advice = "내용이 짧습니다. 부연 설명을 덧붙여 여유 있게 진행하세요."

    return {
        "estimated_duration_sec": duration,
        "pacing_advice": advice
    }


def extract_slide_contents(docai_result: Dict, pages: List[Dict]) -> List[Dict]:
    slides_data = []
    detected_sections = docai_result.get("detected_sections", [])
    section_map = {s['page']: s['section'] for s in detected_sections}

    for idx, page in enumerate(pages):
        page_num = idx + 1
        section_type = section_map.get(page_num, "unknown")

        full_text = ""
        for block in page.get("blocks", []):
            layout = block.get("layout", {})
            for segment in layout.get("textAnchor", {}).get("textSegments", []):
                start = int(segment.get("startIndex", 0))
                end = int(segment.get("endIndex", 0))
                full_text += docai_result.get("text", "")[start:end]

        text_len = len(full_text)
        image_count = len(page.get("image", []))

        est_duration = estimate_speech_duration(full_text)
        visual_analysis = analyze_visual_balance(text_len, image_count)
        voice_guide = generate_voice_guide(text_len, est_duration)

        slides_data.append({
            "page_number": page_num,
            "section_type": section_type,
            "contents": {
                "full_text": full_text,
                "summary": full_text[:100] + "..." if len(full_text) > 100 else full_text,
                "char_count": text_len,
                "image_count": image_count
            },
            "analysis": {
                "visual_balance": visual_analysis,
                "readability": "Low" if text_len > 800 else ("Medium" if text_len > 400 else "High")
            },
            "voice_guide": voice_guide,
            "design_feedback": []
        })

    return slides_data


def analyze_with_gemini(
    gemini: GeminiAnalyst,
    slides_data: List[Dict],
    pitch_strategy: Optional[Dict],
    doc_type: str
) -> Dict:

    print("\n 문서 분석 진행 중...")

    if not gemini.model:
        print("⚠ Gemini 모델 없음 → 기본 분석으로 전환")
        return _get_fallback_analysis(slides_data, pitch_strategy)

    slides_summary = []
    for slide in slides_data:
        slides_summary.append({
            "page": slide["page_number"],
            "section": slide["section_type"],
            "text_preview": slide["contents"]["summary"],
            "char_count": slide["contents"]["char_count"],
            "image_count": slide["contents"]["image_count"],
            "duration_sec": slide["voice_guide"]["estimated_duration_sec"]
        })

    if pitch_strategy:
        strategy_context = f"""
[심사 전략 정보 (공고문 기반)]
- 피칭 유형: {pitch_strategy.get('type', 'N/A')}
- 핵심 평가 기준: {pitch_strategy.get('focus_point', 'N/A')}
- 필수 섹션: {', '.join(pitch_strategy.get('required_sections', []))}
- 평가 배점표: {pitch_strategy.get('evaluation_criteria', [])}
- 킬러 질문: {pitch_strategy.get('killer_question', 'N/A')}
"""
    else:
        strategy_context = "[심사 전략 정보 없음 – 범용 분석 모드]"

    total_duration = sum(s['voice_guide']['estimated_duration_sec'] for s in slides_data)

    prompt = build_ir_analysis_prompt(
        strategy_context=strategy_context,
        slides_summary=slides_summary,
        doc_type=doc_type,
        total_duration=total_duration
    )

    try:
        response = gemini.model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.3
            }
        )

        analysis_result = json.loads(response.text)
        print("Gemini 분석 완료!")
        return analysis_result

    except Exception as e:
        print("Gemini 분석 실패:", e)
        return _get_fallback_analysis(slides_data, pitch_strategy)


def _get_fallback_analysis(slides_data: List[Dict], pitch_strategy: Optional[Dict]) -> Dict:
    print("⚙ 기본 규칙 기반 분석 수행")

    heavy_slides = [s['page_number'] for s in slides_data if s['voice_guide']['estimated_duration_sec'] > 100]
    light_slides = [s['page_number'] for s in slides_data if s['contents']['char_count'] < 30]

    return {
        "diagnosis": {
            "overall_completeness": 50,
            "missing_sections": [],
            "logic_flow_issues": [],
            "priority_issues": ["자동 분석 실패 - 수동 검토 필요"]
        },
        "content_quality": {
            "text_density_avg": sum(s['contents']['char_count'] for s in slides_data) // len(slides_data),
            "visual_balance_avg": 50,
            "slides_too_heavy": heavy_slides,
            "slides_too_light": light_slides
        },
        "slide_feedback": [],
        "recommendations": {
            "critical": [],
            "important": [],
            "suggested": [{"issue": "AI 분석 실패", "action": "문서를 수동 검토", "priority": 3}]
        }
    }



def merge_llm_feedback_to_slides(slides_data: List[Dict], slide_feedback: List[Dict]) -> List[Dict]:
    feedback_map = {item['page']: item['feedbacks'] for item in slide_feedback}

    for slide in slides_data:
        page_num = slide["page_number"]
        if page_num in feedback_map:
            slide["design_feedback"] = feedback_map[page_num]

    return slides_data



def export_final_json(
    docai_result: Dict,
    layoutlm_result: Dict,
    output_path: str,
    pitch_strategy: Optional[Dict] = None
) -> Dict:

    print("\n" + "=" * 80)
    print("📦 [V4 - LLM Powered] 최종 분석 JSON 생성")
    print("=" * 80)

    gemini = GeminiAnalyst()

    pages = docai_result.get("pages", [])
    doc_type = layoutlm_result.get("doc_type", "unknown")

    slides_data = extract_slide_contents(docai_result, pages)

    llm_analysis = analyze_with_gemini(
        gemini, slides_data, pitch_strategy, doc_type
    )

    slides_data = merge_llm_feedback_to_slides(
        slides_data,
        llm_analysis.get("slide_feedback", [])
    )

    total_duration = sum(s['voice_guide']['estimated_duration_sec'] for s in slides_data)

    if pitch_strategy:
        strategy_info = {
            "type": pitch_strategy.get("type", "Unknown"),
            "focus_point": pitch_strategy.get("focus_point", "N/A"),
            "evaluation_criteria": pitch_strategy.get("evaluation_criteria", []),
            "killer_question": pitch_strategy.get("killer_question", "N/A")
        }
    else:
        strategy_info = {
            "type": "General Analysis",
            "focus_point": "범용 문서 분석 (공고문 없음)"
        }

    final_output = {
        "meta": {
            "filename": docai_result.get("metadata", {}).get("filename", "unknown"),
            "doc_type": doc_type,
            "pitch_strategy": strategy_info,
            "total_slides": len(slides_data),
            "total_duration_est": total_duration,
            "analysis_method": "LLM-Powered (Gemini)" if gemini.model else "Rule-Based"
        },
        "diagnosis": llm_analysis.get("diagnosis", {}),
        "content_quality": llm_analysis.get("content_quality", {}),
        "recommendations": llm_analysis.get("recommendations", {}),
        "slides": slides_data
    }

    for slide in final_output["slides"]:
        if "full_text" in slide["contents"]:
            del slide["contents"]["full_text"]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\n완료!")
    print("파일:", output_path)
    print("완성도:", llm_analysis["diagnosis"]["overall_completeness"])

    return final_output