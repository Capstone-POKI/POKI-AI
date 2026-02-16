import re
from typing import Any, Dict, List, Optional, Union

from src.domain.notice.prompts import build_notice_prompt
from src.infrastructure.gemini.client import GeminiJSONClient


Number = Union[int, float]


def analyze_notice(
    gemini: Optional[GeminiJSONClient],
    notice_text: str,
    tables: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if gemini is None:
        return empty_notice_result()

    prompt = build_notice_prompt(notice_text, tables)
    try:
        raw = gemini.generate_json(prompt, temperature=0.2)
        return normalize_notice_result(raw, tables=tables, notice_text=notice_text)
    except Exception:
        return empty_notice_result()


def empty_notice_result() -> Dict[str, Any]:
    return {
        "notice_name": "",
        "host_organization": "",
        "recruitment_type": "",
        "target_audience": "",
        "application_period": "",
        "summary": "",
        "core_requirements": "",
        "source_reference": "",
        "evaluation_structure_type": "NOT_EXPLICIT",
        "extraction_confidence": 0.0,
        "evaluation_criteria": [],
        "ir_deck_guide": "",
    }


def normalize_notice_result(
    raw: Dict[str, Any],
    tables: Optional[List[Dict[str, Any]]] = None,
    notice_text: str = "",
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return empty_notice_result()

    normalized = empty_notice_result()
    normalized["notice_name"] = _to_str(raw.get("notice_name"))
    normalized["host_organization"] = _to_str(raw.get("host_organization"))
    normalized["recruitment_type"] = _to_str(raw.get("recruitment_type"))
    normalized["target_audience"] = _to_str(raw.get("target_audience"))
    normalized["application_period"] = _to_str(raw.get("application_period"))
    normalized["summary"] = _to_str(raw.get("summary"))
    normalized["core_requirements"] = _to_str(raw.get("core_requirements"))
    normalized["source_reference"] = _to_str(raw.get("source_reference"))
    normalized["evaluation_structure_type"] = _to_structure_type(raw.get("evaluation_structure_type"))
    normalized["extraction_confidence"] = _to_confidence(raw.get("extraction_confidence"))
    normalized["evaluation_criteria"] = _normalize_criteria_list(
        raw.get("evaluation_criteria"),
        tables=tables,
        notice_text=notice_text,
    )
    normalized["ir_deck_guide"] = _to_str(raw.get("ir_deck_guide"))

    # Backward compatibility for older prompt outputs.
    basic_info = raw.get("basic_info", {}) if isinstance(raw.get("basic_info"), dict) else {}
    classification = raw.get("classification", {}) if isinstance(raw.get("classification"), dict) else {}
    evaluation = raw.get("evaluation", {}) if isinstance(raw.get("evaluation"), dict) else {}
    eval_items = evaluation.get("items", []) if isinstance(evaluation, dict) else []

    if not normalized["notice_name"]:
        normalized["notice_name"] = _to_str(basic_info.get("program_name"))
    if not normalized["host_organization"]:
        normalized["host_organization"] = _to_str(basic_info.get("organizer"))
    if not normalized["target_audience"]:
        normalized["target_audience"] = _to_str(basic_info.get("target"))
    if not normalized["application_period"]:
        normalized["application_period"] = _to_str(basic_info.get("application_period"))
    if not normalized["recruitment_type"]:
        normalized["recruitment_type"] = _to_str(classification.get("type"))
    if not normalized["core_requirements"]:
        normalized["core_requirements"] = _to_str(classification.get("reason"))
    if not normalized["evaluation_criteria"]:
        normalized["evaluation_criteria"] = _normalize_legacy_eval_items(
            eval_items,
            tables=tables,
            notice_text=notice_text,
        )

    normalized["evaluation_criteria"] = _filter_non_evaluation_criteria(normalized["evaluation_criteria"])

    normalized["extraction_confidence"] = _adjust_confidence_by_points_quality(
        normalized["extraction_confidence"],
        normalized["evaluation_criteria"],
    )
    if normalized["evaluation_structure_type"] == "NOT_EXPLICIT":
        normalized["evaluation_structure_type"] = _infer_structure_type(normalized["evaluation_criteria"])

    return normalized


def _normalize_criteria_list(
    value: Any,
    tables: Optional[List[Dict[str, Any]]] = None,
    notice_text: str = "",
) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    items: List[Dict[str, Any]] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        criteria_name = _to_str(raw_item.get("criteria_name"))
        raw_points_text = _to_str(raw_item.get("raw_points_text"))
        source_snippet = _to_str(raw_item.get("source_snippet"))
        interpretation = _to_str(raw_item.get("pitchcoach_interpretation"))

        points = _to_number(raw_item.get("points"))
        if points is None:
            points = _extract_points_from_text(raw_points_text)
        if points is None:
            points = _extract_points_from_text(source_snippet)
        if points is None:
            points = _extract_points_from_text(interpretation)
        if points is None:
            points = _infer_points_from_tables(criteria_name, tables or [])
        if points is None:
            points = _infer_points_from_notice_text(criteria_name, notice_text)
        items.append(
            {
                "criteria_name": criteria_name,
                "points": points if points is not None else 0,
                "sub_requirements": _to_str_list(raw_item.get("sub_requirements")),
                "pitchcoach_interpretation": interpretation,
            }
        )
    return items


def _normalize_legacy_eval_items(
    value: Any,
    tables: Optional[List[Dict[str, Any]]] = None,
    notice_text: str = "",
) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    items: List[Dict[str, Any]] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        criteria_name = _to_str(raw_item.get("item"))
        description = _to_str(raw_item.get("description"))
        points = _to_number(raw_item.get("weight"))
        if points is None:
            points = _extract_points_from_text(description)
        if points is None:
            points = _infer_points_from_tables(criteria_name, tables or [])
        if points is None:
            points = _infer_points_from_notice_text(criteria_name, notice_text)
        items.append(
            {
                "criteria_name": criteria_name,
                "points": points if points is not None else 0,
                "sub_requirements": _to_str_list(description),
                "pitchcoach_interpretation": description,
            }
        )
    return items


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _to_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        result = []
        for v in value:
            text = _to_str(v)
            if text:
                result.append(text)
        return result
    if isinstance(value, str):
        text = _to_str(value)
        return [text] if text else []
    return []


def _to_number(value: Any) -> Optional[Number]:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            try:
                parsed = float(match.group(0))
                return int(parsed) if parsed.is_integer() else parsed
            except ValueError:
                return None
    return None


def _extract_points_from_text(text: str) -> Optional[Number]:
    cleaned = _to_str(text)
    if not cleaned:
        return None
    if _is_total_text(cleaned):
        return None

    score_match = re.search(r"(-?\d+(?:\.\d+)?)\s*점", cleaned)
    if score_match:
        return _to_number(score_match.group(1))

    percent_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", cleaned)
    if percent_match:
        return _to_number(percent_match.group(1))

    ratio_match = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?", cleaned)
    if ratio_match:
        return _to_number(ratio_match.group(1))

    return _to_number(cleaned)


def _infer_points_from_tables(criteria_name: str, tables: List[Dict[str, Any]]) -> Optional[Number]:
    key = _normalize_text(criteria_name)
    if not key:
        return None

    for table in tables:
        rows = table.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, list):
                continue
            row_text = " ".join(_to_str(cell) for cell in row)
            if _is_total_text(row_text):
                continue
            if not _row_matches_criteria(row, key):
                continue
            for cell in row:
                points = _extract_points_from_text(_to_str(cell))
                if points is not None:
                    return points
            points = _extract_points_from_text(row_text)
            if points is not None:
                return points

    return None


def _infer_points_from_notice_text(criteria_name: str, notice_text: str) -> Optional[Number]:
    cleaned_text = _to_str(notice_text)
    if not cleaned_text:
        return None

    aliases = _criteria_aliases(criteria_name)
    if not aliases:
        return None

    # 1) 평가/심사/배점 키워드 주변 블록 우선 탐색
    for block in _candidate_text_blocks(cleaned_text):
        points = _extract_points_from_block(aliases, block)
        if points is not None:
            return points

    # 2) 전체 텍스트 폴백
    return _extract_points_from_block(aliases, cleaned_text)


def _row_matches_criteria(row: List[Any], normalized_criteria: str) -> bool:
    for cell in row:
        cell_text = _normalize_text(_to_str(cell))
        if not cell_text:
            continue
        if normalized_criteria in cell_text:
            return True
        if cell_text in normalized_criteria and len(cell_text) >= 2:
            return True
    return False


def _is_total_text(text: str) -> bool:
    lowered = _to_str(text).replace(" ", "").lower()
    return any(token in lowered for token in ["총점", "합계", "총합", "total", "sum"])


def _normalize_text(text: str) -> str:
    lowered = _to_str(text).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", lowered)


def _criteria_aliases(criteria_name: str) -> List[str]:
    raw = _to_str(criteria_name)
    if not raw:
        return []

    aliases = {raw}
    aliases.add(re.sub(r"\([^)]*\)", "", raw).strip())
    aliases.add(raw.replace("(", " ").replace(")", " ").strip())

    # Extract parenthesized alternatives (e.g., 창업가(팀) 역량 -> 팀 역량)
    inside = re.findall(r"\(([^)]*)\)", raw)
    for inner in inside:
        inner_text = _to_str(inner)
        if inner_text:
            aliases.add(inner_text)
            aliases.add(f"{inner_text} 역량")

    # Common semantic aliases from notices.
    normalized_key = _normalize_text(raw)
    semantic_map = {
        "혁신성": ["아이템의 혁신성", "사업의 혁신성", "기술 혁신성"],
        "시장성": ["아이템의 시장성", "사업의 시장성", "시장 경쟁력"],
        "성장성": ["아이템의 성장성", "사업의 성장성", "성장 가능성"],
        "창업가팀역량": ["팀 역량", "창업팀 역량", "창업가 역량"],
    }
    for key, values in semantic_map.items():
        if key in normalized_key:
            aliases.update(values)

    # Deduplicate/clean
    cleaned: List[str] = []
    for alias in aliases:
        text = _to_str(alias)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _candidate_text_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    keywords = ["평가항목", "심사기준", "평가기준", "배점", "평가표", "심사표"]
    for kw in keywords:
        for match in re.finditer(re.escape(kw), text):
            start = max(0, match.start() - 400)
            end = min(len(text), match.end() + 700)
            blocks.append(text[start:end])
    return blocks if blocks else [text]


def _extract_points_from_block(aliases: List[str], text_block: str) -> Optional[Number]:
    for alias in aliases:
        if not alias:
            continue
        pattern = _alias_pattern(alias)

        # ex) 혁신성 25점 / 혁신성: 25%
        m = re.search(rf"{pattern}\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(점|%)", text_block)
        if m:
            return _to_number(m.group(1))

        # ex) 혁신성 25/100
        m = re.search(rf"{pattern}\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?", text_block)
        if m:
            return _to_number(m.group(1))

        # ex) 25점 혁신성
        m = re.search(rf"(\d+(?:\.\d+)?)\s*(점|%)\s*{pattern}", text_block)
        if m:
            return _to_number(m.group(1))
    return None


def _alias_pattern(alias: str) -> str:
    # Match loose spacing/newlines between alias tokens.
    tokens = [re.escape(token) for token in re.split(r"\s+", _to_str(alias)) if token]
    return r"\s*".join(tokens)


def _filter_non_evaluation_criteria(criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocked_tokens = [
        "가산점",
        "우대",
        "참여율",
        "출석",
        "자격",
        "자격요건",
        "지원자격",
        "신청자격",
        "제출",
        "접수",
        "의무",
        "지역",
        "청년",
        "거주",
        "소재",
    ]
    filtered: List[Dict[str, Any]] = []
    for item in criteria:
        if not isinstance(item, dict):
            continue
        name = _to_str(item.get("criteria_name"))
        interp = _to_str(item.get("pitchcoach_interpretation"))
        combined = f"{name} {interp}".replace(" ", "")
        if any(token in combined for token in blocked_tokens):
            continue
        filtered.append(item)
    return filtered


def _adjust_confidence_by_points_quality(confidence: float, criteria: List[Dict[str, Any]]) -> float:
    if not criteria:
        return confidence
    point_values = []
    for item in criteria:
        if not isinstance(item, dict):
            continue
        value = item.get("points")
        if isinstance(value, (int, float)):
            point_values.append(float(value))

    if not point_values:
        return confidence

    zero_ratio = sum(1 for p in point_values if p == 0) / len(point_values)
    if zero_ratio < 0.6:
        return confidence

    lowered = max(0.1, 1.0 - zero_ratio)
    return min(confidence, lowered)


def _to_structure_type(value: Any) -> str:
    allowed = {"POINT_BASED", "PERCENT_BASED", "MIXED", "NOT_EXPLICIT"}
    text = _to_str(value).upper()
    return text if text in allowed else "NOT_EXPLICIT"


def _to_confidence(value: Any) -> float:
    number = _to_number(value)
    if not isinstance(number, (int, float)):
        return 0.0
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return float(number)


def _infer_structure_type(criteria: List[Dict[str, Any]]) -> str:
    if not criteria:
        return "NOT_EXPLICIT"

    labels = []
    for item in criteria:
        name = _to_str(item.get("criteria_name"))
        text = f"{name} {_to_str(item.get('pitchcoach_interpretation'))}"
        if "%" in text or "퍼센트" in text:
            labels.append("PERCENT")
        elif "점" in text:
            labels.append("POINT")
        else:
            labels.append("UNKNOWN")

    has_point = any(l == "POINT" for l in labels)
    has_percent = any(l == "PERCENT" for l in labels)

    if has_point and has_percent:
        return "MIXED"
    if has_point:
        return "POINT_BASED"
    if has_percent:
        return "PERCENT_BASED"
    return "NOT_EXPLICIT"
