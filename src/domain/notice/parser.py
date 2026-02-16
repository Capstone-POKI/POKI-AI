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
        return normalize_notice_result(raw)
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


def normalize_notice_result(raw: Dict[str, Any]) -> Dict[str, Any]:
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
    normalized["evaluation_criteria"] = _normalize_criteria_list(raw.get("evaluation_criteria"))
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
        normalized["evaluation_criteria"] = _normalize_legacy_eval_items(eval_items)
    if normalized["evaluation_structure_type"] == "NOT_EXPLICIT":
        normalized["evaluation_structure_type"] = _infer_structure_type(normalized["evaluation_criteria"])

    return normalized


def _normalize_criteria_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    items: List[Dict[str, Any]] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        points = _to_number(raw_item.get("points"))
        items.append(
            {
                "criteria_name": _to_str(raw_item.get("criteria_name")),
                "points": points if points is not None else 0,
                "sub_requirements": _to_str_list(raw_item.get("sub_requirements")),
                "pitchcoach_interpretation": _to_str(raw_item.get("pitchcoach_interpretation")),
            }
        )
    return items


def _normalize_legacy_eval_items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    items: List[Dict[str, Any]] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        points = _to_number(raw_item.get("weight"))
        items.append(
            {
                "criteria_name": _to_str(raw_item.get("item")),
                "points": points if points is not None else 0,
                "sub_requirements": _to_str_list(raw_item.get("description")),
                "pitchcoach_interpretation": _to_str(raw_item.get("description")),
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
