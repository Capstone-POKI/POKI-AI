import re
from typing import Any, Dict, List, Optional, Union

from src.domain.notice.prompts import build_notice_prompt
from src.infrastructure.gemini.client import GeminiJSONClient


Number = Union[int, float]


NOTICE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "notice_name": {"type": "STRING"},
        "host_organization": {"type": "STRING"},
        "recruitment_type": {"type": "STRING"},
        "target_audience": {"type": "STRING"},
        "application_period": {"type": "STRING"},
        "summary": {"type": "STRING"},
        "core_requirements": {"type": "STRING"},
        "source_reference": {"type": "STRING"},
        "evaluation_structure_type": {"type": "STRING"},
        "extraction_confidence": {"type": "NUMBER"},
        "additional_criteria": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "item": {"type": "STRING"},
                    "points": {"type": "NUMBER"},
                },
                "required": ["item", "points"],
            },
        },
        "evaluation_criteria": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "criteria_name": {"type": "STRING"},
                    "points": {"type": "NUMBER"},
                    "source_reference": {"type": "STRING"},
                    "sub_requirements": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "pitchcoach_interpretation": {"type": "STRING"},
                    "ir_guide": {"type": "STRING"},
                },
                "required": [
                    "criteria_name",
                    "points",
                    "pitchcoach_interpretation",
                    "ir_guide",
                ],
            },
        },
        "ir_deck_guide": {"type": "STRING"},
    },
    "required": [
        "notice_name",
        "host_organization",
        "recruitment_type",
        "target_audience",
        "application_period",
        "summary",
        "core_requirements",
        "additional_criteria",
        "evaluation_criteria",
        "ir_deck_guide",
    ],
}


def analyze_notice(
    gemini: Optional[GeminiJSONClient],
    notice_text: str,
    tables: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if gemini is None:
        return empty_notice_result()

    prompt = build_notice_prompt(notice_text, tables)
    try:
        raw = gemini.generate_json(
            prompt,
            temperature=0.2,
            response_schema=NOTICE_RESPONSE_SCHEMA,
        )
        return normalize_notice_result(raw, tables=tables, notice_text=notice_text)
    except Exception as exc:
        import traceback
        print(f"❌ [analyze_notice] Gemini 호출 실패: {exc}")
        traceback.print_exc()
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
    additional_criteria = _normalize_additional_criteria(raw.get("additional_criteria"))
    if additional_criteria:
        normalized["additional_criteria"] = additional_criteria
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

    filtered, bonus_items = _filter_and_capture_bonus_criteria(normalized["evaluation_criteria"])
    normalized["evaluation_criteria"] = filtered
    if bonus_items and not normalized.get("additional_criteria"):
        normalized["additional_criteria"] = bonus_items

    table_criteria, table_bonus_items = _extract_evaluation_criteria_from_tables(tables or [])
    if table_criteria and _should_use_table_criteria(
        current_criteria=normalized["evaluation_criteria"],
        table_criteria=table_criteria,
    ):
        normalized["evaluation_criteria"] = _prefer_table_criteria(
            normalized["evaluation_criteria"],
            table_criteria,
        )
    if table_bonus_items:
        normalized["additional_criteria"] = _merge_additional_criteria(
            normalized.get("additional_criteria", []),
            table_bonus_items,
        )

    normalized["evaluation_criteria"] = _apply_even_point_distribution_if_needed(
        normalized["evaluation_criteria"],
        tables or [],
        notice_text,
    )
    normalized["evaluation_criteria"] = _sanitize_evaluation_criteria(
        normalized["evaluation_criteria"],
    )

    # Fallback: extract 우대사항 from tables/text if still empty
    if not normalized.get("additional_criteria"):
        extracted_additional_criteria = _extract_additional_criteria_from_sources(
            tables or [], notice_text,
        )
        if extracted_additional_criteria:
            normalized["additional_criteria"] = extracted_additional_criteria

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
        ir_guide = _to_str(raw_item.get("ir_guide"))

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
                "ir_guide": ir_guide,
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


def _apply_even_point_distribution_if_needed(
    criteria: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    notice_text: str,
) -> List[Dict[str, Any]]:
    if not criteria:
        return criteria

    point_values = []
    for item in criteria:
        if not isinstance(item, dict):
            continue
        value = _to_number(item.get("points"))
        point_values.append(float(value) if isinstance(value, (int, float)) else 0.0)

    # Apply only when per-item scores were not found at all.
    if any(value > 0 for value in point_values):
        return criteria

    total_points = _infer_total_points_from_sources(tables, notice_text)
    if total_points is None or total_points <= 0:
        return criteria

    count = len(criteria)
    if count == 0:
        return criteria

    base_points = total_points // count
    remainder = total_points % count
    if base_points <= 0:
        return criteria

    distributed: List[Dict[str, Any]] = []
    for index, item in enumerate(criteria):
        next_item = dict(item)
        next_item["points"] = base_points + (1 if index < remainder else 0)
        distributed.append(next_item)
    return distributed


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


def _infer_total_points_from_sources(
    tables: List[Dict[str, Any]],
    notice_text: str,
) -> Optional[int]:
    total_from_tables = _infer_total_points_from_tables(tables)
    if total_from_tables is not None:
        return total_from_tables
    return _infer_total_points_from_notice_text(notice_text)


def _infer_total_points_from_tables(tables: List[Dict[str, Any]]) -> Optional[int]:
    for table in tables:
        rows = table.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, list):
                continue
            row_text = " ".join(_to_str(cell) for cell in row)
            total = _extract_total_points_from_text(row_text)
            if total is not None:
                return total
    return None


def _infer_total_points_from_notice_text(notice_text: str) -> Optional[int]:
    cleaned_text = _to_str(notice_text)
    if not cleaned_text:
        return None

    for block in _candidate_text_blocks(cleaned_text):
        total = _extract_total_points_from_text(block)
        if total is not None:
            return total

    return _extract_total_points_from_text(cleaned_text)


def _extract_total_points_from_text(text: str) -> Optional[int]:
    cleaned = _to_str(text)
    if not cleaned:
        return None

    patterns = [
        r"총\s*(\d+)\s*점",
        r"(\d+)\s*점\s*만점",
        r"평가\s*항목\s*\(\s*(\d+)\s*점\s*만점\s*\)",
        r"심사\s*항목\s*\(\s*(\d+)\s*점\s*만점\s*\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        value = _to_number(match.group(1))
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


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
    lowered = re.sub(r"[^0-9a-z가-힣]", "", _to_str(text).lower())
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


def _filter_and_capture_bonus_criteria(
    criteria: List[Dict[str, Any]],
) -> tuple:
    """Filter out non-evaluation items. Bonus/우대 items are captured as structured list."""
    bonus_tokens = ["가산점", "우대"]
    non_eval_tokens = [
        "참여율", "출석", "자격", "자격요건", "지원자격",
        "신청자격", "제출", "접수", "의무",
    ]

    filtered: List[Dict[str, Any]] = []
    bonus_items: List[Dict[str, Any]] = []

    for item in criteria:
        if not isinstance(item, dict):
            continue
        name = _to_str(item.get("criteria_name"))
        interp = _to_str(item.get("pitchcoach_interpretation"))
        combined = f"{name} {interp}".replace(" ", "")

        if any(token in combined for token in bonus_tokens):
            points = _to_number(item.get("points")) or 0
            bonus_items.append({"item": name, "points": int(points)})
            continue

        if any(token in combined for token in non_eval_tokens):
            continue

        filtered.append(item)

    return filtered, bonus_items


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

    has_point = any(label == "POINT" for label in labels)
    has_percent = any(label == "PERCENT" for label in labels)

    if has_point and has_percent:
        return "MIXED"
    if has_point:
        return "POINT_BASED"
    if has_percent:
        return "PERCENT_BASED"
    return "NOT_EXPLICIT"


def _normalize_additional_criteria(value: Any) -> List[Dict[str, Any]]:
    """Normalize additional_criteria from Gemini: list of {item, points} or string fallback."""
    if isinstance(value, list):
        items: List[Dict[str, Any]] = []
        for raw in value:
            if isinstance(raw, dict):
                item_name = _to_str(raw.get("item"))
                pts = _to_number(raw.get("points")) or 0
                if item_name:
                    items.append({"item": item_name, "points": int(pts)})
        return items
    # Legacy string fallback: parse "항목: N점" patterns
    if isinstance(value, str) and value.strip():
        return _parse_additional_criteria_string(value)
    return []


def _parse_additional_criteria_string(text: str) -> List[Dict[str, Any]]:
    """Parse a free-text additional_criteria string into structured items."""
    items: List[Dict[str, Any]] = []
    # Match patterns like "서초구 본점 소재: 3점" or "서초구 본점 소재 기업(3점)"
    for m in re.finditer(r"([^,;()\d]+?)\s*[:(]\s*(\d+)\s*점?\s*\)?", text):
        name = m.group(1).strip().rstrip(":( ")
        pts = int(m.group(2))
        if name and len(name) >= 2:
            items.append({"item": name, "points": pts})
    return items


def _extract_additional_criteria_from_sources(
    tables: List[Dict[str, Any]],
    notice_text: str,
) -> List[Dict[str, Any]]:
    """Fallback: scan tables and text for 우대사항/가산점 rows."""
    bonus_keywords = ["우대", "가산", "가점", "우대사항", "우대점수", "추가배점"]

    # 1) Search tables for bonus rows
    for table in tables:
        rows = table.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, list):
                continue
            row_text = " ".join(_to_str(cell) for cell in row)
            row_normalized = row_text.replace(" ", "")
            if any(kw in row_normalized for kw in bonus_keywords):
                parsed = _parse_additional_criteria_string(row_text)
                if parsed:
                    return parsed

    # 2) Search text blocks around bonus keywords
    cleaned_text = _to_str(notice_text)
    for kw in bonus_keywords:
        for match in re.finditer(re.escape(kw), cleaned_text):
            start = max(0, match.start() - 50)
            end = min(len(cleaned_text), match.end() + 300)
            block = cleaned_text[start:end].strip()
            parsed = _parse_additional_criteria_string(block)
            if parsed:
                return parsed

    return []


def _extract_evaluation_criteria_from_tables(
    tables: List[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    best_eval: list[Dict[str, Any]] = []
    best_bonus: list[Dict[str, Any]] = []
    best_score = -1.0

    for table in tables:
        rows = table.get("rows", [])
        if not isinstance(rows, list):
            continue
        if not _table_looks_like_evaluation_table(rows):
            continue

        eval_rows: list[Dict[str, Any]] = []
        bonus_rows: list[Dict[str, Any]] = []
        seen_names: set[str] = set()

        for row in rows:
            parsed = _extract_criterion_from_row(row)
            if parsed is None:
                continue
            name, points, source_reference, is_bonus = parsed
            name_key = _normalize_text(name)
            if not name_key or name_key in seen_names:
                continue
            seen_names.add(name_key)

            if is_bonus:
                bonus_rows.append({"item": name, "points": int(points)})
            else:
                eval_rows.append(
                    {
                        "criteria_name": name,
                        "points": int(points),
                        "source_reference": source_reference,
                        "sub_requirements": [],
                        "pitchcoach_interpretation": "",
                        "ir_guide": "",
                    }
                )

        score = _score_table_criteria(eval_rows)
        if score > best_score:
            best_score = score
            best_eval = eval_rows
            best_bonus = bonus_rows

    return best_eval, best_bonus


def _extract_criterion_from_row(
    row: Any,
) -> Optional[tuple[str, int, str, bool]]:
    if not isinstance(row, list):
        return None

    cells = [_to_str(cell) for cell in row if _to_str(cell)]
    if not cells:
        return None

    row_text = " | ".join(cells)
    normalized_row = row_text.replace(" ", "")
    if _is_total_text(row_text):
        return None

    points: Optional[float] = None
    points_cell_index = -1
    for idx, cell in enumerate(cells):
        point_value = _extract_points_from_text(cell)
        if isinstance(point_value, (int, float)) and point_value > 0:
            points = float(point_value)
            points_cell_index = idx
            break
    if points is None:
        return None

    name = ""
    for idx, cell in enumerate(cells):
        if idx == points_cell_index:
            continue
        if _extract_points_from_text(cell) is not None:
            continue
        cleaned = _cleanup_criteria_name(cell)
        if cleaned and not _is_header_like(cleaned):
            name = cleaned
            break
    if not name:
        cleaned = _cleanup_criteria_name(row_text)
        if cleaned and not _is_header_like(cleaned):
            name = cleaned
    if not name:
        return None

    bonus_tokens = ["우대", "가산", "가점", "추가배점", "우대사항", "우대점수"]
    non_eval_tokens = ["참여율", "출석", "자격", "제출", "접수", "의무"]

    is_bonus = any(token in normalized_row for token in bonus_tokens) or any(
        token in _normalize_text(name) for token in bonus_tokens
    )
    if not is_bonus and any(token in normalized_row for token in non_eval_tokens):
        return None

    return name, int(round(points)), row_text, is_bonus


def _cleanup_criteria_name(text: str) -> str:
    cleaned = _to_str(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\([^)]*\d+(?:\.\d+)?\s*(?:점|%)\s*[^)]*\)", "", cleaned)
    cleaned = re.sub(r"\d+(?:\.\d+)?\s*(?:점|%)", "", cleaned)
    cleaned = re.sub(r"[/|:·•\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_header_like(name: str) -> bool:
    normalized = _normalize_text(name)
    header_tokens = {
        "평가항목",
        "심사항목",
        "심사기준",
        "세부기준",
        "항목",
        "배점",
        "점수",
        "내용",
        "비고",
    }
    return normalized in header_tokens


def _score_table_criteria(criteria_rows: List[Dict[str, Any]]) -> float:
    if len(criteria_rows) < 2:
        return -1.0
    total = sum(int(row.get("points", 0)) for row in criteria_rows)
    score = 0.0
    if 2 <= len(criteria_rows) <= 8:
        score += 2.0
    if 80 <= total <= 120:
        score += 2.0
    if all(0 < int(row.get("points", 0)) <= 100 for row in criteria_rows):
        score += 1.0
    score += min(len(criteria_rows), 8) * 0.05
    return score


def _table_looks_like_evaluation_table(rows: List[Any]) -> bool:
    preview = " ".join(
        " ".join(_to_str(cell) for cell in row) if isinstance(row, list) else ""
        for row in rows[:4]
    )
    normalized = preview.replace(" ", "")
    keywords = ["평가", "심사", "배점", "항목", "기준", "점수", "만점"]
    return any(k in normalized for k in keywords)


def _criteria_points_sum(criteria_rows: List[Dict[str, Any]]) -> int:
    return sum(int(_to_number(row.get("points")) or 0) for row in criteria_rows if isinstance(row, dict))


def _is_reliable_table_criteria(criteria_rows: List[Dict[str, Any]]) -> bool:
    count = len(criteria_rows)
    total = _criteria_points_sum(criteria_rows)
    if count < 2:
        return False
    if 95 <= total <= 105:
        return True
    if count >= 3 and 90 <= total <= 110:
        return True
    return False


def _should_use_table_criteria(
    current_criteria: List[Dict[str, Any]],
    table_criteria: List[Dict[str, Any]],
) -> bool:
    if _is_reliable_table_criteria(table_criteria):
        return True
    if not current_criteria:
        return True
    current_sum = _criteria_points_sum(current_criteria)
    if current_sum <= 0:
        return True
    # 현재 결과가 100점 정합성을 만족하면 테이블이 불확실할 때 덮어쓰지 않음
    return not (95 <= current_sum <= 105)


def _prefer_table_criteria(
    llm_criteria: List[Dict[str, Any]],
    table_criteria: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    llm_index: Dict[str, Dict[str, Any]] = {}
    for item in llm_criteria:
        if not isinstance(item, dict):
            continue
        name = _to_str(item.get("criteria_name"))
        if not name:
            continue
        aliases = _criteria_aliases(name) or [name]
        for alias in aliases:
            key = _normalize_text(alias)
            if key and key not in llm_index:
                llm_index[key] = item

    merged: List[Dict[str, Any]] = []
    for t_item in table_criteria:
        name = _to_str(t_item.get("criteria_name"))
        matched: Optional[Dict[str, Any]] = None
        for alias in _criteria_aliases(name) or [name]:
            matched = llm_index.get(_normalize_text(alias))
            if matched:
                break
        merged.append(
            {
                "criteria_name": name,
                "points": int(_to_number(t_item.get("points")) or 0),
                "source_reference": _to_str(t_item.get("source_reference")),
                "sub_requirements": _to_str_list((matched or {}).get("sub_requirements")),
                "pitchcoach_interpretation": _to_str((matched or {}).get("pitchcoach_interpretation")),
                "ir_guide": _to_str((matched or {}).get("ir_guide")),
            }
        )
    return merged


def _merge_additional_criteria(
    current_items: Any,
    new_items: Any,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for raw in list(current_items or []) + list(new_items or []):
        if not isinstance(raw, dict):
            continue
        item = _to_str(raw.get("item"))
        points = int(_to_number(raw.get("points")) or 0)
        key = _normalize_text(item)
        if not key:
            continue
        if key not in merged:
            merged[key] = {"item": item, "points": points}
    return list(merged.values())


def _sanitize_evaluation_criteria(criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in criteria:
        if not isinstance(row, dict):
            continue
        name = _to_str(row.get("criteria_name"))
        if not name or _is_total_text(name):
            continue
        key = _normalize_text(name)
        if not key or key in seen:
            continue
        seen.add(key)
        copied = dict(row)
        copied["points"] = int(_to_number(copied.get("points")) or 0)
        cleaned.append(copied)

    if len(cleaned) < 2:
        return cleaned

    if any(int(row.get("points", 0)) > 0 for row in cleaned):
        cleaned = [row for row in cleaned if int(row.get("points", 0)) > 0]
        if len(cleaned) < 2:
            return cleaned

    total = _criteria_points_sum(cleaned)
    if total <= 110:
        return cleaned

    # 명백한 총점성 항목(100점 단일 행 등) 제거
    high_filtered = [r for r in cleaned if int(r.get("points", 0)) < 90]
    if len(high_filtered) >= 2:
        high_total = _criteria_points_sum(high_filtered)
        if abs(high_total - 100) < abs(total - 100):
            cleaned, total = high_filtered, high_total

    # 상/하위 항목 중복 합산으로 총점이 과도할 때 100점 근사화
    while len(cleaned) >= 3 and total > 110:
        base_delta = abs(total - 100)
        best_idx = -1
        best_delta = base_delta
        for idx, row in enumerate(cleaned):
            next_total = total - int(row.get("points", 0))
            if next_total < 60:
                continue
            delta = abs(next_total - 100)
            if delta < best_delta:
                best_idx = idx
                best_delta = delta
        if best_idx < 0:
            break
        cleaned.pop(best_idx)
        total = _criteria_points_sum(cleaned)

    return cleaned
