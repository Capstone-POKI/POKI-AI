import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.domain.notice.document_ai import run_notice_document_ai
from src.domain.notice.parser import analyze_notice
from src.infrastructure.gemini.client import GeminiJSONClient
from src.common.utils import save_strategy, strategy_output_path


DEFAULT_STRATEGY = {
    "type": "Government Grant",
    "evaluation_criteria": ["사업성(40점)", "실현가능성(30점)", "팀빌딩(30점)"],
    "required_sections": ["problem", "solution", "market", "team"],
    "focus_point": "사업의 실현 가능성과 성과 창출 계획",
    "killer_question": "지원 종료 후 자립 방안은 무엇입니까?",
}


def init_gemini() -> Optional[GeminiJSONClient]:
    client = GeminiJSONClient()
    if client.model is None:
        print("⚠️ Gemini 연결 실패 (규칙 기반 폴백)")
        return None
    print(f"☁️ Gemini 연결 성공! ({client.model_name})")
    return client


def run_notice_analysis(notice_pdf: Path, output_dir: Path, gemini: Optional[GeminiJSONClient] = None) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n🧭 [Notice Analysis] 공고문 분석 시작")

    if not notice_pdf.exists():
        raise FileNotFoundError(f"공고문 파일이 없습니다: {notice_pdf}")

    stage1 = stage1_extract_with_docai(notice_pdf, output_dir)
    stage2 = stage2_parse_with_gemini(stage1, output_dir, notice_pdf.stem, gemini)
    notice_analysis = _strip_internal_fields(stage2)

    final_strategy = build_strategy(notice_analysis)
    strategy_path = strategy_output_path(output_dir, notice_pdf)
    save_strategy(final_strategy, strategy_path, notice_pdf)

    final_analysis_path = output_dir / f"{notice_pdf.stem}_notice_analysis.json"
    _write_json(final_analysis_path, notice_analysis)

    manifest = {
        "source_notice_pdf": str(notice_pdf),
        "artifacts": {
            "stage1_structured": stage1.get("_artifact_path"),
            "stage2_analysis": stage2.get("_artifact_path"),
            "final_analysis": str(final_analysis_path),
            "final_strategy": str(strategy_path),
        },
        "strategy": final_strategy,
    }
    manifest_path = output_dir / f"{notice_pdf.stem}_manifest.json"
    _write_json(manifest_path, manifest)

    print(f"✅ 최종 분석 JSON 저장 완료: {final_analysis_path}")
    print(f"✅ 최종 전략 저장 완료: {strategy_path}")
    print(f"✅ 매니페스트 저장 완료: {manifest_path}")

    return {
        "analysis": notice_analysis,
        "analysis_path": str(final_analysis_path),
        "strategy": final_strategy,
        "strategy_path": str(strategy_path),
        "manifest_path": str(manifest_path),
        "ocr_output": str(output_dir / f"{notice_pdf.stem}_docai.json"),
    }


def stage1_extract_with_docai(notice_pdf: Path, output_dir: Path) -> Dict[str, Any]:
    print("\n[Stage 1] Document AI: 텍스트/표 구조 추출")
    docai = run_notice_document_ai(notice_pdf, output_dir)
    if not docai:
        raise RuntimeError("Stage 1 실패: OCR 결과가 비어 있습니다.")

    structured = {
        "text": docai.get("text", ""),
        "tables": _extract_tables(docai),
        "metadata": docai.get("metadata", {}),
    }
    stage1_path = output_dir / f"{notice_pdf.stem}_stage1_structured.json"
    _write_json(stage1_path, structured)
    structured["_artifact_path"] = str(stage1_path)
    print(f"✅ Stage 1 저장: {stage1_path}")
    return structured


def stage2_parse_with_gemini(
    stage1: Dict[str, Any],
    output_dir: Path,
    source_stem: str,
    gemini: Optional[GeminiJSONClient],
) -> Dict[str, Any]:
    print("\n[Stage 2] Gemini 단일 분석: JSON 파싱")
    result = analyze_notice(
        gemini=gemini,
        notice_text=stage1.get("text", ""),
        tables=stage1.get("tables", []),
    )

    stage2_path = output_dir / f"{source_stem}_stage2_analysis.json"
    _write_json(stage2_path, result)
    result["_artifact_path"] = str(stage2_path)
    print(f"✅ Stage 2 저장: {stage2_path}")
    return result


def build_strategy(analysis: Dict[str, Any]) -> Dict[str, Any]:
    items = analysis.get("evaluation_criteria", []) if isinstance(analysis, dict) else []
    criteria: List[str] = []

    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        name = _to_str(item.get("criteria_name")) or "평가항목"
        points = item.get("points")
        if isinstance(points, (int, float)):
            criteria.append(f"{name}({points})")
        else:
            criteria.append(name)

    strategy_type = _to_str(analysis.get("recruitment_type")) or DEFAULT_STRATEGY["type"]
    focus = _to_str(analysis.get("core_requirements")) or DEFAULT_STRATEGY["focus_point"]

    return {
        "type": strategy_type,
        "evaluation_criteria": criteria or DEFAULT_STRATEGY["evaluation_criteria"],
        "required_sections": DEFAULT_STRATEGY["required_sections"],
        "focus_point": focus,
        "killer_question": DEFAULT_STRATEGY["killer_question"],
    }


def _extract_tables(docai: Dict[str, Any]) -> List[Dict[str, Any]]:
    full_text = docai.get("text", "")
    pages = docai.get("pages", [])
    extracted: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(pages, 1):
        tables = page.get("tables", [])
        for table_idx, table in enumerate(tables, 1):
            rows: List[List[str]] = []
            for row in table.get("headerRows", []):
                rows.append(_extract_row_texts(row, full_text))
            for row in table.get("bodyRows", []):
                rows.append(_extract_row_texts(row, full_text))
            extracted.append(
                {
                    "page": page_idx,
                    "table_index": table_idx,
                    "row_count": len(rows),
                    "rows": rows,
                }
            )

    return extracted


def _extract_row_texts(row: Dict[str, Any], full_text: str) -> List[str]:
    cells = row.get("cells", [])
    values: List[str] = []
    for cell in cells:
        layout = cell.get("layout", {})
        anchor = layout.get("textAnchor", {})
        values.append(_extract_anchor_text(anchor, full_text))
    return values


def _extract_anchor_text(anchor: Dict[str, Any], full_text: str) -> str:
    segments = anchor.get("textSegments", [])
    parts: List[str] = []
    for segment in segments:
        start = int(segment.get("startIndex", 0))
        end = int(segment.get("endIndex", 0))
        parts.append(full_text[start:end])
    return " ".join(parts).strip()
def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _strip_internal_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()
