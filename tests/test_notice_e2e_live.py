import json
import os
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.integration
def test_notice_e2e_live(tmp_path):
    """Live e2e test for notice pipeline (DocAI + Gemini).

    Opt-in only:
    RUN_LIVE_E2E=1 pytest -m integration -q tests/test_notice_e2e_live.py
    """
    if os.getenv("RUN_LIVE_E2E") != "1":
        pytest.skip("Set RUN_LIVE_E2E=1 to run live notice e2e test")

    try:
        docai_spec = importlib.util.find_spec("google.cloud.documentai_v1beta3")
    except ModuleNotFoundError:
        docai_spec = None
    if docai_spec is None:
        pytest.skip("google-cloud-documentai is not installed in this interpreter")
    if importlib.util.find_spec("PyPDF2") is None:
        pytest.skip("PyPDF2 is not installed in this interpreter")

    from src.domain.notice.pipeline import init_gemini, run_notice_analysis

    notice_pdf = Path("data/input/sample_notice.pdf")
    assert notice_pdf.exists(), f"missing sample pdf: {notice_pdf}"

    gemini = init_gemini()
    assert gemini is not None and gemini.model is not None, "Gemini initialization failed"

    result = run_notice_analysis(
        notice_pdf=notice_pdf,
        output_dir=tmp_path,
        gemini=gemini,
    )

    final_json_path = Path(result["analysis_path"])

    # 1) final JSON 존재
    assert final_json_path.exists(), f"final JSON not found: {final_json_path}"

    payload = json.loads(final_json_path.read_text(encoding="utf-8"))

    # 2) 최상위 키 12개 존재
    expected_keys = {
        "notice_name",
        "host_organization",
        "recruitment_type",
        "target_audience",
        "application_period",
        "summary",
        "core_requirements",
        "source_reference",
        "evaluation_structure_type",
        "extraction_confidence",
        "evaluation_criteria",
        "ir_deck_guide",
    }
    assert expected_keys.issubset(payload.keys())

    # 3) evaluation_criteria는 list
    criteria = payload.get("evaluation_criteria")
    assert isinstance(criteria, list)

    # 4) criteria_name은 str
    # 5) points는 number
    for item in criteria:
        if not isinstance(item, dict):
            continue
        assert isinstance(item.get("criteria_name", ""), str)
        assert isinstance(item.get("points", 0), (int, float))

    # 6) hallucination 방지: criteria_name이 빈 문자열만 여러 개인 상태 방지
    if len(criteria) >= 2:
        names = [item.get("criteria_name", "") for item in criteria if isinstance(item, dict)]
        stripped = [name.strip() for name in names if isinstance(name, str)]
        assert not stripped or not all(name == "" for name in stripped), (
            "evaluation_criteria contains multiple items but all criteria_name values are empty"
        )

    # Live e2e must prove Gemini actually produced meaningful output.
    extraction_confidence = payload.get("extraction_confidence", 0.0)
    summary = payload.get("summary", "")
    notice_name = payload.get("notice_name", "")
    source_reference = payload.get("source_reference", "")
    assert (
        (isinstance(extraction_confidence, (int, float)) and extraction_confidence > 0)
        or (isinstance(summary, str) and summary.strip())
        or (isinstance(notice_name, str) and notice_name.strip())
        or (isinstance(source_reference, str) and source_reference.strip())
    ), "Gemini output appears empty (likely fallback result)"
