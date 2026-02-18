import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_ir_e2e_live(tmp_path):
    """Live e2e test for IR pipeline (DocAI + Gemini, opt-in).

    RUN_LIVE_IR_E2E=1 python -m pytest -m integration -q tests/test_ir_e2e_live.py -s
    """
    if os.getenv("RUN_LIVE_IR_E2E") != "1":
        pytest.skip("Set RUN_LIVE_IR_E2E=1 to run live IR e2e test")

    try:
        docai_spec = importlib.util.find_spec("google.cloud.documentai_v1beta3")
    except ModuleNotFoundError:
        docai_spec = None
    if docai_spec is None:
        pytest.skip("google-cloud-documentai is not installed in this interpreter")
    if importlib.util.find_spec("PyPDF2") is None:
        pytest.skip("PyPDF2 is not installed in this interpreter")

    from src.domain.ir.pipeline import run_ir_analysis

    ir_pdf = Path("data/input/sample_irdeck.pdf")
    assert ir_pdf.exists(), f"missing sample pdf: {ir_pdf}"

    result = run_ir_analysis(
        ir_pdf=ir_pdf,
        output_dir=tmp_path,
        strategy=None,
        use_chunking=True,
        pitch_type=os.getenv("IR_TEST_PITCH_TYPE", "COMPETITION"),
    )

    final_json_path = Path(result["final_path"])
    assert final_json_path.exists(), f"final JSON not found: {final_json_path}"

    payload = json.loads(final_json_path.read_text(encoding="utf-8"))

    expected_top_keys = {
        "analysis_version",
        "analysis_method",
        "pitch_type",
        "deck_score",
        "criteria_scores",
        "presentation_guide",
        "slides",
        "meta",
    }
    assert expected_top_keys.issubset(payload.keys())
    assert payload.get("pitch_type") in {"STARTUP_CONTEST", "COMPETITION"}

    assert isinstance(payload["deck_score"].get("total_score"), int)
    assert 0 <= payload["deck_score"]["total_score"] <= 100

    criteria_scores = payload.get("criteria_scores", [])
    assert isinstance(criteria_scores, list)
    assert len(criteria_scores) > 0

    non_empty_feedback = 0
    for item in criteria_scores:
        assert isinstance(item.get("criteria_name", ""), str)
        assert isinstance(item.get("score", 0), int)
        assert 0 <= item["score"] <= 100
        assert item.get("coverage_status") in {"COVERED", "PARTIALLY_COVERED", "NOT_COVERED"}
        if isinstance(item.get("feedback"), str) and item["feedback"].strip():
            non_empty_feedback += 1

    # 라이브 분석은 기준별 피드백이 최소 1개 이상 의미 있게 생성되어야 함.
    assert non_empty_feedback >= 1

    slides = payload.get("slides", [])
    assert isinstance(slides, list)
    assert len(slides) > 0
