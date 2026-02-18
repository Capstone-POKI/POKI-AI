import json
from pathlib import Path

from src.domain.ir.rag_pipeline import run_rag_ir_analysis


def test_ir_rag_pipeline_offline_from_cached_docai(tmp_path):
    """Offline schema smoke test using cached DocAI JSON."""
    candidates = [
        Path("data/output/sample_irdeck_docai.json"),
        Path("data/output/ir_analysis/sample_irdeck_docai.json"),
        Path("data/output/ir_benchmark/sample_irdeck/sample_irdeck_docai.json"),
    ]
    docai_path = next((p for p in candidates if p.exists()), candidates[0])
    assert docai_path.exists(), f"missing cached docai json: {docai_path}"

    payload = json.loads(docai_path.read_text(encoding="utf-8"))
    output_path = tmp_path / "sample_irdeck_final_v1.json"

    result = run_rag_ir_analysis(
        docai_result=payload,
        output_path=str(output_path),
        strategy=None,
        analysis_version=1,
    )

    assert output_path.exists()
    assert isinstance(result, dict)

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
    assert expected_top_keys.issubset(result.keys())

    deck_score = result.get("deck_score", {})
    assert isinstance(deck_score.get("total_score"), int)
    assert 0 <= deck_score["total_score"] <= 100

    criteria_scores = result.get("criteria_scores", [])
    assert isinstance(criteria_scores, list)
    assert len(criteria_scores) > 0

    for item in criteria_scores:
        assert isinstance(item.get("criteria_name", ""), str)
        assert isinstance(item.get("score", 0), int)
        assert 0 <= item["score"] <= 100
        assert item.get("coverage_status") in {"COVERED", "PARTIALLY_COVERED", "NOT_COVERED"}
        related = item.get("related_slides", [])
        assert isinstance(related, list)
        for s in related:
            assert isinstance(s, int)
            assert s >= 1

    slides = result.get("slides", [])
    assert isinstance(slides, list)
    assert len(slides) > 0

    first = slides[0]
    assert "slide_number" in first
    assert "category" in first
    assert "feedback" in first
