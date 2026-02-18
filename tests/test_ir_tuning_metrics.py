from pathlib import Path

from src.domain.ir.tuning_metrics import (
    aggregate_eval,
    evaluate_label,
    find_result_for_label,
    normalize_category,
    normalize_pitch_type,
)


def test_normalizers():
    assert normalize_pitch_type("COMPETITION") == "STARTUP_CONTEST"
    assert normalize_category("PLAN") == "ASK"


def test_evaluate_label_basic(tmp_path):
    result_root = tmp_path / "out"
    stem = "deck_a"
    (result_root / stem).mkdir(parents=True)
    final = result_root / stem / f"{stem}_final.json"
    final.write_text(
        """{
          "pitch_type":"STARTUP_CONTEST",
          "criteria_scores":[{"criteria_id":"PROBLEM","coverage_status":"COVERED","related_slides":[3]}],
          "slides":[{"slide_number":3,"category":"PROBLEM"}]
        }""",
        encoding="utf-8",
    )
    label = {
        "label_id": "GT_X",
        "filename": "deck_a.pdf",
        "pitch_type": "COMPETITION",
        "group_labels": [{"group_id": "PROBLEM", "expected_coverage": "COVERED", "related_slides": [3]}],
        "slide_classification_labels": [{"slide_number": 3, "expected_category": "PROBLEM"}],
    }
    p = find_result_for_label(result_root, label["filename"])
    assert p is not None
    import json

    row = evaluate_label(label, json.loads(p.read_text(encoding="utf-8")))
    summary = aggregate_eval([row])
    assert summary["pitch_type_accuracy"] == 1.0
    assert summary["group_coverage_accuracy"] == 1.0
    assert summary["related_slide_hit_rate"] == 1.0
    assert summary["slide_category_accuracy"] == 1.0

