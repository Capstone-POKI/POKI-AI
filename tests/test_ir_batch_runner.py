import json
from pathlib import Path

from src.domain.ir import batch_runner


def test_run_ir_batch_writes_summary(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (input_dir / "b.pdf").write_bytes(b"%PDF-1.4\n")
    (input_dir / "sample_notice.pdf").write_bytes(b"%PDF-1.4\n")

    def fake_run_ir_analysis(ir_pdf, output_dir, strategy, use_chunking, pitch_type):
        payload = {
            "deck_score": {"total_score": 77},
            "criteria_scores": [
                {"coverage_status": "COVERED"},
                {"coverage_status": "PARTIALLY_COVERED"},
                {"coverage_status": "NOT_COVERED"},
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{Path(ir_pdf).stem}_final.json"
        final_path.write_text(json.dumps(payload), encoding="utf-8")
        return {"final_path": str(final_path)}

    monkeypatch.setattr(batch_runner, "run_ir_analysis", fake_run_ir_analysis)

    config = batch_runner.BatchRunConfig(
        input_dir=input_dir,
        output_root=tmp_path / "out",
        pitch_type="COMPETITION",
        skip_notice_like=True,
    )
    summary = batch_runner.run_ir_batch(config)
    assert summary["total_files"] == 2
    assert summary["completed_files"] == 2
    assert (tmp_path / "out" / "batch_summary.json").exists()
    assert (tmp_path / "out" / "batch_summary.csv").exists()

