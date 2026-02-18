import importlib.util
import os
from pathlib import Path

import pytest

from src.domain.ir.batch_runner import BatchRunConfig, run_ir_batch


@pytest.mark.integration
def test_ir_batch_live():
    if os.getenv("RUN_LIVE_IR_BATCH") != "1":
        pytest.skip("Set RUN_LIVE_IR_BATCH=1 to run batch live IR test")

    try:
        docai_spec = importlib.util.find_spec("google.cloud.documentai_v1beta3")
    except ModuleNotFoundError:
        docai_spec = None
    if docai_spec is None:
        pytest.skip("google-cloud-documentai is not installed in this interpreter")
    if importlib.util.find_spec("PyPDF2") is None:
        pytest.skip("PyPDF2 is not installed in this interpreter")

    input_dir = Path("data/input")
    assert input_dir.exists()
    out_dir = Path("data/output/ir_benchmark/live_batch")
    config = BatchRunConfig(
        input_dir=input_dir,
        output_root=out_dir,
        pitch_type=os.getenv("IR_TEST_PITCH_TYPE", "COMPETITION"),
        glob_pattern=os.getenv("IR_BATCH_GLOB", "*.pdf"),
        max_files=int(os.getenv("IR_BATCH_MAX_FILES", "3")),
        use_chunking=True,
        skip_notice_like=True,
    )
    summary = run_ir_batch(config)
    assert summary["total_files"] >= 1
    assert (out_dir / "batch_summary.json").exists()

