from pathlib import Path

from src.infrastructure.document_ai.pipeline import run_document_ai_pipeline


def test_proactive_chunking_by_page_count(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake")

    called = {"chunk": 0, "merge": 0, "single": 0}

    monkeypatch.setattr(
        "src.infrastructure.document_ai.pipeline._get_pdf_page_count",
        lambda _p: 27,
    )
    monkeypatch.setattr(
        "src.infrastructure.document_ai.pipeline.process_pdf_ocr_in_chunks",
        lambda *_args, **_kwargs: called.__setitem__("chunk", called["chunk"] + 1) or [{"ok": "chunk"}],
    )
    monkeypatch.setattr(
        "src.infrastructure.document_ai.pipeline.merge_chunk_results",
        lambda _chunks, _out: called.__setitem__("merge", called["merge"] + 1) or {"mode": "chunked"},
    )

    def _single(*_args, **_kwargs):
        called["single"] += 1
        return {"mode": "single"}

    monkeypatch.setattr("src.infrastructure.document_ai.pipeline.process_document", _single)

    result = run_document_ai_pipeline(
        pdf_path=pdf_path,
        output_dir=tmp_path,
        use_chunking=False,
        pages_per_chunk=15,
    )

    assert result == {"mode": "chunked"}
    assert called["chunk"] == 1
    assert called["merge"] == 1
    assert called["single"] == 0


def test_page_limit_error_falls_back_to_chunking(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake")

    called = {"chunk": 0, "merge": 0, "single": 0}

    monkeypatch.setattr(
        "src.infrastructure.document_ai.pipeline._get_pdf_page_count",
        lambda _p: 5,
    )

    def _single(*_args, **_kwargs):
        called["single"] += 1
        raise RuntimeError("PAGE_LIMIT_EXCEEDED: document pages exceed limit")

    monkeypatch.setattr("src.infrastructure.document_ai.pipeline.process_document", _single)
    monkeypatch.setattr(
        "src.infrastructure.document_ai.pipeline.process_pdf_ocr_in_chunks",
        lambda *_args, **_kwargs: called.__setitem__("chunk", called["chunk"] + 1) or [{"ok": "chunk"}],
    )
    monkeypatch.setattr(
        "src.infrastructure.document_ai.pipeline.merge_chunk_results",
        lambda _chunks, _out: called.__setitem__("merge", called["merge"] + 1) or {"mode": "chunked_after_error"},
    )

    result = run_document_ai_pipeline(
        pdf_path=pdf_path,
        output_dir=tmp_path,
        use_chunking=False,
        pages_per_chunk=15,
    )

    assert result == {"mode": "chunked_after_error"}
    assert called["single"] == 1
    assert called["chunk"] == 1
    assert called["merge"] == 1
