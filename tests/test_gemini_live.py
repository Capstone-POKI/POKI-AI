import os

import pytest


@pytest.mark.integration
def test_gemini_client_live_smoke():
    """Live Gemini smoke test (opt-in).

    Run only when explicitly enabled:
    RUN_LIVE_GEMINI=1 pytest -m integration -q
    """
    if os.getenv("RUN_LIVE_GEMINI") != "1":
        pytest.skip("Set RUN_LIVE_GEMINI=1 to run live Gemini integration test")

    from src.infrastructure.gemini.client import GeminiJSONClient

    client = GeminiJSONClient()
    assert client.model is not None, "Gemini model init failed"

    result = client.generate_json(
        "Return a JSON object with key ok and value true.",
        temperature=0.0,
    )
    assert isinstance(result, dict)
    assert result.get("ok") is True
