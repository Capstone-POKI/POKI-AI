from app.state_store import load_state, save_state


def test_state_store_round_trip_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_STATE_DIR", str(tmp_path))
    payload = {
        "jobs": {
            "job-1": {
                "status": "IN_PROGRESS",
                "version": 1,
            }
        }
    }

    save_state("test-jobs", payload)

    assert load_state("test-jobs") == payload
    assert list(tmp_path.glob("*.tmp")) == []


def test_state_store_returns_empty_for_corrupt_json(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_STATE_DIR", str(tmp_path))
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")

    assert load_state("broken") == {}
