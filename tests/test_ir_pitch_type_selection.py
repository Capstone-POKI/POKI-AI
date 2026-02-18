from src.domain.ir.rag_pipeline import _normalize_pitch_type, _resolve_pitch_type


def test_normalize_pitch_type_explicit_mapping():
    assert _normalize_pitch_type("VC_DEMO") == "VC_DEMO"
    assert _normalize_pitch_type("GOVERNMENT") == "GOV_SUPPORT"
    assert _normalize_pitch_type("COMPETITION") == "STARTUP_CONTEST"
    assert _normalize_pitch_type("ELEVATOR") == "VC_DEMO"


def test_resolve_pitch_type_prefers_explicit():
    slides = [{"clean_text": "정부지원사업 창업패키지 공고"}]
    assert _resolve_pitch_type(strategy=None, explicit_pitch_type="COMPETITION", slides=slides) == "STARTUP_CONTEST"


def test_resolve_pitch_type_uses_strategy_then_inference():
    assert _resolve_pitch_type(strategy={"type": "Government Grant"}, explicit_pitch_type=None, slides=[]) == "GOV_SUPPORT"
    inferred = _resolve_pitch_type(
        strategy=None,
        explicit_pitch_type=None,
        slides=[{"clean_text": "정부지원사업 공공 지자체 정책 연계 창업패키지"}],
    )
    assert inferred == "GOV_SUPPORT"
