def resolve_strategy(*args, **kwargs):
    from src.domain.ir.pipeline import resolve_strategy as _resolve_strategy

    return _resolve_strategy(*args, **kwargs)


def run_ir_analysis(*args, **kwargs):
    from src.domain.ir.pipeline import run_ir_analysis as _run_ir_analysis

    return _run_ir_analysis(*args, **kwargs)


def run_ir_deck_analysis(*args, **kwargs):
    from src.domain.ir.pipeline import run_ir_deck_analysis as _run_ir_deck_analysis

    return _run_ir_deck_analysis(*args, **kwargs)


__all__ = ["resolve_strategy", "run_ir_analysis", "run_ir_deck_analysis"]
