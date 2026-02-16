def build_analysis_context(*args, **kwargs):
    from src.domain.report.feature_builder import build_analysis_context as _build_analysis_context

    return _build_analysis_context(*args, **kwargs)


def generate_qa(*args, **kwargs):
    from src.domain.report.qa_generator import generate_qa as _generate_qa

    return _generate_qa(*args, **kwargs)


def build_final_report(*args, **kwargs):
    from src.domain.report.report_builder import build_final_report as _build_final_report

    return _build_final_report(*args, **kwargs)


def compute_feature_impacts(*args, **kwargs):
    from src.domain.report.score_engine import compute_feature_impacts as _compute_feature_impacts

    return _compute_feature_impacts(*args, **kwargs)


__all__ = ["build_analysis_context", "generate_qa", "build_final_report", "compute_feature_impacts"]
