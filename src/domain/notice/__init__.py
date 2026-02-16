def init_gemini():
    from src.domain.notice.pipeline import init_gemini as _init_gemini

    return _init_gemini()


def run_notice_analysis(*args, **kwargs):
    from src.domain.notice.pipeline import run_notice_analysis as _run_notice_analysis

    return _run_notice_analysis(*args, **kwargs)


__all__ = ["init_gemini", "run_notice_analysis"]
