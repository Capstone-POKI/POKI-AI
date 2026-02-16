from pydantic import BaseModel


class NoticeAnalyzeRequest(BaseModel):
    notice_pdf: str = "data/input/sample_notice.pdf"
    output_dir: str = "data/output/notice_analysis"
    no_gemini: bool = False
