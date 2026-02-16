from pydantic import BaseModel


class IRAnalyzeRequest(BaseModel):
    ir_pdf: str = "data/input/sample_irdeck.pdf"
    output_dir: str = "data/output/ir_analysis"
    strategy_json: str | None = None
    notice_output_dir: str = "data/output/notice_analysis"
    no_auto_strategy: bool = False
    no_chunking: bool = False
