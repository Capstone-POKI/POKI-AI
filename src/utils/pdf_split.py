# src/utils/pdf_split.py
import os
from PyPDF2 import PdfReader, PdfWriter
from typing import List

def split_pdf(input_pdf: str, output_dir: str, chunk_size: int = 15) -> List[str]:
    """
    PDF를 chunk_size 단위로 분할하여 여러 개 PDF로 저장.
    반환값: 생성된 chunk PDF 경로 리스트
    """
    if not os.path.exists(input_pdf):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_pdf}")

    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)

    os.makedirs(output_dir, exist_ok=True)
    chunks = []

    start = 0
    part = 1
    
    # 원본 파일명 가져오기 (확장자 제외)
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]

    while start < total_pages:
        end = min(start + chunk_size, total_pages)

        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        # 파일명 형식: 원본파일명_chunk_1.pdf
        chunk_filename = f"{base_name}_chunk_{part}.pdf"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        with open(chunk_path, "wb") as f:
            writer.write(f)

        chunks.append(chunk_path)
        start = end
        part += 1

    return chunks