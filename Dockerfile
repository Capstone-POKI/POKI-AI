FROM python:3.11-slim

# System dependencies: poppler (pdf2image), ffmpeg (pydub/librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (production only — no torch/transformers)
COPY requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY src/ ./src/
COPY data/config/ ./data/config/
COPY main.py ./

# Create data directories
RUN mkdir -p data/output/notice_uploads \
             data/output/notice_analysis \
             data/output/ir_uploads \
             data/output/ir_analysis

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
