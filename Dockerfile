FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV / OCR common runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better Docker layer cache
COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/backend/requirements.txt

# Copy source
COPY . /app

EXPOSE 8000

# CloudBase Run compatible command shape
CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend/src --host 0.0.0.0 --port ${PORT:-8000}"]
