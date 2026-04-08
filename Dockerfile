# ─────────────────────────────────────────────────────────────
# NeuralForge — GPU-native Expert Knowledge Platform
# Base: RAPIDS (cuGraph + CUDA 12.5) on Python 3.12
# ─────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.12

WORKDIR /app

# System deps — ffmpeg for media ingestion
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Python deps — separate layer for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create runtime directories
RUN mkdir -p data/logs data/graph

EXPOSE 8090

# Health check against the API root
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8090/health').raise_for_status()" || exit 1

CMD ["uvicorn", "forge.api.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "4"]
