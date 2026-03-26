# ═══════════════════════════════════════════════════════════════
# Dockerfile — UAV Disease Detection
# Works on: Hugging Face Spaces, Render.com, Railway, Fly.io
# ═══════════════════════════════════════════════════════════════

FROM python:3.10-slim

# System deps for Pillow / tifffile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY main.py .
COPY index.html .

# Copy ONNX model if it exists (optional — app works without it)
# COPY dinov2.onnx .

# Create working directories
RUN mkdir -p uploads tiles tiles_raw

# Hugging Face Spaces uses port 7860
# Render.com injects $PORT automatically
# Default fallback: 8000
EXPOSE 7860
ENV PORT=7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
