FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables for model caching
ENV U2NET_HOME=/app/.u2net
ENV HF_HOME=/app/.cache/huggingface
ENV HOME=/app

# Create cache directories
RUN mkdir -p /app/.u2net /app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download withoutBG models (4-stage pipeline ~320MB)
# This prevents download on first request
RUN python -c "from withoutbg import WithoutBG; WithoutBG.opensource()" || echo "withoutbg models will download on first use"

# Pre-download rembg model as fallback
RUN python -c "from rembg import new_session; new_session('isnet-general-use')" || echo "rembg model will download on first use"

COPY app.py .

EXPOSE 5000

# Increased timeout for model loading and processing
# Workers=1 to prevent multiple model loads (saves memory)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--graceful-timeout", "120"]