FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Optional: keep model cache inside container path
ENV U2NET_HOME=/app/.u2net

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Warm rembg model during build (no ARG lines inside here!)
RUN python - <<'PY'
from rembg import new_session
new_session("isnet-general-use")
print("rembg model cache warmed")
PY

COPY app.py .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120"]
