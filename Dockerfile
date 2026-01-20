FROM python:3.11-slim

# Install system dependencies required by OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Run with gunicorn (2 workers for handling concurrent requests)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--max-requests", "1000"]
```

---

## Why These System Packages?

| Package | Required For |
|---------|--------------|
| `libgl1-mesa-glx` | OpenCV image rendering |
| `libglib2.0-0` | OpenCV core dependency |
| `libsm6` | X11 session management (OpenCV) |
| `libxext6` | X11 extension (OpenCV) |
| `libxrender1` | X11 rendering (OpenCV) |
| `libgomp1` | OpenMP for parallel processing in NumPy/OpenCV |

---

## Your requirements.txt is Perfect
```
Flask==3.0.3
Pillow==10.4.0
numpy==1.26.4
gunicorn==22.0.0
Werkzeug==3.0.3
opencv-python-headless==4.10.0.84
```

The `opencv-python-headless` is the right choice for servers (no GUI needed).

---

## File Structure for Deployment

Your repo should look like:
```
image-processor-api/
├── Dockerfile           ← Create this (no extension)
├── app.py               ✓ Already have
├── requirements.txt     ✓ Already have
├── .dockerignore        ← Optional
└── README.md            ✓ Already have
```

---

## Create .dockerignore

This speeds up builds by excluding unnecessary files:
```
__pycache__/
*.pyc
*.pyo
.git/
.gitignore
.env
*.md
render.yaml
.DS_Store
*.log
```

---

## Quick Summary for Coolify Setup

1. **Add DNS:** `imgapi` → your server IP
2. **Create Dockerfile** in your repo (filename: `Dockerfile`, no extension)
3. **Push to GitHub**
4. **In Coolify:**
   - New Application → Select your repo
   - Build Pack: **Dockerfile**
   - Port: **5000**
   - Domain: `https://imgapi.limon.dev`
5. **Environment Variables:**
```
   PORT=5000
   FLASK_DEBUG=False
   API_KEY=your-secure-key
   MAX_FILE_SIZE_MB=15
