# 1. Use Slim Python (Smallest possible base)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install System Deps (Required for RDKit)
# This prevents "ImportError: libXrender.so.1"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Dependencies (CPU Version Only)
# We copy requirements first to leverage Docker caching
COPY requirements.txt .

# CRITICAL: Install CPU-only torch first (Saves ~4GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
# Since we are inside 'backend', this copies main.py directly to /app/main.py
COPY . .

# 6. Run the App
# CRITICAL CHANGE: Use "main:app", NOT "backend.main:app"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]