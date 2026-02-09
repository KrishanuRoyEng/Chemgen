# 1. Base Image (Slim = Tiny)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install System Dependencies (Only what RDKit needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY requirements first (for caching)
COPY requirements.txt .

# 5. Install PyTorch CPU-ONLY (The Magic Step)
# This prevents downloading 4GB of CUDA drivers
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 6. Install other deps
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your code
COPY backend/ ./backend/

# 8. Run
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]