# 1. Use Slim Python (Smallest possible base)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install System Deps
# FIX: 'libgl1-mesa-glx' is dead. We use 'libgl1' now.
# We also add 'libglib2.0-0' just in case RDKit needs it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Dependencies (CPU Version Only)
COPY requirements.txt .

# CRITICAL: Install CPU-only torch first (Saves ~4GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY . .

# 6. Run the App
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]