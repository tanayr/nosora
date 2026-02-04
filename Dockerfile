# Multi-stage build for SoraWatermarkCleaner
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies and Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    git \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY uv.lock ./

# Create virtual environment and install dependencies
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install pip and dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies (without editable install to avoid setuptools issues)
RUN pip install aiofiles aiosqlite diffusers einops "fastapi==0.108.0" ffmpeg-python \
    fire greenlet httpx huggingface-hub loguru matplotlib omegaconf opencv-python \
    pandas pydantic python-multipart requests rich ruptures scikit-learn sqlalchemy \
    tqdm transformers ultralytics uuid uvicorn psutil mmcv

# Copy the rest of the application
COPY . .

# Add sorawm to Python path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Build frontend
WORKDIR /app/frontend
RUN npm install && npm run build

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/working_dir /app/logs /app/resources/checkpoint

# Expose port
EXPOSE 5344

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5344/health || exit 1

# Run the server
CMD ["/app/.venv/bin/python", "start_server.py", "--host", "0.0.0.0", "--port", "5344"]
