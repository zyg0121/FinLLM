# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TA-Lib
RUN pip install --no-cache-dir ta-lib

# Copy application code
COPY . /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=finllm/deployment/api.py

# Expose port
EXPOSE 5000

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Entry point
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "4", "--timeout", "120", "finllm.deployment.api:app"]