FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn lightgbm mlforecast window_ops \
    fastapi "uvicorn[standard]" pydantic \
    && rm -rf /root/.cache/pip

# Copy application code
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
