FROM python:3.11-slim

# Set environment variables for better performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with longer timeout and retries
RUN pip install --no-cache-dir --timeout=1000 --retries=3 -r requirements.txt

# Copy application code
COPY . .

# Don't expose a fixed port - Render will provide PORT environment variable
# EXPOSE 8080

# Run the application using Render's PORT environment variable
# Change this to match your actual app structure
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
