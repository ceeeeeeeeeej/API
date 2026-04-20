# Use a stable, tested Python image
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install optimized dependencies (tflite-runtime)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
# IMPORTANT: Ensure garbage_classifier.tflite is in this directory
COPY . .

# Cloud Run sets the PORT environment variable (default 8080)
ENV PORT 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Use Gunicorn for production serving
# We bind to the port provided by the environment variable
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app
