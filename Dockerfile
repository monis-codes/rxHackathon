# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# ---- ADD THIS BLOCK ----
# Install system-level dependencies required for building Python packages like PyTorch.
# 'build-essential' includes compilers like gcc and g++.
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
# ----------------------

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# This line can be removed if you already did the "bake the model in" step
# COPY ./cross_encoder_model /app/cross_encoder_model

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]