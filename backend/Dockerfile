# REMIX-FND Backend Dockerfile
FROM python:3.10-slim

# Avoid OpenMP oversubscription in multi-lib stacks (same defaults as core/torch_env.py)
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories and copy datasets
RUN mkdir -p /app/data/fact_checking /app/data/ai_detection

# Copy LIAR dataset (12.8K fact-checks)
COPY data_fact_checking/ /app/data/fact_checking/

# Copy HC3 dataset (24K human vs AI samples)
COPY data_ai_detection/ /app/data/ai_detection/

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]
