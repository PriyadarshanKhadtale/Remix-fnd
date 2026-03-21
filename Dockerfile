# Render / cloud: build from repository root (context `.`).
# Bundles backend + repo-root models/ and listens on Render's PORT.
FROM python:3.10-slim

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAX_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/ /app/
COPY models/ /app/models/

# Match paths used in load_liar_dataset / HC3 loaders (see backend Dockerfile)
RUN mkdir -p /app/data/fact_checking /app/data/ai_detection \
    && cp -r /app/data_fact_checking/. /app/data/fact_checking/ 2>/dev/null || true \
    && cp -r /app/data_ai_detection/. /app/data/ai_detection/ 2>/dev/null || true

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn run:app --host 0.0.0.0 --port ${PORT:-8000}"]
