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

# CPU-only PyTorch: default pip on Linux pulls CUDA wheels (~GB + high RAM). Render has no GPU.
COPY backend/requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements-docker.txt

COPY backend/ /app/
COPY models/ /app/models/

# Match paths used in load_liar_dataset / HC3 loaders (see backend Dockerfile)
RUN mkdir -p /app/data/fact_checking /app/data/ai_detection \
    && cp -r /app/data_fact_checking/. /app/data/fact_checking/ 2>/dev/null || true \
    && cp -r /app/data_ai_detection/. /app/data/ai_detection/ 2>/dev/null || true

# Bake DistilRoBERTa into the image so runtime does not wait on Hugging Face download.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface \
    && python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('distilroberta-base'); AutoModel.from_pretrained('distilroberta-base')"

EXPOSE 8000

# Full API: run:app. Free-tier OOM? Set env REMIX_RENDER_LITE=1 on Render to use rule-based lite API.
CMD ["sh", "-c", "if [ \"${REMIX_RENDER_LITE}\" = \"1\" ]; then exec uvicorn run_lite:app --host 0.0.0.0 --port ${PORT:-8000}; else exec uvicorn run:app --host 0.0.0.0 --port ${PORT:-8000}; fi"]
