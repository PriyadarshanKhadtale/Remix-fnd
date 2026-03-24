# REMIX-FND scope and pipeline

This file is the single map from **paper claims** → **code** → **benchmarks**. It complements the manuscript Table A / §2.6.

## API entry points

| Entry | Command (from `backend/`) | Role |
|--------|---------------------------|------|
| **Full stack (canonical)** | `uvicorn run:app --host 0.0.0.0 --port 8000` | All routes: `/detect`, `/explain`, `/ai-detect`, `/evidence`, `/image-analyze`, MC dropout, DSRG, multimodal fusion, early exit. **Docker default** when not using lite mode. |
| **Modular app** | `uvicorn app.main:app` | Slim router layout under `app/routes/`. Feature toggles in [`app/config.py`](backend/app/config.py) (`.env`). Does **not** duplicate the full `run.py` orchestration. |
| **Lite** | `uvicorn run_lite:app` | Minimal memory footprint for low-RAM hosts (e.g. Render). See Dockerfile `REMIX_RENDER_LITE`. |

## `/detect` pipeline order (`run.py`)

Stages run in order when the client enables optional flags:

1. **Text classification** — DistilRoBERTa (+ optional checkpoint: baseline, DANN, DIML-style `model_type`).
2. **Early exit** — If `enable_early_exit` and confidence ≥ 90% (no multimodal payload), return after stage 1.
3. **Image analysis** — If `image_base64` is set.
4. **Multimodal fusion** — If `use_multimodal_fusion` and non-text signals present (`SocialSignals`, `published_at_iso`, or image).
5. **AI detection** — If `check_ai_generated`.
6. **Evidence retrieval** — If `check_evidence` (FAISS + LIAR-derived KB, optional DSRG).
7. **Explanation** — If `include_explanation`.

## Evidence depth: MC (Table 1) vs softmax *k*

| Mode | When | Depth rule |
|------|------|------------|
| **Default** | `mc_dropout_passes=0` (default on `/detect`) | Linear map `k = 5 + floor(u * 15)` with `u = 1 - p_max` from softmax, range **5–20**. |
| **Paper Table 1** | `mc_dropout_passes > 0` and `check_evidence=true` | `depth_override` from calibrated variance across **T** MC passes; optional **fast path** skips retrieval when decisive mean/var satisfy thresholds. |

Env tuning for fast path: `REMIX_MC_FAST_CONF` (default `0.8`), `REMIX_MC_FAST_VAR` (default `0.02`).

Veracity checkpoint:

| Variable | Role |
|----------|------|
| `REMIX_VERACITY_CKPT` | Explicit path to `.pt` (highest priority). |
| `REMIX_VERACITY_RUN_ID` | Folder name under `benchmarks/runs/` (e.g. `20260321T191615Z`). |
| `REMIX_VERACITY_VARIANT` | `dann` (default), `baseline`, or `diml` — resolves to `benchmarks/runs/<id>/<variant>_veracity/best_model.pt` when that file exists. |

If you copy Colab training outputs into the repo, mirror the script layout (`dann_veracity/best_model.pt`, etc.) inside the indexed run folder.

`run.py` and `run_hybrid.py` call `core.veracity_checkpoint.load_env_files` for `backend/.env` and repo-root `.env` before resolving the checkpoint, so the same `REMIX_*` variables work without exporting them in the shell.

## Knowledge base (EVRS)

| Source | Status |
|--------|--------|
| **LIAR** (+ hand-crafted facts) | **Primary** — loaded by `ExpandedKnowledgeBase` in [`features/evidence_retrieval_3/retriever.py`](backend/features/evidence_retrieval_3/retriever.py). |
| **FEVER** | **Optional / developer** — [`load_fever_dataset.py`](backend/features/evidence_retrieval_3/load_fever_dataset.py) is not wired into the default retriever path. |

## AI-generated text detection (ELDS)

The live ensemble in [`features/ai_detection_4/detector.py`](backend/features/ai_detection_4/detector.py) runs **six** lightweight detectors (weights sum to 1.0):

1. Perplexity-style analyzer  
2. Burstiness  
3. Linguistic / regex patterns  
4. Repetition  
5. Vocabulary richness  
6. HC3 corpus similarity (retrieval-style signal)

These are the **deployable** analogue of the manuscript’s six-detector design (not DeBERTa-v3 / DetectGPT++ binaries in the API path).

## Benchmark artifacts

Frozen JSON lives under [`benchmarks/runs/`](benchmarks/runs/) with index [`benchmarks/index.json`](benchmarks/index.json). Training/eval drivers are under [`training/scripts/`](training/scripts/).

## Route → module map (`run.py`)

| HTTP | Primary modules |
|------|-----------------|
| `POST /detect` | `text_analysis_1`, `routing/mc_uncertainty`, `early_exit`, `image_analysis_2`, `multimodal_fusion`, `ai_detection_4`, `evidence_retrieval_3`, `explainability_5` |
| `POST /explain` | `text_analysis_1`, `explainability_5` |
| `POST /ai-detect` | `ai_detection_4` |
| `POST /evidence` | `evidence_retrieval_3` |
| `POST /image-analyze` | `image_analysis_2` |
| `GET /health` | Status only |

`app.main` routes map to individual features per [`app/routes/*.py`](backend/app/routes/) with `ENABLE_*` flags.
