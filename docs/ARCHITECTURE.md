# Architecture

This document matches the **implemented** REMIX-FND stack described in the manuscript (§2.1–2.6, Table A) and the canonical map in **[`SCOPE.md`](../SCOPE.md)**. For API flags and env vars, prefer `SCOPE.md` plus this file.

## API entry points (read this first)

| Entry | Command (from `backend/`) | Role |
|--------|---------------------------|------|
| **Full stack (paper-aligned)** | `python run.py` or `uvicorn run:app --host 0.0.0.0 --port 8000` | `/detect` orchestration: text (+ DANN/DIML checkpoint), optional Monte Carlo dropout, early exit, image, multimodal fusion, six-detector AI stack, evidence + DSRG, tiered explanations. **Docker default** for full stack. |
| **Modular app** | `uvicorn app.main:app` | Slim routers under `app/routes/`; [`app/config.py`](../backend/app/config.py) toggles. **Does not** duplicate full [`run.py`](../backend/run.py) behavior. |
| **Lite** | `uvicorn run_lite:app` | Low-memory hosts (e.g. Render lite). See Dockerfile `REMIX_RENDER_LITE`. |

**Recommendation:** For behavior aligned with the manuscript reference API (§2.6), always run **`run.py`**, not `app.main` alone.

## Three-stage pipeline (manuscript §2.1)

### Stage 1 — Multi-modal feature learning and veracity

- **Text:** DistilRoBERTa encoder + head (`768 → 256 → 2`) in [`run.py`](../backend/run.py) (`TextClassifier`). Checkpoints from training may load [`DomainAdversarialClassifier`](../backend/features/text_analysis_1/domain_adversarial.py) for **domain-adversarial** or **DIML** (`model_type` in checkpoint).
- **Training scripts:** [`train_text_model.py`](../training/scripts/train_text_model.py), [`train_domain_adversarial.py`](../training/scripts/train_domain_adversarial.py), [`train_diml.py`](../training/scripts/train_diml.py).
- **Multimodal fusion:** [`features/multimodal_fusion/fusion.py`](../backend/features/multimodal_fusion/fusion.py) when the client sends `image_base64`, `social_signals`, and/or `published_at_iso` with `use_multimodal_fusion` on `/detect`.
- **Image:** [`features/image_analysis_2/`](../backend/features/image_analysis_2/).

### Stage 2 — Uncertainty-guided evidence retrieval (§2.3, Table 1)

- **Monte Carlo dropout:** [`features/routing/mc_uncertainty.py`](../backend/features/routing/mc_uncertainty.py) — `predict_with_mc_dropout` (e.g. T = 30), `table1_depth_from_fake_variance` → depths **5 / 10 / 20**, `evidence_fast_path` (high decisive probability + low variance).
- **When `mc_dropout_passes = 0` (default):** evidence depth uses **softmax-linear** *k*: `k = 5 + floor(u * 15)` with `u = 1 - p_max`, range **5–20** (see `SCOPE.md`).
- **Evidence stack:** [`features/evidence_retrieval_3/retriever.py`](../backend/features/evidence_retrieval_3/retriever.py), [`dsrg.py`](../backend/features/evidence_retrieval_3/dsrg.py), [`stance_encoder.py`](../backend/features/evidence_retrieval_3/stance_encoder.py). Primary KB: LIAR-derived + hand entries (§2.6).
- **Fast-path env:** `REMIX_MC_FAST_CONF` (default `0.8`), `REMIX_MC_FAST_VAR` (default `0.02`).

### Stage 3 — AI-text ensemble and explanations (§2.4–2.5)

- **Six detectors, fixed simplex weights, 0.55 fused threshold:** [`features/ai_detection_4/detector.py`](../backend/features/ai_detection_4/detector.py) (`AIContentDetector`).
- **Tiers (novice / intermediate / expert):** [`features/explainability_5/explainer.py`](../backend/features/explainability_5/explainer.py).

### Efficiency — early exit

- **Threshold:** max class probability ≥ **0.90** when `enable_early_exit` and no multimodal payload blocking early return ([`early_exit/router.py`](../backend/features/early_exit/router.py), constants in `run.py`).

## `/detect` stage order (`run.py`)

Order when optional flags are enabled (see `SCOPE.md`):

1. Text classification (with optional MC aggregation for routing).
2. Early exit if enabled and confidence ≥ 90% (no multimodal payload).
3. Image analysis if `image_base64` is set.
4. Multimodal fusion if enabled and non-text signals present.
5. AI detection if `check_ai_generated`.
6. Evidence retrieval if `check_evidence` (FAISS + KB; optional DSRG).
7. Explanation if `include_explanation`.

## Key request fields (`POST /detect`, `run.py`)

| Field | Purpose |
|--------|---------|
| `mc_dropout_passes` | `0` = single forward (default, lower latency). `> 0` (e.g. **30**) = MC dropout for Table 1–style variance and depth when combined with `check_evidence`. |
| `check_evidence` | Run evidence retrieval path. |
| `use_evidence_fast_path` | When MC is on, allow skipping heavy retrieval if means/vars satisfy thresholds. |
| `use_multimodal_fusion` | Fuse text with image / social / temporal hints when present. |
| `use_dsrg` | Use DSRG reliability propagation in evidence path. |
| `enable_early_exit` | Allow return after high-confidence text stage. |

## Environment: veracity checkpoint

| Variable | Purpose |
|----------|---------|
| `REMIX_VERACITY_CKPT` | Path to `.pt` veracity checkpoint (wins over run-id). |
| `REMIX_VERACITY_RUN_ID` | Name of a folder under `benchmarks/runs/` (e.g. `20260321T191615Z`). |
| `REMIX_VERACITY_VARIANT` | `dann` (default), `baseline`, or `diml` — loads `benchmarks/runs/<id>/<variant>_veracity/best_model.pt` when present. |
| *(fallback)* | `models/text_classifier/best_model.pt` under repo root, or Docker `/app/models/text_classifier/best_model.pt`. |

`run.py` / `run_hybrid.py` load `backend/.env` and repo-root `.env` into the process environment (via `core.veracity_checkpoint.load_env_files`) before applying the table above. The modular app (`app.main`) uses Pydantic `Settings` with the same variable names.

**Best logged in-domain weighted F1** on the manuscript FakeNewsNet-format test (**n = 3253**) uses the **+DANN** run indexed as **`20260321T191615Z`** in [`benchmarks/index.json`](../benchmarks/index.json). Point `REMIX_VERACITY_CKPT` at the DANN checkpoint file from that export (or your local copy). The **DIML** trainer row uses export **`20260322T135657Z_diml`**; the manuscript notes the archived short DIML schedule **below** +DANN F1 on that split—use DIML only when you intentionally load that checkpoint or retrain with [`train_diml.py`](../training/scripts/train_diml.py).

## Benchmarks and OOD diagnostics

- Frozen JSON under [`benchmarks/runs/`](../benchmarks/runs/) and index [`benchmarks/index.json`](../benchmarks/index.json).
- **Domain-tag subset metrics** (PolitiFact vs GossipCop on `test.json`): [`training/scripts/evaluate_ood_domains.py`](../training/scripts/evaluate_ood_domains.py). **Important:** the manuscript Table 5 footnote states that published REMIX-FND subset numbers used a **baseline** checkpoint from export **`20260322T135702Z_dann`**, **not** the +DANN weights used for the headline in-domain F1—do not mix checkpoints when comparing columns.

## System diagram (logical)

```
┌─────────────────────────────────────────────────────────────────┐
│                     REMIX-FND (run.py)                           │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React) ◄──── HTTP ────► FastAPI (run.py)             │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: text_analysis_1 + optional DomainAdversarial / DIML     │
│           image_analysis_2 · multimodal_fusion                   │
│  Stage 2: routing/mc_uncertainty · early_exit                  │
│           evidence_retrieval_3 (FAISS, KB, DSRG, stance)           │
│  Stage 3: ai_detection_4 (6 detectors) · explainability_5       │
└─────────────────────────────────────────────────────────────────┘
```

## Tech stack

| Layer | Technology |
|-------|------------|
| Frontend | React, Vite |
| Backend | FastAPI, Pydantic |
| ML | PyTorch, Hugging Face Transformers, Sentence-Transformers (evidence), FAISS (optional) |
| Deployment | Docker, uvicorn |

## Related docs

- [`SCOPE.md`](../SCOPE.md) — paper-to-code map, evidence depth modes, KB sources.
- [`docs/API_REFERENCE.md`](API_REFERENCE.md) — HTTP details (if present).
- [`docs/SETUP_GUIDE.md`](SETUP_GUIDE.md) — environment setup.
