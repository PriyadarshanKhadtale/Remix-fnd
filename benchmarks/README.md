# Consolidated benchmark runs

All Colab `run_all_benchmarks.py` exports for this project live under **`runs/<run_id>/`**.

| Folder | Head | Use |
|--------|------|-----|
| `runs/20260321T191615Z` | DANN | Manuscript **Table 5** weighted F1 **~85.47%** (DANN); older full export (no `ood_domains` / `paper_components` JSON). |
| `runs/20260322T135702Z_dann` | DANN | Full suite + **`ood_domains.json`** + **`paper_components_probe.json`**. |
| `runs/20260322T135657Z_diml` | DIML | Same suite with **`veracity_diml_eval.json`** instead of DANN. |

Each run includes **`manifest.json`** (step log, `all_ok`). Metrics: `veracity_*_eval.json`, `stance_liar_test.json`, `ai_ensemble_micro.json`, `veracity_latency.json` where applicable.

**Checkpoints:** `best_model.pt` files may be present locally after copy (~330MB each); they are **gitignored**. Re-run `training/scripts/run_all_benchmarks.py` to regenerate, or keep copies outside Git (Drive / release asset).

**Duplicates** you may still have next to the repo: `benchmark_bundle/`, `benchmark_runs/`, etc. under `my projects/` — safe to delete after confirming this tree.

See **`index.json`** for machine-readable metadata.
