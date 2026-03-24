"""
Veracity checkpoint path resolution (aligned with SCOPE.md and docs/ARCHITECTURE.md).

Priority:
1. REMIX_VERACITY_CKPT — explicit filesystem path.
2. REMIX_VERACITY_RUN_ID + REMIX_VERACITY_VARIANT — under
   ``<repo_root>/benchmarks/runs/<id>/<variant>_veracity/best_model.pt``.
3. First existing file among Docker default, ``<repo_root>/models/...``,
   ``<repo_root>/../models/...``.
4. Default return: ``<repo_root>/models/text_classifier/best_model.pt`` (may not exist).
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_files(*paths: Path, override: bool = False) -> None:
    """
    Parse simple ``KEY=value`` lines from ``.env`` files into ``os.environ``.

    Skips comments and blank lines; does not require ``python-dotenv``. Matches how
    operators expect ``REMIX_*`` vars to work alongside ``app.config.Settings``.
    """
    for path in paths:
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if not key:
                continue
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if override or key not in os.environ:
                os.environ[key] = val


def resolve_veracity_model_path(repo_root: Path) -> Path:
    """
    Resolve the DistilRoBERTa veracity (or DANN/DIML) checkpoint path.

    ``repo_root`` should be the REMIX_FND_v2 project directory (parent of ``backend/``).
    """
    if os.environ.get("REMIX_VERACITY_CKPT"):
        return Path(os.environ["REMIX_VERACITY_CKPT"])

    run_id = os.environ.get("REMIX_VERACITY_RUN_ID", "").strip()
    variant = os.environ.get("REMIX_VERACITY_VARIANT", "dann").strip().lower()
    subdirs = {
        "dann": "dann_veracity",
        "baseline": "baseline_veracity",
        "diml": "diml_veracity",
    }
    if run_id and variant in subdirs:
        candidate = (
            repo_root / "benchmarks" / "runs" / run_id / subdirs[variant] / "best_model.pt"
        )
        if candidate.is_file():
            return candidate
        print(
            f"  ⚠ REMIX_VERACITY_RUN_ID={run_id!r} REMIX_VERACITY_VARIANT={variant!r} "
            f"— missing {candidate}"
        )

    for p in (
        Path("/app/models/text_classifier/best_model.pt"),
        repo_root / "models" / "text_classifier" / "best_model.pt",
        repo_root.parent / "models" / "text_classifier" / "best_model.pt",
    ):
        if p.is_file():
            return p
    return repo_root / "models" / "text_classifier" / "best_model.pt"
