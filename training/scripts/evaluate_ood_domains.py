"""
Cross-domain (OOD-style) evaluation on FakeNewsNet-style JSON with a `domain` field.

- Default: macro metrics on the full file (same as evaluate.py) plus optional per-domain F1.
- With --ood_domain: metrics only on rows from that domain (use when the checkpoint was
  trained without that domain, e.g. train on gossipcop-only then test on politifact).

Manuscript alignment (Table 5 footnote, §3.1): REMIX-FND **PolitiFact / GossipCop subset**
macro-F1 diagnostics were reported using a **baseline** checkpoint from export
**20260322T135702Z_dann**, not the **+DANN** checkpoint used for the headline in-domain
weighted F1 (**20260321T191615Z**). When you cite subset vs full-test numbers, use the
**same** `--model_path` as in the benchmark manifest for that row, or label the checkpoint
explicitly so results are not mixed across exports.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(SCRIPTS))
import core.torch_env  # noqa: E402, F401

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from device_util import device_pretty, resolve_device  # noqa: E402
from evaluate import load_model  # noqa: E402


class RowsDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_length: int = 128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        enc = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
            "domain": item.get("domain") or "unknown",
        }


def load_rows(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_domains(
    rows: List[dict],
    include: Tuple[str, ...] = (),
    exclude: Tuple[str, ...] = (),
) -> List[dict]:
    inc = set(include) if include else None
    exc = set(exclude) if exclude else set()
    out = []
    for r in rows:
        d = r.get("domain") or "unknown"
        if inc is not None and d not in inc:
            continue
        if d in exc:
            continue
        out.append(r)
    return out


@torch.no_grad()
def predict_all(model, loader, device) -> Tuple[List[int], List[int], List[str]]:
    y_true, y_pred, doms = [], [], []
    model.eval()
    for batch in tqdm(loader, desc="eval"):
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        pred = logits.argmax(dim=-1).cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(batch["label"].numpy().tolist())
        doms.extend(batch["domain"])
    return y_true, y_pred, doms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--test_data", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument(
        "--ood_domain",
        type=str,
        default=None,
        help="Restrict evaluation to this domain only (typical OOD holdout)",
    )
    ap.add_argument(
        "--exclude_domains",
        type=str,
        default=None,
        help="Comma-separated domains to drop from evaluation (e.g. gossipcop)",
    )
    ap.add_argument(
        "--include_domains",
        type=str,
        default=None,
        help="Comma-separated allowlist; if set, only these domains are evaluated",
    )
    ap.add_argument("--output_json", type=str, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    path = Path(args.test_data)
    rows = load_rows(path)
    if args.max_samples:
        rows = rows[: args.max_samples]

    if args.ood_domain:
        rows = filter_domains(rows, include=(args.ood_domain.strip(),))
    if args.exclude_domains:
        exc = tuple(s.strip() for s in args.exclude_domains.split(",") if s.strip())
        rows = filter_domains(rows, exclude=exc)
    if args.include_domains:
        inc = tuple(s.strip() for s in args.include_domains.split(",") if s.strip())
        rows = filter_domains(rows, include=inc)

    if not rows:
        raise SystemExit("No rows left after domain filters.")

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    ds = RowsDataset(rows, tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size)

    model = load_model(Path(args.model_path), device)
    y_true, y_pred, doms = predict_all(model, loader, device)

    print(f"\nDevice: {device_pretty(device)}")
    print(f"Samples evaluated: {len(y_true)}")
    if args.ood_domain:
        print(f"OOD filter: domain == {args.ood_domain!r}")
    print("\nOverall classification report:")
    print(classification_report(y_true, y_pred, digits=4))
    macro = float(f1_score(y_true, y_pred, average="macro"))
    print(f"Overall macro F1: {100 * macro:.2f}%")

    by_dom: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for yt, yp, d in zip(y_true, y_pred, doms):
        by_dom[d].append((yt, yp))

    print("\nPer-domain macro F1 (diagnostic):")
    for d in sorted(by_dom.keys()):
        pairs = by_dom[d]
        yt = [a for a, _ in pairs]
        yp = [b for _, b in pairs]
        if len(set(yt)) < 2 and len(pairs) < 2:
            f1d = 0.0
        else:
            f1d = float(f1_score(yt, yp, average="macro", zero_division=0))
        print(f"  {d}: n={len(pairs)} macro_f1={100 * f1d:.2f}%")

    if args.output_json:
        import json as json_mod

        out = {
            "n_evaluated": len(y_true),
            "macro_f1": macro,
            "ood_domain": args.ood_domain,
            "per_domain_macro_f1": {
                d: float(
                    f1_score(
                        [a for a, _ in by_dom[d]],
                        [b for _, b in by_dom[d]],
                        average="macro",
                        zero_division=0,
                    )
                )
                for d in by_dom
            },
        }
        Path(args.output_json).write_text(json_mod.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
