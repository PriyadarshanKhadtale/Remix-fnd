"""
Train domain-adversarial veracity model (shared encoder + veracity + domain heads).
Uses FakeNewsNet JSON shards with a `domain` field (e.g. gossipcop, politifact).
Saves checkpoint compatible with backend/run.py when loaded as domain-adversarial.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import core.torch_env  # noqa: E402, F401

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from features.text_analysis_1.domain_adversarial import DomainAdversarialClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_util import device_pretty, resolve_device  # noqa: E402


def _load_rows(path: Path, max_samples: int | None) -> list:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    return rows


class DomainNewsDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        domain_to_id: dict,
        max_length: int = 128,
        max_samples: int | None = None,
    ):
        self.rows = _load_rows(path, max_samples)
        self.tokenizer = tokenizer
        self.domain_to_id = domain_to_id
        self.max_length = max_length
        self.unk_domain = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        text = item["text"]
        label = int(item["label"])
        dom = item.get("domain") or "unknown"
        dom_id = self.domain_to_id.get(dom, self.unk_domain)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "domain": torch.tensor(dom_id, dtype=torch.long),
        }


def build_domain_vocab(
    train_path: Path, val_path: Path, max_samples: int | None = None
) -> dict:
    domains: Counter = Counter()
    for p in (train_path, val_path):
        rows = _load_rows(p, max_samples)
        for r in rows:
            domains[r.get("domain") or "unknown"] += 1
    # stable ordering by frequency
    ordered = [d for d, _ in domains.most_common()]
    return {d: i for i, d in enumerate(ordered)}


def train_epoch(model, loader, optimizer, criterion_v, criterion_d, device, lambda_d, grl_alpha):
    model.train()
    tot_v, tot_d, n = 0.0, 0.0, 0
    correct, total = 0, 0
    for batch in tqdm(loader, desc="train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        d = batch["domain"].to(device)
        optimizer.zero_grad()
        logits_v, logits_d = model(input_ids, attention_mask, grl_alpha=grl_alpha)
        loss_v = criterion_v(logits_v, y)
        loss_d = criterion_d(logits_d, d)
        loss = loss_v + lambda_d * loss_d
        loss.backward()
        optimizer.step()
        tot_v += loss_v.item()
        tot_d += loss_d.item()
        n += 1
        pred = logits_v.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return tot_v / max(n, 1), tot_d / max(n, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion_v, device):
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(loader, desc="eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        logits_v = model(input_ids, attention_mask, grl_alpha=0.0)
        pred = logits_v.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=str(ROOT / "data/processed/fakenewsnet"))
    ap.add_argument("--output_dir", type=str, default=str(ROOT / "models/text_classifier"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda_domain", type=float, default=0.1)
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu | cuda | mps | auto",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap rows per train/val JSON (quick CPU demo)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device_pretty(device)}")

    train_p = data_dir / "train.json"
    val_p = data_dir / "val.json"
    if not train_p.exists() or not val_p.exists():
        raise SystemExit(f"Missing {train_p} or {val_p}")

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    ms = args.max_samples
    dom_vocab = build_domain_vocab(train_p, val_p, max_samples=ms)
    num_domains = len(dom_vocab)
    print(f"Domains ({num_domains}): {list(dom_vocab.keys())[:8]}...")

    train_ds = DomainNewsDataset(train_p, tokenizer, dom_vocab, max_samples=ms)
    val_ds = DomainNewsDataset(val_p, tokenizer, dom_vocab, max_samples=ms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = DomainAdversarialClassifier(num_domains=num_domains).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion_v = nn.CrossEntropyLoss()
    criterion_d = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(args.epochs):
        grl_alpha = min(1.0, (epoch + 1) / max(args.epochs, 1))
        lv, ld, acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion_v,
            criterion_d,
            device,
            args.lambda_domain,
            grl_alpha,
        )
        vacc = evaluate(model, val_loader, criterion_v, device)
        print(f"Epoch {epoch+1}: loss_v={lv:.4f} loss_d={ld:.4f} train_acc={acc:.1f}% val_acc={vacc:.1f}%")
        if vacc > best:
            best = vacc
            torch.save(
                {
                    "model_type": "domain_adversarial",
                    "model_state_dict": model.state_dict(),
                    "num_domains": num_domains,
                    "domain_vocab": dom_vocab,
                    "val_acc": vacc,
                    "epoch": epoch,
                },
                out_dir / "best_model.pt",
            )
            print("  saved best_model.pt")

    print("Done. Point REMIX_VERACITY_CKPT to this file or replace models/text_classifier/best_model.pt")


if __name__ == "__main__":
    main()
