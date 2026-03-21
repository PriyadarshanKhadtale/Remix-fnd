"""
Train a 3-way stance classifier (supports / refutes / neutral) on LIAR (statement + context).
Checkpoint: models/stance_cross_encoder/best_model.pt — loaded by evidence retriever when present.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import core.torch_env  # noqa: E402, F401

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def liar_label_to_stance(label: str) -> Optional[int]:
    label = label.strip().lower()
    if label in ("true", "mostly-true"):
        return 0  # supports (fact-check aligns with claim being substantially true)
    if label in ("false", "pants-fire", "pants on fire"):
        return 1  # refutes
    if label in ("half-true", "barely-true"):
        return 2  # neutral / mixed
    return None


class LiarStanceDataset(Dataset):
    def __init__(
        self,
        tsv_paths: list,
        tokenizer,
        max_length: int = 256,
        max_samples: Optional[int] = None,
    ):
        self.samples: list[tuple[str, str, int]] = []
        for p in tsv_paths:
            if not p.exists():
                continue
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 14:
                        continue
                    label = row[1].strip().lower()
                    statement = row[2].strip()
                    context = row[13].strip() if len(row) > 13 else ""
                    y = liar_label_to_stance(label)
                    if y is None or len(statement) < 10:
                        continue
                    evidence = context if len(context) > 20 else row[2][:200]
                    if len(evidence) < 10:
                        continue
                    self.samples.append((statement, evidence, y))
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break
            if max_samples is not None and len(self.samples) >= max_samples:
                break
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        claim, ev, y = self.samples[idx]
        enc = self.tokenizer(
            claim,
            ev,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.long),
        }


def find_liar_dir() -> Path:
    for p in (
        ROOT / "backend/data_fact_checking",
        ROOT / "data/fact_checking",
    ):
        if (p / "train.tsv").exists():
            return p
    raise SystemExit("LIAR TSV not found under backend/data_fact_checking or data/fact_checking")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liar_dir", type=str, default="")
    ap.add_argument("--output_dir", type=str, default=str(ROOT / "models/stance_cross_encoder"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--model_name", type=str, default="distilroberta-base")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap training pairs (quick CPU demo)",
    )
    args = ap.parse_args()

    liar_dir = Path(args.liar_dir) if args.liar_dir else find_liar_dir()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tsvs = [liar_dir / "train.tsv", liar_dir / "valid.tsv"]
    ds = LiarStanceDataset(tsvs, tokenizer, max_samples=args.max_samples)
    if len(ds) < 50:
        raise SystemExit(f"Too few stance samples ({len(ds)}). Check LIAR paths.")

    n_train = int(0.9 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1} train"):
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                pred = logits.argmax(dim=-1)
                correct += (pred == batch["labels"].to(device)).sum().item()
                total += pred.size(0)
        acc = 100.0 * correct / max(total, 1)
        print(f"Epoch {epoch+1} val acc: {acc:.1f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_name": args.model_name,
                    "num_labels": 3,
                    "model_state_dict": model.state_dict(),
                    "val_acc": acc,
                    "label_names": ["supports", "refutes", "neutral"],
                },
                out_dir / "best_model.pt",
            )
            print("  saved best_model.pt")

    print("Integration: retriever loads models/stance_cross_encoder/best_model.pt automatically.")


if __name__ == "__main__":
    main()
