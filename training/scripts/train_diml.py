"""
Train REMIX-FND DIML: joint DANN (L_cls + L_adv) + episodic meta-loss (L_meta) on query sets.

Implements a practical first-order MAML variant: K inner SGD steps on veracity + domain heads
only (encoder shared, gradients flow to encoder via query forward). Checkpoints match
DomainAdversarialClassifier and load in backend/run.py as model_type diml | domain_adversarial.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import core.torch_env  # noqa: E402, F401

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
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


def build_domain_vocab(
    train_path: Path, val_path: Path, max_samples: int | None = None
) -> dict:
    domains: Counter = Counter()
    for p in (train_path, val_path):
        rows = _load_rows(p, max_samples)
        for r in rows:
            domains[r.get("domain") or "unknown"] += 1
    ordered = [d for d, _ in domains.most_common()]
    return {d: i for i, d in enumerate(ordered)}


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


def domain_index_map(dataset: DomainNewsDataset) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, row in enumerate(dataset.rows):
        d = row.get("domain") or "unknown"
        out.setdefault(d, []).append(i)
    return out


def _adapt_param_keys(model: nn.Module) -> List[str]:
    return [
        n
        for n, _ in model.named_parameters()
        if n.startswith("veracity.") or n.startswith("domain.")
    ]


def _params_and_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    d: Dict[str, torch.Tensor] = {}
    for k, v in model.named_parameters():
        d[k] = v
    for k, v in model.named_buffers():
        d[k] = v
    return d


def meta_loss_episode(
    model: DomainAdversarialClassifier,
    param_state: Dict[str, torch.Tensor],
    adapt_keys: List[str],
    batch_s: dict,
    batch_q: dict,
    device: torch.device,
    inner_steps: int,
    inner_lr: float,
    lambda_domain: float,
    grl_alpha: float,
    criterion_v: nn.Module,
    criterion_d: nn.Module,
) -> torch.Tensor:
    """One inner-loop adaptation on support, then veracity CE on query."""
    ids_s = batch_s["input_ids"].to(device)
    mask_s = batch_s["attention_mask"].to(device)
    y_s = batch_s["label"].to(device)
    d_s = batch_s["domain"].to(device)

    ids_q = batch_q["input_ids"].to(device)
    mask_q = batch_q["attention_mask"].to(device)
    y_q = batch_q["label"].to(device)

    state = {k: v for k, v in param_state.items()}

    for _ in range(inner_steps):
        model.train()
        logits_v, logits_d = functional_call(
            model,
            state,
            (ids_s, mask_s),
            {"grl_alpha": grl_alpha},
        )
        loss_inner = criterion_v(logits_v, y_s) + lambda_domain * criterion_d(logits_d, d_s)
        adapt_params = [state[k] for k in adapt_keys]
        grads = torch.autograd.grad(
            loss_inner,
            adapt_params,
            create_graph=True,
            allow_unused=True,
        )
        for k, g in zip(adapt_keys, grads):
            if g is None:
                continue
            state[k] = state[k] - inner_lr * g

    was_training = model.training
    model.eval()
    logits_v_q = functional_call(
        model,
        state,
        (ids_q, mask_q),
        {"grl_alpha": 0.0},
    )
    model.train(was_training)
    return criterion_v(logits_v_q, y_q)


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


def train_epoch(
    model,
    loader,
    optimizer,
    criterion_v,
    criterion_d,
    device,
    lambda_domain: float,
    lambda_meta: float,
    grl_alpha: float,
    train_ds: DomainNewsDataset,
    domain_to_indices: Dict[str, List[int]],
    domains_eligible: List[str],
    adapt_keys: List[str],
    inner_steps: int,
    inner_lr: float,
    tasks_per_step: int,
    support_n: int,
    query_n: int,
) -> Tuple[float, float, float, float]:
    model.train()
    tot_dann, tot_meta = 0.0, 0.0
    n_batches = 0
    n_meta_batches = 0
    sum_task_count = 0
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
        loss_dann = loss_v + lambda_domain * loss_d

        loss_meta_acc: Optional[torch.Tensor] = None
        meta_count = 0
        if domains_eligible and lambda_meta > 0 and tasks_per_step > 0:
            picked = random.sample(
                domains_eligible,
                k=min(tasks_per_step, len(domains_eligible)),
            )
            param_state = _params_and_buffers(model)
            for dom in picked:
                idxs = domain_to_indices[dom]
                need = support_n + query_n
                if len(idxs) < need:
                    continue
                sel = random.sample(idxs, k=need)
                sup_i, q_i = sel[:support_n], sel[support_n:]
                batch_s = default_collate([train_ds[i] for i in sup_i])
                batch_q = default_collate([train_ds[i] for i in q_i])
                try:
                    lm = meta_loss_episode(
                        model,
                        param_state,
                        adapt_keys,
                        batch_s,
                        batch_q,
                        device,
                        inner_steps,
                        inner_lr,
                        lambda_domain,
                        grl_alpha,
                        criterion_v,
                        criterion_d,
                    )
                except RuntimeError:
                    continue
                loss_meta_acc = lm if loss_meta_acc is None else loss_meta_acc + lm
                meta_count += 1

        if loss_meta_acc is not None and meta_count > 0:
            loss = loss_dann + lambda_meta * (loss_meta_acc / meta_count)
            tot_meta += float((loss_meta_acc / meta_count).detach().item())
            n_meta_batches += 1
            sum_task_count += meta_count
        else:
            loss = loss_dann

        loss.backward()
        optimizer.step()

        tot_dann += float(loss_dann.detach().item())
        n_batches += 1

        pred = logits_v.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_meta = tot_meta / max(n_meta_batches, 1) if (lambda_meta > 0 and n_meta_batches) else 0.0
    avg_tasks = sum_task_count / max(n_meta_batches, 1) if n_meta_batches else 0.0
    return (
        tot_dann / max(n_batches, 1),
        avg_meta,
        100.0 * correct / max(total, 1),
        avg_tasks,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=str(ROOT / "data/processed/fakenewsnet"))
    ap.add_argument("--output_dir", type=str, default=str(ROOT / "models/text_classifier"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda_domain", type=float, default=0.1)
    ap.add_argument("--lambda_meta", type=float, default=0.2)
    ap.add_argument("--inner_steps", type=int, default=5, help="MAML inner steps K (paper Fig. 2)")
    ap.add_argument("--inner_lr", type=float, default=0.05, help="Inner SGD step size on heads")
    ap.add_argument("--tasks_per_step", type=int, default=2, help="Episodes per optimizer step")
    ap.add_argument("--support_n", type=int, default=8)
    ap.add_argument("--query_n", type=int, default=16)
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu | cuda | mps | auto",
    )
    ap.add_argument("--max_samples", type=int, default=None)
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

    dom_map = domain_index_map(train_ds)
    need = args.support_n + args.query_n
    domains_eligible = [d for d, ix in dom_map.items() if len(ix) >= need]
    if not domains_eligible:
        print(
            f"⚠️ No domain has ≥{need} train samples; meta term disabled. "
            "Lower --support_n/--query_n or use more data."
        )

    model = DomainAdversarialClassifier(num_domains=num_domains).to(device)
    adapt_keys = _adapt_param_keys(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion_v = nn.CrossEntropyLoss()
    criterion_d = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(args.epochs):
        grl_alpha = min(1.0, (epoch + 1) / max(args.epochs, 1))
        ldann, lmeta, acc, avg_tasks = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion_v,
            criterion_d,
            device,
            args.lambda_domain,
            args.lambda_meta,
            grl_alpha,
            train_ds,
            dom_map,
            domains_eligible,
            adapt_keys,
            args.inner_steps,
            args.inner_lr,
            args.tasks_per_step,
            args.support_n,
            args.query_n,
        )
        vacc = evaluate(model, val_loader, criterion_v, device)
        print(
            f"Epoch {epoch+1}: dann_loss={ldann:.4f} meta_loss={lmeta:.4f} "
            f"train_acc={acc:.1f}% val_acc={vacc:.1f}% avg_meta_tasks={avg_tasks:.2f}"
        )
        if vacc > best:
            best = vacc
            torch.save(
                {
                    "model_type": "diml",
                    "model_state_dict": model.state_dict(),
                    "num_domains": num_domains,
                    "domain_vocab": dom_vocab,
                    "val_acc": vacc,
                    "epoch": epoch,
                    "diml_config": {
                        "lambda_domain": args.lambda_domain,
                        "lambda_meta": args.lambda_meta,
                        "inner_steps": args.inner_steps,
                        "inner_lr": args.inner_lr,
                        "support_n": args.support_n,
                        "query_n": args.query_n,
                    },
                },
                out_dir / "best_model.pt",
            )
            print("  saved best_model.pt")

    print("Done. DIML checkpoint is compatible with backend/run.py (model_type diml).")


if __name__ == "__main__":
    main()
