"""
Train Text Classification Model
===============================
Training script for the text-based fake news detector.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import core.torch_env  # noqa: E402, F401

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np


class TextClassifier(nn.Module):
    """Text classifier model."""
    def __init__(self, model_name="distilroberta-base", num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


class NewsDataset(Dataset):
    """Dataset for news articles."""
    def __init__(self, data_path, tokenizer, max_length=128, max_samples=None):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    return total_loss / len(loader), 100. * correct / total, f1


def main():
    parser = argparse.ArgumentParser(description="Train text classifier")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('--output_dir', type=str, default='../../models/text_classifier')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap rows per split (quick CPU demo)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("REMIX-FND Text Model Training")
    print("=" * 60)
    
    # Setup
    device = args.device
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    
    # Load tokenizer & data
    print("\nLoading data...")
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    
    ms = args.max_samples
    train_dataset = NewsDataset(data_dir / 'train.json', tokenizer, max_samples=ms)
    val_dataset = NewsDataset(data_dir / 'val.json', tokenizer, max_samples=ms)
    test_dataset = NewsDataset(data_dir / 'test.json', tokenizer, max_samples=ms)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    print("\nInitializing model...")
    model = TextClassifier().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%, F1: {val_f1:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_f1': val_f1
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model")
    
    # Test
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.1f}%")
    print(f"Test F1: {test_f1:.1f}%")
    print("\n✅ Training Complete!")


if __name__ == "__main__":
    main()

