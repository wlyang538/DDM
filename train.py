# train.py
"""
The training script (offline friendly) is used to train the DDMClassifier with the insertion mask strategy (M [masks] inserted after CLS).
After training, the best model is stored in outputs/best_model/ and the token frequency statistics are stored in outputs/token_freq.json.
The training and evaluation processes do not rely on online resources, and all models and data are loaded locally.

"""

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from datasets import load_from_disk

from config import Config
from model import DDMClassifier
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_dataset(dataset, tokenizer, split_name: str):
    """
    Tokenize a datasets.Dataset object in batched mode using the provided tokenizer.
    Returns a tokenized dataset (datasets.Dataset) with torch tensors format.
    """
    def _tok(batch):
        # detect text field: common keys are 'sentence', 'text', 'review'
        if "sentence" in batch:
            text_key = "sentence"
        elif "text" in batch:
            text_key = "text"
        elif "review" in batch:
            text_key = "review"
        else:
            # fallback: take first non-label column
            text_key = [k for k in batch.keys() if k != "label"][0]

        enc = tokenizer(batch[text_key],
                        padding="max_length",
                        truncation=True,
                        max_length=Config.max_seq_len)
        return enc

    tokenized = dataset.map(_tok, batched=True, remove_columns=[c for c in dataset.column_names if c not in ("label",)])
    # Ensure label exists and is named "label"
    if "label" not in tokenized.column_names:
        raise ValueError("Dataset does not contain 'label' column. Please ensure label column exists.")

    tokenized.set_format(type="torch",
                         columns=["input_ids", "attention_mask", "label"])
    print(f"[info] Tokenized {split_name} dataset, num rows: {len(tokenized)}")
    return tokenized


def build_dataloader(tokenized_dataset, split: str, shuffle: bool, batch_size: int):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)


def compute_and_save_token_freq(tokenized_train_dataset, tokenizer, out_path: str):
    """
    Compute token frequency across the *train* dataset and save as JSON mapping token_id -> freq
    """
    freq = {}
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.mask_token_id}
    print("[info] Computing token frequency on train set...")
    for ex in tqdm(tokenized_train_dataset):
        input_ids = ex["input_ids"].tolist()
        attention_mask = ex["attention_mask"].tolist()
        for tid, attn in zip(input_ids, attention_mask):
            if attn == 0:
                continue
            if tid in special_ids:
                continue
            freq[tid] = freq.get(tid, 0) + 1
    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(freq, f)
    print(f"[info] Token freq saved to {out_path} (vocab ids count: {len(freq)})")
    return freq


def evaluate(model: DDMClassifier, dataloader: DataLoader, device: torch.device, freq_dict: dict = None):
    """
    Evaluate on clean data (no inference masking) and return accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # No masking for clean eval
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None,
                        apply_train_mask=False, insert_mode=True)
            logits = out["logits"]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc


def train(args):
    set_seed(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # Load datasets from disk (offline)
    if args.dataset.lower() == "sst2":
        data_path = Config.sst2_path
    elif args.dataset.lower() == "mr":
        data_path = Config.mr_path
    else:
        raise ValueError("Unsupported dataset. Choose 'sst2' or 'mr'.")

    print(f"[info] Loading dataset from {data_path}")
    ds = load_from_disk(data_path)  # expects a DatasetDict with splits
    # Some disk-saved datasets may have 'train', 'validation', 'test'
    train_ds = ds["train"]
    val_ds = ds.get("validation", ds.get("dev", None))
    if val_ds is None:
        # fallback: split train a bit
        print("[warn] No validation split found. Splitting train 90/10 for validation.")
        split = train_ds.train_test_split(test_size=0.1, seed=Config.seed)
        train_ds = split["train"]
        val_ds = split["test"]

    # Initialize model and tokenizer (local only)
    temp_model = DDMClassifier(num_labels=args.num_labels)  # loads tokenizer inside
    tokenizer = temp_model.tokenizer  # reuse
    model = temp_model.to(device)

    # Tokenize datasets (offline)
    tokenized_train = tokenize_dataset(train_ds, tokenizer, split_name="train")
    tokenized_val = tokenize_dataset(val_ds, tokenizer, split_name="validation")

    # Compute & save token freq (for suspicious scoring in inference)
    freq_out = os.path.join(Config.output_dir, "token_freq.json")
    if args.force_recompute_freq or not os.path.exists(freq_out):
        freq_dict = compute_and_save_token_freq(tokenized_train, tokenizer, freq_out)
    else:
        with open(freq_out, "r", encoding="utf-8") as f:
            freq_dict = json.load(f)
        print(f"[info] Loaded existing token freq from {freq_out}")

    # DataLoaders
    train_loader = build_dataloader(tokenized_train, "train", shuffle=True, batch_size=Config.batch_size)
    val_loader = build_dataloader(tokenized_val, "validation", shuffle=False, batch_size=Config.batch_size)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

    best_val_acc = 0.0
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        print(f"=== Epoch {epoch+1}/{args.epochs} ===")
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for batch in pbar:
            model.train()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # -----------------------
            # Use a plug-in mask during training
            # -----------------------
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        apply_train_mask=True,
                        train_mask_ratio=Config.mask_ratio,
                        insert_mode=True)  # insert_mode=True
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % args.log_interval == 0:
                pbar.set_postfix({"loss": float(loss.item())})

        # End epoch -> evaluate on validation (clean)
        val_acc = evaluate(model, val_loader, device, freq_dict=freq_dict)
        print(f"[info] Validation accuracy (clean) after epoch {epoch+1}: {val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_dir = os.path.join(Config.output_dir, "best_model")
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"[info] New best val acc {best_val_acc:.4f} -> saving model to {ckpt_dir}")
            model.save(ckpt_dir)
            # Also save freq dict (already saved before), ensure it lives with model
            with open(os.path.join(ckpt_dir, "token_freq.json"), "w", encoding="utf-8") as f:
                json.dump(freq_dict, f)

    print(f"[info] Training finished. Best val acc: {best_val_acc:.4f}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", help="Dataset to train on: sst2 or mr")
    parser.add_argument("--epochs", type=int, default=Config.num_epochs, help="Number of epochs")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels (2 for sst2/mr binary)")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging interval in steps")
    parser.add_argument("--force_recompute_freq", action="store_true", help="Force recompute token freq")
    args = parser.parse_args()

    train(args)
