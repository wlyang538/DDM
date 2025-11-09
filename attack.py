# attack.py
"""
Offline attack & evaluation utilities for DDM experiment.

Features:
- Load dataset from disk (original raw text) and the trained model (from outputs/best_model)
- Two lightweight attack generators:
  1) token_lowfreq_replacement_attack: replace some tokens by sampling from global low-frequency token pool
  2) char_level_bug_attack: character-level corruptions (deletion / swap) inside selected words
- Evaluate model on clean/adversarial data with/without DDM inference-time masking
- Compute metrics: CLA (clean acc), CAA (acc on adversarial), SUCC (attack success rate)
- CLI to run different modes.

Notes:
- Entirely offline-friendly (no internet). Requires:
  - token_freq.json saved (train.py)
  - model checkpoint saved under Config.output_dir / "best_model" (or pass path)
  - datasets saved via datasets.save_to_disk at Config.sst2_path / Config.mr_path
"""

import os
import json
import random
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict

import math
import string

import torch
from transformers import BertTokenizer
from datasets import load_from_disk

from config import Config
from model import DDMClassifier

import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from collections import defaultdict

# -------------------------
# Utilities
# -------------------------
def load_freq_dict(freq_path: str) -> Dict[int, int]:
    with open(freq_path, "r", encoding="utf-8") as f:
        freq_raw = json.load(f)
    # convert keys to int if necessary
    freq = {int(k): int(v) for k, v in freq_raw.items()}
    return freq

def load_model_from_ckpt(ckpt_dir: str, num_labels: int, device: torch.device):
    model = DDMClassifier(num_labels=num_labels)
    model.load(ckpt_dir)
    model.to(device)
    model.eval()
    return model

def load_raw_dataset(dataset_name: str):
    if dataset_name.lower() == "sst2":
        path = Config.sst2_path
    elif dataset_name.lower() == "mr":
        path = Config.mr_path
    else:
        raise ValueError("Unsupported dataset")
    ds = load_from_disk(path)  # DatasetDict
    return ds

# -------------------------
# Attack generators
# -------------------------

def load_adv_examples_by_mode(dataset_name: str,
                              attack_mode: str,
                              adv_path_arg: str = None) -> Tuple[List[str], List[int]]:
    """
    通用的按 dataset + attack_mode 加载预先生成的对抗样本。

    优先级（依次尝试）:
      1) {dataset_name}_{attack_mode}_adv.json
      2) {dataset_name}_adv.json
      3) adv_path_arg (如果传入)
    JSON 期望格式:
      [
        {"label": 1, "text": "it 's a tempt nad often affecting journey ."},
        {"label": 0, "text": "unflinchinglyy raw and heroic"},
        ...
      ]
    返回 (texts, labels)
    """
    dataset_name = dataset_name.lower()
    attack_mode = attack_mode.lower() if attack_mode is not None else ""

    candidates = []
    if dataset_name and attack_mode:
        candidates.append(f"data/adv/{dataset_name}_{attack_mode}_adv.json")
    if dataset_name:
        candidates.append(f"data/adv/{dataset_name}_adv.json")
    if adv_path_arg:
        candidates.append(adv_path_arg)

    found = None
    for p in candidates:
        if p and os.path.exists(p):
            found = p
            break

    if found is None:
        raise FileNotFoundError(
            "No adversarial file found. Tried: "
            + ", ".join(candidates) +
            ". Provide a precomputed adv json or use --adv_path to specify a path."
        )

    # 读取 JSON
    with open(found, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"adv file format error: expected list, got {type(data)} in {found}")

    texts = []
    labels = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"adv file format error at index {idx} in {found}: expected dict, got {type(item)}")
        if "text" not in item or "label" not in item:
            raise ValueError(f"adv file missing 'text' or 'label' at index {idx} in {found}: {item}")
        texts.append(str(item["text"] or ""))
        labels.append(int(item["label"]))

    print(f"[INFO] Loaded {len(texts)} adversarial examples from {found}")
    return texts, labels


# -------------------------
# Tokenize helper for raw adv texts
# -------------------------
def tokenize_texts(texts: List[str], tokenizer: BertTokenizer, max_length=128):
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

# -------------------------
# Evaluation utilities
# -------------------------
def batch_predict(model: DDMClassifier,
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor,
                  device: torch.device,
                  apply_defense: bool = False,
                  freq_dict: Dict[int, int] = None,
                  suspicious_ratio: float = None) -> torch.Tensor:
    """
    Predict labels for a batch. If apply_defense True, apply inference-time insertion masking (model.mask_suspicious_tokens_insert)
    using freq_dict or batch fallback; then forward.
    Returns predicted labels (torch.Tensor on CPU).
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if apply_defense:
        # let model perform masking-insertion using freq_dict
        with torch.no_grad():
            input_ids_masked, attention_mask_masked = model.mask_suspicious_tokens_insert(input_ids, attention_mask,
                                                                                          suspicious_ratio=suspicious_ratio,
                                                                                          freq_dict=freq_dict)
            out = model(input_ids=input_ids_masked, attention_mask=attention_mask_masked,
                        labels=None, apply_train_mask=False, insert_mode=True, apply_inference_mask=True)
            logits = out["logits"]
    else:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=None, apply_train_mask=False, insert_mode=True)
            logits = out["logits"]
    preds = torch.argmax(logits, dim=-1)
    return preds.cpu()

def evaluate_attack(model: DDMClassifier,
                    raw_texts: List[str],
                    labels: List[int],
                    tokenizer: BertTokenizer,
                    device: torch.device,
                    freq_dict: Dict[int, int],
                    apply_defense: bool = True,
                    suspicious_ratio: float = None,
                    batch_size: int = 32) -> Dict[str, float]:
    """
    Compute accuracy on given raw_texts with/without defense.
    Returns dict: {"acc":...,}
    """
    model.eval()
    total = 0
    correct = 0
    n = len(raw_texts)
    for i in range(0, n, batch_size):
        batch_texts = raw_texts[i:i+batch_size]

        batch_labels = torch.tensor(labels[i:i+batch_size], dtype=torch.long)
        input_ids, attention_mask = tokenize_texts(batch_texts, tokenizer, max_length=Config.max_seq_len)
        preds = batch_predict(model, input_ids, attention_mask, device, apply_defense=apply_defense,
                              freq_dict=freq_dict, suspicious_ratio=suspicious_ratio)
        correct += (preds == batch_labels).sum().item()
        total += len(batch_texts)
    acc = correct / total if total > 0 else 0.0
    return {"acc": acc, "correct": correct, "total": total}

# -------------------------
# High level runner
# -------------------------
def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")

    # load freq dict
    freq_path = os.path.join(Config.output_dir, "token_freq.json")
    if not os.path.exists(freq_path):
        raise FileNotFoundError(f"token_freq.json not found at {freq_path}. Run train.py first.")
    freq_dict = load_freq_dict(freq_path)

    # load model
    ckpt_dir = args.model_dir if args.model_dir is not None else os.path.join(Config.output_dir, "best_model")
    print(ckpt_dir)
    
    model = load_model_from_ckpt(ckpt_dir, num_labels=args.num_labels, device=device)

    # load raw dataset
    ds = load_raw_dataset(args.dataset)

    # Prefer validation (dev) if exists, otherwise fallback to test. But if selected split has placeholder labels (e.g. -1),
    # automatically fall back to validation or train.
    # --- 选 split ---
    selected_split = None

    # 如果用户显式要求 --test，就优先用 test split
    if args.test and "test" in ds:
        selected_split = "test"
    else:
        preferred_splits = []
        if "validation" in ds:
            preferred_splits.append("validation")
        if "dev" in ds:
            preferred_splits.append("dev")
        if "test" in ds:
            preferred_splits.append("test")
        if "train" in ds:
            preferred_splits.append("train")
        for s in preferred_splits:
            selected_split = s
            break


    if selected_split is None:
        raise ValueError("No suitable split found in dataset (expected one of validation/dev/test/train).")

    raw_split = ds[selected_split]

    # determine text field key
    possible_text = None
    for k in ["sentence", "text", "review"]:
        if k in raw_split.column_names:
            possible_text = k
            break
    if possible_text is None:
        possible_text = [c for c in raw_split.column_names if c != "label"][0]

    # coerce to plain python lists
    raw_texts = [str(x) for x in raw_split[possible_text]]
    # attempt to coerce labels; if labels are placeholders (e.g., all -1), fallback
    labels = [int(x) for x in raw_split["label"]]

    # detect placeholder labels (common sentinel is -1); if so, try alternative splits
    def is_placeholder_label_list(lbls):
        if len(lbls) == 0:
            return True
        # If all labels are equal and equal to -1, treat as placeholder
        all_same = all((l == lbls[0]) for l in lbls)
        if all_same and lbls[0] == -1:
            return True
        # also treat if label values are out-of-range (e.g., <0)
        if all(l < 0 for l in lbls):
            return True
        return False

    if is_placeholder_label_list(labels):
        # try other splits (validation/dev preferred)
        fallback_found = False
        for s in ["validation", "dev", "train", "test"]:
            if s == selected_split:
                continue
            if s in ds:
                cand = ds[s]
                # check cand labels
                try:
                    cand_labels = [int(x) for x in cand["label"]]
                except Exception:
                    continue
                if not is_placeholder_label_list(cand_labels):
                    print(f"[warn] Selected split '{selected_split}' has placeholder labels; falling back to '{s}'.")
                    selected_split = s
                    raw_texts = [str(x) for x in cand[possible_text]]
                    labels = cand_labels
                    fallback_found = True
                    break
        if not fallback_found:
            raise ValueError(f"No valid split with real labels found in dataset. Checked selected '{selected_split}' and others.")

    # apply limit if requested
    if args.limit > 0:
        raw_texts = raw_texts[:args.limit]
        labels = labels[:args.limit]

    print(f"[info] Loaded {len(raw_texts)} examples from split '{selected_split}', text field '{possible_text}'")


    # optionally limit size for quick runs
    if args.limit > 0:
        raw_texts = raw_texts[:args.limit]
        labels = labels[:args.limit]

    # compute clean acc
    tokenizer = model.tokenizer
    clean_res = evaluate_attack(model, raw_texts, labels, tokenizer, device, freq_dict,
                                apply_defense=False, suspicious_ratio=None, batch_size=args.batch_size)
    print(f"[clean] acc: {clean_res['acc']:.4f} ({clean_res['correct']}/{clean_res['total']})")


    # -------------------------
    # Load adversarial samples (precomputed) for the chosen attack_mode
    # -------------------------
    if args.attack_mode == "load":
        # 原来 load 模式让用户提供 adv_path（制式：label \t text），我们保留这条通道以兼容旧流程
        if not args.adv_path:
            raise ValueError("adv_path required for load mode")
        adv_texts, adv_labels = load_adversarial_from_file(args.adv_path)
    else:
        # 对于其它 mode，我们统一从预计算的 json 中加载：
        # 尝试文件名： {dataset}_{attack_mode}_adv.json -> {dataset}_adv.json -> args.adv_path
        adv_texts, adv_labels = load_adv_examples_by_mode(dataset_name=args.dataset,
                                                        attack_mode=args.attack_mode,
                                                        adv_path_arg=args.adv_path)


    # evaluate adversarial without defense
    adv_res_no_def = evaluate_attack(model, adv_texts, adv_labels, tokenizer, device, freq_dict,
                                     apply_defense=False, suspicious_ratio=None, batch_size=args.batch_size)
    print(f"[adv no defense] acc: {adv_res_no_def['acc']:.4f} ({adv_res_no_def['correct']}/{adv_res_no_def['total']})")

    # evaluate adversarial with DDM inference-time defense (insertion masking)
    adv_res_def = evaluate_attack(model, adv_texts, adv_labels, tokenizer, device, freq_dict,
                                  apply_defense=True, suspicious_ratio=args.suspicious_ratio, batch_size=args.batch_size)
    print(f"[adv with DDM] acc: {adv_res_def['acc']:.4f} ({adv_res_def['correct']}/{adv_res_def['total']})")

    # compute SUCC (attack success rate): proportion of originally-correct examples that become incorrect under attack
    # For that we need per-example behavior; compute preds per example
    # Quick re-run to collect per-example preds (no defense)
    orig_preds = []
    adv_preds_no_def = []
    adv_preds_def = []
    model.eval()
    n = len(raw_texts)
    for i in range(0, n, args.batch_size):
        batch_texts = raw_texts[i:i+args.batch_size]
        batch_labels = labels[i:i+args.batch_size]
        input_ids, attention_mask = tokenize_texts(batch_texts, tokenizer, max_length=Config.max_seq_len)
        p = batch_predict(model, input_ids, attention_mask, device, apply_defense=False, freq_dict=None)
        orig_preds.extend(p.tolist())
    for i in range(0, n, args.batch_size):
        batch_texts = adv_texts[i:i+args.batch_size]
        input_ids, attention_mask = tokenize_texts(batch_texts, tokenizer, max_length=Config.max_seq_len)
        p_adv_no = batch_predict(model, input_ids, attention_mask, device, apply_defense=False, freq_dict=None)
        p_adv_def = batch_predict(model, input_ids, attention_mask, device, apply_defense=True, freq_dict=freq_dict,
                                  suspicious_ratio=args.suspicious_ratio)
        adv_preds_no_def.extend(p_adv_no.tolist())
        adv_preds_def.extend(p_adv_def.tolist())

    # compute SUCC_no_def and SUCC_def
    orig_correct_inds = [i for i, (pred, lab) in enumerate(zip(orig_preds, labels)) if pred == lab]
    if len(orig_correct_inds) == 0:
        succ_no_def = 0.0
        succ_def = 0.0
    else:
        broken_no_def = sum(1 for i in orig_correct_inds if adv_preds_no_def[i] != labels[i])
        broken_def = sum(1 for i in orig_correct_inds if adv_preds_def[i] != labels[i])
        succ_no_def = broken_no_def / len(orig_correct_inds)
        succ_def = broken_def / len(orig_correct_inds)

    print(f"[SUCC] attack success rate (no defense) on originally-correct: {succ_no_def:.4f}")
    print(f"[SUCC] attack success rate (with DDM) on originally-correct: {succ_def:.4f}")

    # Save adv samples if requested
    if args.save_adv:
        outf = args.save_adv if isinstance(args.save_adv, str) else os.path.join(Config.output_dir, "adv_examples.txt")
        with open(outf, "w", encoding="utf-8") as f:
            for lab, text in zip(adv_labels, adv_texts):
                f.write(f"{lab}\t{text}\n")
        print(f"[info] Saved adv samples to {outf}")

# -------------------------
# Helper to load adv file if user precomputed adversarial samples
# -------------------------
def load_adversarial_from_file(path: str) -> Tuple[List[str], List[int]]:
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 1:
                # fallback: assume only text present and labels unknown -> set dummy 0
                labels.append(0)
                texts.append(parts[0])
            else:
                labels.append(int(parts[0]))
                texts.append(parts[1])
    return texts, labels

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", help="sst2 or mr")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to saved model dir (if not provided, uses outputs/best_model)")
    parser.add_argument(
        "--attack_mode",
        type=str,
        default="token_repl",
        choices=["token_repl", "char_bug", "textfooler", "deepwordbug", "textfooler_orig", "load"],
        help="Which attack to run; supported: token_repl, char_bug, textfooler, deepwordbug, load"
    )
    parser.add_argument("--replace_ratio", type=float, default=0.2, help="Proportion of tokens/words to corrupt")
    parser.add_argument("--top_k_lowfreq", type=int, default=200, help="Low-frequency pool size for token replacement")
    parser.add_argument("--suspicious_ratio", type=float, default=Config.suspicious_ratio, help="DDM suspicious ratio for inference masking")
    parser.add_argument("--limit", type=int, default=1000, help="Limit dataset size (0 = all)")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size, help="Batch size for evaluation")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels")
    parser.add_argument("--save_adv", type=str, default=None, help="If set, save generated adv samples to this path")
    parser.add_argument("--adv_path", type=str, default=None, help="If attack_mode=load, path to adv file")
    parser.add_argument("--sim_threshold", type=float, default=0.8, help="semantic similarity threshold for TextFooler original")
    parser.add_argument("--sentence_encoder", type=str, default=None,
                        help="(optional) sentence-transformers model name or local path to use for semantic similarity. Must be installed and cached by user; code will not download models automatically.")
    parser.add_argument("--test", action="store_true",
                    help="If set, force evaluation on the test split instead of validation/dev/train.")
    args = parser.parse_args()

    run_eval(args)
