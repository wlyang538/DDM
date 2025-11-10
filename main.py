# main.py
"""
Main pipeline for DDM experiment (offline version).

Process:
1. train: call train.py's interface (or internal function) to train the model, generating best_model/ and token_freq.json.
2. attack evaluation phase: call attack.py function and evaluate under clean/adv (no defense)/adv (with DDM);
3. Output the final metrics table and save the results to outputs/metrics.json.
"""

import os
import json
import argparse
import torch

from train import train
from attack import run_eval
from config import Config


def main(args):
    print("========== [Stage 1] Training ==========")
    # Training phase
    train_args = argparse.Namespace(
        dataset=args.dataset,
        epochs=args.epochs,
        num_labels=args.num_labels,
        log_interval=args.log_interval,
        force_recompute_freq=args.force_recompute_freq,
    )
    train(train_args)

    print("\n========== [Stage 2] Attack & Defense Evaluation ==========")
    attack_args = argparse.Namespace(
        dataset=args.dataset,
        model_dir=os.path.join(Config.output_dir, "best_model"),
        attack_mode=args.attack_mode,
        replace_ratio=args.replace_ratio,
        top_k_lowfreq=args.top_k_lowfreq,
        suspicious_ratio=args.suspicious_ratio,
        limit=args.limit,
        batch_size=args.batch_size,
        num_labels=args.num_labels,
        save_adv=None,
        adv_path=None,
        debug=False
    )

    # Attack & Defense Evaluation phase
    run_eval(attack_args)

    print("\n========== [Pipeline Done] ==========")
    print(f"All outputs saved under: {Config.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", help="sst2 or mr")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--force_recompute_freq", action="store_true")

    # attack & defense eval args
    parser.add_argument("--attack_mode", type=str, default="token_repl", choices=["textfooler", "deepwordbug", "pwws", "load", "token_repl"]) 
    parser.add_argument("--replace_ratio", type=float, default=0.2)
    parser.add_argument("--top_k_lowfreq", type=int, default=200)
    parser.add_argument("--suspicious_ratio", type=float, default=Config.suspicious_ratio)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)

    args = parser.parse_args()

    main(args)
