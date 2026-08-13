#!/usr/bin/env python3
"""
Aggregate results across the 20 LOSO folds produced by
review-finetune-motortask_R2_loso.sh.

For each fold directory under
  ./models_weights/MotorTask/exp_author_config_R2_LOSO_val-all_patch_reps-all-subject_independent/
reads config.yaml (val_subject/test_subject) and training_summary.json
(final_results at the epoch selected by validation BACC), then reports a
per-fold table plus mean +/- SD across folds -- the number the reviewer asked
for ("leave-several-subjects-out cross-validation ... report mean/SD").

Usage:
  python3 aggregate_loso_results_R2.py
  python3 aggregate_loso_results_R2.py --parent_dir <other dir> --out summary.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml

DEFAULT_PARENT_DIR = (
    "./models_weights/MotorTask/"
    "exp_author_config_R2_LOSO_val-all_patch_reps-all-subject_independent"
)


def load_fold(fold_dir):
    cfg_path = os.path.join(fold_dir, "config.yaml")
    summary_path = os.path.join(fold_dir, "training_summary.json")
    if not (os.path.exists(cfg_path) and os.path.exists(summary_path)):
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    with open(summary_path) as f:
        summary = json.load(f)
    fr = summary.get("final_results")
    if fr is None:
        return None
    return {
        "fold_dir": os.path.basename(fold_dir),
        "val_subject": cfg.get("val_subject"),
        "test_subject": cfg.get("test_subject"),
        "best_epoch": fr["best_epoch"],
        "val_acc": fr["val_acc"],
        "val_bacc": fr["val_bacc"],
        "test_acc": fr["test_acc"],
        "test_bacc": fr["test_bacc"],
        "test_kappa": fr["test_kappa"],
        "test_f1": fr["test_f1"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type=str, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--out", type=str, default="loso_results_R2.csv")
    args = parser.parse_args()

    fold_dirs = sorted(glob.glob(os.path.join(args.parent_dir, "fold*_test*")))
    rows = []
    missing = []
    for fd in fold_dirs:
        row = load_fold(fd)
        if row is None:
            missing.append(os.path.basename(fd))
        else:
            rows.append(row)

    print(f"Parent dir: {args.parent_dir}")
    print(f"Found {len(fold_dirs)} fold directories, {len(rows)} complete, "
          f"{len(missing)} incomplete/missing.")
    if missing:
        print("  Incomplete folds (no training_summary.json yet):")
        for m in missing:
            print(f"    - {m}")

    if not rows:
        print("No completed folds to aggregate yet.")
        return

    header = (f"{'fold':10s} {'val':7s} {'test':7s} {'ep':>3s} "
              f"{'test_acc':>9s} {'test_bacc':>9s} {'test_kappa':>10s} {'test_f1':>8s}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        val_label = r['val_subject'] if r['val_subject'] else "-"
        print(f"{r['fold_dir']:10s} {val_label:7s} {r['test_subject']:7s} "
              f"{r['best_epoch']:3d} {r['test_acc']:9.4f} {r['test_bacc']:9.4f} "
              f"{r['test_kappa']:10.4f} {r['test_f1']:8.4f}")

    def mean_sd(key):
        vals = np.array([r[key] for r in rows])
        return vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0

    print("\n" + "=" * len(header))
    print(f"N folds = {len(rows)}")
    for key, label in [("test_acc", "Test Accuracy"), ("test_bacc", "Test BACC"),
                        ("test_kappa", "Test Kappa"), ("test_f1", "Test F1")]:
        m, sd = mean_sd(key)
        print(f"  {label:14s}: {m:.4f} +/- {sd:.4f}")

    # save CSV
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-fold results saved to {args.out}")


if __name__ == "__main__":
    main()
