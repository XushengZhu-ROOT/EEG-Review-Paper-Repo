#!/usr/bin/env python3
"""
Aggregate results across the 20 LOSO folds produced by
scripts/review-finetune-motor6class-LOSO.sh (see save_loso_fold_results() in
src/train_gpt.py).

Reads directly from the {task}_{model}_fold{i:02d}.json / .npz pair written
by save_loso_fold_results() -- everything needed (test_subject, val_subject,
best_epoch, balanced_accuracy) is already in the json, and accuracy/kappa/
macro_f1 are recomputed fresh from the sibling .npz so the printed table
isn't just repeating the json's one saved number. Recomputed
balanced_accuracy is checked against the json's saved value (same
self-consistency check as compute_metrics_from_npz.py) and raises if they
disagree.

Ported from labram_finetune/aggregate_loso_results.py / eegpt_finetune's /
biot_finetune's (same npz/json schema, same logic) with defaults switched to
the neurogpt fold_results directory/model name.

Usage:
  python3 aggregate_loso_results.py
  python3 aggregate_loso_results.py --fold_results_dir ./fold_results_neurogpt --task motor --model neurogpt --out loso_results_neurogpt.csv
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, confusion_matrix,
)


def load_fold(json_path, n_classes):
    npz_path = os.path.splitext(json_path)[0] + ".npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{json_path}: sibling npz not found at {npz_path}")

    with open(json_path) as f:
        meta = json.load(f)

    data = np.load(npz_path)
    y_true, y_pred = data["y_true"], data["y_pred"]

    bacc = balanced_accuracy_score(y_true, y_pred)
    recorded = meta.get("balanced_accuracy")
    if recorded is not None and abs(bacc - recorded) > 1e-6:
        raise RuntimeError(
            f"{json_path}: recomputed balanced_accuracy ({bacc}) does not match "
            f"sidecar json ({recorded}); saved data may be inconsistent."
        )

    # Fix labels=0..n_classes-1 explicitly (not inferred from this fold's
    # y_true/y_pred): some subjects are missing a class entirely, so an
    # inferred label set would produce a smaller matrix for that fold and
    # break summation across folds.
    labels = list(range(n_classes))

    # kappa/macro_f1 只在这一折真实出现过的类别上取平均：某个类完全没有真实样本时
    # (如 Sub05 没有 Horizontal)，recall 分母为 0，这一类是"算不出来"而不是"算出来是0"，
    # 跟 balanced_accuracy_score 的现有行为保持一致（sklearn 内部对没有真实样本的类做
    # NaN 跳过，只在剩下的类里平均），避免模型因为一个本来就不可能预测对的类被扣分。
    # confusion_matrix/per_class_recall 仍用完整的 0..n_classes-1，因为要跨折求和/按位置 zip，形状必须固定。
    labels_present = sorted(set(y_true.tolist()))

    return {
        "fold": meta["fold"],
        "val_subject": meta.get("val_subject"),
        "test_subject": meta["test_subject"],
        "best_epoch": meta["best_epoch"],
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(bacc),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels_present)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels_present, zero_division=0)),
        "train_time_sec": meta.get("train_time_sec"),
        "peak_gpu_mem_mb": meta.get("peak_gpu_mem_mb"),
        "gpu_name": meta.get("gpu_name"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_results_dir", type=str, default="./fold_results_neurogpt")
    parser.add_argument("--task", type=str, default="motor")
    parser.add_argument("--model", type=str, default="neurogpt")
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--out", type=str, default="loso_results_neurogpt.csv")
    parser.add_argument("--cm_out", type=str, default="loso_confusion_matrix_neurogpt.npy",
                        help="where to save the confusion matrix summed across all folds")
    args = parser.parse_args()

    pattern = os.path.join(args.fold_results_dir, f"{args.task}_{args.model}_fold*.json")
    json_paths = sorted(glob.glob(pattern))
    if not json_paths:
        raise FileNotFoundError(f"No files matching {pattern}")

    rows = [load_fold(p, args.n_classes) for p in json_paths]

    header = (f"{'fold':>4s} {'val':7s} {'test':7s} {'ep':>3s} {'n':>5s} "
              f"{'acc':>7s} {'bacc':>7s} {'kappa':>7s} {'f1':>7s} {'time_min':>9s} {'peak_mem_mb':>11s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        val_label = r["val_subject"] if r["val_subject"] else "-"
        time_min = f"{r['train_time_sec']/60:.1f}" if r["train_time_sec"] is not None else "-"
        mem_mb = f"{r['peak_gpu_mem_mb']:.0f}" if r["peak_gpu_mem_mb"] is not None else "-"
        print(f"{r['fold']:4d} {val_label:7s} {r['test_subject']:7s} {r['best_epoch']:3d} "
              f"{r['n_samples']:5d} {r['accuracy']:7.4f} {r['balanced_accuracy']:7.4f} "
              f"{r['kappa']:7.4f} {r['macro_f1']:7.4f} {time_min:>9s} {mem_mb:>11s}")

    def mean_sd(key):
        vals = np.array([r[key] for r in rows])
        return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)

    print("\n" + "=" * len(header))
    print(f"N folds = {len(rows)}  (expected 20 for the full Motor LOSO sweep)")
    for key, label in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced Accuracy"),
                        ("kappa", "Kappa"), ("macro_f1", "Macro F1")]:
        m, sd = mean_sd(key)
        print(f"  {label:18s}: {m:.4f} +/- {sd:.4f}")

    # ===== confusion matrix summed across all folds =====
    cm_sum = sum(r["confusion_matrix"] for r in rows)
    print("\n" + "=" * len(header))
    print(f"Confusion matrix summed across {len(rows)} folds (rows=true, cols=pred, classes 0..{args.n_classes-1}):")
    print(cm_sum)
    np.save(args.cm_out, cm_sum)
    print(f"Saved summed confusion matrix to {args.cm_out}")

    # ===== compute resource usage summary =====
    times = [r["train_time_sec"] for r in rows if r["train_time_sec"] is not None]
    mems = [r["peak_gpu_mem_mb"] for r in rows if r["peak_gpu_mem_mb"] is not None]
    gpu_names = sorted(set(r["gpu_name"] for r in rows if r["gpu_name"]))
    print("\n" + "=" * len(header))
    print("Compute resource usage:")
    if times:
        print(f"  train_time_sec : total={sum(times)/3600:.2f}h  mean/fold={np.mean(times)/60:.1f}min  "
              f"min={min(times)/60:.1f}min  max={max(times)/60:.1f}min")
    if mems:
        print(f"  peak_gpu_mem_mb: mean={np.mean(mems):.0f}MB  max={max(mems):.0f}MB")
    if gpu_names:
        print(f"  gpu_name(s)    : {', '.join(gpu_names)}")

    # per-fold csv (confusion_matrix excluded -- it's a 2D array, saved separately above)
    import csv
    csv_rows = [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in rows]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nPer-fold results saved to {args.out}")


if __name__ == "__main__":
    main()
