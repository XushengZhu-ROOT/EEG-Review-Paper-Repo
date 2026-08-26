#!/usr/bin/env python3
"""
Aggregate results across the 17-fold Stress LOSO sweep
(scripts/review-finetune-stress-LOSO.sh / scripts/STTransformer-review-finetune-stress-LOSO.sh),
reading directly from the {task}_{model}_fold{i:02d}.json / .npz pairs saved by
save_stress_fold_results() in run_binary_supervised.py.

Ported from eegpt_finetune/aggregate_loso_results_stress.py / cbramod_finetune's
(same npz/json schema, same self-consistency check against the sidecar json's
balanced_accuracy, same POOLED section) with defaults switched to biot's fold_results
directory/model name. Works for both biot and st -- pass --model biot/st and matching
--fold_results_dir.

Two Stress-specific additions on top of the plain Motor/Motion aggregate_loso_results.py:

1. roc_auc/pr_auc per fold, computed from the stored y_prob (N,2) array. BIOT/ST's
   Stress head outputs a single logit (BCEWithLogitsLoss), so y_prob is reconstructed
   as [1-sigmoid, sigmoid] by save_stress_fold_results() -- same (N,2) schema as the
   genuine-softmax models, so y_prob[:, 1] is usable directly here without change.
2. A POOLED section: 11 of the 17 stress subjects only ever recorded one condition
   (increase-only or normal-only -- see stress_data/subject_edf_mapping.csv), so
   whichever subject is held out as test/val in a given fold is often single-class.
   cohen_kappa_score/roc_auc_score are undefined on single-class data, so a per-fold
   mean +/- SD is NaN-polluted or misleading for Kappa/ROC-AUC/PR-AUC. The POOLED
   section concatenates all 17 folds' predictions (every subject contributes to test
   exactly once, the defining property of LOSO) and computes ONE overall
   accuracy/balanced_accuracy/kappa/macro_f1/per_class_recall/confusion_matrix/
   roc_auc/pr_auc from that -- this is the number to report, not the per-fold mean.
   As a sanity check, the per-fold-summed confusion matrix must exactly equal the
   pooled confusion matrix.

Usage:
  python3 aggregate_loso_results_stress.py --fold_results_dir ./fold_results_biot_stress --task stress --model biot --out loso_results_biot_stress.csv --cm_out loso_confusion_matrix_biot_stress.npy
  python3 aggregate_loso_results_stress.py --fold_results_dir ./fold_results_st_stress --task stress --model st --out loso_results_st_stress.csv --cm_out loso_confusion_matrix_st_stress.npy
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
    confusion_matrix, recall_score, roc_auc_score, precision_recall_curve, auc,
)

N_CLASSES = 2  # Stress is binary (0=normal, 1=increase); not configurable via CLI.


def load_fold(json_path):
    npz_path = os.path.splitext(json_path)[0] + ".npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{json_path}: sibling npz not found at {npz_path}")

    with open(json_path) as f:
        meta = json.load(f)

    data = np.load(npz_path)
    y_true, y_pred, y_prob = data["y_true"], data["y_pred"], data["y_prob"]
    if y_prob.shape[1] != N_CLASSES:
        raise ValueError(f"{npz_path}: expected y_prob shape (N,{N_CLASSES}), got {y_prob.shape}")

    bacc = balanced_accuracy_score(y_true, y_pred)
    recorded = meta.get("balanced_accuracy")
    if recorded is not None and abs(bacc - recorded) > 1e-6:
        raise RuntimeError(
            f"{json_path}: recomputed balanced_accuracy ({bacc}) does not match "
            f"sidecar json ({recorded}); saved data may be inconsistent."
        )

    labels_present = sorted(set(y_true.tolist()))
    if len(labels_present) < 2:
        roc_auc = float('nan')
        pr_auc = float('nan')
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1], pos_label=1)
        pr_auc = float(auc(recall, precision))

    return {
        "fold": meta["fold"],
        "val_subject": meta.get("val_subject"),
        "test_subject": meta["test_subject"],
        "best_epoch": meta["best_epoch"],
        "n_samples": int(len(y_true)),
        "n_classes_present": len(labels_present),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(bacc),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels_present)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels_present, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "train_time_sec": meta.get("train_time_sec"),
        "peak_gpu_mem_mb": meta.get("peak_gpu_mem_mb"),
        "gpu_name": meta.get("gpu_name"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def mean_sd_nan_safe(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return float('nan'), float('nan'), 0
    return float(valid.mean()), (float(valid.std(ddof=1)) if len(valid) > 1 else 0.0), int(len(valid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_results_dir", type=str, default="./fold_results_biot_stress")
    parser.add_argument("--task", type=str, default="stress")
    parser.add_argument("--model", type=str, default="biot")
    parser.add_argument("--out", type=str, default="loso_results_biot_stress.csv")
    parser.add_argument("--cm_out", type=str, default="loso_confusion_matrix_biot_stress.npy")
    args = parser.parse_args()

    pattern = os.path.join(args.fold_results_dir, f"{args.task}_{args.model}_fold*.json")
    json_paths = sorted(glob.glob(pattern))
    if not json_paths:
        raise FileNotFoundError(f"No files matching {pattern}")

    rows = [load_fold(p) for p in json_paths]

    header = (f"{'fold':>4s} {'val':7s} {'test':7s} {'ep':>3s} {'n':>5s} {'ncls':>4s} "
              f"{'acc':>7s} {'bacc':>7s} {'kappa':>7s} {'f1':>7s} {'roc_auc':>7s} {'pr_auc':>7s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        val_label = r["val_subject"] if r["val_subject"] else "-"
        print(f"{r['fold']:4d} {val_label:7s} {r['test_subject']:7s} {r['best_epoch']:3d} "
              f"{r['n_samples']:5d} {r['n_classes_present']:4d} "
              f"{r['accuracy']:7.4f} {r['balanced_accuracy']:7.4f} "
              f"{r['kappa']:7.4f} {r['macro_f1']:7.4f} {r['roc_auc']:7.4f} {r['pr_auc']:7.4f}")

    print("\n" + "=" * len(header))
    n_single_class = sum(1 for r in rows if r["n_classes_present"] < 2)
    print(f"N folds = {len(rows)}  (expected 17 for the full Stress LOSO sweep; "
          f"{n_single_class} fold(s) have a single-class test subject -- kappa/roc_auc/pr_auc "
          f"are NaN for those, excluded from the per-fold mean below)")
    for key, label in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced Accuracy"),
                        ("kappa", "Kappa"), ("macro_f1", "Macro F1"),
                        ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC")]:
        m, sd, n_valid = mean_sd_nan_safe(rows, key)
        print(f"  {label:18s}: {m:.4f} +/- {sd:.4f}  (valid folds: {n_valid}/{len(rows)})")
    print("  ^ per-fold mean -- for Stress, prefer the POOLED numbers below instead "
          "(every subject tested exactly once, no single-class degeneracy).")

    # ===== POOLED: concatenate every fold's test predictions, compute ONE overall
    # set of metrics. This is the standard LOSO report (equivalent to k-fold CV
    # where the folds partition the whole dataset with no overlap). =====
    y_true_all = np.concatenate([r["y_true"] for r in rows])
    y_pred_all = np.concatenate([r["y_pred"] for r in rows])
    y_prob_all = np.concatenate([r["y_prob"] for r in rows], axis=0)

    pooled_acc = accuracy_score(y_true_all, y_pred_all)
    pooled_bacc = balanced_accuracy_score(y_true_all, y_pred_all)
    pooled_kappa = cohen_kappa_score(y_true_all, y_pred_all)
    pooled_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    pooled_recall = recall_score(y_true_all, y_pred_all, labels=[0, 1], average=None, zero_division=0)
    if len(set(y_true_all.tolist())) < 2:
        # Only possible if aggregating a partial/incomplete set of folds (e.g. mid-sweep,
        # or someone passes --fold_results_dir at a subset). With all 17 real folds this
        # can't happen -- both classes are guaranteed present in the pooled set.
        print("\nWARNING: pooled y_true is single-class (incomplete fold set?) -- "
              "roc_auc/pr_auc are undefined, reporting NaN instead of computing.")
        pooled_roc_auc = float('nan')
        pooled_pr_auc = float('nan')
    else:
        pooled_roc_auc = roc_auc_score(y_true_all, y_prob_all[:, 1])
        precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all[:, 1], pos_label=1)
        pooled_pr_auc = auc(recall, precision)
    pooled_cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

    cm_sum = sum(r["confusion_matrix"] for r in rows)
    if not np.array_equal(cm_sum, pooled_cm):
        raise RuntimeError(
            "Sanity check failed: per-fold-summed confusion matrix != pooled confusion "
            "matrix. Since every fold's test set should be a disjoint subject, these must "
            "match exactly; a mismatch means some sample was double-counted or dropped."
        )

    print("\n" + "=" * len(header))
    print(f"POOLED (all {len(rows)} folds concatenated, {len(y_true_all)} samples total "
          f"-- every subject tested exactly once; report these numbers, not the per-fold mean):")
    print(f"  accuracy={pooled_acc:.4f}  balanced_accuracy={pooled_bacc:.4f}  "
          f"kappa={pooled_kappa:.4f}  macro_f1={pooled_f1:.4f}")
    print(f"  per_class_recall: normal(0)={pooled_recall[0]:.4f}  increase(1)={pooled_recall[1]:.4f}")
    print(f"  roc_auc={pooled_roc_auc:.4f}  pr_auc={pooled_pr_auc:.4f}")
    print(f"  confusion_matrix (rows=true, cols=pred, [normal, increase]):")
    print(f"  {pooled_cm}")

    np.save(args.cm_out, pooled_cm)
    print(f"\nSaved pooled confusion matrix to {args.cm_out}")

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

    # per-fold csv (confusion_matrix/y_true/y_pred/y_prob excluded -- not scalar)
    csv_rows = [{k: v for k, v in r.items() if k not in ("confusion_matrix", "y_true", "y_pred", "y_prob")}
                for r in rows]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nPer-fold results saved to {args.out}")


if __name__ == "__main__":
    main()
