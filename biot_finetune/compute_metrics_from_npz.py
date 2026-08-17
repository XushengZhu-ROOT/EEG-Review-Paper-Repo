#!/usr/bin/env python3
"""
[R2] 从 save_fold_predictions_npz 存的 {task}_{model}_fold{i:02d}.npz 里，
纯粹靠 y_true / y_pred / y_prob 重新算所有下游指标——不读训练日志、不碰模型、不重跑训练。

用法：
  python3 compute_metrics_from_npz.py --npz fold_results/motortask_cbramod_fold00.npz
  python3 compute_metrics_from_npz.py --npz_dir fold_results_st --task motion --model st
  python3 compute_metrics_from_npz.py --npz_dir fold_results_biot --task motion --model biot

如果同目录下有同名 .json（sidecar 元信息），会额外把它里面记录的 balanced_accuracy
拿出来和这里重新算出来的做比对，两边理论上应该完全一致（因为 json 里的数字
本来就是训练脚本在同一次推理里算出来的）——如果对不上，说明保存/复现流程有问题。
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, confusion_matrix, recall_score,
)


def compute_metrics(npz_path, n_classes):
    data = np.load(npz_path, allow_pickle=False)
    for key in ("sample_id", "y_true", "y_pred", "y_prob", "subject_id"):
        if key not in data:
            raise ValueError(f"{npz_path}: missing required key '{key}'")

    sample_id = data["sample_id"]
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_prob = data["y_prob"]
    subject_id = data["subject_id"]

    n = len(sample_id)
    if not (len(y_true) == len(y_pred) == len(y_prob) == len(subject_id) == n):
        raise ValueError(f"{npz_path}: array length mismatch")

    # sample_id 必须是排好序的（save_fold_predictions_npz 写入前排过序）
    if list(sample_id) != sorted(sample_id):
        raise ValueError(f"{npz_path}: sample_id is not sorted; file may be corrupted or hand-edited")

    # y_prob 的 argmax 应该等于 y_pred（自洽性检查）
    prob_argmax = y_prob.argmax(axis=1)
    if not np.array_equal(prob_argmax, y_pred):
        n_mismatch = int((prob_argmax != y_pred).sum())
        raise ValueError(f"{npz_path}: y_prob.argmax() != y_pred for {n_mismatch}/{n} samples")

    # labels 显式固定为 0..n_classes-1：某个 fold 的 test/val 受试者可能完全没有
    # 某一类的样本（如 MotorTask 的 Sub05 没有 Horizontal），不传 labels 的话
    # sklearn 会按这个文件里实际出现的类别数推断形状，per_class_recall/
    # confusion_matrix 就可能比 n_classes 小一维，跨文件比较或求和时对不上。
    labels = list(range(n_classes))

    # kappa/macro_f1 只在这一折真实出现过的类别上取平均：某个类完全没有真实样本时
    # (如 Sub05 没有 Horizontal)，recall 分母为 0，这一类是"算不出来"而不是"算出来是0"，
    # 跟 balanced_accuracy_score 的现有行为保持一致（sklearn 内部对没有真实样本的类做
    # NaN 跳过，只在剩下的类里平均），避免模型因为一个本来就不可能预测对的类被扣分。
    # confusion_matrix/per_class_recall 仍用完整的 0..n_classes-1，因为要跨折求和/按位置 zip，形状必须固定。
    labels_present = sorted(set(y_true.tolist()))

    metrics = {
        "n_samples": n,
        "n_subjects": int(len(np.unique(subject_id))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels_present)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels_present, zero_division=0)),
        "per_class_recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz", type=str, help="Path to a single .npz file")
    group.add_argument("--npz_dir", type=str, help="Directory containing {task}_{model}_fold*.npz files")
    parser.add_argument("--task", type=str, default=None, help="Filter by task name (used with --npz_dir)")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name (used with --npz_dir)")
    parser.add_argument("--n_classes", type=int, default=6, help="Fixed number of classes (0..n_classes-1)")
    args = parser.parse_args()

    if args.npz:
        npz_paths = [args.npz]
    else:
        pattern = f"{args.task or '*'}_{args.model or '*'}_fold*.npz"
        npz_paths = sorted(glob.glob(os.path.join(args.npz_dir, pattern)))
        if not npz_paths:
            raise FileNotFoundError(f"No npz files matching {pattern} under {args.npz_dir}")

    all_bacc = []
    for npz_path in npz_paths:
        print(f"\n=== {npz_path} ===")
        m = compute_metrics(npz_path, args.n_classes)
        print(f"  n_samples={m['n_samples']}  n_subjects={m['n_subjects']}")
        print(f"  accuracy={m['accuracy']:.4f}  balanced_accuracy={m['balanced_accuracy']:.4f}  "
              f"kappa={m['kappa']:.4f}  macro_f1={m['macro_f1']:.4f}")
        print(f"  per_class_recall={['%.4f' % r for r in m['per_class_recall']]}")

        json_path = os.path.splitext(npz_path)[0] + ".json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            recorded = meta.get("balanced_accuracy")
            if recorded is not None:
                match = abs(recorded - m["balanced_accuracy"]) < 1e-6
                status = "OK, matches" if match else "MISMATCH!"
                print(f"  sidecar json balanced_accuracy={recorded:.6f} vs recomputed "
                      f"{m['balanced_accuracy']:.6f}  -> {status}")
                if not match:
                    raise RuntimeError(
                        f"{npz_path}: recomputed balanced_accuracy does not match sidecar json "
                        f"({m['balanced_accuracy']} vs {recorded}); saved data may be inconsistent."
                    )
        all_bacc.append(m["balanced_accuracy"])

    if len(all_bacc) > 1:
        arr = np.array(all_bacc)
        print(f"\n=== Aggregate over {len(all_bacc)} files ===")
        print(f"balanced_accuracy: mean={arr.mean():.4f}  sd={arr.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
