#!/usr/bin/env python3
"""
从 save_loso_fold_results() 存的 {task}_{model}_fold{i:02d}.npz 里，
纯粹靠 y_true / y_pred / y_prob 重新算所有下游指标——不读训练日志、不碰模型、不重跑训练。

用法：
  python3 compute_metrics_from_npz.py --npz fold_results_labram/motor_labram_fold00.npz
  python3 compute_metrics_from_npz.py --npz_dir fold_results_labram --task motor --model labram

如果同目录下有同名 .json（sidecar 元信息），会额外把它里面记录的 balanced_accuracy
拿出来和这里重新算出来的做比对，两边理论上应该完全一致（因为 json 里的数字
本来就是训练脚本在同一次推理里算出来的）——如果对不上，说明保存/复现流程有问题。

与 cbramod_finetune/compute_metrics_from_npz.py、biot_finetune/compute_metrics_from_npz.py、
eegpt_finetune/compute_metrics_from_npz.py 逻辑完全一致（npz/json 是同一套 schema），
只是拷到 labram_finetune 下方便直接用。
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, confusion_matrix, recall_score,
)

# [Sleep] ISRUC 5 分类的可读标签，仅用于打印；npz/json 里仍然存 0..4 的 int
SLEEP_STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def stage_labels(n_classes):
    if n_classes == len(SLEEP_STAGE_NAMES):
        return SLEEP_STAGE_NAMES
    return [str(i) for i in range(n_classes)]


def bootstrap_ci_by_subject(y_true, y_pred, subject_id, n_classes, n_bootstrap=1000, seed=0, alpha=0.05):
    """[Sleep] 按受试者做 block bootstrap：每次从这份 npz 里出现过的 subject_id 集合中
    有放回重采样同样多个受试者，取被抽中受试者的全部样本汇总重新算一次指标，重复
    n_bootstrap 次，取 (alpha/2, 1-alpha/2) 分位数。

    用受试者而不是逐样本做 bootstrap，是因为 Sleep 的切分是按 epoch 整体分层随机切分
    （不是 LOSO），同一受试者相邻 epoch 高度自相关，逐样本 bootstrap 会低估真实不确定性、
    给出偏窄的假 CI；受试者数少（比如 ISRUC 只有 10 个）时这里算出来的 CI 天然会比较宽，
    这是如实反映小样本不确定性，不是 bug。
    """
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject_id)
    n_subj = len(subjects)
    labels = list(range(n_classes))
    subj_to_idx = {s: np.where(subject_id == s)[0] for s in subjects}

    bacc_samples, kappa_samples, macro_f1_samples, per_class_recall_samples = [], [], [], []
    # 小样本 bootstrap 时，某次重采样很常见地不会覆盖全部 n_classes 个类别（比如
    # ISRUC 只有 10 个受试者，重采样出来的子集完全没有 REM 很正常），这时
    # recall_score/f1_score 对"预测了但真实标签里没有的类别"会打印
    # UserWarning，重复 n_bootstrap 次会刷屏；zero_division=0 已经把这种情况
    # 正确处理成 0（不是错误），这里只是压掉多余的警告文本，不影响计算结果。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for _ in range(n_bootstrap):
            chosen = rng.choice(subjects, size=n_subj, replace=True)
            idx = np.concatenate([subj_to_idx[s] for s in chosen])
            yt, yp = y_true[idx], y_pred[idx]
            labels_present = sorted(set(yt.tolist()))
            if len(labels_present) == 0:
                continue
            bacc_samples.append(balanced_accuracy_score(yt, yp))
            kappa_samples.append(cohen_kappa_score(yt, yp, labels=labels_present))
            macro_f1_samples.append(f1_score(yt, yp, average="macro", labels=labels_present, zero_division=0))
            per_class_recall_samples.append(recall_score(yt, yp, labels=labels, average=None, zero_division=0))

    if len(bacc_samples) == 0:
        raise RuntimeError("bootstrap_ci_by_subject: every resample was empty; check subject_id/y_true.")

    def _pct(samples):
        lo, hi = np.percentile(np.asarray(samples, dtype=np.float64), [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return [float(lo), float(hi)]

    per_class_recall_arr = np.stack(per_class_recall_samples, axis=0)  # (n_bootstrap_eff, n_classes)
    return {
        "ci_n_bootstrap": n_bootstrap,
        "ci_n_subjects": int(n_subj),
        "balanced_accuracy_ci95": _pct(bacc_samples),
        "kappa_ci95": _pct(kappa_samples),
        "macro_f1_ci95": _pct(macro_f1_samples),
        "per_class_recall_ci95": [_pct(per_class_recall_arr[:, c]) for c in range(n_classes)],
    }


def compute_metrics(npz_path, n_classes, ci=False, n_bootstrap=1000, ci_seed=0):
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

    # sample_id 必须是排好序的（save_loso_fold_results 写入前排过序）
    if list(sample_id) != sorted(sample_id):
        raise ValueError(f"{npz_path}: sample_id is not sorted; file may be corrupted or hand-edited")

    # y_prob 的 argmax 应该等于 y_pred（自洽性检查）
    prob_argmax = y_prob.argmax(axis=1)
    if not np.array_equal(prob_argmax, y_pred):
        n_mismatch = int((prob_argmax != y_pred).sum())
        raise ValueError(f"{npz_path}: y_prob.argmax() != y_pred for {n_mismatch}/{n} samples")

    # labels 显式固定为 0..n_classes-1：某个 fold 的 test/val 受试者可能完全没有
    # 某一类的样本（如 Sub05 没有 Horizontal），不传 labels 的话 sklearn 会按这个
    # 文件里实际出现的类别数推断形状，per_class_recall/confusion_matrix 就可能
    # 比 n_classes 小一维，跨文件比较或求和时对不上。
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

    if ci:
        metrics.update(bootstrap_ci_by_subject(
            y_true, y_pred, subject_id, n_classes, n_bootstrap=n_bootstrap, seed=ci_seed,
        ))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz", type=str, help="Path to a single .npz file")
    group.add_argument("--npz_dir", type=str, help="Directory containing {task}_{model}_fold*.npz files")
    parser.add_argument("--task", type=str, default=None, help="Filter by task name (used with --npz_dir)")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name (used with --npz_dir)")
    parser.add_argument("--n_classes", type=int, default=6, help="Fixed number of classes (0..n_classes-1)")
    parser.add_argument("--ci", action="store_true",
                        help="Also compute subject-level block-bootstrap 95%% CI (Sleep-style npz needs subject_id)")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap resamples for --ci")
    parser.add_argument("--ci_seed", type=int, default=0, help="RNG seed for --ci bootstrap")
    args = parser.parse_args()

    if args.npz:
        npz_paths = [args.npz]
    else:
        # 兼容两种命名：LOSO 的 {task}_{model}_fold{i:02d}.npz 和 Sleep 的
        # {task}_{model}_val.npz / {task}_{model}_test.npz
        pattern = f"{args.task or '*'}_{args.model or '*'}_*.npz"
        npz_paths = sorted(glob.glob(os.path.join(args.npz_dir, pattern)))
        if not npz_paths:
            raise FileNotFoundError(f"No npz files matching {pattern} under {args.npz_dir}")

    labels_for_print = stage_labels(args.n_classes)
    all_bacc = []
    for npz_path in npz_paths:
        print(f"\n=== {npz_path} ===")
        m = compute_metrics(npz_path, args.n_classes, ci=args.ci, n_bootstrap=args.n_bootstrap, ci_seed=args.ci_seed)
        print(f"  n_samples={m['n_samples']}  n_subjects={m['n_subjects']}")
        print(f"  accuracy={m['accuracy']:.4f}  balanced_accuracy={m['balanced_accuracy']:.4f}  "
              f"kappa={m['kappa']:.4f}  macro_f1={m['macro_f1']:.4f}")
        print(f"  per_class_recall={dict(zip(labels_for_print, ['%.4f' % r for r in m['per_class_recall']]))}")
        if 'balanced_accuracy_ci95' in m:
            lo, hi = m['balanced_accuracy_ci95']
            print(f"  balanced_accuracy 95% CI (n_subjects={m['ci_n_subjects']}, "
                  f"n_bootstrap={m['ci_n_bootstrap']}): [{lo:.4f}, {hi:.4f}]")
            lo, hi = m['macro_f1_ci95']
            print(f"  macro_f1 95% CI: [{lo:.4f}, {hi:.4f}]")
            lo, hi = m['kappa_ci95']
            print(f"  kappa 95% CI: [{lo:.4f}, {hi:.4f}]")
            for name, (lo, hi) in zip(labels_for_print, m['per_class_recall_ci95']):
                print(f"  per_class_recall[{name}] 95% CI: [{lo:.4f}, {hi:.4f}]")

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
