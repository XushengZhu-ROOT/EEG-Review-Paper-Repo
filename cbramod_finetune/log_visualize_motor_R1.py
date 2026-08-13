#!/usr/bin/env python3
"""
MotorTask 训练日志可视化（R1 / 旧实验均可）。

用法示例：
  # R1 受试者独立（默认）
  python log_visualize_motor_R1.py

  # 换回旧的随机 epoch 结果对比
  python log_visualize_motor_R1.py \
    --exp_dir ./models_weights/MotorTask/exp_author_config-all_patch_reps-all

  # 无显示窗口时保存图片到目录
  python log_visualize_motor_R1.py --save_dir ./viz_R1 --no_show
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import yaml

# 无显示器时用 Agg；有显示器且需要弹窗时可用 --no_show 关掉显示
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


MOTOR_CLASS_NAMES = [
    "Walking (straight)",
    "Walking (turning)",
    "Head shaking",
    "Head nodding",
    "Picking up an object",
    "Stairs climbing and descending",
]


def plot_metric(epochs, y_values, ylabel, title, save_path=None, show=False):
    plt.figure(figsize=(5, 3))
    plt.plot(epochs, y_values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_trainNval_metric(epochs, y_values_val, y_values_test, ylabel, title, save_path=None, show=False):
    plt.figure(figsize=(5, 3))
    plt.plot(epochs, y_values_val, marker="o", label="val")
    plt.plot(epochs, y_values_test, marker="o", label="test")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm, class_names, title, save_path=None, show=False, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage (%)"},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()
    print(f"\n{title} - Raw Counts:")
    print(cm)


def get_metrics_motor(log_file):
    metrics = {
        "train_loss": [],
        "val_acc": [],
        "val_bacc": [],
        "val_kappa": [],
        "val_f1": [],
        "test_acc": [],
        "test_bacc": [],
        "test_kappa": [],
        "test_f1": [],
        "learning_rate": [],
        "epoch_time_seconds": [],
        "epoch_time_minutes": [],
    }
    epochs = []
    with open(log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)
        for line in logs:
            epochs.append(line["epoch"])
            for key in metrics.keys():
                metrics[key].append(line[key] if key in line else None)
    return metrics, epochs


def format_sci(value):
    if value is None:
        return ""
    if value < 1:
        exp = int(np.floor(np.log10(value)))
        coeff = value / (10 ** exp)
        if abs(coeff - round(coeff)) < 0.01:
            coeff = int(round(coeff))
        else:
            coeff = round(coeff, 1)
        return f"{coeff}E{exp:02d}"
    return f"{value:.5f}"


def main():
    parser = argparse.ArgumentParser(description="Visualize MotorTask training logs")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="./models_weights/MotorTask/exp_author_config_R1-all_patch_reps-all-subject_independent",
        help="Experiment directory containing training_logs.json / training_summary.json / config.yaml",
    )
    # 旧实验对照：
    # --exp_dir ./models_weights/MotorTask/exp_author_config-all_patch_reps-all
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./viz_R1",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not call plt.show() (recommended on server)",
    )
    args = parser.parse_args()
    show = not args.no_show

    exp_dir = args.exp_dir.rstrip("/")
    log_file = os.path.join(exp_dir, "training_logs.json")
    summary_file = os.path.join(exp_dir, "training_summary.json")
    config_file = os.path.join(exp_dir, "config.yaml")

    os.makedirs(args.save_dir, exist_ok=True)

    metrics, epochs = get_metrics_motor(log_file)

    lr = weight_decay = batch_size = None
    split_mode = "random_epoch"
    single_fold_debug = val_subject = test_subject = None

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
            lr = summary["training_info"].get("learning_rate")
            batch_size = summary["training_info"].get("batch_size")
    except Exception:
        pass

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if weight_decay is None:
                weight_decay = config.get("weight_decay")
            if lr is None:
                lr = config.get("lr")
            if batch_size is None:
                batch_size = config.get("batch_size")
            split_mode = config.get("split_mode", "random_epoch")
            single_fold_debug = config.get("single_fold_debug", None)
            val_subject = config.get("val_subject", None)
            test_subject = config.get("test_subject", None)
    except Exception:
        pass

    best_epoch_val_bacc = int(np.argmax(metrics["val_bacc"]))
    best_val_bacc = metrics["val_bacc"][best_epoch_val_bacc]
    best_test_bacc = metrics["test_bacc"][best_epoch_val_bacc]
    best_epoch = best_epoch_val_bacc + 1

    lr_str = format_sci(lr) if lr is not None and lr < 1 else (f"{lr:.5f}" if lr is not None else "")
    if weight_decay is not None:
        wd_str = format_sci(weight_decay) if weight_decay < 0.0001 else f"{weight_decay:.5f}"
    else:
        wd_str = ""
    bs_str = str(batch_size) if batch_size is not None else ""

    f1_name = "Macro F1" if split_mode == "subject_independent" else "F1 (weighted)"

    print(f"exp_dir: {exp_dir}")
    print(
        f"split_mode: {split_mode}; single_fold_debug: {single_fold_debug}; "
        f"val_subject: {val_subject}; test_subject: {test_subject}"
    )
    print(
        f"lr: {lr_str}; weight decay: {wd_str}; batch size: {bs_str}; "
        f"Best val bacc: {best_val_bacc * 100:.2f}%; "
        f"对应的test bacc: {best_test_bacc * 100:.2f}%; best epoch: {best_epoch}"
    )
    print("=" * 60)
    print("Training Results Summary")
    print("=" * 60)
    print(f"\nBest Epoch (based on Validation Balanced Accuracy): {best_epoch}")
    print(f"  Validation Accuracy: {metrics['val_acc'][best_epoch_val_bacc]:.4f}")
    print(f"  Validation Balanced Accuracy: {metrics['val_bacc'][best_epoch_val_bacc]:.4f}")
    print(f"  Validation Kappa: {metrics['val_kappa'][best_epoch_val_bacc]:.4f}")
    print(f"  Validation {f1_name}: {metrics['val_f1'][best_epoch_val_bacc]:.4f}")
    print(f"  Test Accuracy: {metrics['test_acc'][best_epoch_val_bacc]:.4f}")
    print(f"  Test Balanced Accuracy: {metrics['test_bacc'][best_epoch_val_bacc]:.4f}")
    print(f"  Test Kappa: {metrics['test_kappa'][best_epoch_val_bacc]:.4f}")
    print(f"  Test {f1_name}: {metrics['test_f1'][best_epoch_val_bacc]:.4f}")

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_for_recall = json.load(f)
        test_cm_tmp = np.array(summary_for_recall["final_results"]["test_cm"])
        per_class_recall = test_cm_tmp.diagonal() / test_cm_tmp.sum(axis=1).clip(min=1)
        print("\n[R1] Test Per-class Recall (best epoch by val bacc):")
        for name, rec in zip(MOTOR_CLASS_NAMES, per_class_recall):
            print(f"  {name}: {rec:.4f}")
        print(f"  Mean (= Balanced Acc): {per_class_recall.mean():.4f}")
    except Exception as e:
        print(f"\n[R1] Skip per-class recall from CM: {e}")

    best_epoch_test_bacc = int(np.argmax(metrics["test_bacc"]))
    best_epoch_test_kappa = int(np.argmax(metrics["test_kappa"]))
    best_epoch_test_f1 = int(np.argmax(metrics["test_f1"]))
    print("\n[For Comparison Only] Best Test Metrics:")
    print(
        f"  Best Test Balanced Accuracy: epoch {best_epoch_test_bacc + 1}, "
        f"value: {metrics['test_bacc'][best_epoch_test_bacc]:.4f}"
    )
    print(
        f"  Best Test Kappa: epoch {best_epoch_test_kappa + 1}, "
        f"value: {metrics['test_kappa'][best_epoch_test_kappa]:.4f}"
    )
    print(
        f"  Best Test {f1_name}: epoch {best_epoch_test_f1 + 1}, "
        f"value: {metrics['test_f1'][best_epoch_test_f1]:.4f}"
    )
    print("=" * 60)

    save = args.save_dir
    plot_metric(
        epochs, metrics["train_loss"], "Train Loss",
        f"Training Loss [{split_mode}]",
        save_path=os.path.join(save, "01_train_loss.png"), show=show,
    )
    plot_trainNval_metric(
        epochs, metrics["val_acc"], metrics["test_acc"], "Accuracy",
        f"Accuracy (Val vs Test) [{split_mode}]",
        save_path=os.path.join(save, "02_accuracy.png"), show=show,
    )
    plot_trainNval_metric(
        epochs, metrics["val_bacc"], metrics["test_bacc"], "Balanced Accuracy",
        f"Balanced Accuracy (Val vs Test) [{split_mode}]",
        save_path=os.path.join(save, "03_balanced_accuracy.png"), show=show,
    )
    plot_trainNval_metric(
        epochs, metrics["val_kappa"], metrics["test_kappa"], "Kappa",
        f"Kappa (Val vs Test) [{split_mode}]",
        save_path=os.path.join(save, "04_kappa.png"), show=show,
    )
    plot_trainNval_metric(
        epochs, metrics["val_f1"], metrics["test_f1"], f1_name,
        f"{f1_name} (Val vs Test) [{split_mode}]",
        save_path=os.path.join(save, "05_f1.png"), show=show,
    )
    plot_metric(
        epochs, metrics["learning_rate"], "Learning Rate",
        f"Learning Rate Schedule [{split_mode}]",
        save_path=os.path.join(save, "06_learning_rate.png"), show=show,
    )

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        best_epoch_cm = summary["final_results"]["best_epoch"]
        val_cm = np.array(summary["final_results"]["val_cm"])
        test_cm = np.array(summary["final_results"]["test_cm"])
        print(f"\n{'=' * 60}")
        print(f"Confusion Matrices (Best Epoch: {best_epoch_cm}) [{split_mode}]")
        print(f"{'=' * 60}")
        plot_confusion_matrix(
            val_cm, MOTOR_CLASS_NAMES,
            f"Validation CM (Epoch {best_epoch_cm}) [{split_mode}]",
            save_path=os.path.join(save, "07_val_cm.png"), show=show,
        )
        plot_confusion_matrix(
            test_cm, MOTOR_CLASS_NAMES,
            f"Test CM (Epoch {best_epoch_cm}) [{split_mode}]",
            save_path=os.path.join(save, "08_test_cm.png"), show=show,
        )
    except FileNotFoundError:
        print(f"\nWarning: {summary_file} not found. Skipping confusion matrix visualization.")
    except KeyError as e:
        print(f"\nWarning: Key {e} not found in summary file. Skipping confusion matrix visualization.")

    print(f"\nDone. Figures saved under: {os.path.abspath(save)}")


if __name__ == "__main__":
    main()
