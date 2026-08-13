#!/usr/bin/env python3
"""
Build the paper's classification-performance figure (subject-independent protocol)
in the same panel layout as the original Fig. 7 (acc.pdf):
  left column  : validation confusion matrix (top), test confusion matrix (bottom)
  right column : training loss (top), learning rate schedule (middle),
                 balanced accuracy val-vs-test (bottom)

Data source: models_weights/MotorTask/exp_author_config_R1-all_patch_reps-all-subject_independent/
  - training_logs.json      -> per-epoch train_loss / learning_rate / val_bacc / test_bacc
  - training_summary.json   -> final_results.best_epoch / val_cm / test_cm (at best epoch, chosen by val BACC)

Output: cbramod_subject_independent.pdf (upload this to the Overleaf project,
replacing the old acc.pdf reference for Fig. 7).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

EXP_DIR = "./models_weights/MotorTask/exp_author_config_R1-all_patch_reps-all-subject_independent"
OUT_PATH = "./cbramod_subject_independent.pdf"

# Same class order already used in log_visualize_motor_R1.py (MOTOR_CLASS_NAMES),
# reworded here to match the task names used elsewhere in the manuscript
# (e.g. Fig. 5 caption: straight walking, curved walking, head shaking, head
# nodding, object pick-up, stair climbing and descending).
CLASS_NAMES = [
    "Straight walking",
    "Curved walking",
    "Head shaking",
    "Head nodding",
    "Object pick-up",
    "Stairs climb/descend",
]


def load_data():
    with open(os.path.join(EXP_DIR, "training_logs.json")) as f:
        logs = json.load(f)
    with open(os.path.join(EXP_DIR, "training_summary.json")) as f:
        summary = json.load(f)
    return logs, summary


def plot_cm(ax, cm, title):
    cm = np.array(cm, dtype=float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(
        cm_pct, annot=True, fmt=".0f", cmap="Blues", cbar=True,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        vmin=0, vmax=100, ax=ax,
        cbar_kws={"label": "%"},
        annot_kws={"fontsize": 8},
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    ax.tick_params(axis="x", labelrotation=40, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")


def main():
    logs, summary = load_data()
    epochs = [e["epoch"] for e in logs]
    train_loss = [e["train_loss"] for e in logs]
    lr = [e["learning_rate"] for e in logs]
    val_bacc = [e["val_bacc"] for e in logs]
    test_bacc = [e["test_bacc"] for e in logs]

    best_epoch = summary["final_results"]["best_epoch"]
    val_cm = summary["final_results"]["val_cm"]
    test_cm = summary["final_results"]["test_cm"]

    fig = plt.figure(figsize=(11, 8))
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 1.15], wspace=0.05)

    # ---- left column: confusion matrices ----
    left_axes = subfigs[0].subplots(2, 1)
    plot_cm(left_axes[0], val_cm, f"Validation confusion matrix (epoch {best_epoch})")
    plot_cm(left_axes[1], test_cm, f"Test confusion matrix (epoch {best_epoch})")
    subfigs[0].subplots_adjust(hspace=0.55, left=0.18, right=0.98, top=0.95, bottom=0.08)

    # ---- right column: training curves ----
    right_axes = subfigs[1].subplots(3, 1)

    ax = right_axes[0]
    ax.plot(epochs, train_loss, color="tab:blue")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Training loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.grid(alpha=0.3)

    ax = right_axes[1]
    ax.plot(epochs, lr, color="tab:orange")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Learning rate schedule", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Learning rate", fontsize=9)
    ax.grid(alpha=0.3)

    ax = right_axes[2]
    ax.plot(epochs, val_bacc, color="tab:green", label="Validation")
    ax.plot(epochs, test_bacc, color="tab:red", label="Test")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
               label=f"Selected epoch ({best_epoch})")
    ax.axhline(1 / 6, color="black", linestyle=":", linewidth=1, label="Chance (16.7%)")
    ax.set_title("Balanced accuracy (validation vs. test)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Balanced accuracy", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    subfigs[1].subplots_adjust(hspace=0.6, left=0.12, right=0.95, top=0.95, bottom=0.08)

    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
