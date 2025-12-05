#!/usr/bin/env python3

import os
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import torch
from transformers import TrainingArguments, TrainerCallback
from trainer.base import Trainer
import torch.nn.functional as F


class CSVLogCallback(TrainerCallback):

    def __init__(self):
        super().__init__()
        self.train_log_filepath = None
        # self.eval_log_filepath = None
        # self.eval_keys = None
        self.eval_log_files = {}

    def on_log(self, args, state, control, model, **kwargs) -> None:

        if args.local_rank not in {-1, 0}:
            return

        latest_log = state.log_history[-1]  # 最新的日誌

        if self.train_log_filepath is None:
            self.train_log_filepath = os.path.join(args.output_dir, "train_history.csv")

            with open(self.train_log_filepath, "a") as f:
                f.write("step,loss,lr\n")

        eval_keys = [
            k
            for k in latest_log.keys()
            if k.startswith("eval_")
            and not any(x in k for x in ["runtime", "second", "epoch", "step"])
        ]
        # is_eval = any("eval_" in k for k in latest_log.keys())

        if len(eval_keys) > 0:
            dataset_name = "metrics"  # 預設名稱
            first_key = eval_keys[0]

            if "_validation_" in first_key:
                dataset_name = "validation"
            elif "_test_" in first_key:
                dataset_name = "test"

            if dataset_name not in self.eval_log_files:
                filename = f"eval_{dataset_name}_history.csv"
                filepath = os.path.join(args.output_dir, filename)

                sorted_keys = sorted(eval_keys)

                self.eval_log_files[dataset_name] = {
                    "path": filepath,
                    "keys": sorted_keys,
                    "initialized": False,
                }

            log_info = self.eval_log_files[dataset_name]
            filepath = log_info["path"]
            target_keys = log_info["keys"]

            if not log_info["initialized"]:
                if not os.path.exists(filepath):
                    header = "step," + ",".join(target_keys) + "\n"
                    with open(filepath, "w") as f:
                        f.write(header)
                log_info["initialized"] = True

            data_values = [str(state.global_step)]
            for k in target_keys:
                val = latest_log.get(k, np.nan)
                data_values.append(str(val))

            with open(filepath, "a") as f:
                f.write(",".join(data_values) + "\n")
        else:

            with open(self.train_log_filepath, "a") as f:
                f.write(
                    "{},{},{}\n".format(
                        state.global_step,
                        (
                            latest_log["loss"]
                            if "loss" in latest_log
                            else latest_log["train_loss"]
                        ),
                        (
                            latest_log["learning_rate"]
                            if "learning_rate" in latest_log
                            else None
                        ),
                    )
                )


def _cat_data_collator(features: List) -> Dict[str, torch.tensor]:

    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    return {
        k: torch.cat([f[k] for f in features])
        for k in features[0].keys()
        if not k.startswith("__")
    }


def make_decoding_accuracy_metrics(num_classes: int):
    def decoding_accuracy_metrics(eval_preds):
        preds_logits, labels = eval_preds

        # 檢查是否為單 Logit 輸出 (適用於 BCEWithLogitsLoss)
        is_binary_logit_output = preds_logits.shape[-1] == 1 or preds_logits.ndim == 1

        if num_classes <= 2 and is_binary_logit_output:
            # ==== 單 Logit 輸出 ====

            # 展平 Logits，並計算 y_pred (Logit > 0 即預測為 1)
            logits_1d = preds_logits.flatten()
            y_pred = (logits_1d > 0).astype(int)

            # 使用 Sigmoid 獲取類別 1 的機率，用於 AUC 指標
            y_score_binary = torch.sigmoid(torch.from_numpy(logits_1d)).numpy()

        else:
            # ==== 多分類或多 Logit 輸出 ====

            # 計算 y_pred
            y_pred = preds_logits.argmax(axis=-1)

            # 使用 Softmax 獲取 y_score_full
            y_score_full = F.softmax(torch.from_numpy(preds_logits), dim=-1).numpy()

            if num_classes <= 2:
                # 假設類別 1 的分數在 index 1
                y_score_binary = y_score_full[:, 1]

        metrics = {}
        metrics["accuracy"] = accuracy_score(labels, y_pred)
        metrics["bacc"] = balanced_accuracy_score(labels, y_pred)

        if num_classes <= 2:
            # ROC-AUC
            try:
                metrics["roc_auc"] = roc_auc_score(labels, y_score_binary)
            except ValueError:
                metrics["roc_auc"] = np.nan

            # PR-AUC
            precision, recall, _ = precision_recall_curve(labels, y_score_binary)
            metrics["pr_auc"] = auc(recall, precision)

        elif num_classes > 2:
            # Kappa
            metrics["kappa"] = cohen_kappa_score(labels, y_pred)

            # F1-score (weighted for imbalanced classes)
            f1_sc = f1_score(labels, y_pred, average="weighted", zero_division=0)
            metrics["f1_weighted"] = f1_sc

            try:
                metrics["roc_auc_weighted"] = roc_auc_score(
                    labels, y_score_full, average="weighted", multi_class="ovr"
                )
            except ValueError:
                metrics["roc_auc_weighted"] = np.nan

        formatted_metrics = {f"{k}": round(v, 4) for k, v in metrics.items()}
        return formatted_metrics

    return decoding_accuracy_metrics


def get_compute_metrics(training_style: str, num_classes: int = None):

    if training_style == "decoding":
        if num_classes is None:
            # TODO: regression downstream tasks
            print(
                "WARNING: 'decoding' task requires 'num_classes' parameter for metric calculation."
            )
            return None
        return make_decoding_accuracy_metrics(num_classes)

    else:
        return None


def decoding_accuracy_metrics(eval_preds):
    preds, labels = eval_preds
    preds = preds.argmax(axis=-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": round(accuracy, 3)}


def make_trainer(
    model_init,
    training_style,
    train_dataset,
    validation_dataset,
    num_decoding_classes: int = None,  # for recording metrics
    do_train: bool = True,
    do_eval: bool = True,
    run_name: str = None,
    output_dir: str = None,
    overwrite_output_dir: bool = True,
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
        None,
        None,
    ),
    optim: str = "adamw_hf",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    per_device_train_batch_size: int = 64,
    per_device_eval_batch_size: int = 64,
    dataloader_num_workers: int = 0,
    max_steps: int = 400000,
    num_train_epochs: int = 1,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.01,
    evaluation_strategy: str = "steps",
    prediction_loss_only: bool = False,
    logging_strategy: str = "steps",
    save_strategy: str = "steps",
    save_total_limit: int = 5,
    save_steps: int = 10000,
    logging_steps: int = 10000,
    eval_steps: int = None,
    logging_first_step: bool = True,
    greater_is_better: bool = True,
    seed: int = 1,
    fp16: bool = True,
    deepspeed: str = None,
    compute_metrics=None,
    **kwargs,
) -> Trainer:
    """
    Make a Trainer object for training a model.
    Returns an instance of transformers.Trainer.

    See the HuggingFace transformers documentation for more details
    on input arguments:
    https://huggingface.co/transformers/main_classes/trainer.html

    Custom arguments:
    ---
    model_init: callable
        A callable that does not require any arguments and
        returns model that is to be trained (see scripts.train.model_init)
    training_style: str
        The training style (ie., framework) to use.
        One of: 'BERT', 'CSM', 'NetBERT', 'autoencoder',
        'decoding'.
    train_dataset: src.batcher.dataset
        The training dataset, as generated by src.batcher.dataset
    validation_dataset: src.batcher.dataset
        The validation dataset, as generated by src.batcher.dataset

    Returns
    ----
    trainer: transformers.Trainer
    """
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        do_train=do_train,
        do_eval=do_eval,
        overwrite_output_dir=overwrite_output_dir,
        prediction_loss_only=prediction_loss_only,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        optim=optim,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        greater_is_better=greater_is_better,
        save_steps=save_steps,
        logging_strategy=logging_strategy,
        logging_first_step=logging_first_step,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps if eval_steps is not None else logging_steps,
        seed=seed,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        deepspeed=deepspeed,
        **kwargs,
    )

    data_collator = _cat_data_collator
    is_deepspeed = deepspeed is not None
    # TODO: custom compute_metrics so far not working in multi-gpu setting
    # compute_metrics = (
    #     decoding_accuracy_metrics
    #     if training_style == "decoding" and compute_metrics is None
    #     else compute_metrics
    # )
    if compute_metrics is None:
        compute_metrics = get_compute_metrics(training_style, num_decoding_classes)

    trainer = Trainer(
        args=trainer_args,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        is_deepspeed=is_deepspeed,
    )

    trainer.add_callback(CSVLogCallback)

    return trainer
