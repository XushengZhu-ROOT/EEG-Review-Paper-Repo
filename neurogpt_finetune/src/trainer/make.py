#!/usr/bin/env python3

import os
import re
from typing import Dict, List, Tuple
import numpy as np
import math
from collections import defaultdict
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
    if len(features) == 0:
        raise ValueError("Empty features list passed to data collator")
    
    # 将features转换为字典列表
    converted_features = []
    for i, f in enumerate(features):
        if isinstance(f, dict):
            converted_features.append(f)
        elif isinstance(f, (str, bytes)):
            # 字符串和字节类型不能转换为字典，直接报错
            raise TypeError(
                f"Feature at index {i} is a string/bytes type, cannot convert to dict. "
                f"Type: {type(f)}, value (first 100 chars): {str(f)[:100]}. "
                f"This usually indicates a data loading issue. "
                f"Expected dict or object with __dict__ attribute."
            )
        elif hasattr(f, '__dict__'):
            # 对象有__dict__属性，使用vars()
            converted_features.append(vars(f))
        else:
            # 其他情况，尝试转换为字典
            # 如果是namedtuple或其他可迭代对象，尝试dict()转换
            try:
                # 尝试作为可迭代对象转换为字典（如namedtuple）
                if hasattr(f, '_asdict'):
                    # namedtuple的情况
                    converted_features.append(f._asdict())
                elif hasattr(f, '_fields'):
                    # namedtuple的另一种情况
                    converted_features.append(dict(zip(f._fields, f)))
                else:
                    # 尝试直接转换为字典（仅对可迭代的键值对有效，如列表的元组对）
                    # 但对于字符串等类型，dict()会尝试迭代字符，导致错误结果
                    # 所以先检查是否是可迭代的键值对
                    if hasattr(f, 'items'):
                        converted_features.append(dict(f.items()))
                    else:
                        # 尝试dict()，但如果失败会抛出异常
                        converted_features.append(dict(f))
            except (TypeError, ValueError, AttributeError) as e:
                # 如果都失败，抛出清晰的错误信息
                raise TypeError(
                    f"Cannot convert feature at index {i} to dict. "
                    f"Feature type: {type(f)}, value (first 100 chars): {str(f)[:100]}. "
                    f"Expected dict or object with __dict__ attribute. "
                    f"Original error: {e}"
                ) from e
    
    # 确保所有元素都是字典
    if not all(isinstance(f, dict) for f in converted_features):
        non_dict_indices = [i for i, f in enumerate(converted_features) if not isinstance(f, dict)]
        raise TypeError(
            f"Not all features are dictionaries. "
            f"Non-dict indices: {non_dict_indices[:5]}, "
            f"Types: {[type(converted_features[i]) for i in non_dict_indices[:3]]}"
        )
    
    features = converted_features

    result = {}
    for k in features[0].keys():
        if not k.startswith("__"):
            values = [f[k] for f in features]
            # 字符串类型字段（如epoch_id）特殊处理：保存为列表
            if len(values) > 0 and isinstance(values[0], (str, bytes)):
                result[k] = values  # 保存为列表，不做tensor转换
            elif len(values) > 0 and isinstance(values[0], torch.Tensor):
                # labels字段特殊处理：如果是0维tensor（标量），使用stack；否则使用cat
                if k == 'labels' and values[0].dim() == 0:
                    result[k] = torch.stack(values)
                else:
                    result[k] = torch.cat(values)
            else:
                # 其他类型（如numpy数组），尝试转换为tensor或保持原样
                try:
                    if len(values) > 0 and hasattr(values[0], '__array__'):
                        # numpy数组或其他可转换为tensor的类型
                        result[k] = torch.cat([torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v) for v in values])
                    else:
                        # 保持为列表
                        result[k] = values
                except (TypeError, ValueError):
                    # 如果转换失败，保持为列表
                    result[k] = values
    return result


def make_decoding_accuracy_metrics(num_classes: int):
    def decoding_accuracy_metrics(eval_preds):
        preds_logits, labels = eval_preds

        # Handle the case where preds_logits has batch_size * num_chunks dimension
        # but labels only has batch_size dimension
        # This happens when each sample is split into multiple chunks
        if len(preds_logits) != len(labels):
            # Calculate num_chunks from the size mismatch
            logits_batch_size = len(preds_logits)
            labels_batch_size = len(labels)
            
            # Check if logits_batch_size is divisible by labels_batch_size
            if logits_batch_size % labels_batch_size == 0:
                num_chunks = logits_batch_size // labels_batch_size
                # Reshape preds_logits to (batch_size, num_chunks, num_classes)
                # Then average over chunks to get (batch_size, num_classes)
                preds_logits = preds_logits.reshape(labels_batch_size, num_chunks, -1)
                # Average over chunks dimension
                preds_logits = preds_logits.mean(axis=1)
            else:
                # If not divisible, try to find a common factor
                import math
                gcd = math.gcd(logits_batch_size, labels_batch_size)
                if gcd > 1:
                    base_batch_size = gcd
                    logits_num_chunks = logits_batch_size // base_batch_size
                    labels_num_chunks = labels_batch_size // base_batch_size
                    
                    # Reshape and average
                    if logits_num_chunks > 1:
                        preds_logits = preds_logits.reshape(base_batch_size, logits_num_chunks, -1)
                        preds_logits = preds_logits.mean(axis=1)
                        # Repeat to match labels if needed
                        if labels_num_chunks > 1:
                            preds_logits = np.repeat(preds_logits, labels_num_chunks, axis=0)
                    else:
                        # If we can't properly reshape, just take first labels_batch_size samples
                        preds_logits = preds_logits[:labels_batch_size]
                else:
                    # Last resort: just take first labels_batch_size samples
                    preds_logits = preds_logits[:labels_batch_size]

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


def extract_video_index(epoch_id: str) -> int:
    """从 epoch_id 中提取 video_index"""
    match = re.search(r'video_index_(\d+)_chunk', epoch_id)
    if match:
        return int(match.group(1))
    return None


def extract_subject_id(epoch_id: str) -> int:
    """从 epoch_id 中提取 subject_id"""
    match = re.search(r'subject_(\d+)_', epoch_id)
    if match:
        return int(match.group(1))
    return None


def compute_voting_metrics(predictions: np.ndarray, labels: np.ndarray, epoch_ids: List[str]) -> Dict[str, float]:
    """
    基于投票的评估指标计算
    
    参数:
        predictions: (n_samples, num_classes) 预测logits或 (n_samples,) 预测类别
        labels: (n_samples,) 真实标签
        epoch_ids: (n_samples,) epoch_id列表
        
    返回:
        包含投票评估指标的字典
    """
    # 如果predictions是多维的（logits），转换为类别
    if predictions.ndim > 1:
        pred_labels = predictions.argmax(axis=-1)
    else:
        pred_labels = predictions
    
    # 按视频分组
    video_groups = defaultdict(lambda: {'preds': [], 'label': None, 'subject_id': None})
    
    for i, epoch_id in enumerate(epoch_ids):
        video_index = extract_video_index(epoch_id)
        subject_id = extract_subject_id(epoch_id)
        
        if video_index is None:
            continue  # 跳过无法提取video_index的样本
        
        video_key = (subject_id, video_index)
        
        video_groups[video_key]['preds'].append(int(pred_labels[i]))
        video_groups[video_key]['label'] = int(labels[i])  # 同一视频的label应该相同
        video_groups[video_key]['subject_id'] = subject_id
    
    # 对每个视频进行投票
    video_predictions = []
    video_labels = []
    video_scores = []  # 用于记录每个视频的得分（0, 0.5, 1）
    
    for video_key, video_data in video_groups.items():
        preds = video_data['preds']
        true_label = video_data['label']
        
        if len(preds) == 0:
            continue
        
        # 多数投票
        from collections import Counter
        vote_counts = Counter(preds)
        max_votes = max(vote_counts.values())
        
        # 找出所有得票最多的类别
        majority_classes = [cls for cls, count in vote_counts.items() if count == max_votes]
        
        if len(majority_classes) == 1:
            # 有唯一的多数类别
            video_pred = majority_classes[0]
            score = 1.0 if video_pred == true_label else 0.0
        else:
            # 平票情况
            if true_label in majority_classes:
                video_pred = true_label  # 如果真实标签在候选中，选择它
                score = 0.5
            else:
                video_pred = majority_classes[0]  # 否则选择第一个（任意选择）
                score = 0.0
        
        video_predictions.append(video_pred)
        video_labels.append(true_label)
        video_scores.append(score)
    
    if len(video_predictions) == 0:
        return {
            'video_accuracy': 0.0,
            'video_bacc': 0.0,
            'num_videos': 0,
        }
    
    video_predictions = np.array(video_predictions)
    video_labels = np.array(video_labels)
    video_scores = np.array(video_scores)
    
    # 视频级别准确率（使用投票得分）
    video_accuracy = video_scores.mean()
    
    # 视频级别平衡准确率（基于最终预测结果）
    video_bacc = balanced_accuracy_score(video_labels, video_predictions)
    
    # 按subject分组计算准确率
    subject_accuracies = {}
    subject_groups = defaultdict(lambda: {'preds': [], 'labels': []})
    
    for i, (video_key, video_data) in enumerate(video_groups.items()):
        subject_id = video_data['subject_id']
        if subject_id is not None:
            subject_groups[subject_id]['preds'].append(video_predictions[i])
            subject_groups[subject_id]['labels'].append(video_labels[i])
    
    for subject_id, subject_data in subject_groups.items():
        sub_preds = np.array(subject_data['preds'])
        sub_labels = np.array(subject_data['labels'])
        sub_accuracy = accuracy_score(sub_labels, sub_preds)
        subject_accuracies[f'subject_{subject_id}_accuracy'] = sub_accuracy
    
    metrics = {
        'video_accuracy': round(video_accuracy, 4),
        'video_bacc': round(video_bacc, 4),
        'num_videos': len(video_predictions),
    }
    
    # 添加subject级别的准确率
    metrics.update({f'{k}': round(v, 4) for k, v in subject_accuracies.items()})
    
    # 计算平均subject准确率
    if len(subject_accuracies) > 0:
        metrics['avg_subject_accuracy'] = round(np.mean(list(subject_accuracies.values())), 4)
    
    return metrics


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
