import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import (
    confusion_matrix, 
    balanced_accuracy_score, 
    classification_report,
    accuracy_score,
    f1_score,
    cohen_kappa_score
)


def multiclass_metrics_fn(y_true, y_pred_proba, metrics=None):
    """
    使用 sklearn 实现多分类指标计算
    替换 pyhealth.metrics.multiclass_metrics_fn
    
    Args:
        y_true: 真实标签 (1D array)
        y_pred_proba: 预测概率 (2D array, shape: [n_samples, n_classes])
        metrics: 要计算的指标列表
    
    Returns:
        dict: 包含各项指标的字典
    """
    if metrics is None:
        metrics = ["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"]
    
    # 获取预测类别
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    result = {}
    
    if "accuracy" in metrics:
        result["accuracy"] = accuracy_score(y_true, y_pred)
    
    if "f1_macro" in metrics:
        result["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    if "f1_weighted" in metrics:
        result["f1_weighted"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if "cohen_kappa" in metrics:
        result["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    
    return result

from model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
    Ada_BIOT,
    Labram_style_BIOTClassifier,
    Labram_style_Ada_BIOT,
    CBraMod_3lyStyle_LayerNorm_BIOT,
    CBraMod_3lyStyle_LayerNorm_Ada_BIOT,
)
from utils import TUEVLoader, HARLoader, MotionLoader, SEEDLoader, collate_fn_seed_with_epoch_id

import json
import yaml
import re
from collections import defaultdict, Counter

# ============ SEED视频投票评估辅助函数 ============
def extract_video_index(epoch_id):
    """从 epoch_id 中提取 video_index
    例如: 'subject_1_video_index_3_chunk001' -> 3
    """
    match = re.search(r'video_index_(\d+)_chunk', epoch_id)
    if match:
        return int(match.group(1))
    return None

def extract_subject_id(epoch_id):
    """从 epoch_id 中提取 subject_id
    例如: 'subject_1_video_index_3_chunk001' -> 1
    """
    match = re.search(r'subject_(\d+)_', epoch_id)
    if match:
        return int(match.group(1))
    return None

def majority_voting_with_tie_handling(chunk_predictions, true_label):
    """
    对视频的chunks进行多数投票，处理平票情况
    
    Args:
        chunk_predictions: list of int, 同一视频所有chunks的预测类别
        true_label: int, 真实标签
    
    Returns:
        float: 得分 (1.0=完全正确, 0.5=平票且真实标签在候选中, 0.0=错误)
        int or list: 投票结果（单个类别或平票类别列表）
    """
    if len(chunk_predictions) == 0:
        return 0.0, None
    
    # 统计每个类别的票数
    vote_counts = Counter(chunk_predictions)
    max_votes = max(vote_counts.values())
    
    # 找出得票最多的类别
    winners = [label for label, count in vote_counts.items() if count == max_votes]
    
    if len(winners) == 1:
        # 只有一个最高票数，正常情况
        predicted_label = winners[0]
        score = 1.0 if predicted_label == true_label else 0.0
        return score, predicted_label
    else:
        # 平票情况
        if true_label in winners:
            # 真实标签在平票候选中，得0.5分
            return 0.5, winners
        else:
            # 真实标签不在平票候选中，得0.0分
            return 0.0, winners

def compute_video_level_metrics(preds, targets, epoch_ids):
    """
    计算视频级别的评估指标（基于多数投票）
    
    Args:
        preds: numpy array, shape (N, n_classes), 预测概率
        targets: numpy array, shape (N,), 真实标签
        epoch_ids: list of str, shape (N,), epoch_id列表
    
    Returns:
        dict: 包含视频级别指标的字典
    """
    # 获取chunk级别的预测类别
    chunk_pred_classes = np.argmax(preds, axis=1)
    
    # 按视频分组：{video_key: {'chunk_preds': [...], 'true_label': int, 'subject_id': int}}
    # video_key = f"subject_{subject_id}_video_{video_index}"
    video_groups = defaultdict(lambda: {'chunk_preds': [], 'true_label': None, 'subject_id': None})
    
    for i, epoch_id in enumerate(epoch_ids):
        video_idx = extract_video_index(epoch_id)
        subject_id = extract_subject_id(epoch_id)
        
        if video_idx is None or subject_id is None:
            print(f"Warning: Could not extract video_index or subject_id from epoch_id: {epoch_id}")
            continue
        
        video_key = f"subject_{subject_id}_video_{video_idx}"
        video_groups[video_key]['chunk_preds'].append(chunk_pred_classes[i])
        video_groups[video_key]['true_label'] = targets[i]  # 同一视频的所有chunks应该有相同的true_label
        video_groups[video_key]['subject_id'] = subject_id
    
    # 对每个视频进行投票
    video_preds = []
    video_targets = []
    video_scores = []
    video_details = []  # 用于详细日志
    
    for video_key, video_data in video_groups.items():
        chunk_preds = video_data['chunk_preds']
        true_label = video_data['true_label']
        
        score, vote_result = majority_voting_with_tie_handling(chunk_preds, true_label)
        video_scores.append(score)
        video_targets.append(true_label)
        
        # vote_result可能是int（单个结果）或list（平票）
        if isinstance(vote_result, list):
            # 平票情况，使用第一个作为预测（但分数已经是0.5或0.0）
            video_preds.append(vote_result[0])
        else:
            video_preds.append(vote_result)
        
        video_details.append({
            'video_key': video_key,
            'chunk_predictions': chunk_preds,
            'vote_result': vote_result,
            'true_label': int(true_label),
            'score': score
        })
    
    video_preds = np.array(video_preds)
    video_targets = np.array(video_targets)
    video_scores = np.array(video_scores)
    
    # 计算视频级别指标
    video_accuracy = np.mean(video_scores)  # 平均得分（考虑0.5分的情况）
    video_strict_accuracy = accuracy_score(video_targets, video_preds)  # 严格准确率（不考虑0.5分）
    video_f1_macro = f1_score(video_targets, video_preds, average='macro', zero_division=0)
    video_f1_weighted = f1_score(video_targets, video_preds, average='weighted', zero_division=0)
    video_cohen_kappa = cohen_kappa_score(video_targets, video_preds)
    video_balanced_accuracy = balanced_accuracy_score(video_targets, video_preds)
    
    # 计算混淆矩阵
    video_cm = confusion_matrix(video_targets, video_preds)
    
    # 按subject计算准确率
    subject_accuracies = {}
    subject_groups = defaultdict(lambda: {'scores': [], 'count': 0})
    
    for video_key, video_data in video_groups.items():
        subject_id = video_data['subject_id']
        chunk_preds = video_data['chunk_preds']
        true_label = video_data['true_label']
        score, _ = majority_voting_with_tie_handling(chunk_preds, true_label)
        subject_groups[subject_id]['scores'].append(score)
        subject_groups[subject_id]['count'] += 1
    
    for subject_id, data in subject_groups.items():
        if data['count'] > 0:
            subject_accuracies[f"subject_{subject_id}"] = {
                'accuracy': np.mean(data['scores']),
                'total_videos': data['count']
            }
    
    return {
        'video_accuracy': float(video_accuracy),
        'video_strict_accuracy': float(video_strict_accuracy),
        'video_f1_macro': float(video_f1_macro),
        'video_f1_weighted': float(video_f1_weighted),
        'video_cohen_kappa': float(video_cohen_kappa),
        'video_balanced_accuracy': float(video_balanced_accuracy),
        'video_confusion_matrix': video_cm.tolist(),
        'subject_accuracies': subject_accuracies,
        'total_videos': len(video_groups),
        'video_details': video_details  # 详细日志（可选，可能很大）
    }

class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model, test_loader=None):
        super().__init__()
        self.model = model
        self.args = args
        self.test_loader = test_loader
        self.criterion = torch.nn.CrossEntropyLoss()

        # ---- NEW: buffers for epoch-end hooks in PL v2 ----
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_epoch_ids = []  # 用于存储val的epoch_id
        self.test_epoch_ids = []  # 用于存储test的epoch_id

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)              # shape: (B, n_classes)
        loss = self.criterion(logits, y.long())  # y shape: (B,)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # SEED数据集在test/val时会返回(X, y, epoch_id)，训练时只返回(X, y)
        if len(batch) == 3:
            X, y, epoch_ids = batch
            # epoch_ids是list of str，需要保存
            self.val_epoch_ids.extend(epoch_ids)
        else:
            X, y = batch
        
        # Lightning 會自動在 val loop 關閉 grad，其實不用 no_grad，但保留也沒關係
        with torch.no_grad():
            logits = self.model(X)
            prob = torch.softmax(logits, dim=1)
            step_result = prob.detach().cpu().numpy()
            step_gt = y.detach().cpu().numpy()

        # ---- NEW: save outputs to buffer, do not rely on outputs param ----
        self.val_step_outputs.append((step_result, step_gt))
        # 這邊不用 return 也可以，但保留 return 也不會錯
        return step_result, step_gt

    # ---- REPLACED: validation_epoch_end -> on_validation_epoch_end ----
    def on_validation_epoch_end(self):
        if len(self.val_step_outputs) == 0:
            return

        preds = []
        targets = []

        for pred, tgt in self.val_step_outputs:
            preds.append(pred)   # pred shape: (B, n_classes)
            targets.append(tgt)  # tgt  shape: (B,)

        preds = np.concatenate(preds, axis=0)      # (N, n_classes)
        targets = np.concatenate(targets, axis=0)  # (N,)

        # 清 buffer，避免 memory 疊加
        self.val_step_outputs.clear()

        # 获取预测类别
        pred_classes = np.argmax(preds, axis=1)
        
        # === SEED数据集：计算视频级别的投票评估 ===
        is_seed_dataset = self.args.dataset in ["SEED", "SEED-ST"]
        if is_seed_dataset and len(self.val_epoch_ids) == len(targets):
            # 使用视频级别的投票评估
            video_metrics = compute_video_level_metrics(preds, targets, self.val_epoch_ids)
            
            # 记录视频级别的指标
            val_result = {
                "chunk_accuracy": multiclass_metrics_fn(targets, preds, metrics=["accuracy"])["accuracy"],
                "video_accuracy": video_metrics["video_accuracy"],
                "video_strict_accuracy": video_metrics["video_strict_accuracy"],
                "video_f1_macro": video_metrics["video_f1_macro"],
                "video_f1_weighted": video_metrics["video_f1_weighted"],
                "video_cohen_kappa": video_metrics["video_cohen_kappa"],
                "video_balanced_accuracy": video_metrics["video_balanced_accuracy"],
                "video_confusion_matrix": video_metrics["video_confusion_matrix"],
                "subject_accuracies": video_metrics["subject_accuracies"],
                "total_videos": video_metrics["total_videos"],
            }
            
            # 记录chunk级别的指标作为参考
            chunk_result = multiclass_metrics_fn(
                targets,
                preds,
                metrics=["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
            )
            val_result["chunk_f1_macro"] = chunk_result["f1_macro"]
            val_result["chunk_f1_weighted"] = chunk_result["f1_weighted"]
            val_result["chunk_cohen_kappa"] = chunk_result["cohen_kappa"]
            val_result["chunk_confusion_matrix"] = confusion_matrix(targets, pred_classes).tolist()
            
            # 清空epoch_ids
            self.val_epoch_ids.clear()
            
            # 打印subject准确率
            print("\n" + "="*50)
            print("验证集 - Subject级别准确率:")
            print("="*50)
            for subject_key, acc_data in video_metrics["subject_accuracies"].items():
                print(f"{subject_key}: {acc_data['accuracy']:.4f} ({acc_data['total_videos']} videos)")
            print("="*50 + "\n")
        else:
            # 非SEED数据集或没有epoch_id，使用chunk级别评估
            val_result = multiclass_metrics_fn(
                targets,
                preds,
                metrics=["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
            )
            val_result["balanced_accuracy"] = balanced_accuracy_score(targets, pred_classes)
            cm = confusion_matrix(targets, pred_classes)
            val_result["confusion_matrix"] = cm.tolist()
            
            if len(self.val_epoch_ids) > 0:
                self.val_epoch_ids.clear()  # 清空，防止污染下次评估
        
        # 根据是否有视频级别指标来决定log哪个
        if is_seed_dataset and "video_accuracy" in val_result:
            self.log("val_acc", val_result["video_accuracy"], sync_dist=True, prog_bar=True)
            self.log("val_video_strict_acc", val_result["video_strict_accuracy"], sync_dist=True, prog_bar=True)
            self.log("val_chunk_acc", val_result["chunk_accuracy"], sync_dist=True)
            self.log("val_f1_macro", val_result["video_f1_macro"], sync_dist=True, prog_bar=True)
            self.log("val_f1_weighted", val_result["video_f1_weighted"], sync_dist=True)
            self.log("val_cohen_kappa", val_result["video_cohen_kappa"], sync_dist=True)
            self.log("val_bacc", val_result["video_balanced_accuracy"], sync_dist=True, prog_bar=True)
        else:
            self.log("val_acc", val_result["accuracy"], sync_dist=True, prog_bar=True)
            self.log("val_f1_macro", val_result["f1_macro"], sync_dist=True, prog_bar=True)
            self.log("val_f1_weighted", val_result["f1_weighted"], sync_dist=True)
            self.log("val_cohen_kappa", val_result["cohen_kappa"], sync_dist=True)
            self.log("val_bacc", val_result["balanced_accuracy"], sync_dist=True, prog_bar=True)

        # 在每個 val epoch 結束跑一次 test（你原本的邏輯）
        test_results = self._run_test_epoch()

        # 保存验证集的混淆矩阵
        if self.logger:
            log_dir = self.logger.log_dir
            try:
                # SEED数据集使用video_confusion_matrix，非SEED数据集使用confusion_matrix
                if "video_confusion_matrix" in val_result:
                    # SEED数据集：保存视频级别混淆矩阵
                    val_cm_file = os.path.join(log_dir, f"val_video_confusion_matrix_epoch_{self.current_epoch}.npy")
                    val_cm = np.array(val_result["video_confusion_matrix"])
                    np.save(val_cm_file, val_cm)
                    # 同时保存chunk级别的混淆矩阵作为参考
                    if "chunk_confusion_matrix" in val_result:
                        chunk_cm_file = os.path.join(log_dir, f"val_chunk_confusion_matrix_epoch_{self.current_epoch}.npy")
                        chunk_cm = np.array(val_result["chunk_confusion_matrix"])
                        np.save(chunk_cm_file, chunk_cm)
                elif "confusion_matrix" in val_result:
                    # 非SEED数据集：保存标准混淆矩阵
                    val_cm_file = os.path.join(log_dir, f"val_confusion_matrix_epoch_{self.current_epoch}.npy")
                    val_cm = np.array(val_result["confusion_matrix"])
                    np.save(val_cm_file, val_cm)
            except Exception as e:
                print(f"Warning: Could not save validation confusion matrix: {e}")
        
        # 把 val + test 的結果一起寫到 jsonl
        if self.logger:  # 確保 logger 存在
            log_dir = self.logger.log_dir
            log_file = os.path.join(log_dir, "training_logs.jsonl")

            log_entry = {
                "epoch": int(self.current_epoch),
                "step": int(self.global_step),
                "type": "validation+test",
                "val_metrics": val_result,
                "test_metrics": test_results,
            }

            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"Warning: Could not write to training_logs.jsonl: {e}")

    def _run_test_epoch(self):
        """Run one test epoch manually during validation."""
        self.model.eval()
        preds, targets = [], []

        test_loader = self.test_loader
        if test_loader is None:
            print("Warning: No test dataloader found, skipping test evaluation.")
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "cohen_kappa": 0.0,
                "balanced_accuracy": 0.0,
                "confusion_matrix": [],
            }

        # 临时存储epoch_ids（用于_run_test_epoch）
        temp_test_epoch_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                # SEED数据集在test时会返回(X, y, epoch_id)
                if len(batch) == 3:
                    X, y, epoch_ids = batch
                    temp_test_epoch_ids.extend(epoch_ids)
                else:
                    X, y = batch
                
                X = X.to(self.device)
                y = y.to(self.device)

                logits = self.model(X)
                prob = torch.softmax(logits, dim=1)
                preds.append(prob.cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        # 获取预测类别
        pred_classes = np.argmax(preds, axis=1)
        
        # === SEED数据集：计算视频级别的投票评估 ===
        is_seed_dataset = self.args.dataset in ["SEED", "SEED-ST"]
        if is_seed_dataset and len(temp_test_epoch_ids) == len(targets):
            # 使用视频级别的投票评估
            video_metrics = compute_video_level_metrics(preds, targets, temp_test_epoch_ids)
            
            # 记录视频级别的指标
            test_result = {
                "chunk_accuracy": multiclass_metrics_fn(targets, preds, metrics=["accuracy"])["accuracy"],
                "video_accuracy": video_metrics["video_accuracy"],
                "video_strict_accuracy": video_metrics["video_strict_accuracy"],
                "video_f1_macro": video_metrics["video_f1_macro"],
                "video_f1_weighted": video_metrics["video_f1_weighted"],
                "video_cohen_kappa": video_metrics["video_cohen_kappa"],
                "video_balanced_accuracy": video_metrics["video_balanced_accuracy"],
                "video_confusion_matrix": video_metrics["video_confusion_matrix"],
                "subject_accuracies": video_metrics["subject_accuracies"],
                "total_videos": video_metrics["total_videos"],
            }
            
            # 记录chunk级别的指标作为参考
            chunk_result = multiclass_metrics_fn(
                targets,
                preds,
                metrics=["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
            )
            test_result["chunk_f1_macro"] = chunk_result["f1_macro"]
            test_result["chunk_f1_weighted"] = chunk_result["f1_weighted"]
            test_result["chunk_cohen_kappa"] = chunk_result["cohen_kappa"]
            test_result["chunk_confusion_matrix"] = confusion_matrix(targets, pred_classes).tolist()
        else:
            # 非SEED数据集或没有epoch_id，使用chunk级别评估
            test_result = multiclass_metrics_fn(
                targets,
                preds,
                metrics=["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
            )
            test_result["balanced_accuracy"] = balanced_accuracy_score(targets, pred_classes)
            cm = confusion_matrix(targets, pred_classes)
            test_result["confusion_matrix"] = cm.tolist()

        # 根据是否有视频级别指标来决定log哪个
        if is_seed_dataset and "video_accuracy" in test_result:
            self.log("test_acc", test_result["video_accuracy"], sync_dist=True)
            self.log("test_video_strict_acc", test_result["video_strict_accuracy"], sync_dist=True)
            self.log("test_chunk_acc", test_result["chunk_accuracy"], sync_dist=True)
            self.log("test_f1_macro", test_result["video_f1_macro"], sync_dist=True)
            self.log("test_f1_weighted", test_result["video_f1_weighted"], sync_dist=True)
            self.log("test_cohen_kappa", test_result["video_cohen_kappa"], sync_dist=True)
            self.log("test_bacc", test_result["video_balanced_accuracy"], sync_dist=True)
        else:
            self.log("test_acc", test_result["accuracy"], sync_dist=True)
            self.log("test_f1_macro", test_result["f1_macro"], sync_dist=True)
            self.log("test_f1_weighted", test_result["f1_weighted"], sync_dist=True)
            self.log("test_cohen_kappa", test_result["cohen_kappa"], sync_dist=True)
            self.log("test_bacc", test_result["balanced_accuracy"], sync_dist=True)

        self.model.train()
        return test_result

    def test_step(self, batch, batch_idx):
        # SEED数据集在test时会返回(X, y, epoch_id)
        if len(batch) == 3:
            X, y, epoch_ids = batch
            # epoch_ids是list of str，需要保存
            self.test_epoch_ids.extend(epoch_ids)
        else:
            X, y = batch
        
        with torch.no_grad():
            logits = self.model(X)
            prob = torch.softmax(logits, dim=1)
            step_result = prob.detach().cpu().numpy()
            step_gt = y.detach().cpu().numpy()

        # ---- NEW: buffer for on_test_epoch_end ----
        self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    # ---- REPLACED: test_epoch_end -> on_test_epoch_end ----
    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            return

        preds = []
        targets = []

        for pred, tgt in self.test_step_outputs:
            preds.append(pred)   # (B, n_classes)
            targets.append(tgt)  # (B,)

        preds = np.concatenate(preds, axis=0)      # (N, n_classes)
        targets = np.concatenate(targets, axis=0)  # (N,)

        # 清 buffer
        self.test_step_outputs.clear()

        # 获取预测类别
        pred_classes = np.argmax(preds, axis=1)
        
        # === SEED数据集：计算视频级别的投票评估 ===
        is_seed_dataset = self.args.dataset in ["SEED", "SEED-ST"]
        if is_seed_dataset and len(self.test_epoch_ids) == len(targets):
            # 使用视频级别的投票评估
            video_metrics = compute_video_level_metrics(preds, targets, self.test_epoch_ids)
            
            # 记录视频级别的指标
            test_result = {
                "chunk_accuracy": multiclass_metrics_fn(targets, preds, metrics=["accuracy"])["accuracy"],
                "video_accuracy": video_metrics["video_accuracy"],
                "video_strict_accuracy": video_metrics["video_strict_accuracy"],
                "video_f1_macro": video_metrics["video_f1_macro"],
                "video_f1_weighted": video_metrics["video_f1_weighted"],
                "video_cohen_kappa": video_metrics["video_cohen_kappa"],
                "video_balanced_accuracy": video_metrics["video_balanced_accuracy"],
                "video_confusion_matrix": video_metrics["video_confusion_matrix"],
                "subject_accuracies": video_metrics["subject_accuracies"],
                "total_videos": video_metrics["total_videos"],
            }
            
            # 记录chunk级别的指标作为参考
            chunk_result = multiclass_metrics_fn(
                targets,
                preds,
                metrics=["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
            )
            test_result["chunk_f1_macro"] = chunk_result["f1_macro"]
            test_result["chunk_f1_weighted"] = chunk_result["f1_weighted"]
            test_result["chunk_cohen_kappa"] = chunk_result["cohen_kappa"]
            chunk_cm = confusion_matrix(targets, pred_classes)
            test_result["chunk_confusion_matrix"] = chunk_cm.tolist()
            
            # 计算分类报告（基于视频级别）
            # 从video_groups重建预测和标签
            try:
                video_preds_for_report = []
                video_targets_for_report = []
                for video_detail in video_metrics.get("video_details", []):
                    # video_detail是字典，包含vote_result和true_label
                    vote_result = video_detail.get('vote_result', None)
                    true_label = video_detail.get('true_label', None)
                    
                    if vote_result is not None and true_label is not None:
                        # vote_result可能是int（单个结果）或list（平票）
                        if isinstance(vote_result, list):
                            video_preds_for_report.append(vote_result[0])
                        else:
                            video_preds_for_report.append(vote_result)
                        video_targets_for_report.append(int(true_label))
                
                if len(video_preds_for_report) > 0:
                    class_report = classification_report(
                        video_targets_for_report, video_preds_for_report,
                        output_dict=True,
                        zero_division=0
                    )
                    test_result["video_classification_report"] = class_report
            except Exception as e:
                print(f"Warning: Could not compute video classification report: {e}")
                import traceback
                traceback.print_exc()
                test_result["video_classification_report"] = {}
            
            # 清空epoch_ids
            self.test_epoch_ids.clear()
            
            # 打印视频级别混淆矩阵和subject准确率
            video_cm = np.array(video_metrics["video_confusion_matrix"])
            print("\n" + "="*50)
            print("测试集 - 视频级别混淆矩阵 (Video-level Confusion Matrix):")
            print("="*50)
            print(video_cm)
            print("="*50)
            print("\n测试集 - Subject级别准确率:")
            print("="*50)
            for subject_key, acc_data in video_metrics["subject_accuracies"].items():
                print(f"{subject_key}: {acc_data['accuracy']:.4f} ({acc_data['total_videos']} videos)")
            print("="*50)
            print(f"\n视频级别准确率 (考虑平票): {video_metrics['video_accuracy']:.4f}")
            print(f"视频级别准确率 (严格): {video_metrics['video_strict_accuracy']:.4f}")
            print(f"Chunk级别准确率 (参考): {test_result['chunk_accuracy']:.4f}\n")
        else:
            # 非SEED数据集或没有epoch_id，使用chunk级别评估
            test_result = multiclass_metrics_fn(
                targets,
                preds,
                metrics=["accuracy", "f1_macro", "f1_weighted", "cohen_kappa"],
            )
            test_result["balanced_accuracy"] = balanced_accuracy_score(targets, pred_classes)
            cm = confusion_matrix(targets, pred_classes)
            test_result["confusion_matrix"] = cm.tolist()
            
            # 计算分类报告
            try:
                class_report = classification_report(
                    targets, pred_classes, 
                    output_dict=True, 
                    zero_division=0
                )
                test_result["classification_report"] = class_report
            except Exception as e:
                print(f"Warning: Could not compute classification report: {e}")
                test_result["classification_report"] = {}
            
            if len(self.test_epoch_ids) > 0:
                self.test_epoch_ids.clear()  # 清空，防止污染下次评估
            
            # 打印混淆矩阵
            print("\n" + "="*50)
            print("混淆矩阵 (Confusion Matrix):")
            print("="*50)
            print(cm)
            print("="*50 + "\n")
        
        # 根据是否有视频级别指标来决定log哪个
        if is_seed_dataset and "video_accuracy" in test_result:
            self.log("test_acc", test_result["video_accuracy"], sync_dist=True)
            self.log("test_video_strict_acc", test_result["video_strict_accuracy"], sync_dist=True)
            self.log("test_chunk_acc", test_result["chunk_accuracy"], sync_dist=True)
            self.log("test_f1_macro", test_result["video_f1_macro"], sync_dist=True)
            self.log("test_f1_weighted", test_result["video_f1_weighted"], sync_dist=True)
            self.log("test_cohen_kappa", test_result["video_cohen_kappa"], sync_dist=True)
            self.log("test_bacc", test_result["video_balanced_accuracy"], sync_dist=True)
        else:
            self.log("test_acc", test_result["accuracy"], sync_dist=True)
            self.log("test_f1_macro", test_result["f1_macro"], sync_dist=True)
            self.log("test_f1_weighted", test_result["f1_weighted"], sync_dist=True)
            self.log("test_cohen_kappa", test_result["cohen_kappa"], sync_dist=True)
            self.log("test_bacc", test_result["balanced_accuracy"], sync_dist=True)

        if self.logger:
            log_dir = self.logger.log_dir
            log_file = os.path.join(log_dir, "training_logs.jsonl")
            
            # 保存混淆矩阵为numpy文件（便于后续可视化）
            try:
                # SEED数据集使用video_confusion_matrix，非SEED数据集使用confusion_matrix
                if "video_confusion_matrix" in test_result:
                    # SEED数据集：保存视频级别混淆矩阵
                    video_cm_file = os.path.join(log_dir, f"test_video_confusion_matrix_epoch_{self.current_epoch}.npy")
                    video_cm = np.array(test_result["video_confusion_matrix"])
                    np.save(video_cm_file, video_cm)
                    # 同时保存chunk级别的混淆矩阵作为参考
                    if "chunk_confusion_matrix" in test_result:
                        chunk_cm_file = os.path.join(log_dir, f"test_chunk_confusion_matrix_epoch_{self.current_epoch}.npy")
                        chunk_cm = np.array(test_result["chunk_confusion_matrix"])
                        np.save(chunk_cm_file, chunk_cm)
                elif "confusion_matrix" in test_result:
                    # 非SEED数据集：保存标准混淆矩阵
                    cm_file = os.path.join(log_dir, f"test_confusion_matrix_epoch_{self.current_epoch}.npy")
                    cm = np.array(test_result["confusion_matrix"])
                    np.save(cm_file, cm)
            except Exception as e:
                print(f"Warning: Could not save confusion matrix: {e}")
            
            log_entry = {
                "epoch": int(self.current_epoch),
                "step": int(self.global_step),
                "type": "test",
                "metrics": test_result,
            }
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"Warning: Could not write to training_logs.jsonl: {e}")

        return test_result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        return optimizer

def prepare_TUEV_dataloader(args):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_HAR_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/HAR/processed/"

    train_files = os.listdir(os.path.join(root, "train"))
    test_files = os.listdir(os.path.join(root, "test"))
    val_files = os.listdir(os.path.join(root, "val"))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_Motion_dataloader(args):
    # === 固定 random seed ===
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # === 你的 dataset 根目錄 ===
    root = "../../AllSubjects_Epochs"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print("train/val/test:", len(train_files), len(val_files), len(test_files))

    # === DataLoaders ===
    train_loader = torch.utils.data.DataLoader(
        MotionLoader(
            os.path.join(root, "train"),
            train_files,
            sampling_rate=args.sampling_rate,
            in_channels=args.in_channels
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        # num_workers=args.num_workers,
        num_workers=0,
        persistent_workers=False,
    )

    val_loader = torch.utils.data.DataLoader(
        MotionLoader(
            os.path.join(root, "val"),
            val_files,
            sampling_rate=args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
        num_workers=0,
        persistent_workers=False,
    )

    test_loader = torch.utils.data.DataLoader(
        MotionLoader(
            os.path.join(root, "test"),
            test_files,
            sampling_rate=args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
        num_workers=0,
        persistent_workers=False,
    )

    print("Loader sizes:", len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_SEED_dataloader(args):
    """
    准备SEED数据集的数据加载器
    
    根据process_seed.ipynb的预处理：
    - 数据已经resample到200Hz (BIOT) 或 250Hz (STTransformer)
    - 原始label是0-6（7分类），但会过滤掉neutral (label=2)，重新映射为0-5（6分类）
    - 数据已经预处理过，只需要95%分位数归一化
    
    标签映射（移除neutral，从7分类变为6分类）：
    - 原始: happy=0, sad=1, neutral=2, disgust=3, fear=4, surprise=5, anger=6
    - 新的: happy=0, sad=1, disgust=2, fear=3, surprise=4, anger=5 (跳过neutral=2)
    
    数据结构：
    - biot_seed_data: train/subject_5/, val/subject_6/, test/subject_4/
    - st_seed_data: train/subject_4/, val/subject_5/, test/subject_6/
    两个数据集都已经按照train/val/test划分好了
    """
    # === 固定 random seed ===
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # === 数据集根目录 ===
    # 根据args.dataset判断使用哪个数据文件夹
    if args.dataset == "SEED":
        # 使用biot_seed_data (BIOT预处理，200Hz)
        root = "biot_seed_data"
    elif args.dataset == "SEED-ST":
        # 使用st_seed_data (STTransformer预处理，250Hz)
        root = "st_seed_data"
    else:
        raise ValueError(f"Unknown SEED dataset: {args.dataset}")

    # === 获取文件列表 ===
    # 两个数据集都已经按照train/val/test划分好了，结构相同
    # 过滤掉neutral标签（label=2）的数据，只保留6分类
    def filter_non_neutral_files(root, files):
        """过滤掉label=2 (neutral)的文件"""
        filtered = []
        for f in files:
            file_path = os.path.join(root, f) if root else f
            try:
                sample = pickle.load(open(file_path, "rb"))
                label = int(sample["label"])
                if label != 2:  # 跳过neutral
                    filtered.append(f)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}, skipping...")
        return filtered
    
    train_files = []
    train_dir = os.path.join(root, "train")
    if os.path.exists(train_dir):
        for subj_dir in os.listdir(train_dir):
            subj_path = os.path.join(train_dir, subj_dir)
            if os.path.isdir(subj_path):
                subj_files = os.listdir(subj_path)
                # 只保留pickle文件
                subj_files = [f for f in subj_files if f.endswith('.pickle')]
                full_paths = [os.path.join("train", subj_dir, f) for f in subj_files]
                train_files.extend(full_paths)
    
    val_files = []
    val_dir = os.path.join(root, "val")
    if os.path.exists(val_dir):
        for subj_dir in os.listdir(val_dir):
            subj_path = os.path.join(val_dir, subj_dir)
            if os.path.isdir(subj_path):
                subj_files = os.listdir(subj_path)
                # 只保留pickle文件
                subj_files = [f for f in subj_files if f.endswith('.pickle')]
                full_paths = [os.path.join("val", subj_dir, f) for f in subj_files]
                val_files.extend(full_paths)
    
    test_files = []
    test_dir = os.path.join(root, "test")
    if os.path.exists(test_dir):
        for subj_dir in os.listdir(test_dir):
            subj_path = os.path.join(test_dir, subj_dir)
            if os.path.isdir(subj_path):
                subj_files = os.listdir(subj_path)
                # 只保留pickle文件
                subj_files = [f for f in subj_files if f.endswith('.pickle')]
                full_paths = [os.path.join("test", subj_dir, f) for f in subj_files]
                test_files.extend(full_paths)
    
    # 过滤掉neutral标签的文件
    train_files = filter_non_neutral_files(root, train_files)
    val_files = filter_non_neutral_files(root, val_files)
    test_files = filter_non_neutral_files(root, test_files)

    np.random.shuffle(train_files)
    print("train/val/test files:", len(train_files), len(val_files), len(test_files))

    # === DataLoaders ===
    # 数据已经预处理过，不需要label_offset和重采样
    # 训练时不需要epoch_id，测试和验证时需要epoch_id用于视频投票评估
    train_loader = torch.utils.data.DataLoader(
        SEEDLoader(
            root,
            train_files,
            sampling_rate=args.sampling_rate,
            return_epoch_id=False  # 训练时不需要
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = torch.utils.data.DataLoader(
        SEEDLoader(
            root,
            val_files,
            sampling_rate=args.sampling_rate,
            return_epoch_id=True  # 验证时需要epoch_id用于视频投票
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn_seed_with_epoch_id,  # 使用自定义collate函数处理epoch_id
    )

    test_loader = torch.utils.data.DataLoader(
        SEEDLoader(
            root,
            test_files,
            sampling_rate=args.sampling_rate,
            return_epoch_id=True  # 测试时需要epoch_id用于视频投票
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn_seed_with_epoch_id,  # 使用自定义collate函数处理epoch_id
    )

    print("Loader sizes:", len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def supervised(args):
    # get data loaders
    if args.dataset == "TUEV":
        train_loader, test_loader, val_loader = prepare_TUEV_dataloader(args)

    elif args.dataset == "Motion":
        train_loader, test_loader, val_loader = prepare_Motion_dataloader(args)

    elif args.dataset in ["SEED", "SEED-ST"]:
        train_loader, test_loader, val_loader = prepare_SEED_dataloader(args)

    else:
        raise NotImplementedError

    # define the model
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sample_length * args.sampling_rate),
            n_classes=args.n_classes,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        )

    elif args.model == "ContraWR":
        model = ContraWR(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
        )

    elif args.model == "CNNTransformer":
        model = CNNTransformer(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.sampling_rate,
            steps=args.hop_length // 5,
            dropout=0.2,
            nhead=4,
            emb_size=256,
            n_segments=4 if args.dataset == "HAR" else 5,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sample_length * args.sampling_rate),
            shrink_steps=16 if args.dataset == "HAR" else 20,
        )

    elif args.model == "STTransformer":
        model = STTransformer(
            emb_size=256,
            depth=4,
            n_classes=args.n_classes,
            channel_legnth=int(
                args.sampling_rate * args.sample_length
            ),  # (sampling_rate * duration)
            n_channels=args.in_channels,
        )

    elif args.model == "BIOT":
        if args.dataset_channels ==  args.in_channels:
            model = BIOTClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        else:
            model = Ada_BIOT(
                input_chan_size=args.dataset_channels,
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")
    elif args.model == "LabramClassifier-BIOT":
        if args.dataset_channels ==  args.in_channels:
            model = Labram_style_BIOTClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        else:
            model = Labram_style_Ada_BIOT(
                input_chan_size=args.dataset_channels,
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")
    elif args.model == "CBraMod_3lyStyle_LayerNorm-BIOT":
        if args.dataset_channels ==  args.in_channels:
            model = CBraMod_3lyStyle_LayerNorm_BIOT(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        else:
            model = CBraMod_3lyStyle_LayerNorm_Ada_BIOT(
                input_chan_size=args.dataset_channels,
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")
    else:
        raise NotImplementedError
    
    if args.freeze_backbone:
        print("Freezing parameters for model.biot...")
        for param in model.biot.parameters():
            param.requires_grad = False
        print("Parameters frozen.")
    
    lightning_model = LitModel_finetune(args, model, test_loader=test_loader)

    # logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log",
    )
    # 將所有參數 (args) 儲存為 config.yaml
    log_dir = os.path.join("./log", version)
    os.makedirs(log_dir, exist_ok=True)
    args_dict = vars(args)
    args_yaml_path = os.path.join(log_dir, "config.yaml")
    try:
        with open(args_yaml_path, 'w') as f:
            yaml.dump(args_dict, f, sort_keys=False)
        print(f"Configuration saved to {args_yaml_path}")
    except Exception as e:
        print(f"Could not save config.yaml: {e}. (Is 'pyyaml' installed?)")
    # 將模型結構儲存為 model_structure.txt
    model_txt_path = os.path.join(log_dir, "model_structure.txt")
    try:
        with open(model_txt_path, 'w') as f:
            f.write(str(lightning_model.model))
        print(f"Model structure saved to {model_txt_path}")
    except Exception as e:
        print(f"Error saving model_structure.txt: {e}")

    # 取消早停机制
    # early_stop_callback = EarlyStopping(
    #     monitor="val_f1_macro", patience=5, verbose=True, mode="max"
    # )

    trainer = pl.Trainer(
        # devices=[0],
        accelerator="gpu",
        # strategy=DDPStrategy(find_unused_parameters=False),
        strategy="auto",
        devices=1,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[],  # 移除早停回调
    )
    # trainer = pl.Trainer(
    #     accelerator="cpu",
    #     devices=1,
    #     precision=32,
    #     logger=logger,
    #     max_epochs=args.epochs,
    #     callbacks=[early_stop_callback],
    #     num_sanity_val_steps=0,
    # )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print("\n" + "="*50)
    print("最终测试结果:")
    print("="*50)
    print(pretrain_result)
    
    # 如果结果中包含混淆矩阵，也打印出来
    if "confusion_matrix" in pretrain_result:
        print("\n" + "="*50)
        print("测试集混淆矩阵:")
        print("="*50)
        cm = np.array(pretrain_result["confusion_matrix"])
        print(cm)
        print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="finetune", help="experiment name")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
    parser.add_argument(
        "--dataset_channels", type=int, default=None,  # 如果為 None，則使用 in_channels
        help="actual number of channels in dataset (if different from pretrained model)"
    )
    parser.add_argument('--pos_weight', default=None, type=float)
    parser.add_argument(
        "--model", type=str, default="SPaRCNet", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=12, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=6, help="number of output classes"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="", help="pretrained model path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./", help="saved model path"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the backbone during training (default: False)."
    )
    args = parser.parse_args()
    print(args)

    supervised(args)
