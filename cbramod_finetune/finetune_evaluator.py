import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error, recall_score
from tqdm import tqdm
import re
from collections import defaultdict, Counter


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_metrics_for_multiclass(self, model):
        model.eval()

        truths = []
        preds = []
        for batch in tqdm(self.data_loader, mininterval=1):
            # 兼容处理：batch 可能是 (x,y) / (x,y,epoch_ids) / (x,y,epoch_ids,sample_ids)；
            # 这里不需要 epoch_ids/sample_ids，只取前两项
            x, y = batch[0], batch[1]

            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = accuracy_score(truths, preds)
        bacc = balanced_accuracy_score(truths, preds)
        # [保留] weighted F1，用于与旧实验日志对齐
        f1_weighted = f1_score(truths, preds, average='weighted')
        # ===== [R1] Macro F1 + Per-class Recall（审稿要求的扩展指标）=====
        f1_macro = f1_score(truths, preds, average='macro')
        # labels 显式固定为 0..num_of_classes-1：某些受试者某个类别的样本数
        # 为 0（如 MotorTask 的 Sub05 完全没有 Horizontal 这一类），若不传
        # labels，sklearn 会按 truths/preds 里实际出现的类别数推断形状，
        # 使 per_class_recall 少一维、cm 变成非 num_of_classes x num_of_classes——
        # 前者会让下游按位置 zip(class_names, per_class_recall) 错位，
        # 后者会让下游按固定 class_names 画热力图时形状对不上。
        labels = list(range(self.params.num_of_classes))
        per_class_recall = recall_score(truths, preds, labels=labels, average=None, zero_division=0)
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds, labels=labels)

        # 默认仍返回 weighted F1，保持旧 random_epoch 实验日志可对比；
        # subject_independent 时额外打印审稿指标，并返回 macro F1。
        split_mode = getattr(self.params, 'split_mode', 'random_epoch')
        if split_mode == 'subject_independent':
            print("[R1 Expanded Test/Val Metrics]")
            print(f"  Balanced Accuracy: {bacc:.5f}")
            print(f"  Macro F1:          {f1_macro:.5f}")
            print(f"  Weighted F1:       {f1_weighted:.5f}")
            print(f"  Per-class Recall:  {np.array2string(per_class_recall, precision=5, separator=', ')}")
            print("  Confusion Matrix:")
            print(cm)
            f1 = f1_macro
        else:
            f1 = f1_weighted
        return acc, bacc, kappa, f1, cm

    def save_test_predictions_for_mcnemar(self, model, save_path):
        """在最佳 val bacc 时调用，保存 test 逐样本预测（与 get_metrics 使用相同 data_loader）"""
        model.eval()
        results = []
        with torch.no_grad():
            for batch in tqdm(self.data_loader, mininterval=1):
                if len(batch) >= 3:
                    x, y, epoch_ids = batch[0], batch[1], batch[2]
                else:
                    break  # 需要 epoch_ids
                x = x.cuda()
                pred = model(x)
                pred_y = torch.max(pred, dim=-1)[1]
                for i, eid in enumerate(epoch_ids):
                    p = int(pred_y[i].cpu().item())
                    t = int(y[i].cpu().item())
                    c = 1 if p == t else 0
                    results.append((eid, p, t, c))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('epoch_id\tpred\ttrue\tcorrect\n')
            for eid, p, t, c in results:
                f.write(f'{eid}\t{p}\t{t}\t{c}\n')
        print(f"Saved test predictions for McNemar to {save_path}")

    def save_fold_predictions_npz(self, model, task, model_name, fold_idx, save_dir):
        """
        [R2] 对 self.data_loader（调用方传入 test loader）跑一遍推理，
        按 sample_id 排序后保存为 {task}_{model_name}_fold{fold_idx:02d}.npz，
        字段：sample_id / y_true / y_pred / y_prob(softmax，全部类别) / subject_id。
        目的：所有下游指标事后都能从这个 npz 重新算，不需要重跑训练。

        要求 dataset 的 collate 返回 (x, y, epoch_ids, sample_ids) 四元组
        （见 datasets/motortask_dataset.py 的 compute_sample_id）；
        任何异常（batch 里没有 sample_id、结果为空、写文件失败、写完读不回来）
        都直接抛异常退出，不静默跳过——保存失败必须让调用方知道。
        """
        model.eval()
        sample_ids, y_true, y_pred, y_prob, subject_ids = [], [], [], [], []
        subj_re = re.compile(r'^S(\d+)_')
        # 二分类任务（CustomStress/KaggleERN）的模型输出是每个样本一个标量 logit
        # （给 BCEWithLogitsLoss 用，形状 (batch,)），不是多分类的 (batch, n_classes)。
        # 直接对标量 logits 做 softmax(dim=-1)/argmax(dim=-1) 会退化成对整个 batch
        # 求一个 0-dim 结果，而不是逐样本的预测——这里按 downstream_dataset 分支，
        # 二分类走 sigmoid+0.5 阈值（跟 get_metrics_for_binaryclass 一致），并把
        # y_prob 拼成 (N, 2) 的 [P(class0), P(class1)]，这样 compute_metrics_from_npz.py
        # 里 y_prob.argmax(axis=1) == y_pred 的自洽性检查不用改，两种任务共用同一套 npz schema。
        is_binary = self.params.downstream_dataset in ('CustomStress', 'KaggleERN')
        with torch.no_grad():
            for batch in tqdm(self.data_loader, mininterval=1, desc="save_fold_predictions_npz"):
                if len(batch) < 4:
                    raise RuntimeError(
                        f"save_fold_predictions_npz requires (x, y, epoch_ids, sample_ids) batches "
                        f"(got {len(batch)} elements) — dataset collate must return sample_id."
                    )
                x, y, epoch_ids, batch_sample_ids = batch[0], batch[1], batch[2], batch[3]
                x = x.cuda()
                logits = model(x)
                if is_binary:
                    prob_pos = torch.sigmoid(logits)
                    preds = (prob_pos > 0.5).long()
                    probs = torch.stack([1 - prob_pos, prob_pos], dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                for i, sid in enumerate(batch_sample_ids):
                    m = subj_re.match(sid)
                    if not m:
                        raise ValueError(f"Cannot parse subject_id from sample_id: {sid!r}")
                    sample_ids.append(sid)
                    y_true.append(int(y[i].item()))
                    y_pred.append(int(preds[i].cpu().item()))
                    y_prob.append(probs[i].cpu().numpy())
                    subject_ids.append(int(m.group(1)))

        if len(sample_ids) == 0:
            raise RuntimeError("save_fold_predictions_npz: no samples collected, refusing to save an empty file.")

        sample_ids_arr = np.array(sample_ids)
        order = np.argsort(sample_ids_arr)
        sample_ids_arr = sample_ids_arr[order]
        y_true_arr = np.array(y_true, dtype=np.int64)[order]
        y_pred_arr = np.array(y_pred, dtype=np.int64)[order]
        y_prob_arr = np.array(y_prob, dtype=np.float32)[order]
        subject_id_arr = np.array(subject_ids, dtype=np.int64)[order]

        os.makedirs(save_dir, exist_ok=True)
        npz_path = os.path.join(save_dir, f"{task}_{model_name}_fold{fold_idx:02d}.npz")
        np.savez(
            npz_path,
            sample_id=sample_ids_arr,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_prob=y_prob_arr,
            subject_id=subject_id_arr,
        )
        if not os.path.exists(npz_path):
            raise RuntimeError(f"save_fold_predictions_npz: failed to write {npz_path}")
        # 保存完立刻回读校验，保存失败/损坏要当场报错，而不是留到事后分析才发现
        check = np.load(npz_path)
        for key in ("sample_id", "y_true", "y_pred", "y_prob", "subject_id"):
            if key not in check:
                raise RuntimeError(f"save_fold_predictions_npz: {npz_path} missing key '{key}' after save")
            if len(check[key]) != len(sample_ids_arr):
                raise RuntimeError(f"save_fold_predictions_npz: {npz_path} key '{key}' length mismatch after save")

        print(f"Saved fold predictions npz to {npz_path} ({len(sample_ids_arr)} samples)")
        return npz_path, sample_ids_arr, y_true_arr, y_pred_arr, y_prob_arr, subject_id_arr

    def get_metrics_for_binaryclass(self, model):
        model.eval()

        truths = []
        preds = []
        scores = []
        for batch in tqdm(self.data_loader, mininterval=1):
            # 兼容处理：batch 可能是 (x,y) / (x,y,epoch_ids,sample_ids)（LOSO test set）
            x, y = batch[0], batch[1]
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = accuracy_score(truths, preds)
        bacc = balanced_accuracy_score(truths, preds)
        # LOSO 场景下某一折的 val/test 受试者可能只有单一类别（stress 数据里
        # 17 个受试者中有 11 个只做过 increase 或只做过 normal），roc_auc_score
        # 在 y_true 只有一个类别时会直接抛异常；bacc 的模型选择逻辑不受影响，
        # 这里只对不可定义的 roc_auc/pr_auc 返回 NaN，不让训练崩溃。
        if len(np.unique(truths)) < 2:
            roc_auc = float('nan')
            pr_auc = float('nan')
        else:
            roc_auc = roc_auc_score(truths, scores)
            precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
            pr_auc = auc(recall, precision)
        # 显式传 labels=[0,1]：否则当 truths/preds 恰好都只出现一个类别时，
        # confusion_matrix 会退化成 1x1，跟其它折的 2x2 形状对不上。
        cm = confusion_matrix(truths, preds, labels=[0, 1])
        return acc, bacc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse

    def extract_video_index(self, epoch_id):
        """从 epoch_id 中提取 video_index"""
        match = re.search(r'video_index_(\d+)_chunk', epoch_id)
        if match:
            return int(match.group(1))
        return None

    def extract_subject_id(self, epoch_id):
        """从 epoch_id 中提取 subject_id"""
        match = re.search(r'subject_(\d+)_', epoch_id)
        if match:
            return int(match.group(1))
        return None

    def vote_majority(self, preds_list):
        """
        多数投票，处理平票情况
        
        Args:
            preds_list: 同一视频的所有chunks的预测标签列表
            
        Returns:
            (voted_label, score): 
            - voted_label: 投票结果（如果有平票，返回平票候选项列表）
            - score: 1.0(完全正确), 0.5(平票且真实标签在候选中), 0.0(错误)
        """
        if not preds_list:
            return None, 0.0
        
        # 统计每个类别的得票数
        vote_counts = Counter(preds_list)
        max_votes = max(vote_counts.values())
        
        # 找出得票最多的类别（可能多个）
        winners = [label for label, count in vote_counts.items() if count == max_votes]
        
        if len(winners) == 1:
            # 没有平票，返回单一结果
            return winners[0], None
        else:
            # 有平票，返回候选项列表
            return winners, None

    def get_metrics_with_voting(self, model):
        """
        使用投票机制的评估方法（适用于SEED-Emotion数据集）
        
        评估流程：
        1. 收集所有chunks的预测和真实标签，以及对应的epoch_id
        2. 按video_index分组（同一个subject的同一个video）
        3. 对每个视频的chunks进行多数投票
        4. 处理平票：如果真实标签在平票候选中，算0.5分，否则0.0分
        5. 按subject分组，计算每个subject的准确率（正确数量/80个视频）
        
        Returns:
            dict: 包含以下指标
                - video_acc: 视频级别的准确率（考虑平票）
                - subject_accs: 每个subject的准确率字典
                - overall_subject_acc: 所有subject的平均准确率
                - video_results: 详细的视频级别结果
        """
        model.eval()
        
        # 收集所有chunks的预测、真实标签和epoch_id
        truths = []
        preds = []
        epoch_ids = []
        
        for batch in tqdm(self.data_loader, mininterval=1, desc="Collecting predictions"):
            if len(batch) == 3:
                x, y, batch_epoch_ids = batch
                epoch_ids.extend(batch_epoch_ids)
            else:
                raise ValueError("Dataset should return (x, y, epoch_id) for voting evaluation. "
                               "Please check if dataset collate function returns epoch_ids.")
            
            x = x.cuda()
            y = y.cuda()
            
            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]
            
            truths.extend(y.cpu().squeeze().numpy().tolist())
            preds.extend(pred_y.cpu().squeeze().numpy().tolist())
        
        # 按video分组
        video_groups = defaultdict(lambda: {'preds': [], 'truths': [], 'subject_id': None, 'video_index': None})
        
        for pred, truth, epoch_id in zip(preds, truths, epoch_ids):
            subject_id = self.extract_subject_id(epoch_id)
            video_index = self.extract_video_index(epoch_id)
            
            if subject_id is None or video_index is None:
                print(f"Warning: Failed to extract subject_id or video_index from {epoch_id}, skipping...")
                continue
            
            # 使用(subject_id, video_index)作为唯一标识
            key = (subject_id, video_index)
            video_groups[key]['preds'].append(pred)
            video_groups[key]['truths'].append(truth)
            video_groups[key]['subject_id'] = subject_id
            video_groups[key]['video_index'] = video_index
        
        print(f"\n找到 {len(video_groups)} 个视频")
        
        # 对每个视频进行投票
        video_results = []
        video_correct = 0
        video_total = 0
        
        subject_results = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # 用于混淆矩阵：收集视频级别的真实标签和投票预测
        video_truths = []
        video_preds = []
        
        for (subject_id, video_index), group in sorted(video_groups.items()):
            preds_list = group['preds']
            # 同一视频的所有chunks应该有相同的真实标签
            truth_label = group['truths'][0]
            
            # 验证：同一视频的所有chunks应该有相同的真实标签
            if not all(t == truth_label for t in group['truths']):
                print(f"Warning: Video (subject_{subject_id}, video_{video_index}) has inconsistent truth labels!")
            
            # 投票
            voted_label, _ = self.vote_majority(preds_list)
            
            # 计算得分和用于混淆矩阵的预测标签
            # 对于平票情况，选择第一个候选项用于混淆矩阵（或真实标签如果在候选中）
            if isinstance(voted_label, list):
                # 平票情况
                if truth_label in voted_label:
                    score = 0.5
                    # 如果真实标签在候选中，使用真实标签（这样混淆矩阵对角线会增加）
                    cm_pred_label = truth_label
                else:
                    score = 0.0
                    # 否则使用第一个候选项
                    cm_pred_label = voted_label[0]
                vote_result_str = f"{voted_label} (tie)"
            else:
                # 单一结果
                if voted_label == truth_label:
                    score = 1.0
                else:
                    score = 0.0
                cm_pred_label = voted_label
                vote_result_str = str(voted_label)
            
            video_correct += score
            video_total += 1
            
            subject_results[subject_id]['correct'] += score
            subject_results[subject_id]['total'] += 1
            
            # 收集用于混淆矩阵的数据
            video_truths.append(int(truth_label))
            video_preds.append(int(cm_pred_label))
            
            video_results.append({
                'subject_id': subject_id,
                'video_index': video_index,
                'truth': int(truth_label),
                'voted_pred': voted_label if not isinstance(voted_label, list) else voted_label,
                'chunks_preds': preds_list,
                'score': score,
                'num_chunks': len(preds_list)
            })
        
        # 计算指标
        video_acc = video_correct / video_total if video_total > 0 else 0.0
        
        # 计算每个subject的准确率
        subject_accs = {}
        for subject_id in sorted(subject_results.keys()):
            result = subject_results[subject_id]
            acc = result['correct'] / result['total'] if result['total'] > 0 else 0.0
            subject_accs[subject_id] = acc
        
        overall_subject_acc = np.mean(list(subject_accs.values())) if subject_accs else 0.0
        
        # 计算混淆矩阵（视频级别）
        video_truths_array = np.array(video_truths)
        video_preds_array = np.array(video_preds)
        video_cm = confusion_matrix(video_truths_array, video_preds_array)
        
        print(f"\n视频级别准确率: {video_acc:.4f} ({video_correct:.1f}/{video_total})")
        print(f"\n各Subject准确率:")
        for subject_id, acc in sorted(subject_accs.items()):
            print(f"  Subject {subject_id}: {acc:.4f} ({subject_results[subject_id]['correct']:.1f}/{subject_results[subject_id]['total']})")
        print(f"\n平均Subject准确率: {overall_subject_acc:.4f}")
        print(f"\n视频级别混淆矩阵:")
        print(video_cm)
        
        return {
            'video_acc': video_acc,
            'video_correct': video_correct,
            'video_total': video_total,
            'subject_accs': subject_accs,
            'overall_subject_acc': overall_subject_acc,
            'video_cm': video_cm,
            'video_results': video_results
        }