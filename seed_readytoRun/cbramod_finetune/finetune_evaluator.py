import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error
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
            # 兼容处理：batch可能是(x, y)或(x, y, epoch_ids)
            if len(batch) == 3:
                x, y, epoch_ids = batch
            else:
                x, y = batch
            
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
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, bacc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model):
        model.eval()

        truths = []
        preds = []
        scores = []
        for x, y in tqdm(self.data_loader, mininterval=1):
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
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
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