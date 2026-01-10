import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from finetune_evaluator import Evaluator
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
import umap
from sklearn.decomposition import PCA
import copy
import os
import json
import time


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a', 'TUSL', 'HMC', 'SEED-Emotion']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB', 'siena']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()


        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        bacc_best = 0
        cm_best = None
        self.best_val_voting_results = None
        self.best_test_voting_results = None
        self.training_logs = []
        self.training_start_time = time.time()
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for batch in tqdm(self.data_loader['train'], mininterval=10):
                # 兼容处理：batch可能是(x, y)或(x, y, epoch_ids)
                if len(batch) == 3:
                    x, y, _ = batch  # 忽略epoch_ids，训练时不需要
                else:
                    x, y = batch
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, bacc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                
                # 如果是SEED-Emotion数据集，额外进行投票评估
                val_voting_results = None
                if self.params.downstream_dataset == 'SEED-Emotion':
                    print("\n--- 投票评估 (Val) ---")
                    val_voting_results = self.val_eval.get_metrics_with_voting(self.model)
                
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, bacc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins | model: {}, seed: {}, lr: {}, weight_decay: {}, dropout: {}, foundation_dir: {}".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        bacc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60,
                        self.params.model,
                        self.params.seed,
                        self.params.lr,
                        self.params.weight_decay,
                        self.params.dropout,
                        self.params.foundation_dir
                    )
                )
                print(cm)
                if val_voting_results is not None:
                    print(f"  Val Voting  - video_acc: {val_voting_results['video_acc']:.5f}, subject_acc: {val_voting_results['overall_subject_acc']:.5f}")
                    print("Val Voting CM (video level):")
                    print(val_voting_results['video_cm'])
                
                # 保存 epoch 训练日志
                epoch_log = {
                    'epoch': epoch + 1,
                    'train_loss': float(np.mean(losses)),
                    'val_acc': float(acc),
                    'val_bacc': float(bacc),
                    'val_kappa': float(kappa),
                    'val_f1': float(f1),
                    'learning_rate': float(optim_state['param_groups'][0]['lr']),
                    'epoch_time_seconds': float((timer() - start_time)),
                    'epoch_time_minutes': float((timer() - start_time) / 60),
                }
                
                # 如果是SEED-Emotion，添加投票评估结果
                if val_voting_results is not None:
                    epoch_log.update({
                        'val_voting_video_acc': float(val_voting_results['video_acc']),
                        'val_voting_subject_acc': float(val_voting_results['overall_subject_acc']),
                    })
                
                self.training_logs.append(epoch_log)
                
                if bacc > bacc_best: # 使用 bacc 作为最佳模型选择标准
                    print("BACC increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, bacc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        bacc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    bacc_best = bacc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    if val_voting_results is not None:
                        self.best_val_voting_results = val_voting_results
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            test_acc, test_bacc, test_kappa, test_f1, test_cm = self.test_eval.get_metrics_for_multiclass(self.model)
            
            # 如果是SEED-Emotion数据集，额外进行投票评估
            test_voting_results = None
            if self.params.downstream_dataset == 'SEED-Emotion':
                print("\n--- 投票评估 (Test) ---")
                test_voting_results = self.test_eval.get_metrics_with_voting(self.model)
                self.best_test_voting_results = test_voting_results
            
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, bacc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    test_acc,
                    test_bacc,
                    test_kappa,
                    test_f1,
                )
            )
            print(test_cm)
            if test_voting_results is not None:
                print(f"\nTest Voting  - video_acc: {test_voting_results['video_acc']:.5f}, subject_acc: {test_voting_results['overall_subject_acc']:.5f}")
                print("Test Voting CM (video level):")
                print(test_voting_results['video_cm'])
            
            if not os.path.exists(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            
            # 保存最终结果
            final_results = {
                'best_epoch': best_f1_epoch,
                'val_acc': float(acc_best),
                'val_bacc': float(bacc_best),
                'val_kappa': float(kappa_best),
                'val_f1': float(f1_best),
                'val_cm': cm_best.tolist() if cm_best is not None else None,
                'test_acc': float(test_acc),
                'test_bacc': float(test_bacc),
                'test_kappa': float(test_kappa),
                'test_f1': float(test_f1),
                'test_cm': test_cm.tolist(),
            }
            
            # 如果是SEED-Emotion，添加投票评估结果
            if self.best_val_voting_results is not None and self.best_test_voting_results is not None:
                final_results.update({
                    'val_voting_video_acc': float(self.best_val_voting_results['video_acc']),
                    'val_voting_subject_acc': float(self.best_val_voting_results['overall_subject_acc']),
                    'val_voting_subject_accs': {k: float(v) for k, v in self.best_val_voting_results['subject_accs'].items()},
                    'val_voting_video_cm': self.best_val_voting_results['video_cm'].tolist(),
                    'test_voting_video_acc': float(self.best_test_voting_results['video_acc']),
                    'test_voting_subject_acc': float(self.best_test_voting_results['overall_subject_acc']),
                    'test_voting_subject_accs': {k: float(v) for k, v in self.best_test_voting_results['subject_accs'].items()},
                    'test_voting_video_cm': self.best_test_voting_results['video_cm'].tolist(),
                })
            
            # 保存模型
            model_path = self.params.model_dir + "/epoch{}_valBacc_{:.5f}_testBacc_{:.5f}.pth".format(best_f1_epoch, bacc_best, test_bacc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            
            # 保存训练日志
            self.save_training_logs(final_results)
    
    def save_training_logs(self, final_results=None):
        """儲存訓練日誌"""
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        
        # 儲存每個epoch的訓練日誌
        log_path = os.path.join(self.params.model_dir, 'training_logs.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_logs, f, indent=4)
        print(f"Training logs saved to {log_path}")
        
        # 儲存最終結果摘要
        if final_results is not None:
            summary_path = os.path.join(self.params.model_dir, 'final_results.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=4)
            print(f"Final results saved to {summary_path}")
            
            # 打印最終結果摘要
            print("\n" + "="*70)
            print("FINAL RESULTS SUMMARY")
            print("="*70)
            print(f"Best Epoch: {final_results['best_epoch']}")
            print(f"\nValidation Set:")
            print(f"  Accuracy:  {final_results['val_acc']:.5f}")
            print(f"  BACC:      {final_results['val_bacc']:.5f}")
            print(f"  Kappa:     {final_results['val_kappa']:.5f}")
            print(f"  F1-Score:  {final_results['val_f1']:.5f}")
            print(f"\nTest Set:")
            print(f"  Accuracy:  {final_results['test_acc']:.5f}")
            print(f"  BACC:      {final_results['test_bacc']:.5f}")
            print(f"  Kappa:     {final_results['test_kappa']:.5f}")
            print(f"  F1-Score:  {final_results['test_f1']:.5f}")
            
            # 如果是SEED-Emotion，打印投票结果
            if 'val_voting_video_acc' in final_results:
                print(f"\nVoting Results (Val):")
                print(f"  Video Acc:    {final_results['val_voting_video_acc']:.5f}")
                print(f"  Subject Acc:  {final_results['val_voting_subject_acc']:.5f}")
                print(f"\nVoting Results (Test):")
                print(f"  Video Acc:    {final_results['test_voting_video_acc']:.5f}")
                print(f"  Subject Acc:  {final_results['test_voting_subject_acc']:.5f}")
                if 'test_voting_subject_accs' in final_results:
                    print(f"\n  Per-Subject Accuracies:")
                    for subject_id, acc in sorted(final_results['test_voting_subject_accs'].items()):
                        print(f"    Subject {subject_id}: {acc:.5f}")
            print("="*70 + "\n")


    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins | model: {}, seed: {}, lr: {}, weight_decay: {}, dropout: {}".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60,
                        self.params.model,
                        self.params.seed,
                        self.params.lr,
                        self.params.weight_decay,
                        self.params.dropout
                    )
                )
                print(cm)
                if acc > acc_best: # zhouyc
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)

            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)


    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)