import copy
import os
import json
import time
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

import psutil
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available. Install with: pip install gputil")

from finetune_evaluator import Evaluator


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB', 'CustomStress']:
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

        # ===== 新增：訓練記錄 =====
        self.training_start_time = None
        self.training_logs = []  # 儲存每個 epoch 的訓練日誌
        self.resource_logs = []  # 儲存資源使用記錄

    def log_gpu_usage(self, epoch, phase='train', step=None):
        """
        記錄GPU和系統資源使用情況
        1. GPU 記憶體：使用 torch.cuda.memory_allocated() 獲取已分配記憶體（單位：bytes -> GB）
        2. GPU 使用率：使用 GPUtil 獲取 GPU 負載百分比
        3. CPU：使用 psutil.cpu_percent() 獲取 CPU 使用率
        4. RAM：使用 psutil.virtual_memory() 獲取記憶體使用情況
        """
        if not torch.cuda.is_available():
            return None
            
        gpu_stats = {
            'epoch': epoch,
            'phase': phase,
            'step': step,
            'timestamp': time.time() - self.training_start_time,
        }
        
        # === GPU 記憶體統計（使用 torch.cuda API）===
        # 計算方式：torch.cuda.memory_allocated(device_id) 返回當前分配的記憶體（bytes）
        # 除以 1024^3 轉換為 GB
        for i in range(torch.cuda.device_count()):
            gpu_stats[f'gpu_{i}_memory_allocated_GB'] = torch.cuda.memory_allocated(i) / 1024**3
            gpu_stats[f'gpu_{i}_memory_reserved_GB'] = torch.cuda.memory_reserved(i) / 1024**3
            gpu_stats[f'gpu_{i}_max_memory_allocated_GB'] = torch.cuda.max_memory_allocated(i) / 1024**3
        
        # === GPU 使用率統計（使用 GPUtil）===
        # 計算方式：GPUtil.getGPUs() 返回 GPU 物件列表，每個 GPU 物件有 load 屬性（0-1之間）
        # 乘以 100 轉換為百分比
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_stats[f'gpu_{i}_utilization_%'] = gpu.load * 100  # load 是 0-1 的浮點數
                    gpu_stats[f'gpu_{i}_temperature_C'] = gpu.temperature
                    gpu_stats[f'gpu_{i}_memory_used_GB'] = gpu.memoryUsed / 1024  # memoryUsed 單位是 MB
                    gpu_stats[f'gpu_{i}_memory_total_GB'] = gpu.memoryTotal / 1024
            except Exception as e:
                print(f"Warning: Failed to get GPU stats from GPUtil: {e}")
        
        # === CPU 統計（使用 psutil）===
        # 計算方式：psutil.cpu_percent(interval=0.1) 返回 CPU 使用百分比
        # interval=0.1 表示在 0.1 秒內測量
        gpu_stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        
        # === RAM 統計（使用 psutil）===
        # 計算方式：psutil.virtual_memory() 返回記憶體統計物件
        # .used 屬性是已使用記憶體（bytes），除以 1024^3 轉換為 GB
        # .percent 屬性直接返回使用百分比
        memory_info = psutil.virtual_memory()
        gpu_stats['ram_used_GB'] = memory_info.used / 1024**3
        gpu_stats['ram_percent'] = memory_info.percent
        
        self.resource_logs.append(gpu_stats)
        return gpu_stats


    def save_training_logs(self, final_results=None):
        """
        儲存訓練日誌和資源使用記錄
        生成：
        1. training_logs.json - 每個 epoch 的訓練指標
        2. resource_logs.csv - 詳細的資源使用記錄
        3. training_summary.json - 訓練摘要統計
        """
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        
        # 儲存訓練日誌
        log_path = os.path.join(self.params.model_dir, 'training_logs.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_logs, f, indent=4)
        print(f"Training logs saved to {log_path}")
        
        # 儲存資源使用記錄
        if self.resource_logs:
            import pandas as pd
            df = pd.DataFrame(self.resource_logs)
            csv_path = os.path.join(self.params.model_dir, 'resource_logs.csv')
            df.to_csv(csv_path, index=False)
            print(f"Resource logs saved to {csv_path}")
            
            # 生成摘要統計
            summary = self._generate_training_summary(df, final_results)
            summary_path = os.path.join(self.params.model_dir, 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"Training summary saved to {summary_path}")
            
            # 打印摘要
            self._print_training_summary(summary)
    
    def _generate_training_summary(self, resource_df, final_results):
        """生成訓練摘要統計"""
        summary = {
            'training_info': {
                'total_epochs': self.params.epochs,
                'total_training_time_seconds': time.time() - self.training_start_time,
                'average_time_per_epoch_seconds': (time.time() - self.training_start_time) / self.params.epochs,
                'dataset': self.params.downstream_dataset,
                'optimizer': self.params.optimizer,
                'learning_rate': self.params.lr,
                'batch_size': self.params.batch_size if hasattr(self.params, 'batch_size') else 'N/A',
            }
        }
        
        # GPU 統計
        gpu_stats = {}
        for i in range(torch.cuda.device_count()):
            if f'gpu_{i}_memory_allocated_GB' in resource_df.columns:
                gpu_stats[f'gpu_{i}'] = {
                    'avg_memory_allocated_GB': float(resource_df[f'gpu_{i}_memory_allocated_GB'].mean()),
                    'max_memory_allocated_GB': float(resource_df[f'gpu_{i}_max_memory_allocated_GB'].max()),
                }
            if f'gpu_{i}_utilization_%' in resource_df.columns:
                gpu_stats[f'gpu_{i}']['avg_utilization_%'] = float(resource_df[f'gpu_{i}_utilization_%'].mean())
                gpu_stats[f'gpu_{i}']['max_utilization_%'] = float(resource_df[f'gpu_{i}_utilization_%'].max())
        
        summary['gpu_stats'] = gpu_stats
        
        # CPU 和 RAM 統計
        if 'cpu_percent' in resource_df.columns:
            summary['cpu_stats'] = {
                'avg_cpu_percent': float(resource_df['cpu_percent'].mean()),
                'max_cpu_percent': float(resource_df['cpu_percent'].max()),
            }
        
        if 'ram_used_GB' in resource_df.columns:
            summary['ram_stats'] = {
                'avg_ram_used_GB': float(resource_df['ram_used_GB'].mean()),
                'max_ram_used_GB': float(resource_df['ram_used_GB'].max()),
                'avg_ram_percent': float(resource_df['ram_percent'].mean()),
            }
        
        # 最終結果
        if final_results:
            summary['final_results'] = final_results
        
        return summary
    
    def _print_training_summary(self, summary):
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        print("\n[Training Info]")
        for key, value in summary['training_info'].items():
            if 'time' in key and isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        if 'gpu_stats' in summary:
            print("\n[GPU Statistics]")
            for gpu_id, stats in summary['gpu_stats'].items():
                print(f"  {gpu_id.upper()}:")
                for stat_name, stat_value in stats.items():
                    print(f"    {stat_name}: {stat_value:.2f}")
        
        if 'cpu_stats' in summary:
            print("\n[CPU Statistics]")
            for key, value in summary['cpu_stats'].items():
                print(f"  {key}: {value:.2f}")
        
        if 'ram_stats' in summary:
            print("\n[RAM Statistics]")
            for key, value in summary['ram_stats'].items():
                print(f"  {key}: {value:.2f}")
        
        if 'final_results' in summary:
            print("\n[Final Results]")
            for key, value in summary['final_results'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.5f}")
                else:
                    print(f"  {key}: {value}")
        
        print("="*70 + "\n")

    def train_for_multiclass(self):
        # 初始化訓練記錄
        self.training_start_time = time.time()
        self.log_gpu_usage(epoch=-1, phase='initial', step=0)

        f1_best = 0
        kappa_best = 0
        acc_best = 0
        bacc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()

            start_time = timer()
            epoch_start_time = time.time()
            # 記錄 epoch 開始時的資源狀態
            self.log_gpu_usage(epoch=epoch, phase='train_start', step=0)
            
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
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

            # 訓練階段結束時的資源狀態
            self.log_gpu_usage(epoch=epoch, phase='train_end')

            with torch.no_grad():

                # 驗證階段開始
                self.log_gpu_usage(epoch=epoch, phase='val_start')

                acc, bacc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)

                # 驗證階段結束
                self.log_gpu_usage(epoch=epoch, phase='val_end')
                
                epoch_time = time.time() - epoch_start_time
                
                # 儲存 epoch 訓練日誌
                epoch_log = {
                    'epoch': epoch + 1,
                    'train_loss': float(np.mean(losses)),
                    'val_acc': float(acc),
                    'val_bacc': float(bacc),
                    'val_kappa': float(kappa),
                    'val_f1': float(f1),
                    'learning_rate': float(optim_state['param_groups'][0]['lr']),
                    'epoch_time_seconds': float(epoch_time),
                    'epoch_time_minutes': float(epoch_time / 60),
                }
                self.training_logs.append(epoch_log)

                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, balanced acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        bacc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, balanced acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
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
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            self.log_gpu_usage(epoch=self.params.epochs, phase='test_start')
            acc, bacc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            self.log_gpu_usage(epoch=self.params.epochs, phase='test_end')
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, balanced acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    bacc,
                    kappa,
                    f1,
                )
            )
            print(cm)

            # 儲存結果
            final_results = {
                'best_epoch': best_f1_epoch,
                'test_acc': float(acc),
                'test_bacc': float(bacc),
                'test_kappa': float(kappa),
                'test_f1': float(f1),
                'best_val_acc': float(acc_best),
                'best_val_bacc': float(bacc_best),
                'best_val_kappa': float(kappa_best),
                'best_val_f1': float(f1_best),
            }
            
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_bacc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, bacc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

            # 儲存所有訓練記錄
            self.save_training_logs(final_results)

    def train_for_binaryclass(self):
        # 初始化訓練記錄
        self.training_start_time = time.time()
        self.log_gpu_usage(epoch=-1, phase='initial', step=0)

        acc_best = 0
        bacc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()

            start_time = timer()
            epoch_start_time = time.time()
            # 記錄 epoch 開始時的資源狀態
            self.log_gpu_usage(epoch=epoch, phase='train_start', step=0)

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

            # 訓練階段結束時的資源狀態
            self.log_gpu_usage(epoch=epoch, phase='train_end')

            with torch.no_grad():
                # 記錄驗證階段開始
                self.log_gpu_usage(epoch=epoch, phase='val_start')
                acc, bacc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                # 記錄驗證階段結束
                self.log_gpu_usage(epoch=epoch, phase='val_end')
                epoch_time = time.time() - epoch_start_time

                # 儲存 epoch 訓練日誌
                epoch_log = {
                    'epoch': epoch + 1,
                    'train_loss': float(np.mean(losses)),
                    'val_acc': float(acc),
                    'val_bacc': float(bacc),
                    'val_pr_auc': float(pr_auc),
                    'val_roc_auc': float(roc_auc),
                    'learning_rate': float(optim_state['param_groups'][0]['lr']),
                    'epoch_time_seconds': float(epoch_time),
                    'epoch_time_minutes': float(epoch_time / 60),
                }
                self.training_logs.append(epoch_log)

                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, balanced acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        bacc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, balanced acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        bacc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    bacc_best = bacc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            self.log_gpu_usage(epoch=self.params.epochs, phase='test_start')
            acc, bacc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            self.log_gpu_usage(epoch=self.params.epochs, phase='test_end')
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, balanced acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    bacc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)

            # 儲存最終結果
            final_results = {
                'best_epoch': best_f1_epoch,
                'test_acc': float(acc),
                'test_bacc': float(bacc),
                'test_pr_auc': float(pr_auc),
                'test_roc_auc': float(roc_auc),
                'best_val_acc': float(acc_best),
                'best_val_bacc': float(bacc_best),
                'best_val_pr_auc': float(pr_auc_best),
                'best_val_roc_auc': float(roc_auc_best),
            }
            
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_bacc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, bacc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

            # 儲存所有訓練記錄
            self.save_training_logs(final_results)

    def train_for_regression(self):
        # 初始化訓練記錄
        self.training_start_time = time.time()
        self.log_gpu_usage(epoch=-1, phase='initial', step=0)

        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()

            start_time = timer()
            epoch_start_time = time.time()
            # 記錄 epoch 開始時的資源狀態
            self.log_gpu_usage(epoch=epoch, phase='train_start', step=0)

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

            # 記錄訓練階段結束時的資源狀態
            self.log_gpu_usage(epoch=epoch, phase='train_end')

            with torch.no_grad():
                # 記錄驗證階段開始
                self.log_gpu_usage(epoch=epoch, phase='val_start')
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                # 記錄驗證階段結束
                self.log_gpu_usage(epoch=epoch, phase='val_end')
                
                epoch_time = time.time() - epoch_start_time
                
                # 儲存 epoch 訓練日誌
                epoch_log = {
                    'epoch': epoch + 1,
                    'train_loss': float(np.mean(losses)),
                    'val_corrcoef': float(corrcoef),
                    'val_r2': float(r2),
                    'val_rmse': float(rmse),
                    'learning_rate': float(optim_state['param_groups'][0]['lr']),
                    'epoch_time_seconds': float(epoch_time),
                    'epoch_time_minutes': float(epoch_time / 60),
                }
                self.training_logs.append(epoch_log)

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
                    print("r2 increasing....saving weights !! ")
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
            self.log_gpu_usage(epoch=self.params.epochs, phase='test_start')
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            self.log_gpu_usage(epoch=self.params.epochs, phase='test_end')
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            # 儲存最終結果
            final_results = {
                'best_epoch': best_r2_epoch,
                'test_corrcoef': float(corrcoef),
                'test_r2': float(r2),
                'test_rmse': float(rmse),
                'best_val_corrcoef': float(corrcoef_best),
                'best_val_r2': float(r2_best),
                'best_val_rmse': float(rmse_best),
            }

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            
            # 儲存所有訓練記錄
            self.save_training_logs(final_results)
