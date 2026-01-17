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
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB', 'CustomStress', 'KaggleERN']:
            if self.params.pos_weight:
                self.criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([self.params.pos_weight])).cuda()
            else:
                self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
                param.requires_grad = not params.frozen
            else:
                other_params.append(param)

        # 設定優化器
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

        # 設定學習率排程器
        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

        # 訓練記錄初始化
        self.training_start_time = None
        self.training_logs = []
        self.resource_logs = []

    def train_for_multiclass(self):
        """多類別分類訓練"""
        self.training_start_time = time.time()
        self.log_gpu_usage(epoch=-1, phase='initial', step=0)

        best_metrics = {
            'val_acc': 0,
            'val_bacc': 0,
            'val_kappa': 0,
            'val_f1': 0,
            'val_cm': None,
            'test_acc': 0,
            'test_bacc': 0,
            'test_kappa': 0,
            'test_f1': 0,
            'test_cm': None,
            'epoch': 0
        }
        
        for epoch in range(self.params.epochs):
            self.model.train()
            epoch_start_time = time.time()
            self.log_gpu_usage(epoch=epoch, phase='train_start', step=0)
            
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                # if self.params.downstream_dataset == 'ISRUC':
                #     print('y shape:', y.shape)
                #     print('y', y)
                #     print('pred shape:', pred.shape)
                #     loss = self.criterion(pred.transpose(1, 2), y)
                # else:
                #     loss = self.criterion(pred, y)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                
                self.optimizer.step()
                self.optimizer_scheduler.step()

            self.log_gpu_usage(epoch=epoch, phase='train_end')

            with torch.no_grad():
                self.log_gpu_usage(epoch=epoch, phase='val_start')
                val_acc, val_bacc, val_kappa, val_f1, val_cm = self.val_eval.get_metrics_for_multiclass(self.model)
                self.log_gpu_usage(epoch=epoch, phase='val_end')
                
                self.log_gpu_usage(epoch=epoch, phase='test_start')
                test_acc, test_bacc, test_kappa, test_f1, test_cm = self.test_eval.get_metrics_for_multiclass(self.model)
                self.log_gpu_usage(epoch=epoch, phase='test_end')

                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                # 儲存 epoch 訓練日誌
                epoch_log = {
                    'epoch': epoch + 1,
                    'train_loss': float(np.mean(losses)),
                    'val_acc': float(val_acc),
                    'val_bacc': float(val_bacc),
                    'val_kappa': float(val_kappa),
                    'val_f1': float(val_f1),
                    'test_acc': float(test_acc),
                    'test_bacc': float(test_bacc),
                    'test_kappa': float(test_kappa),
                    'test_f1': float(test_f1),
                    'learning_rate': float(current_lr),
                    'epoch_time_seconds': float(epoch_time),
                    'epoch_time_minutes': float(epoch_time / 60),
                }
                self.training_logs.append(epoch_log)


                print(f"Epoch {epoch + 1}: Training Loss: {np.mean(losses):.5f}")
                print(f"  Val  - acc: {val_acc:.5f}, bacc: {val_bacc:.5f}, kappa: {val_kappa:.5f}, f1: {val_f1:.5f}")
                print(f"  Test - acc: {test_acc:.5f}, bacc: {test_bacc:.5f}, kappa: {test_kappa:.5f}, f1: {test_f1:.5f}")
                print(f"  LR: {current_lr:.5f}, Time: {epoch_time/60:.2f} mins")
                print("Val CM:")
                print(val_cm)
                print("Test CM:")
                print(test_cm)
            
                # 更新最佳模型（根據驗證集的 kappa）
                if val_bacc > best_metrics['val_bacc']:
                    print(">>> Val Balanced Accuracy increasing... saving weights!")
                    best_metrics.update({
                        'val_acc': val_acc,
                        'val_bacc': val_bacc,
                        'val_kappa': val_kappa,
                        'val_f1': val_f1,
                        'val_cm': val_cm,
                        'test_acc': test_acc,
                        'test_bacc': test_bacc,
                        'test_kappa': test_kappa,
                        'test_f1': test_f1,
                        'test_cm': test_cm,
                        'epoch': epoch + 1
                    })
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best model from Epoch {best_metrics['epoch']}")
        print(f"Best Val  - acc: {best_metrics['val_acc']:.5f}, bacc: {best_metrics['val_bacc']:.5f}, \n", \
                f"kappa: {best_metrics['val_kappa']:.5f}, f1: {best_metrics['val_f1']:.5f}")
        print(f"Best Test - acc: {best_metrics['test_acc']:.5f}, bacc: {best_metrics['test_bacc']:.5f}, \n", \
                f"kappa: {best_metrics['test_kappa']:.5f}, f1: {best_metrics['test_f1']:.5f}")
        print("="*70 + "\n")

        # 準備最終結果
        final_results = {
            'best_epoch': best_metrics['epoch'],
            'val_acc': float(best_metrics['val_acc']),
            'val_bacc': float(best_metrics['val_bacc']),
            'val_kappa': float(best_metrics['val_kappa']),
            'val_f1': float(best_metrics['val_f1']),
            'val_cm': (best_metrics['val_cm'].tolist()) if best_metrics['val_cm'] is not None else None,
            'test_acc': float(best_metrics['test_acc']),
            'test_bacc': float(best_metrics['test_bacc']),
            'test_kappa': float(best_metrics['test_kappa']),
            'test_f1': float(best_metrics['test_f1']),
            'test_cm': (best_metrics['test_cm'].tolist()) if best_metrics['test_cm'] is not None else None,
        }
        
        # 儲存模型
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        
        model_path = os.path.join(
            self.params.model_dir, 
            f"best_model_epoch{best_metrics['epoch']}_testKappa{best_metrics['test_kappa']:.5f}_testF1{best_metrics['test_f1']:.5f}.pth"
        )
        if self.best_model_states is not None:
            torch.save(self.best_model_states, model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No best model found (metrics did not improve). Model not saved.")
        
        # 儲存訓練記錄
        self.save_training_logs(final_results)

    def train_for_binaryclass(self):
        """二分類訓練"""
        self.training_start_time = time.time()
        self.log_gpu_usage(epoch=-1, phase='initial', step=0)

        best_metrics = {
            'epoch': 0,
            'val_acc': 0,
            'val_bacc': 0,
            'val_pr_auc': 0,
            'val_roc_auc': 0,
            'val_cm': None,
            'test_acc': 0,
            'test_bacc': 0,
            'test_pr_auc': 0,
            'test_roc_auc': 0,
            'test_cm': None,
        }

        for epoch in range(self.params.epochs):
            self.model.train()
            epoch_start_time = time.time()
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

            self.log_gpu_usage(epoch=epoch, phase='train_end')

            with torch.no_grad():
                self.log_gpu_usage(epoch=epoch, phase='val_start')
                val_acc, val_bacc, val_pr_auc, val_roc_auc, val_cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                self.log_gpu_usage(epoch=epoch, phase='val_end')
                
                self.log_gpu_usage(epoch=epoch, phase='test_start')
                test_acc, test_bacc, test_pr_auc, test_roc_auc, test_cm  = self.test_eval.get_metrics_for_binaryclass(self.model)
                self.log_gpu_usage(epoch=epoch, phase='test_end')
                
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                epoch_log = {
                    'epoch': epoch + 1,
                    'train_loss': float(np.mean(losses)),
                    'val_acc': float(val_acc),
                    'val_bacc': float(val_bacc),
                    'val_pr_auc': float(val_pr_auc),
                    'val_roc_auc': float(val_roc_auc),
                    'val_cm': val_cm.tolist(),
                    'test_acc': float(test_acc),
                    'test_bacc': float(test_bacc),
                    'test_pr_auc': float(val_pr_auc),
                    'test_roc_auc': float(test_roc_auc),
                    'test_cm': test_cm.tolist(),
                    'learning_rate': float(current_lr),
                    'epoch_time_seconds': float(epoch_time),
                    'epoch_time_minutes': float(epoch_time / 60),
                }
                self.training_logs.append(epoch_log)

                print(f"Epoch {epoch + 1}: Training Loss: {np.mean(losses):.5f}")
                print(f"  Val  - acc: {val_acc:.5f}, bacc: {val_bacc:.5f}, pr_auc: {val_pr_auc:.5f}, roc_auc: {val_roc_auc:.5f}")
                print(f"  Test - acc: {test_acc:.5f}, bacc: {test_bacc:.5f}, pr_auc: {test_pr_auc:.5f}, roc_auc: {test_roc_auc:.5f}")
                print(f"  LR: {current_lr:.5f}, Time: {epoch_time/60:.2f} mins")

                if val_roc_auc > best_metrics['val_roc_auc']:
                    print(">>> Val ROC AUC increasing... saving weights!")
                    best_metrics.update({
                        'epoch': epoch + 1,
                        'val_acc': val_acc,
                        'val_bacc': val_bacc,
                        'val_pr_auc': val_pr_auc,
                        'val_roc_auc': val_roc_auc,
                        'val_cm': val_cm,
                        'test_acc': test_acc,
                        'test_bacc': test_bacc,
                        'test_pr_auc': test_pr_auc,
                        'test_roc_auc': test_roc_auc,
                        'test_cm': test_cm,
                    })
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                    
        # 最終報告
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best model from Epoch {best_metrics['epoch']}")
        print(f"Best Val  - acc: {best_metrics['val_acc']:.5f}, bacc: {best_metrics['val_bacc']:.5f}, \n", \
              f"pr_auc: {best_metrics['val_pr_auc']:.5f}, roc_auc: {best_metrics['val_roc_auc']:.5f}")
        print(f"Best Test  - acc: {best_metrics['test_acc']:.5f}, bacc: {best_metrics['test_bacc']:.5f}, \n", \
              f"pr_auc: {best_metrics['test_pr_auc']:.5f}, roc_auc: {best_metrics['test_roc_auc']:.5f}")

        print("="*70 + "\n")

        final_results = {
            'best_epoch': best_metrics['epoch'],
            'val_acc': float(best_metrics['val_acc']),
            'val_bacc': float(best_metrics['val_bacc']),
            'val_pr_auc': float(best_metrics['val_pr_auc']),
            'val_roc_auc': float(best_metrics['val_roc_auc']),
            'val_cm': (best_metrics['val_cm'].tolist()),
            'test_acc': float(best_metrics['test_acc']),
            'test_bacc': float(best_metrics['test_bacc']),
            'test_pr_auc': float(best_metrics['test_pr_auc']),
            'test_roc_auc': float(best_metrics['test_roc_auc']),
            'test_cm': (best_metrics['test_cm'].tolist()),
        }

        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        
        model_path = os.path.join(
            self.params.model_dir, 
            f"best_model_epoch{best_metrics['epoch']}_testAcc{test_acc:.5f}_testBacc{test_bacc:.5f}.pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        self.save_training_logs(final_results)

    def train_for_regression(self):
        """迴歸訓練"""
        self.training_start_time = time.time()
        self.log_gpu_usage(epoch=-1, phase='initial', step=0)

        best_metrics = {
            'val_r2': -float('inf'),
            'val_corrcoef': 0,
            'val_rmse': float('inf'),
            'test_r2': -float('inf'),
            'test_corrcoef': 0,
            'test_rmse': float('inf'),
            'epoch': 0
        }

        for epoch in range(self.params.epochs):
            self.model.train()
            epoch_start_time = time.time()
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

            self.log_gpu_usage(epoch=epoch, phase='train_end')

            with torch.no_grad():
                self.log_gpu_usage(epoch=epoch, phase='val_start')
                val_corrcoef, val_r2, val_rmse = self.val_eval.get_metrics_for_regression(self.model)
                self.log_gpu_usage(epoch=epoch, phase='val_end')
                
                # === 每個 epoch 都執行測試 ===
                self.log_gpu_usage(epoch=epoch, phase='test_start')
                test_corrcoef, test_r2, test_rmse = self.test_eval.get_metrics_for_regression(self.model)
                self.log_gpu_usage(epoch=epoch, phase='test_end')
                
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                
                epoch_log = {
                    'epoch': epoch + 1,
                    'train_loss': float(np.mean(losses)),
                    'val_corrcoef': float(val_corrcoef),
                    'val_r2': float(val_r2),
                    'val_rmse': float(val_rmse),
                    'test_corrcoef': float(test_corrcoef),
                    'test_r2': float(test_r2),
                    'test_rmse': float(test_rmse),
                    'learning_rate': float(current_lr),
                    'epoch_time_seconds': float(epoch_time),
                    'epoch_time_minutes': float(epoch_time / 60),
                }
                self.training_logs.append(epoch_log)

                print(f"Epoch {epoch + 1}: Training Loss: {np.mean(losses):.5f}")
                print(f"  Val  - corrcoef: {val_corrcoef:.5f}, r2: {val_r2:.5f}, rmse: {val_rmse:.5f}")
                print(f"  Test - corrcoef: {test_corrcoef:.5f}, r2: {test_r2:.5f}, rmse: {test_rmse:.5f}")
                print(f"  LR: {current_lr:.5f}, Time: {epoch_time/60:.2f} mins")

                if val_r2 > best_metrics['val_r2']:
                    print(">>> Val R2 increasing... saving weights!")
                    best_metrics.update({
                        'val_corrcoef': val_corrcoef,
                        'val_r2': val_r2,
                        'val_rmse': val_rmse,
                        'test_corrcoef': test_corrcoef,
                        'test_r2': test_r2,
                        'test_rmse': test_rmse,
                        'epoch': epoch + 1
                    })
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                 
        # 最終報告
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best model from Epoch {best_metrics['epoch']}")
        print(f"Best Val  - corrcoef: {best_metrics['val_corrcoef']:.5f}, r2: {best_metrics['val_r2']:.5f}, \n", \
              f"rmse: {best_metrics['val_rmse']:.5f}")
        print(f"Best Test - corrcoef: {best_metrics['test_corrcoef']:.5f}, r2: {best_metrics['test_r2']:.5f}, \n", \
              f"rmse: {best_metrics['test_rmse']:.5f}")
        print("="*70 + "\n")

        final_results = {
            'best_epoch': best_metrics['epoch'],
            'val_corrcoef': float(best_metrics['val_corrcoef']),
            'val_r2': float(best_metrics['val_r2']),
            'val_rmse': float(best_metrics['val_rmse']),
            'test_corrcoef': float(best_metrics['test_corrcoef']),
            'test_r2': float(best_metrics['test_r2']),
            'test_rmse': float(best_metrics['test_rmse']),
        }

        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        
        model_path = os.path.join(
            self.params.model_dir, 
            f"best_model_epoch{best_metrics['epoch']}_testR2{test_r2:.5f}_testRmse{test_rmse:.5f}.pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        self.save_training_logs(final_results)

    def log_gpu_usage(self, epoch, phase='train', step=None):
        """記錄GPU和系統資源使用情況"""
        if not torch.cuda.is_available():
            return None
            
        gpu_stats = {
            'epoch': epoch,
            'phase': phase,
            'step': step,
            'timestamp': time.time() - self.training_start_time,
        }
        
        # GPU 記憶體統計
        for i in range(torch.cuda.device_count()):
            gpu_stats[f'gpu_{i}_memory_allocated_GB'] = torch.cuda.memory_allocated(i) / 1024**3
            gpu_stats[f'gpu_{i}_memory_reserved_GB'] = torch.cuda.memory_reserved(i) / 1024**3
            gpu_stats[f'gpu_{i}_max_memory_allocated_GB'] = torch.cuda.max_memory_allocated(i) / 1024**3
        
        # GPU 使用率統計
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
        
        # CPU 統計
        gpu_stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        
        # RAM 統計
        memory_info = psutil.virtual_memory()
        gpu_stats['ram_used_GB'] = memory_info.used / 1024**3
        gpu_stats['ram_percent'] = memory_info.percent
        
        self.resource_logs.append(gpu_stats)
        return gpu_stats

    def save_training_logs(self, final_results=None):
        """儲存訓練日誌和資源使用記錄"""
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
