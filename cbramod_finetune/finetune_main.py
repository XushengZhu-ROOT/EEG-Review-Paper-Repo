import argparse
import random
import yaml
import os

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')

from datasets import custom_stress_dataset, kaggleern_dataset, \
    motortask_dataset, seed_emotion_dataset, sleep_dataset
from finetune_trainer import Trainer
from models import model_for_custom_stress, model_for_kaggleern, \
    model_for_motortask, model_for_seed_emotion, model_for_sleep

def save_config(params):
    """Saves the configuration parameters to a YAML file."""
    try:
        os.makedirs(params.model_dir, exist_ok=True)
        config_path = os.path.join(params.model_dir, 'config.yaml')
        params_dict = vars(params)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(params_dict, f, default_flow_style=False, allow_unicode=True)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

def save_architecture(model, model_dir):
    """Saves the model architecture to a text file."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        arch_path = os.path.join(model_dir, 'model_architecture.txt')
        with open(arch_path, 'w', encoding='utf-8') as f:
            f.write(str(model))
        print(f"Model architecture saved to {arch_path}")
    except Exception as e:
        print(f"Error saving model architecture: {e}")

def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps, '
                             'Labram_style_classifier, Labram_style_classifier2]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, required=True,
                        help='[CustomStress, KaggleERN, MotorTask, SEED-Emotion, Sleep]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/datasets/BigDownstream/Faced/processed',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='/data/wjq/models_weights/Big/BigFaced', help='model_dir')
    parser.add_argument('--channel_size', type=int, default=30, help='channel size, for classifier')
    parser.add_argument('--window_size', type=int, default=5, help='window size, for classifier')
    parser.add_argument('--pos_weight', default=None, type=float)

    # ===== [R1] MotorTask 划分策略开关（默认保留旧的 random_epoch）=====
    # split_mode:
    #   - random_epoch: 使用磁盘上已有的 train/val/test 随机 epoch 划分（旧方法，有泄漏风险）
    #   - subject_independent: 按受试者严格划分，避免同一受试者跨集合泄漏
    parser.add_argument('--split_mode', type=str, default='random_epoch',
                        choices=['random_epoch', 'subject_independent'],
                        help='MotorTask data split strategy (default: random_epoch)')
    # single_fold_debug=True: 只跑第一折 dry run（train 18 / val 1 / test 1），不做完整 LOSO
    parser.add_argument('--single_fold_debug', type=str2bool, default=True,
                        help='If True, only run the first subject-independent fold and stop')
    # 可选：手动指定 val/test 受试者（例如 Sub20 / Sub21）；留空则用排序后末尾两名
    parser.add_argument('--val_subject', type=str, default=None,
                        help='Optional val subject id for subject_independent split, e.g. Sub20')
    parser.add_argument('--test_subject', type=str, default=None,
                        help='Optional test subject id for subject_independent split, e.g. Sub21')
    # ===== [R2] 无验证子集的经典 LOSO：train = 除 test_subject 外全部受试者，无 val =====
    # 需要同时传 --test_subject。默认 False，不影响 R1（18-1-1）行为。
    parser.add_argument('--no_val_subject', type=str2bool, default=False,
                        help='If True (with --test_subject set), use classic LOSO: '
                             'N-1 train / 1 test, no separate validation subject. '
                             'Reported metrics are fixed to the final epoch.')
    # ===== [R2] 事后可复现指标所需的存档元数据（仅影响保存，不影响训练）=====
    parser.add_argument('--model_name', type=str, default='cbramod',
                        help='Model name used in saved {task}_{model}_fold{i}.npz/json filenames')
    parser.add_argument('--task_name', type=str, default=None,
                        help='Task name used in saved filenames; defaults to lowercased downstream_dataset')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='LOSO fold index (0-based), used in saved filenames and the fold JSON')
    parser.add_argument('--fold_results_dir', type=str, default=None,
                        help='Where to save {task}_{model}_fold{i}.npz/json; defaults to model_dir')

    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=10, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=True,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen',
                    action='store_true',
                    help='Freeze the model (if this flag is present, set to True)')
    parser.add_argument('--use_pretrained_weights', type=bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')

    params = parser.parse_args()
    print(params)
    save_config(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset == 'CustomStress':
        load_dataset = custom_stress_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_custom_stress.Model(params)
        save_architecture(model, params.model_dir)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'KaggleERN':
        load_dataset = kaggleern_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_kaggleern.Model(params)
        save_architecture(model, params.model_dir)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'MotorTask':
        load_dataset = motortask_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_motortask.Model(params)
        save_architecture(model, params.model_dir)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-Emotion':
        load_dataset = seed_emotion_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seed_emotion.Model(params)
        save_architecture(model, params.model_dir)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'Sleep':
        load_dataset = sleep_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_sleep.Model(params)
        save_architecture(model, params.model_dir)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
