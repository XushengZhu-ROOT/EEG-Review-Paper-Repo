"""
从保存的模型计算混淆矩阵的脚本
使用方法：
python compute_confusion_matrix.py --model_path path/to/checkpoint-best.pth --dataset_path path/to/dataset --output_dir path/to/output
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_for_finetuning import evaluate
import utils
import modeling_finetune
from timm.models import create_model


def get_args():
    parser = argparse.ArgumentParser('从保存的模型计算混淆矩阵', add_help=False)
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径 (例如: checkpoint-best.pth)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，用于保存混淆矩阵')
    parser.add_argument('--model', type=str, default='labram_base_patch200_200_cbramod3lyclassifier',
                        help='模型名称')
    parser.add_argument('--input_size', type=int, default=200,
                        help='输入大小')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda 或 cpu)')
    parser.add_argument('--channel_size', type=int, default=20,
                        help='通道数量')
    return parser.parse_args()


def load_model_from_checkpoint(model_path, model_name='labram_base_patch16_224', 
                               input_size=200, nb_classes=6, device='cuda'):
    """从checkpoint加载模型"""
    # 创建模型
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=nb_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
    )
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 尝试不同的key来加载模型权重
    model_key = 'model'
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    elif 'model_without_ddp' in checkpoint:
        checkpoint_model = checkpoint['model_without_ddp']
    else:
        checkpoint_model = checkpoint
    
    # 移除不匹配的head权重（如果类别数不同）
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"移除不匹配的权重: {k}")
            del checkpoint_model[k]
    
    # 加载权重
    utils.load_state_dict(model, checkpoint_model, prefix='')
    
    model.to(device)
    model.eval()
    
    return model


def compute_confusion_matrix_from_model(model_path, dataset_path, output_dir=None,
                                       model_name='labram_base_patch16_224',
                                       input_size=200, batch_size=512, 
                                       channel_size=20, device='cuda'):
    """从保存的模型计算混淆矩阵"""
    
    # 准备数据集
    print("准备数据集...")
    train_dataset, test_dataset, val_dataset = utils.prepare_Motor_dataset(dataset_path)
    
    ch_names = ['F7','FP1','FP2','F8','F3','FZ','F4','C3','CZ','P8','P7','PZ','P4','T3','P3','O1','O2','C4','T4','A2']
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
    nb_classes = 6
    
    # 创建数据加载器
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    # 加载模型
    print(f"从 {model_path} 加载模型...")
    model = load_model_from_checkpoint(model_path, model_name, input_size, nb_classes, device)
    
    # 收集预测和真实标签
    print("计算预测结果...")
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader_test:
            EEG = batch[0]
            target = batch[-1]
            EEG = EEG.float().to(device, non_blocking=True) / 100
            EEG = EEG.unsqueeze(2)  # motor
            target = target.to(device, non_blocking=True)
            
            input_chans = utils.get_input_chans(ch_names)
            
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(EEG, input_chans=input_chans)
            else:
                output = model(EEG, input_chans=input_chans)
            
            # 获取预测类别
            preds = torch.argmax(output, dim=1).cpu().numpy()
            labels = target.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 计算混淆矩阵
    print("计算混淆矩阵...")
    cm = confusion_matrix(all_labels, all_preds)
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存混淆矩阵为numpy文件
        cm_path = os.path.join(output_dir, 'confusion_matrix.npy')
        np.save(cm_path, cm)
        print(f"混淆矩阵已保存到: {cm_path}")
        
        # 保存为文本文件（便于查看）
        cm_txt_path = os.path.join(output_dir, 'confusion_matrix.txt')
        np.savetxt(cm_txt_path, cm, fmt='%d', delimiter='\t')
        print(f"混淆矩阵（文本）已保存到: {cm_txt_path}")
        
        # 保存预测结果和真实标签
        results = {
            'y_true': all_labels,
            'y_pred': all_preds,
            'confusion_matrix': cm.tolist()
        }
        results_path = os.path.join(output_dir, 'predictions.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"预测结果已保存到: {results_path}")
    
    return cm, all_labels, all_preds


if __name__ == '__main__':
    args = get_args()
    
    cm, y_true, y_pred = compute_confusion_matrix_from_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name=args.model,
        input_size=args.input_size,
        batch_size=args.batch_size,
        channel_size=args.channel_size,
        device=args.device
    )
    
    print("\n混淆矩阵:")
    print(cm)
    print(f"\n测试集大小: {len(y_true)}")
    print(f"类别数量: {cm.shape[0]}")

