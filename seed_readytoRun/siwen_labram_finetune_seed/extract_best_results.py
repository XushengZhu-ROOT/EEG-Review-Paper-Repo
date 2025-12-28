import json
import os
import re
from pathlib import Path

def extract_hyperparameters(exp_name):
    """从实验名称中提取超参数"""
    lr = None
    weight_decay = None
    batch_size = None
    
    # 匹配格式: lr0.0009_wd0.05_bs128
    lr_match = re.search(r'lr([\d.]+)', exp_name)
    wd_match = re.search(r'wd([\d.]+)', exp_name)
    bs_match = re.search(r'bs(\d+)', exp_name)
    
    if lr_match:
        lr = float(lr_match.group(1))
    if wd_match:
        weight_decay = float(wd_match.group(1))
    if bs_match:
        batch_size = int(bs_match.group(1))
    
    return lr, weight_decay, batch_size

def get_metrics_from_log(log_file):
    """从log文件中提取所有epoch的指标"""
    epoch_data = {}
    
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                epoch = entry['epoch']
                # 对于相同的epoch，保留最后一次出现的记录
                epoch_data[epoch] = entry
            except json.JSONDecodeError:
                continue
    
    return epoch_data

def find_best_val_acc(epoch_data):
    """找到val_accuracy最大的epoch及其对应的test_accuracy"""
    if not epoch_data:
        return None
    
    best_epoch = None
    best_val_acc = -1
    best_test_acc = None
    
    for epoch, entry in epoch_data.items():
        val_acc = entry.get('val_accuracy', 0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_test_acc = entry.get('test_accuracy', 0)
    
    return {
        'epoch': best_epoch,
        'val_accuracy': best_val_acc,
        'test_accuracy': best_test_acc
    }

def main():
    base_dir = Path('checkpoints/Motor-posWeight1.0-swien_config')
    
    if not base_dir.exists():
        print(f"错误: 目录不存在: {base_dir}")
        return
    
    # 获取所有实验目录，按名称排序
    exp_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    results = []
    
    print("=" * 100)
    print(f"{'实验名称':<55} {'Epoch':<8} {'Best Val Acc':<18} {'Test Acc':<18}")
    print("=" * 100)
    
    for exp_dir in exp_dirs:
        log_file = exp_dir / 'log.txt'
        exp_name = exp_dir.name
        
        epoch_data = get_metrics_from_log(log_file)
        
        if epoch_data is None:
            print(f"{exp_name:<55} {'N/A':<8} {'文件不存在':<18} {'N/A':<18}")
            continue
        
        # 检查是否有足够的epoch数据（至少10个epoch才认为是完整的实验）
        num_epochs = len(epoch_data)
        if num_epochs < 10:
            print(f"{exp_name:<55} {'N/A':<8} {'数据不足':<18} {'N/A':<18} (仅{num_epochs}个epoch)")
            continue
        
        best_result = find_best_val_acc(epoch_data)
        
        if best_result:
            lr, weight_decay, batch_size = extract_hyperparameters(exp_name)
            results.append({
                'experiment': exp_name,
                'lr': lr,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                **best_result
            })
            print(f"{exp_name:<55} {best_result['epoch']:<8} {best_result['val_accuracy']:<18.6f} {best_result['test_accuracy']:<18.6f}")
        else:
            print(f"{exp_name:<55} {'N/A':<8} {'无数据':<18} {'N/A':<18}")
    
    print("=" * 100)
    print(f"\n总共处理了 {len(results)} 个完整实验")
    
    # 按实验名称排序（hpo_exp在前，然后是expauthor）
    results_sorted = sorted(results, key=lambda x: (
        'hpo_exp' not in x['experiment'],  # hpo_exp在前
        x['experiment']
    ))
    
    # 保存结果到CSV文件，将准确率转换为百分比格式
    import csv
    output_file = 'best_val_acc_results.csv'
    
    # 准备写入的数据，将准确率转换为百分比格式
    csv_data = []
    for result in results_sorted:
        csv_row = {
            'experiment': result['experiment'],
            'lr': result['lr'],
            'weight_decay': result['weight_decay'],
            'batch_size': result['batch_size'],
            'epoch': result['epoch'],
            'val_accuracy': f"{result['val_accuracy'] * 100:.2f}%",
            'test_accuracy': f"{result['test_accuracy'] * 100:.2f}%"
        }
        csv_data.append(csv_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'lr', 'weight_decay', 'batch_size', 'epoch', 'val_accuracy', 'test_accuracy'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

