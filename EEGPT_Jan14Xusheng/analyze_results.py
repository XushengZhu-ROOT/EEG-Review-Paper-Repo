#!/usr/bin/env python3
"""
分析训练结果脚本
功能：
1. 找到验证集balanced_accuracy最高的epoch
2. 输出该epoch的验证集、测试集指标
3. 输出测试集混淆矩阵
4. 支持分析单个实验或自动分析output目录下所有实验
5. 提供汇总表格，按验证集BAcc排序显示所有实验

使用方法：
  # 分析output目录下所有实验（推荐）
  python analyze_results.py
  
  # 分析指定实验
  python analyze_results.py output/实验目录名
  
  # 只显示汇总表格，不显示详细信息
  python analyze_results.py --no-details
  
  # 保存结果到JSON文件
  python analyze_results.py --output summary.json
"""

import json
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional


def load_epoch_results(result_dir: Path, epoch: int, split: str) -> Optional[Dict]:
    """加载指定epoch和split的结果文件"""
    file_path = result_dir / f"epoch_{epoch}_{split}.json"
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return json.load(f)


def find_best_epoch(result_dir: Path) -> Tuple[int, float, Dict]:
    """
    找到验证集balanced_accuracy最高的epoch
    
    Returns:
        (best_epoch, best_val_bacc, best_val_data)
    """
    best_epoch = -1
    best_val_bacc = -1.0
    best_val_data = None
    
    # 遍历所有epoch文件
    epoch = 0
    while True:
        val_file = result_dir / f"epoch_{epoch}_valid.json"
        if not val_file.exists():
            break
        
        val_data = load_epoch_results(result_dir, epoch, "valid")
        if val_data is None:
            break
        
        val_bacc = val_data["metrics"]["balanced_accuracy"]
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_epoch = epoch
            best_val_data = val_data
        
        epoch += 1
    
    return best_epoch, best_val_bacc, best_val_data


def print_confusion_matrix(cm: List[List[int]], class_names: Optional[List[str]] = None):
    """美化打印混淆矩阵"""
    cm = np.array(cm)
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # 计算每行的总数（用于显示百分比）
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    cm_percent = (cm / row_sums * 100).round(1)
    
    # 打印表头
    print("\n混淆矩阵 (Confusion Matrix):")
    print("=" * (12 * (n_classes + 1) + 5))
    true_pred_label = "True\\Pred"
    header = f"{true_pred_label:>12}"
    for name in class_names:
        header += f"{name:>12}"
    print(header)
    print("-" * (12 * (n_classes + 1) + 5))
    
    # 打印每一行
    for i in range(n_classes):
        row_str = f"{class_names[i]:>12}"
        for j in range(n_classes):
            count = cm[i, j]
            percent = cm_percent[i, j]
            row_str += f"{count:>8}({percent:>4.1f}%)"
        print(row_str)
    
    print("=" * (12 * (n_classes + 1) + 5))
    
    # 打印每类的准确率
    print("\n各类别准确率 (Per-class Accuracy):")
    for i in range(n_classes):
        if row_sums[i, 0] > 0:
            acc = cm[i, i] / row_sums[i, 0] * 100
            print(f"  {class_names[i]}: {acc:.2f}% ({cm[i, i]}/{int(row_sums[i, 0])})")


def analyze_experiment(result_dir: Path, print_details: bool = True):
    """
    分析单个实验的结果
    
    Args:
        result_dir: 实验结果目录路径
        print_details: 是否打印详细信息
    """
    result_dir = Path(result_dir)
    
    if not result_dir.exists():
        print(f"错误：目录不存在: {result_dir}")
        return None
    
    # 找到最佳epoch
    best_epoch, best_val_bacc, best_val_data = find_best_epoch(result_dir)
    
    if best_epoch == -1:
        print(f"错误：在 {result_dir} 中未找到任何验证集结果文件")
        return None
    
    # 加载对应的测试集和训练集结果
    test_data = load_epoch_results(result_dir, best_epoch, "test")
    train_data = load_epoch_results(result_dir, best_epoch, "train")
    
    # 构建结果字典
    results = {
        "experiment_dir": str(result_dir),
        "best_epoch": best_epoch,
        "best_val_bacc": best_val_bacc,
        "val_metrics": best_val_data["metrics"] if best_val_data else None,
        "train_metrics": train_data["metrics"] if train_data else None,
        "test_metrics": test_data["metrics"] if test_data else None,
        "test_confusion_matrix": test_data["confusion_matrix"] if test_data else None,
    }
    
    # 打印结果
    if print_details:
        print("\n" + "=" * 80)
        print(f"实验结果分析: {result_dir.name}")
        print("=" * 80)
        print(f"\n最佳验证集 Balanced Accuracy 出现在 Epoch {best_epoch}")
        print(f"最佳验证集 BAcc: {best_val_bacc:.4f}")
        
        if train_data:
            train_metrics = train_data["metrics"]
            print(f"\n训练集指标 (Train Metrics @ Epoch {best_epoch}):")
            print(f"  Accuracy:        {train_metrics['accuracy']:.4f}")
            print(f"  Balanced Acc:    {train_metrics['balanced_accuracy']:.4f}")
            print(f"  Cohen's Kappa:   {train_metrics['cohen_kappa']:.4f}")
            print(f"  F1 (Macro):      {train_metrics['f1_macro']:.4f}")
            print(f"  F1 (Weighted):   {train_metrics['f1_weighted']:.4f}")
            print(f"  F1 (Micro):      {train_metrics['f1_micro']:.4f}")
        else:
            print(f"\n注意：未找到 Epoch {best_epoch} 的训练集结果文件")
            print(f"      训练集结果需要修改训练代码保存后才能获取")
        
        if best_val_data:
            val_metrics = best_val_data["metrics"]
            print(f"\n验证集指标 (Validation Metrics @ Epoch {best_epoch}):")
            print(f"  Accuracy:        {val_metrics['accuracy']:.4f}")
            print(f"  Balanced Acc:    {val_metrics['balanced_accuracy']:.4f}")
            print(f"  Cohen's Kappa:   {val_metrics['cohen_kappa']:.4f}")
            print(f"  F1 (Macro):      {val_metrics['f1_macro']:.4f}")
            print(f"  F1 (Weighted):   {val_metrics['f1_weighted']:.4f}")
            print(f"  F1 (Micro):      {val_metrics['f1_micro']:.4f}")
        
        if test_data:
            test_metrics = test_data["metrics"]
            print(f"\n测试集指标 (Test Metrics @ Epoch {best_epoch}):")
            print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
            print(f"  Balanced Acc:    {test_metrics['balanced_accuracy']:.4f}")
            print(f"  Cohen's Kappa:   {test_metrics['cohen_kappa']:.4f}")
            print(f"  F1 (Macro):      {test_metrics['f1_macro']:.4f}")
            print(f"  F1 (Weighted):   {test_metrics['f1_weighted']:.4f}")
            print(f"  F1 (Micro):      {test_metrics['f1_micro']:.4f}")
            
            # 打印测试集混淆矩阵
            if test_data["confusion_matrix"]:
                print_confusion_matrix(test_data["confusion_matrix"])
        else:
            print(f"\n警告：未找到 Epoch {best_epoch} 的测试集结果文件")
        
        print("=" * 80 + "\n")
    
    return results


def find_all_experiments(output_dir: Path = Path("output")) -> List[Path]:
    """
    查找output目录下所有实验目录
    
    Args:
        output_dir: output目录路径
    
    Returns:
        实验目录列表
    """
    if not output_dir.exists():
        return []
    
    experiments = []
    for item in output_dir.iterdir():
        if item.is_dir():
            # 检查是否包含epoch结果文件（至少有一个valid文件）
            valid_files = list(item.glob("epoch_*_valid.json"))
            if valid_files:
                experiments.append(item)
    
    return sorted(experiments)


def print_summary_table(all_results: List[Dict]):
    """打印所有实验的汇总表格"""
    if not all_results:
        return
    
    print("\n" + "=" * 120)
    print("所有实验汇总 (Summary of All Experiments)")
    print("=" * 120)
    
    # 表头
    header = f"{'实验名称':<50} {'Epoch':<8} {'Val BAcc':<10} {'Test BAcc':<10} {'Train BAcc':<10}"
    print(header)
    print("-" * 120)
    
    # 按验证集BAcc排序
    sorted_results = sorted(all_results, key=lambda x: x.get("best_val_bacc", -1), reverse=True)
    
    for result in sorted_results:
        exp_name = Path(result["experiment_dir"]).name
        if len(exp_name) > 47:
            exp_name = exp_name[:44] + "..."
        
        epoch = result.get("best_epoch", -1)
        val_bacc = result.get("best_val_bacc", 0)
        test_bacc = result.get("test_metrics", {}).get("balanced_accuracy", 0) if result.get("test_metrics") else 0
        train_bacc = result.get("train_metrics", {}).get("balanced_accuracy", 0) if result.get("train_metrics") else 0
        
        print(f"{exp_name:<50} {epoch:<8} {val_bacc:<10.4f} {test_bacc:<10.4f} {train_bacc:<10.4f}")
    
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="分析训练实验结果")
    parser.add_argument(
        "result_dir",
        type=str,
        nargs="?",
        default=None,
        help="实验结果目录路径（可选，如果不提供则分析output目录下所有实验）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="将结果保存到JSON文件（可选，如果分析多个实验，会保存汇总结果）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="output目录路径（默认：output）"
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="不打印每个实验的详细信息，只显示汇总表格"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 如果提供了result_dir，只分析该实验
    if args.result_dir:
        results = analyze_experiment(args.result_dir, print_details=not args.no_details)
        all_results = [results] if results else []
    else:
        # 自动查找所有实验
        experiments = find_all_experiments(output_dir)
        
        if not experiments:
            print(f"在 {output_dir.absolute()} 中未找到任何实验目录")
            return
        
        print(f"找到 {len(experiments)} 个实验，开始分析...\n")
        
        all_results = []
        for i, exp_dir in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] 分析实验: {exp_dir.name}")
            result = analyze_experiment(exp_dir, print_details=not args.no_details)
            if result:
                all_results.append(result)
        
        # 打印汇总表格
        if all_results:
            print_summary_table(all_results)
    
    # 保存到文件（如果指定）
    if args.output and all_results:
        output_path = Path(args.output)
        if len(all_results) == 1:
            # 单个实验，保存详细结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results[0], f, indent=2, ensure_ascii=False)
        else:
            # 多个实验，保存汇总结果
            summary = {
                "total_experiments": len(all_results),
                "experiments": all_results
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
