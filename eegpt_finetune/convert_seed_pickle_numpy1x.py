#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
转换SEED pickle文件：从NumPy 2.0格式转换为NumPy 1.x兼容格式

使用方法：
1. 在NumPy 2.0环境中运行此脚本（用于读取原始文件）
2. 脚本会重新保存为NumPy 1.x兼容格式

运行示例：
    python convert_seed_pickle_numpy1x.py \
        --input_dir ./seed_data/train \
        --output_dir ./seed_data_numpy1x/train \
        --recursive
"""

import os
import argparse
import pickle
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm


def convert_pickle_file(input_path, output_path):
    """
    转换单个pickle文件：加载后重新保存为NumPy 1.x兼容格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    try:
        # 加载原始文件（可能用NumPy 2.0保存）
        with open(input_path, 'rb') as f:
            obj = pickle.load(f)
        
        # 提取数据并转换为NumPy 1.x兼容格式
        # 关键：使用numpy 1.x的数组格式重新创建
        converted_obj = {}
        
        if 'signal' in obj:
            # 转换为numpy 1.x兼容的数组
            signal = np.asarray(obj['signal'], dtype=np.float32)
            # 确保是contiguous array（NumPy 1.x兼容）
            if not signal.flags['C_CONTIGUOUS']:
                signal = np.ascontiguousarray(signal)
            converted_obj['signal'] = signal
        
        if 'label' in obj:
            converted_obj['label'] = int(obj['label'])
        
        if 'epoch_id' in obj:
            converted_obj['epoch_id'] = str(obj['epoch_id'])
        
        # 保存为NumPy 1.x兼容格式
        # 使用protocol=4确保兼容性
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(converted_obj, f, protocol=4)
        
        return True, None
    except Exception as e:
        return False, str(e)


def convert_directory(input_dir, output_dir, recursive=True):
    """
    转换整个目录的pickle文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        recursive: 是否递归处理子目录
    """
    # 查找所有pickle文件
    if recursive:
        pattern = os.path.join(input_dir, "**", "*.pickle")
        all_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(input_dir, "*.pickle")
        all_files = glob.glob(pattern)
    
    print(f"找到 {len(all_files)} 个pickle文件")
    
    if len(all_files) == 0:
        print(f"警告: 在 {input_dir} 中没有找到任何pickle文件")
        return
    
    # 转换每个文件
    success_count = 0
    error_count = 0
    errors = []
    
    for input_path in tqdm(all_files, desc="转换文件"):
        # 计算相对路径
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        success, error = convert_pickle_file(input_path, output_path)
        
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append((input_path, error))
            if error_count <= 5:  # 只显示前5个错误
                print(f"\n错误: {input_path}")
                print(f"  原因: {error}")
    
    # 打印统计信息
    print(f"\n转换完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    
    if error_count > 5:
        print(f"  (还有 {error_count - 5} 个错误未显示)")
    
    if errors:
        print(f"\n前5个错误详情:")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="转换SEED pickle文件为NumPy 1.x兼容格式"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='输入目录路径（包含pickle文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='递归处理子目录（默认：True）'
    )
    parser.add_argument(
        '--no-recursive',
        dest='recursive',
        action='store_false',
        help='不递归处理子目录'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"递归模式: {args.recursive}")
    print()
    
    convert_directory(args.input_dir, args.output_dir, args.recursive)


if __name__ == "__main__":
    main()
