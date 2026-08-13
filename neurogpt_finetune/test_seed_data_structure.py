#!/usr/bin/env python3
"""
测试脚本：检查seed_data的文件结构和内容
用于了解epoch_id的格式和位置
"""

import os
import pickle
import glob
import re

def extract_video_index(epoch_id):
    """从 epoch_id 中提取 video_index
    例如: 'subject_1_video_index_3_chunk001' -> 3
    """
    match = re.search(r'video_index_(\d+)_chunk', epoch_id)
    if match:
        return int(match.group(1))
    return None

def extract_subject_id(epoch_id):
    """从 epoch_id 中提取 subject_id
    例如: 'subject_1_video_index_3_chunk001' -> 1
    """
    match = re.search(r'subject_(\d+)_', epoch_id)
    if match:
        return int(match.group(1))
    return None

def test_seed_data_structure():
    """测试seed_data的文件结构"""
    
    # 可能的路径
    possible_paths = [
        "../seed_data",
        "./seed_data",
        "/dsmlp/home-fs03/14/114/xuz056/LEM/motor/test_pipeline/NeuroGPT/neurogpt_finetune/seed_data",
    ]
    
    seed_data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            seed_data_path = path
            break
    
    if seed_data_path is None:
        print("⚠️  未找到seed_data目录，请提供正确的路径")
        return
    
    print(f"✓ 找到seed_data目录: {seed_data_path}")
    
    # 检查目录结构
    test_path = os.path.join(seed_data_path, "test")
    val_path = os.path.join(seed_data_path, "val")
    
    for split_name, split_path in [("test", test_path), ("val", val_path)]:
        if not os.path.exists(split_path):
            print(f"⚠️  {split_name}目录不存在: {split_path}")
            continue
        
        print(f"\n=== 检查 {split_name} 目录 ===")
        
        # 查找所有pickle文件
        pickle_files = glob.glob(os.path.join(split_path, "**", "*.pickle"), recursive=True)
        pickle_files += glob.glob(os.path.join(split_path, "**", "*.pkl"), recursive=True)
        
        if len(pickle_files) == 0:
            print(f"⚠️  未找到pickle文件")
            continue
        
        print(f"✓ 找到 {len(pickle_files)} 个pickle文件")
        
        # 检查前5个文件
        print(f"\n--- 检查前5个文件 ---")
        for i, file_path in enumerate(pickle_files[:5]):
            print(f"\n[{i+1}] 文件路径: {file_path}")
            
            # 提取文件名
            filename = os.path.basename(file_path)
            print(f"    文件名: {filename}")
            
            # 提取相对路径
            rel_path = os.path.relpath(file_path, split_path)
            print(f"    相对路径: {rel_path}")
            
            # 尝试从文件名/路径提取信息
            subject_id = extract_subject_id(filename)
            video_index = extract_video_index(filename)
            
            if subject_id is not None:
                print(f"    提取的subject_id: {subject_id}")
            if video_index is not None:
                print(f"    提取的video_index: {video_index}")
            
            # 检查pickle内容
            try:
                with open(file_path, "rb") as f:
                    sample = pickle.load(f)
                
                print(f"    Pickle keys: {list(sample.keys())}")
                
                # 检查是否有epoch_id
                if "epoch_id" in sample:
                    print(f"    epoch_id: {sample['epoch_id']}")
                    subject_id_from_epoch = extract_subject_id(sample['epoch_id'])
                    video_index_from_epoch = extract_video_index(sample['epoch_id'])
                    if subject_id_from_epoch:
                        print(f"    从epoch_id提取的subject_id: {subject_id_from_epoch}")
                    if video_index_from_epoch:
                        print(f"    从epoch_id提取的video_index: {video_index_from_epoch}")
                
                # 检查数据结构
                if "signal" in sample:
                    print(f"    signal shape: {sample['signal'].shape}")
                if "label" in sample:
                    print(f"    label: {sample['label']}")
                
            except Exception as e:
                print(f"    ❌ 读取pickle文件失败: {e}")
        
        # 分析文件命名模式
        print(f"\n--- 分析文件命名模式 ---")
        filename_patterns = {}
        for file_path in pickle_files:
            filename = os.path.basename(file_path)
            rel_path = os.path.relpath(file_path, split_path)
            
            # 检查是否包含subject信息
            if "subject" in filename.lower() or "subject" in rel_path.lower():
                filename_patterns.setdefault("包含subject", []).append(rel_path)
            
            # 检查是否包含video信息
            if "video" in filename.lower():
                filename_patterns.setdefault("包含video", []).append(rel_path)
            
            # 检查是否包含chunk信息
            if "chunk" in filename.lower():
                filename_patterns.setdefault("包含chunk", []).append(rel_path)
        
        for pattern, files in filename_patterns.items():
            print(f"    {pattern}: {len(files)} 个文件")
            if len(files) <= 3:
                for f in files:
                    print(f"      - {f}")
            else:
                for f in files[:3]:
                    print(f"      - {f}")
                print(f"      ... 还有 {len(files)-3} 个文件")

if __name__ == "__main__":
    test_seed_data_structure()

