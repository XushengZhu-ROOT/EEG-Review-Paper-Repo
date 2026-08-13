import pickle
import numpy as np
import os

# 加载pickle文件
pickle_path = "subject_1_video_index_3_chunk010.pickle"

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

print("=" * 60)
print("Pickle文件数据结构分析")
print("=" * 60)

# 1. 数据类型
print(f"\n1. 数据类型: {type(data)}")

# 2. 如果是字典，查看键值
if isinstance(data, dict):
    print(f"\n2. 字典的键: {list(data.keys())}")
    print(f"   键的数量: {len(data.keys())}")
    
    # 遍历所有键值
    for key, value in data.items():
        print(f"\n   [{key}]")
        print(f"   类型: {type(value)}")
        
        if isinstance(value, np.ndarray):
            print(f"   Shape: {value.shape}")
            print(f"   Dtype: {value.dtype}")
            print(f"   Min: {np.min(value):.6f}")
            print(f"   Max: {np.max(value):.6f}")
            print(f"   Mean: {np.mean(value):.6f}")
            print(f"   Std: {np.std(value):.6f}")
            print(f"   Contains NaN: {np.isnan(value).any()}")
            print(f"   Contains Inf: {np.isinf(value).any()}")
            
            # 如果是信号数据，显示更多统计信息
            if len(value.shape) >= 2:
                print(f"   Channel-wise stats (first 5 channels):")
                for i in range(min(5, value.shape[0])):
                    print(f"      Channel {i}: mean={np.mean(value[i]):.6f}, std={np.std(value[i]):.6f}")
        
        elif isinstance(value, (int, float)):
            print(f"   值: {value}")
        
        elif isinstance(value, str):
            print(f"   值: {value}")
            print(f"   长度: {len(value)}")
        
        elif isinstance(value, (list, tuple)):
            print(f"   长度: {len(value)}")
            if len(value) > 0:
                print(f"   第一个元素类型: {type(value[0])}")
                if len(value) <= 5:
                    print(f"   所有值: {value}")
                else:
                    print(f"   前5个值: {value[:5]}")
        
        else:
            print(f"   详细信息: {value}")

# 3. 如果是numpy数组
elif isinstance(data, np.ndarray):
    print(f"\n2. Array信息:")
    print(f"   Shape: {data.shape}")
    print(f"   Dtype: {data.dtype}")
    print(f"   Min: {np.min(data):.6f}")
    print(f"   Max: {np.max(data):.6f}")
    print(f"   Mean: {np.mean(data):.6f}")
    print(f"   Std: {np.std(data):.6f}")

# 4. 文件大小
file_size = os.path.getsize(pickle_path)
print(f"\n3. 文件大小: {file_size / 1024:.2f} KB ({file_size / 1024 / 1024:.2f} MB)")

# 5. 根据文件名推测数据信息
print(f"\n4. 文件信息:")
print(f"   文件名: {pickle_path}")
if "subject" in pickle_path:
    parts = pickle_path.replace(".pickle", "").split("_")
    print(f"   推测信息: 受试者数据")
    for part in parts:
        if "subject" in part.lower():
            print(f"     受试者: {part}")
        elif "video" in part.lower() or "vid" in part.lower():
            print(f"     视频相关: {part}")
        elif "chunk" in part.lower():
            print(f"     数据块: {part}")

print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)
