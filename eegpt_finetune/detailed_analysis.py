import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载pickle文件
pickle_path = "subject_1_video_index_3_chunk010.pickle"

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

print("=" * 70)
print("详细数据分析报告")
print("=" * 70)

signal = data["signal"]
label = data["label"]
epoch_id = data["epoch_id"]

print(f"\n【基本信息】")
print(f"  Epoch ID: {epoch_id}")
print(f"  标签 (Label): {label}")
print(f"  信号维度: {signal.shape}")
print(f"  数据类型: {signal.dtype}")

print(f"\n【信号统计信息】")
print(f"  通道数: {signal.shape[0]}")
print(f"  时间点数: {signal.shape[1]}")
print(f"  采样率推测: 如果数据长度为4秒，采样率约为 {signal.shape[1]/4:.0f} Hz")
print(f"  全局统计:")
print(f"    最小值: {np.min(signal):.4f}")
print(f"    最大值: {np.max(signal):.4f}")
print(f"    均值: {np.mean(signal):.6f}")
print(f"    标准差: {np.std(signal):.4f}")
print(f"    中位数: {np.median(signal):.4f}")

print(f"\n【数据质量检查】")
print(f"  NaN值数量: {np.isnan(signal).sum()}")
print(f"  Inf值数量: {np.isinf(signal).sum()}")
print(f"  零值数量: {(signal == 0).sum()} ({100*(signal == 0).sum()/signal.size:.2f}%)")

print(f"\n【通道统计信息】（前10个通道）")
for i in range(min(10, signal.shape[0])):
    ch = signal[i]
    print(f"  通道 {i:2d}: 均值={np.mean(ch):8.4f}, 标准差={np.std(ch):8.4f}, "
          f"范围=[{np.min(ch):7.2f}, {np.max(ch):7.2f}]")

print(f"\n【时间序列统计】")
# 分析时间维度上的变化
time_means = np.mean(signal, axis=0)
time_stds = np.std(signal, axis=0)
print(f"  时间均值范围: [{np.min(time_means):.4f}, {np.max(time_means):.4f}]")
print(f"  时间标准差范围: [{np.min(time_stds):.4f}, {np.max(time_stds):.4f}]")

print(f"\n【通道间相关性】（前5个通道）")
if signal.shape[0] >= 5:
    corr_matrix = np.corrcoef(signal[:5])
    print("  前5个通道的相关系数矩阵:")
    for i in range(5):
        row_str = "    "
        for j in range(5):
            row_str += f"{corr_matrix[i,j]:6.3f}  "
        print(row_str)

print(f"\n【频域分析】（前3个通道的功率谱峰值频率）")
try:
    from scipy import signal as scipy_signal
    has_scipy = True
except:
    has_scipy = False
    print("  (需要scipy进行频域分析)")

if has_scipy:
    # 假设采样率为256Hz（根据常见EEG采样率）
    fs = 256
    for ch_idx in range(min(3, signal.shape[0])):
        freqs, psd = scipy_signal.welch(signal[ch_idx], fs=fs, nperseg=min(256, signal.shape[1]))
        peak_freq_idx = np.argmax(psd[1:]) + 1  # 跳过DC分量
        peak_freq = freqs[peak_freq_idx]
        print(f"  通道 {ch_idx}: 峰值频率 ≈ {peak_freq:.2f} Hz")

print(f"\n【数据分布】")
# 计算分位数
percentiles = [1, 5, 25, 50, 75, 95, 99]
print("  分位数统计:")
for p in percentiles:
    val = np.percentile(signal, p)
    print(f"    {p:3d}%: {val:8.4f}")

# 检查是否有异常值（使用IQR方法）
q1, q3 = np.percentile(signal, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = np.sum((signal < lower_bound) | (signal > upper_bound))
print(f"\n  异常值检查 (IQR方法):")
print(f"    Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
print(f"    异常值数量: {outliers} ({100*outliers/signal.size:.2f}%)")

print(f"\n【文件信息】")
file_size = os.path.getsize(pickle_path)
print(f"  文件大小: {file_size / 1024:.2f} KB")
print(f"  数据压缩比: {file_size / (signal.nbytes):.2f}x")

print("\n" + "=" * 70)
