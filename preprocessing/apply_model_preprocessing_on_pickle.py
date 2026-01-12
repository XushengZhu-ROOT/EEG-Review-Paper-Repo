"""
对已有的pickle文件应用NeuroGPT预处理
从numpy数组加载数据，应用预处理，然后保存回pickle文件
"""

import os
import pickle
import numpy as np
import mne
from scipy.signal import detrend
from tqdm import tqdm


def remove_dc_offset(raw):
    """Remove DC offset per channel."""
    data, times = raw.get_data(return_times=True)
    data -= np.mean(data, axis=1, keepdims=True)
    raw._data = data
    return raw


def remove_linear_trend(raw):
    """Remove linear trend per channel."""
    try:
        data = detrend(raw.get_data(), axis=1, type='linear')
    except Exception as e:
        print(f"❌ detrend failed: {e}")
    raw._data = data
    return raw

def filter_full(raw, l_freq, h_freq, line_noise=50, target_sfreq=200):
    sf = raw.info['sfreq']
    raw.filter(l_freq, h_freq, n_jobs=-1)
    raw.notch_filter(line_noise, filter_length='auto', n_jobs=-1)
    if target_sfreq and target_sfreq != sf:
        raw.resample(target_sfreq, npad='auto', verbose=False)
    return raw

def preprocess_stress_neurogpt(raw, line_noise=50):
    """NeuroGPT preprocessing for Stress dataset"""
    raw.resample(250, npad="auto", verbose="error")
    raw.filter(l_freq=0.5, h_freq=100.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)
    raw = remove_dc_offset(raw)
    raw = remove_linear_trend(raw)
    raw.set_eeg_reference(ref_channels="average")
    return raw


def preprocess_stress_sttransformer(raw, line_noise=50):
    """ST-Transformer preprocessing for Stress dataset"""
    raw.resample(250, npad="auto", verbose="error")
    raw.filter(l_freq=4.0, h_freq=40.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)
    return raw

  
def preprocess_stress_cbramod(raw, line_noise=50):
    """CBraMod preprocessing for Stress dataset"""
    raw.resample(200)
    raw = filter_full(raw, 0.3, 75.0, line_noise=line_noise, target_sfreq=200)
    return raw

def preprocess_stress_labram(raw, line_noise=50):
    """LaBraM preprocessing for Stress dataset"""
    raw.resample(200)
    raw.filter(l_freq=0.1, h_freq=75.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)
    return raw

def preprocess_stress_neurolm(raw, line_noise=50):
    """NeuroLM preprocessing for Stress dataset"""
    raw.resample(200, npad="auto", verbose="error")
    raw.filter(l_freq=0.1, h_freq=75.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)
    return raw

def preprocess_stress_biot(raw, line_noise=50):
    """BIOT preprocessing for Stress dataset"""
    raw.resample(200, npad="auto", verbose="error")
    # No filtering for BIOT
    return raw

def preprocess_stress_eegpt(raw, line_noise=50):
    """EEGPT preprocessing for Stress dataset"""
    raw.resample(256, npad="auto", verbose="error")
    raw.filter(l_freq=0.1, h_freq=75.0, n_jobs=1)
    raw.notch_filter(line_noise, n_jobs=1)
    raw = remove_dc_offset(raw)
    raw.set_eeg_reference(ref_channels='average', projection=False)
    return raw


# 预处理函数字典
PREPROCESSORS = {
    'neurogpt': preprocess_stress_neurogpt,
    'sttransformer': preprocess_stress_sttransformer,
    'cbramod': preprocess_stress_cbramod,
    'labram': preprocess_stress_labram,
    'neurolm': preprocess_stress_neurolm,
    'biot': preprocess_stress_biot,
    'eegpt': preprocess_stress_eegpt,
}

def numpy_to_raw(data, ch_names, sfreq=200):
    """
    将numpy数组转换为MNE Raw对象
    
    Parameters:
    -----------
    data : numpy.ndarray
        形状为 (n_channels, n_times) 的EEG数据
    ch_names : list
        通道名称列表
    sfreq : float
        采样频率，默认200 Hz
    
    Returns:
    --------
    raw : mne.io.RawArray
        MNE Raw对象
    """
    # 确保数据形状正确
    if data.ndim != 2:
        raise ValueError(f"数据必须是2维数组 (n_channels, n_times)，当前形状: {data.shape}")
    
    # 确保通道数量匹配
    if data.shape[0] != len(ch_names):
        raise ValueError(f"通道数量不匹配: 数据有 {data.shape[0]} 个通道，但提供了 {len(ch_names)} 个通道名")
    
    # 创建MNE info对象
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # 创建Raw对象
    raw = mne.io.RawArray(data, info, verbose=False)
    
    return raw


def process_pickle_file(input_path, output_path, ch_names, original_sfreq=200, line_noise=50, preprocess_func=None):
    """
    处理单个pickle文件：加载、预处理、保存
    
    Parameters:
    -----------
    input_path : str
        输入pickle文件路径
    output_path : str
        输出pickle文件路径
    ch_names : list
        通道名称列表
    original_sfreq : float
        原始采样频率，默认200 Hz
    line_noise : float
        工频噪声频率，默认50 Hz
    preprocess_func : callable
        预处理函数，默认使用preprocess_stress_neurogpt
    """
    # 加载pickle文件
    with open(input_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # 检查数据格式
    if 'X' not in data_dict:
        raise ValueError(f"pickle文件缺少'X'键: {input_path}")
    if 'y' not in data_dict:
        raise ValueError(f"pickle文件缺少'y'键: {input_path}")
    
    # 获取数据
    X = data_dict['X']  # (n_channels, n_times)
    y = data_dict['y']
    
    # 转换为MNE Raw对象
    raw = numpy_to_raw(X, ch_names, sfreq=original_sfreq)
    
    # 应用预处理
    if preprocess_func is None:
        preprocess_func = preprocess_stress_neurogpt
    raw = preprocess_func(raw, line_noise=line_noise)
    
    # 转换回numpy数组
    X_processed = raw.get_data()  # (n_channels, n_times)
    
    # 创建新的数据字典
    processed_dict = {
        'X': X_processed,
        'y': y
    }
    
    # 保存处理后的数据
    with open(output_path, 'wb') as f:
        pickle.dump(processed_dict, f)


def process_directory(input_dir, output_dir, ch_names, original_sfreq=200, line_noise=50, 
                      overwrite=False, preprocess_func=None):
    """
    批量处理目录中的所有pickle文件
    
    Parameters:
    -----------
    input_dir : str
        输入目录路径
    output_dir : str
        输出目录路径
    ch_names : list
        通道名称列表
    original_sfreq : float
        原始采样频率，默认200 Hz
    line_noise : float
        工频噪声频率，默认50 Hz
    overwrite : bool
        是否覆盖已存在的文件，默认False
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有pickle文件
    pickle_files = [f for f in os.listdir(input_dir) if f.endswith('.pickle')]
    
    if len(pickle_files) == 0:
        print(f"警告: 在 {input_dir} 中没有找到pickle文件")
        return
    
    print(f"找到 {len(pickle_files)} 个pickle文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"原始采样率: {original_sfreq} Hz")
    print(f"预处理后采样率: 250 Hz (NeuroGPT)")
    print("-" * 80)
    
    # 处理每个文件
    success_count = 0
    error_count = 0
    
    for filename in tqdm(pickle_files, desc="处理进度"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path) and not overwrite:
            print(f"跳过已存在的文件: {filename}")
            continue
        
        try:
            process_pickle_file(input_path, output_path, ch_names, original_sfreq, line_noise, preprocess_func)
            success_count += 1
        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {e}")
            error_count += 1
    
    print("-" * 80)
    print(f"处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")


if __name__ == "__main__":
    # 配置参数
    # 30通道的通道名称（根据stress_preprocess.ipynb中的subset_channels）
    CHANNEL_NAMES = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 
                     'FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ', 
                     'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']

    # 原始采样频率（根据你的pickle文件）
    ORIGINAL_SFREQ = 200  # Hz
    
    # 工频噪声频率
    LINE_NOISE = 50  # Hz
    
    # ========== 选择预处理模型 ==========
    # 可选: 'neurogpt', 'sttransformer', 'cbramod', 'labram', 'neurolm', 'biot', 'eegpt'
    MODEL = 'neurogpt'
    
    # 获取预处理函数
    if MODEL not in PREPROCESSORS:
        raise ValueError(f"未知的模型: {MODEL}，可选: {list(PREPROCESSORS.keys())}")
    preprocess_func = PREPROCESSORS[MODEL]
    
    # 处理多个目录（train, val, test）
    SPLITS = ['train', 'val', 'test']
    BASE_INPUT_DIR = './augmented_data/Stress_noleak_30chan_no400up_swien42'
    BASE_OUTPUT_DIR = f'./augmented_data/{MODEL}_Stress_noleak_30chan'
    
    print(f"使用预处理模型: {MODEL}")
    print(f"输出目录: {BASE_OUTPUT_DIR}")
    print("-" * 80)
    
    # 处理每个split
    for split in SPLITS:
        input_dir = os.path.join(BASE_INPUT_DIR, split)
        output_dir = os.path.join(BASE_OUTPUT_DIR, split)
        
        if os.path.exists(input_dir):
            print(f"\n{'='*80}")
            print(f"处理 {split} 数据集")
            print(f"{'='*80}")
            process_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                ch_names=CHANNEL_NAMES,
                original_sfreq=ORIGINAL_SFREQ,
                line_noise=LINE_NOISE,
                overwrite=False,  # 设置为True以覆盖已存在的文件
                preprocess_func=preprocess_func
            )
        else:
            print(f"警告: 输入目录不存在: {input_dir}")
    
    print("\n所有处理完成!")
