import os
import numpy as np
import tempfile
import soundfile as sf
import torch
import librosa

def load_audio(file_path, target_sr=16000):
    """
    加载音频文件并转换为指定采样率
    
    参数:
        file_path: 音频文件路径
        target_sr: 目标采样率，默认16000Hz
        
    返回:
        (numpy array, int): 音频数据和采样率
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        print(f"加载音频文件失败: {e}")
        return None, None

def save_audio(audio_data, sample_rate, file_path):
    """
    保存音频数据到文件
    
    参数:
        audio_data: 音频数据 (numpy array)
        sample_rate: 采样率
        file_path: 保存路径
    
    返回:
        bool: 是否保存成功
    """
    try:
        sf.write(file_path, audio_data, sample_rate)
        return True
    except Exception as e:
        print(f"保存音频文件失败: {e}")
        return False

def audio_to_tensor(audio_data, sample_rate=16000):
    """
    将音频数据转换为张量
    
    参数:
        audio_data: 音频数据 (numpy array)
        sample_rate: 采样率
        
    返回:
        torch.Tensor: 音频张量
    """
    # 确保音频是单声道
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # 转换为torch张量
    audio_tensor = torch.from_numpy(audio_data).float()
    
    return audio_tensor, sample_rate

def tensor_to_audio(audio_tensor, sample_rate=16000):
    """
    将音频张量转换为numpy数组
    
    参数:
        audio_tensor: 音频张量
        sample_rate: 采样率
        
    返回:
        (numpy array, int): 音频数据和采样率
    """
    if isinstance(audio_tensor, torch.Tensor):
        audio_data = audio_tensor.detach().cpu().numpy()
    else:
        audio_data = audio_tensor
    
    return audio_data, sample_rate

def get_temp_file(suffix=".wav"):
    """
    生成临时文件路径
    
    参数:
        suffix: 文件后缀
        
    返回:
        str: 临时文件路径
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = temp_file.name
    temp_file.close()
    return temp_path
