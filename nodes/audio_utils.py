#!/usr/bin/env python3
"""
IndexTTS2 音频工具函数
Audio utility functions for IndexTTS2
"""

import os
import torch
import torchaudio

def load_audio_for_comfyui(audio_path: str) -> dict:
    """
    加载音频文件，确保格式兼容ComfyUI
    Load audio file with ComfyUI compatible format
    
    Args:
        audio_path (str): 音频文件路径
        
    Returns:
        dict: ComfyUI AUDIO格式的字典
        
    Raises:
        RuntimeError: 如果加载失败
    """
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 记录原始维度
        original_shape = waveform.shape
        print(f"[IndexTTS2] 原始音频维度: {original_shape}")
        
        # 确保音频格式正确：[channels, samples]
        if waveform.dim() == 1:
            # [samples] -> [1, samples] (单声道)
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            # [batch, channels, samples] -> [channels, samples]
            waveform = waveform.squeeze(0)
        elif waveform.dim() > 3:
            # 处理更高维度的情况
            while waveform.dim() > 2:
                waveform = waveform.squeeze(0)

        # 确保是单声道或立体声 (最多2个声道)
        if waveform.shape[0] > 2:
            print(f"[IndexTTS2] 警告: 音频有{waveform.shape[0]}个声道，只保留前2个")
            waveform = waveform[:2]

        # 确保数据类型正确
        if waveform.dtype != torch.float32:
            waveform = waveform.float()

        # 最终维度检查
        final_shape = waveform.shape
        print(f"[IndexTTS2] 调整后音频维度: {final_shape}")
        print(f"[IndexTTS2] 采样率: {sample_rate}")

        # 验证维度正确性
        if waveform.dim() != 2:
            raise ValueError(f"Invalid audio dimensions: {waveform.shape}. Expected [channels, samples]")

        if waveform.shape[1] == 0:
            raise ValueError("Audio has no samples")

        # 特别处理ComfyUI Save Audio兼容性
        # ComfyUI Save Audio期望能够执行 waveform.movedim(0, 1)
        # 确保维度至少是 [1, samples] 以支持movedim操作
        if waveform.shape[0] == 1 and waveform.shape[1] > 0:
            # 对于单声道音频，确保格式正确
            print(f"[IndexTTS2] 单声道音频，维度: {waveform.shape}")
        elif waveform.shape[0] == 2:
            # 立体声音频
            print(f"[IndexTTS2] 立体声音频，维度: {waveform.shape}")

        # 应用ComfyUI兼容性修复
        waveform = fix_comfyui_audio_compatibility(waveform)

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "path": audio_path
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio '{audio_path}': {str(e)}")

def save_audio_with_info(waveform: torch.Tensor, sample_rate: int, output_path: str) -> str:
    """
    保存音频文件并返回信息
    Save audio file and return info
    
    Args:
        waveform (torch.Tensor): 音频波形 [channels, samples]
        sample_rate (int): 采样率
        output_path (str): 输出路径
        
    Returns:
        str: 保存信息
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存音频
        torchaudio.save(output_path, waveform, sample_rate)
        
        # 获取文件信息
        file_size = os.path.getsize(output_path)
        duration = waveform.shape[1] / sample_rate
        
        info = f"音频已保存: {output_path}\n"
        info += f"时长: {duration:.2f}秒\n"
        info += f"采样率: {sample_rate}Hz\n"
        info += f"声道数: {waveform.shape[0]}\n"
        info += f"文件大小: {file_size / 1024:.1f}KB"
        
        print(f"[IndexTTS2] {info}")
        
        return info
        
    except Exception as e:
        error_msg = f"Failed to save audio: {str(e)}"
        print(f"[IndexTTS2 Error] {error_msg}")
        raise RuntimeError(error_msg)

def fix_comfyui_audio_compatibility(waveform: torch.Tensor) -> torch.Tensor:
    """
    修复ComfyUI Save Audio节点的兼容性问题
    Fix compatibility issues with ComfyUI Save Audio node

    ComfyUI Save Audio执行: waveform.movedim(0, 1).reshape(1, -1)
    这要求waveform必须是至少2D张量

    Args:
        waveform (torch.Tensor): 输入音频张量

    Returns:
        torch.Tensor: 兼容ComfyUI的音频张量
    """
    original_shape = waveform.shape

    # 确保至少是2D张量
    if waveform.dim() == 1:
        # [samples] -> [1, samples]
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 3:
        # [batch, channels, samples] -> [channels, samples]
        waveform = waveform.squeeze(0)
    elif waveform.dim() > 3:
        # 压缩到2D
        while waveform.dim() > 2:
            waveform = waveform.squeeze(0)

    # 确保是[channels, samples]格式
    if waveform.dim() == 2:
        channels, samples = waveform.shape

        # 如果第二个维度比第一个维度小，可能是[samples, channels]格式，需要转置
        if samples < channels and samples <= 2:
            print(f"[IndexTTS2] 检测到可能的[samples, channels]格式，转置: {waveform.shape}")
            waveform = waveform.transpose(0, 1)
            channels, samples = waveform.shape

        # 限制声道数
        if channels > 2:
            print(f"[IndexTTS2] 限制声道数从{channels}到2")
            waveform = waveform[:2]

        # 确保有足够的样本
        if samples == 0:
            raise ValueError("Audio has no samples")

    # 测试ComfyUI Save Audio的操作
    try:
        test_moved = waveform.movedim(0, 1)
        test_reshaped = test_moved.reshape(1, -1)
        print(f"[IndexTTS2] ComfyUI兼容性测试通过: {original_shape} -> {waveform.shape} -> movedim: {test_moved.shape} -> reshape: {test_reshaped.shape}")
    except Exception as e:
        raise ValueError(f"Failed to make audio compatible with ComfyUI Save Audio: {e}. Shape: {waveform.shape}")

    return waveform

def validate_audio_tensor(waveform: torch.Tensor, name: str = "audio") -> torch.Tensor:
    """
    验证和修复音频张量格式
    Validate and fix audio tensor format

    Args:
        waveform (torch.Tensor): 音频张量
        name (str): 音频名称（用于日志）

    Returns:
        torch.Tensor: 修复后的音频张量
    """
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(waveform)}")
    
    original_shape = waveform.shape
    
    # 处理不同的维度情况
    if waveform.dim() == 1:
        # [samples] -> [1, samples]
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 3:
        # [batch, channels, samples] -> [channels, samples]
        waveform = waveform.squeeze(0)
    elif waveform.dim() > 3:
        # 处理更高维度
        while waveform.dim() > 2:
            waveform = waveform.squeeze(0)
    elif waveform.dim() == 0:
        raise ValueError(f"{name} cannot be a scalar")
    
    # 确保至少有2个维度
    if waveform.dim() != 2:
        raise ValueError(f"{name} must have 2 dimensions [channels, samples], got {waveform.shape}")
    
    # 限制声道数
    if waveform.shape[0] > 2:
        print(f"[IndexTTS2] 警告: {name}有{waveform.shape[0]}个声道，只保留前2个")
        waveform = waveform[:2]
    
    # 确保有样本
    if waveform.shape[1] == 0:
        raise ValueError(f"{name} has no samples")
    
    # 确保数据类型
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
    
    if original_shape != waveform.shape:
        print(f"[IndexTTS2] {name}维度调整: {original_shape} -> {waveform.shape}")
    
    return waveform

if __name__ == "__main__":
    # 测试音频工具函数
    print("音频工具函数测试")
    print("=" * 30)
    
    # 测试validate_audio_tensor
    print("1. 测试validate_audio_tensor...")
    
    # 测试不同维度的张量
    test_cases = [
        torch.randn(1000),           # 1D: [samples]
        torch.randn(1, 1000),        # 2D: [channels, samples]
        torch.randn(1, 1, 1000),     # 3D: [batch, channels, samples]
        torch.randn(2, 1000),        # 2D: [channels, samples] - stereo
    ]
    
    for i, tensor in enumerate(test_cases):
        try:
            result = validate_audio_tensor(tensor, f"test_{i}")
            print(f"  测试{i}: {tensor.shape} -> {result.shape} ✓")
        except Exception as e:
            print(f"  测试{i}: {tensor.shape} -> 错误: {e}")
    
    print("\n音频工具函数测试完成")
