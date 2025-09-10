#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声音一致性增强器
Voice Consistency Enhancer

用于改善多人对话中每个说话人与参考音频的一致性
Improves consistency between each speaker and their reference audio in multi-speaker conversations
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import torch.nn.functional as F


class VoiceConsistencyEnhancer:
    """声音一致性增强器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def enhance_reference_audio(self, waveform: torch.Tensor, sample_rate: int, 
                               consistency_level: float = 1.0) -> Tuple[torch.Tensor, int]:
        """
        增强参考音频以提高声音克隆的一致性
        
        Args:
            waveform: 音频波形 [channels, samples]
            sample_rate: 采样率
            consistency_level: 一致性级别 (0.1-2.0)
        
        Returns:
            增强后的音频波形和采样率
        """
        try:
            if consistency_level <= 1.0:
                return waveform, sample_rate
            
            # 确保音频长度足够
            min_length = int(sample_rate * 2.0)  # 至少2秒
            if waveform.shape[-1] < min_length:
                waveform = self._extend_audio(waveform, min_length)
            
            # 应用增强处理
            enhanced_waveform = self._apply_enhancement(waveform, consistency_level)
            
            # 音频质量优化
            enhanced_waveform = self._optimize_audio_quality(enhanced_waveform, sample_rate)
            
            return enhanced_waveform, sample_rate
            
        except Exception as e:
            print(f"[VoiceEnhancer] 增强失败，返回原始音频: {e}")
            return waveform, sample_rate
    
    def _extend_audio(self, waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """扩展音频长度"""
        current_length = waveform.shape[-1]
        if current_length >= target_length:
            return waveform
        
        # 计算需要重复的次数
        repeat_times = (target_length // current_length) + 1
        
        # 重复音频
        extended = waveform.repeat(1, repeat_times)
        
        # 裁剪到目标长度
        return extended[:, :target_length]
    
    def _apply_enhancement(self, waveform: torch.Tensor, consistency_level: float) -> torch.Tensor:
        """应用音频增强"""
        enhancement_factor = min(consistency_level, 2.0)
        
        # 1. 频谱增强
        enhanced = self._spectral_enhancement(waveform, enhancement_factor)
        
        # 2. 动态范围优化
        enhanced = self._dynamic_range_optimization(enhanced, enhancement_factor)
        
        # 3. 噪声抑制
        enhanced = self._noise_suppression(enhanced, enhancement_factor)
        
        return enhanced
    
    def _spectral_enhancement(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """频谱增强 - 提高语音清晰度"""
        try:
            # 创建高频增强滤波器
            kernel_size = 5
            kernel = torch.tensor([[-0.05, -0.1, 0.7, -0.1, -0.05]], dtype=waveform.dtype)
            kernel = kernel.unsqueeze(0)
            
            enhanced_channels = []
            for ch in range(waveform.shape[0]):
                ch_data = waveform[ch:ch+1].unsqueeze(0)
                
                # 应用卷积滤波
                filtered = F.conv1d(ch_data, kernel, padding=kernel_size//2)
                
                # 混合原始和滤波后的信号
                mix_ratio = (factor - 1.0) * 0.2  # 限制增强强度
                enhanced = ch_data * (1 - mix_ratio) + filtered * mix_ratio
                enhanced_channels.append(enhanced.squeeze(0))
            
            return torch.cat(enhanced_channels, dim=0)
            
        except Exception:
            return waveform
    
    def _dynamic_range_optimization(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """动态范围优化"""
        try:
            # 计算RMS能量
            rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True))
            
            # 目标RMS级别
            target_rms = 0.1 * factor
            
            # 应用增益控制
            if rms.max() > 0:
                gain = target_rms / (rms + 1e-8)
                # 限制增益范围
                gain = torch.clamp(gain, 0.5, 2.0)
                waveform = waveform * gain
            
            return waveform
            
        except Exception:
            return waveform
    
    def _noise_suppression(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """简单的噪声抑制"""
        try:
            # 计算信号的能量阈值
            energy = torch.abs(waveform)
            threshold = torch.quantile(energy, 0.1)  # 10%分位数作为噪声阈值
            
            # 应用软阈值
            suppression_factor = 0.1 * (factor - 1.0)
            mask = torch.sigmoid((energy - threshold) * 10)  # 软掩码
            
            # 应用噪声抑制
            suppressed = waveform * (1 - suppression_factor * (1 - mask))
            
            return suppressed
            
        except Exception:
            return waveform
    
    def _optimize_audio_quality(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """音频质量优化"""
        try:
            # 1. 削峰处理
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0.95:
                waveform = waveform * (0.95 / max_val)
            
            # 2. DC偏移移除
            waveform = waveform - torch.mean(waveform, dim=-1, keepdim=True)
            
            # 3. 轻微的平滑处理
            if waveform.shape[-1] > 3:
                smoothing_kernel = torch.tensor([[0.25, 0.5, 0.25]], dtype=waveform.dtype)
                smoothing_kernel = smoothing_kernel.unsqueeze(0)
                
                smoothed_channels = []
                for ch in range(waveform.shape[0]):
                    ch_data = waveform[ch:ch+1].unsqueeze(0)
                    smoothed = F.conv1d(ch_data, smoothing_kernel, padding=1)
                    # 轻微混合
                    mixed = ch_data * 0.8 + smoothed * 0.2
                    smoothed_channels.append(mixed.squeeze(0))
                
                waveform = torch.cat(smoothed_channels, dim=0)
            
            return waveform
            
        except Exception:
            return waveform
    
    def analyze_voice_consistency(self, reference_audio: torch.Tensor, 
                                 generated_audio: torch.Tensor, 
                                 sample_rate: int) -> dict:
        """
        分析生成音频与参考音频的一致性
        
        Returns:
            包含一致性分析结果的字典
        """
        try:
            # 确保音频长度一致
            min_length = min(reference_audio.shape[-1], generated_audio.shape[-1])
            ref_audio = reference_audio[:, :min_length]
            gen_audio = generated_audio[:, :min_length]
            
            # 计算基本统计特征
            ref_rms = torch.sqrt(torch.mean(ref_audio ** 2))
            gen_rms = torch.sqrt(torch.mean(gen_audio ** 2))
            
            ref_peak = torch.max(torch.abs(ref_audio))
            gen_peak = torch.max(torch.abs(gen_audio))
            
            # 计算相似度指标
            correlation = torch.corrcoef(torch.stack([
                ref_audio.flatten(), gen_audio.flatten()
            ]))[0, 1].item()
            
            # 能量比较
            energy_ratio = (gen_rms / (ref_rms + 1e-8)).item()
            peak_ratio = (gen_peak / (ref_peak + 1e-8)).item()
            
            return {
                "correlation": correlation,
                "energy_ratio": energy_ratio,
                "peak_ratio": peak_ratio,
                "reference_rms": ref_rms.item(),
                "generated_rms": gen_rms.item(),
                "consistency_score": (correlation + 1) * 0.5  # 0-1范围
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "consistency_score": 0.0
            }


def enhance_speaker_audio(audio_path: str, output_path: str, 
                         consistency_level: float = 1.5) -> bool:
    """
    增强说话人音频文件
    
    Args:
        audio_path: 输入音频路径
        output_path: 输出音频路径
        consistency_level: 一致性级别
    
    Returns:
        是否成功
    """
    try:
        enhancer = VoiceConsistencyEnhancer()
        
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 增强音频
        enhanced_waveform, enhanced_sr = enhancer.enhance_reference_audio(
            waveform, sample_rate, consistency_level
        )
        
        # 保存增强后的音频
        torchaudio.save(output_path, enhanced_waveform, enhanced_sr)
        
        print(f"音频增强完成: {audio_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"音频增强失败: {e}")
        return False


if __name__ == "__main__":
    # 测试代码
    enhancer = VoiceConsistencyEnhancer()
    print("声音一致性增强器初始化完成")
