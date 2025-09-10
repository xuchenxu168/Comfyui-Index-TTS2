#!/usr/bin/env python3
"""
参考音频质量检查工具
Reference Audio Quality Checker
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import torchaudio
import numpy as np

def analyze_reference_audio(audio_path):
    """分析参考音频质量"""
    print(f"\n🎤 分析参考音频: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        return False
    
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        print(f"📊 基本信息:")
        print(f"  文件大小: {os.path.getsize(audio_path)} bytes")
        print(f"  音频维度: {waveform.shape}")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  时长: {waveform.shape[-1] / sample_rate:.2f} 秒")
        print(f"  声道数: {waveform.shape[0]}")
        
        # 转换为单声道进行分析
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        audio_data = waveform.squeeze().numpy()
        
        # 音频质量分析
        max_val = np.abs(audio_data).max()
        mean_val = np.abs(audio_data).mean()
        std_val = np.std(audio_data)
        rms = np.sqrt(np.mean(audio_data**2))
        
        print(f"\n🔊 音频质量分析:")
        print(f"  最大振幅: {max_val:.6f}")
        print(f"  平均振幅: {mean_val:.6f}")
        print(f"  RMS值: {rms:.6f}")
        print(f"  标准差: {std_val:.6f}")
        print(f"  动态范围: {20 * np.log10(max_val / (mean_val + 1e-8)):.2f} dB")
        
        # 频谱分析
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # 找到主要频率成分
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # 基频估计
        peak_idx = np.argmax(positive_magnitude[1:]) + 1  # 跳过DC分量
        fundamental_freq = positive_freqs[peak_idx]
        
        print(f"\n🎵 频谱分析:")
        print(f"  估计基频: {fundamental_freq:.1f} Hz")
        
        # 频带能量分析
        low_freq_energy = np.sum(positive_magnitude[(positive_freqs >= 80) & (positive_freqs <= 300)])
        mid_freq_energy = np.sum(positive_magnitude[(positive_freqs >= 300) & (positive_freqs <= 3000)])
        high_freq_energy = np.sum(positive_magnitude[(positive_freqs >= 3000) & (positive_freqs <= 8000)])
        
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        
        if total_energy > 0:
            print(f"  低频能量 (80-300Hz): {low_freq_energy/total_energy*100:.1f}%")
            print(f"  中频能量 (300-3000Hz): {mid_freq_energy/total_energy*100:.1f}%")
            print(f"  高频能量 (3000-8000Hz): {high_freq_energy/total_energy*100:.1f}%")
        
        # 质量评估
        issues = []
        recommendations = []
        
        # 时长检查
        duration = waveform.shape[-1] / sample_rate
        if duration < 2:
            issues.append("音频时长过短 (< 2秒)")
            recommendations.append("使用3-10秒的参考音频")
        elif duration > 15:
            issues.append("音频时长过长 (> 15秒)")
            recommendations.append("截取3-10秒的清晰片段")
        
        # 振幅检查
        if max_val < 0.1:
            issues.append("音频音量过小")
            recommendations.append("增加音频音量或使用更清晰的录音")
        elif max_val > 0.95:
            issues.append("音频可能存在削波失真")
            recommendations.append("降低音频音量，避免失真")
        
        # 动态范围检查
        if std_val < 0.01:
            issues.append("音频动态范围过小")
            recommendations.append("使用更有表现力的语音")
        
        # 采样率检查
        if sample_rate < 16000:
            issues.append(f"采样率过低 ({sample_rate}Hz)")
            recommendations.append("使用至少16kHz的采样率")
        
        # 频谱检查
        if mid_freq_energy / total_energy < 0.3:
            issues.append("中频能量不足，可能影响语音清晰度")
            recommendations.append("使用更清晰的语音录音")
        
        # 静音检查
        silence_threshold = 0.01
        silence_ratio = np.sum(np.abs(audio_data) < silence_threshold) / len(audio_data)
        if silence_ratio > 0.5:
            issues.append(f"静音比例过高 ({silence_ratio*100:.1f}%)")
            recommendations.append("移除过多的静音部分")
        
        print(f"\n📋 质量评估:")
        if len(issues) == 0:
            print("✅ 参考音频质量良好")
            return True
        else:
            print("⚠️  发现问题:")
            for issue in issues:
                print(f"    - {issue}")
            
            print("\n💡 改进建议:")
            for rec in recommendations:
                print(f"    - {rec}")
            
            return False
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return False

def suggest_good_reference_audio():
    """建议好的参考音频特征"""
    print(f"\n🎯 优质参考音频特征:")
    
    features = [
        "📏 时长: 3-10秒",
        "🔊 音量: 适中，无削波失真",
        "🎤 录音质量: 清晰，无背景噪音",
        "🗣️  语音特征: 发音清晰，语调自然",
        "📊 采样率: 16kHz以上 (推荐24kHz)",
        "🎵 频谱: 中频能量充足 (300-3000Hz)",
        "⏸️  静音: 最小化开头和结尾的静音",
        "🎭 情感: 与目标语音情感相匹配"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print(f"\n❌ 避免的音频特征:")
    avoid_features = [
        "🔇 背景噪音、回声、混响",
        "📻 压缩音频 (如MP3低码率)",
        "🎵 背景音乐",
        "⚡ 电流声、爆音",
        "🗣️  多人对话",
        "📱 电话录音质量",
        "🔄 重复的词语或短语"
    ]
    
    for feature in avoid_features:
        print(f"  {feature}")

def main():
    print("🎤 参考音频质量检查工具")
    print("=" * 50)
    
    # 这里可以添加您使用的参考音频路径进行检查
    # 由于我们不知道具体的参考音频路径，提供通用建议
    
    print("📝 使用方法:")
    print("1. 将您的参考音频路径添加到下面的列表中")
    print("2. 运行脚本检查音频质量")
    print("3. 根据建议改进参考音频")
    
    # 示例：检查常见的参考音频位置
    possible_paths = [
        "examples/voice_01.wav",
        "test_audio.wav",
        "reference.wav"
    ]
    
    found_audio = False
    for path in possible_paths:
        if os.path.exists(path):
            analyze_reference_audio(path)
            found_audio = True
    
    if not found_audio:
        print("\n⚠️  未找到参考音频文件")
        print("请手动指定参考音频路径进行检查")
    
    # 提供通用建议
    suggest_good_reference_audio()
    
    print("\n" + "=" * 50)
    print("🎯 下一步操作:")
    print("1. 检查您使用的参考音频质量")
    print("2. 根据建议改进参考音频")
    print("3. 重新测试IndexTTS2生成")
    print("4. 如果问题持续，检查模型配置")

if __name__ == "__main__":
    main()
