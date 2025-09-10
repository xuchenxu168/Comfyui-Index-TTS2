#!/usr/bin/env python3
"""
IndexTTS2音频质量诊断和修复工具
Audio Quality Diagnosis and Fix Tool for IndexTTS2
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import torchaudio
import tempfile
from omegaconf import OmegaConf

def diagnose_audio_file(audio_path):
    """诊断音频文件质量"""
    print(f"\n🔍 诊断音频文件: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        return False
    
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        print(f"📊 音频信息:")
        print(f"  文件大小: {os.path.getsize(audio_path)} bytes")
        print(f"  音频维度: {waveform.shape}")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  时长: {waveform.shape[-1] / sample_rate:.2f} 秒")
        print(f"  声道数: {waveform.shape[0]}")
        
        # 检查音频内容
        max_val = waveform.abs().max().item()
        mean_val = waveform.abs().mean().item()
        std_val = waveform.std().item()
        
        print(f"  最大振幅: {max_val:.6f}")
        print(f"  平均振幅: {mean_val:.6f}")
        print(f"  标准差: {std_val:.6f}")
        
        # 质量评估
        issues = []
        
        if max_val < 0.001:
            issues.append("音频振幅过小，可能为静音")
        elif max_val > 0.99:
            issues.append("音频可能存在削波失真")
        
        if mean_val < 0.0001:
            issues.append("音频平均振幅过小")
        
        if std_val < 0.001:
            issues.append("音频动态范围过小")
        
        if sample_rate not in [16000, 22050, 24000, 44100, 48000]:
            issues.append(f"非标准采样率: {sample_rate}")
        
        if len(issues) == 0:
            print("✅ 音频质量正常")
            return True
        else:
            print("⚠️  发现问题:")
            for issue in issues:
                print(f"    - {issue}")
            return False
            
    except Exception as e:
        print(f"❌ 加载音频失败: {e}")
        return False

def test_indextts2_inference():
    """测试IndexTTS2推理质量"""
    print("\n🧪 测试IndexTTS2推理质量...")
    
    try:
        from indextts.infer_v2 import IndexTTS2
        from nodes.model_utils import get_indextts2_model_path, validate_model_path
        
        # 获取模型路径
        model_dir = get_indextts2_model_path()
        config_path = os.path.join(model_dir, "config.yaml")
        
        if not validate_model_path(model_dir):
            print("❌ 模型文件不完整")
            return False
        
        print(f"📁 模型目录: {model_dir}")
        
        # 检查配置
        cfg = OmegaConf.load(config_path)
        print(f"📋 配置信息:")
        print(f"  采样率: {cfg.dataset.sample_rate}")
        print(f"  Mel频谱: {cfg.dataset.mel.n_mels} mels")
        print(f"  FFT大小: {cfg.dataset.mel.n_fft}")
        print(f"  跳跃长度: {cfg.dataset.mel.hop_length}")
        
        # 初始化模型
        print("\n🚀 初始化IndexTTS2模型...")
        model = IndexTTS2(
            cfg_path=config_path,
            model_dir=model_dir,
            is_fp16=False,  # 使用fp32确保质量
            device=None,
            use_cuda_kernel=False
        )
        
        print("✅ 模型初始化成功")
        
        # 测试推理参数
        test_params = [
            {"name": "默认参数", "params": {}},
            {"name": "高质量参数", "params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_text_tokens_per_sentence": 120
            }},
            {"name": "保守参数", "params": {
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 30,
                "max_text_tokens_per_sentence": 100
            }}
        ]
        
        print(f"\n📝 推荐的推理参数:")
        for param_set in test_params:
            print(f"  {param_set['name']}:")
            if param_set['params']:
                for key, value in param_set['params'].items():
                    print(f"    {key}: {value}")
            else:
                print(f"    使用模型默认参数")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def suggest_quality_improvements():
    """建议质量改进方案"""
    print("\n💡 音频质量改进建议:")
    
    suggestions = [
        "1. 推理参数优化:",
        "   - temperature: 0.7 (降低随机性)",
        "   - top_p: 0.9 (nucleus sampling)",
        "   - top_k: 50 (限制候选数量)",
        "",
        "2. 音频处理优化:",
        "   - 使用fp32精度而非fp16",
        "   - 禁用CUDA kernel避免兼容性问题",
        "   - 确保采样率一致性(24kHz)",
        "",
        "3. 参考音频质量:",
        "   - 使用高质量、清晰的参考音频",
        "   - 参考音频长度建议3-10秒",
        "   - 避免背景噪音和回声",
        "",
        "4. 文本处理:",
        "   - 使用标准中文标点符号",
        "   - 避免过长的句子",
        "   - 适当添加停顿标记",
        "",
        "5. 环境配置:",
        "   - 确保足够的GPU内存",
        "   - 使用稳定的CUDA版本",
        "   - 检查依赖库版本兼容性"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

def create_quality_test_script():
    """创建质量测试脚本"""
    print("\n📝 创建质量测试脚本...")
    
    test_script = '''
# IndexTTS2质量测试脚本
# 在ComfyUI中使用以下参数进行测试:

# 1. 基础TTS节点参数:
text = "你好，这是IndexTTS2的音频质量测试。"
language = "zh"
speed = 1.0
use_fp16 = False
use_cuda_kernel = False

# 2. 推理质量参数 (在basic_tts_node.py中已添加):
temperature = 0.7
top_p = 0.9
top_k = 50
max_text_tokens_per_sentence = 120
interval_silence = 200

# 3. 测试不同的参考音频:
# - 使用清晰、无噪音的音频
# - 长度3-10秒
# - 采样率24kHz或以上

# 4. 检查生成的音频:
# - 使用音频编辑软件查看波形
# - 检查频谱分析
# - 听觉测试音质
'''
    
    with open("quality_test_guide.txt", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ 质量测试指南已保存到: quality_test_guide.txt")

def main():
    print("🎵 IndexTTS2音频质量诊断工具")
    print("=" * 50)
    
    # 检查生成的音频文件
    test_audio_path = r"C:\Users\ASUS\Desktop\node\output_with_audio_loader.wav"
    if os.path.exists(test_audio_path):
        diagnose_audio_file(test_audio_path)
    else:
        print(f"⚠️  测试音频文件不存在: {test_audio_path}")
    
    # 测试IndexTTS2推理
    test_indextts2_inference()
    
    # 提供改进建议
    suggest_quality_improvements()
    
    # 创建测试指南
    create_quality_test_script()
    
    print("\n" + "=" * 50)
    print("🎯 下一步操作:")
    print("1. 重启ComfyUI以加载优化的推理参数")
    print("2. 使用高质量的参考音频进行测试")
    print("3. 检查生成音频的波形和频谱")
    print("4. 根据建议调整参数和环境")

if __name__ == "__main__":
    main()
