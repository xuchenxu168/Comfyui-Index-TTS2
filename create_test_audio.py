#!/usr/bin/env python3
"""
创建测试音频文件
Create Test Audio Files

为测试音频加载器创建一些示例音频文件
Create some sample audio files for testing the audio loader
"""

import os
import sys
import numpy as np
from scipy.io import wavfile

def create_test_audio_files():
    """创建测试音频文件"""
    print("创建测试音频文件...")
    print("Creating test audio files...")
    
    # 确保音频目录存在
    audio_dir = "../../input/audio"
    os.makedirs(audio_dir, exist_ok=True)
    
    # 创建不同的测试音频
    test_files = [
        {
            "name": "test_sine_440hz.wav",
            "description": "440Hz正弦波 (1秒)",
            "duration": 1.0,
            "frequency": 440
        },
        {
            "name": "test_sine_880hz.wav", 
            "description": "880Hz正弦波 (2秒)",
            "duration": 2.0,
            "frequency": 880
        },
        {
            "name": "test_chirp.wav",
            "description": "频率扫描 (3秒)",
            "duration": 3.0,
            "frequency": None  # 特殊处理
        }
    ]
    
    sample_rate = 44100
    
    for test_file in test_files:
        file_path = os.path.join(audio_dir, test_file["name"])
        
        if os.path.exists(file_path):
            print(f"   跳过已存在的文件: {test_file['name']}")
            continue
            
        print(f"   创建: {test_file['name']} - {test_file['description']}")
        
        duration = test_file["duration"]
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        if test_file["frequency"]:
            # 正弦波
            frequency = test_file["frequency"]
            audio_data = np.sin(2 * np.pi * frequency * t)
        else:
            # 频率扫描（chirp）
            f0, f1 = 200, 2000  # 从200Hz到2000Hz
            audio_data = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
        
        # 添加淡入淡出效果
        fade_samples = int(0.1 * sample_rate)  # 0.1秒淡入淡出
        if len(audio_data) > 2 * fade_samples:
            # 淡入
            audio_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # 淡出
            audio_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # 转换为16位整数
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # 保存文件
        try:
            wavfile.write(file_path, sample_rate, audio_data)
            print(f"      ✓ 成功创建: {file_path}")
        except Exception as e:
            print(f"      ✗ 创建失败: {e}")
    
    print(f"\n音频文件已创建在: {os.path.abspath(audio_dir)}")
    print(f"Audio files created in: {os.path.abspath(audio_dir)}")

def list_audio_files():
    """列出现有的音频文件"""
    audio_dir = "../../input/audio"
    
    print(f"\n当前音频目录中的文件:")
    print(f"Files in current audio directory:")
    print(f"目录: {os.path.abspath(audio_dir)}")
    print(f"Directory: {os.path.abspath(audio_dir)}")
    
    if not os.path.exists(audio_dir):
        print("   目录不存在")
        print("   Directory does not exist")
        return
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    audio_files = []
    
    try:
        for file in os.listdir(audio_dir):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                file_path = os.path.join(audio_dir, file)
                file_size = os.path.getsize(file_path)
                audio_files.append((file, file_size))
        
        if audio_files:
            print(f"\n找到 {len(audio_files)} 个音频文件:")
            print(f"Found {len(audio_files)} audio files:")
            for file, size in sorted(audio_files):
                size_kb = size / 1024
                print(f"   📄 {file} ({size_kb:.1f} KB)")
        else:
            print("   没有找到音频文件")
            print("   No audio files found")
            
    except Exception as e:
        print(f"   读取目录时出错: {e}")
        print(f"   Error reading directory: {e}")

if __name__ == "__main__":
    print("IndexTTS2 测试音频文件创建工具")
    print("IndexTTS2 Test Audio File Creation Tool")
    print("=" * 50)
    
    try:
        import scipy.io.wavfile
        print("✓ scipy 可用，将创建测试音频文件")
        print("✓ scipy available, will create test audio files")
        create_test_audio_files()
    except ImportError:
        print("⚠️  scipy 不可用，跳过音频文件创建")
        print("⚠️  scipy not available, skipping audio file creation")
        print("可以手动将音频文件放入 input/audio 目录进行测试")
        print("You can manually place audio files in input/audio directory for testing")
    
    # 列出现有文件
    list_audio_files()
    
    print("\n" + "=" * 50)
    print("完成！现在可以测试音频加载器了")
    print("Done! You can now test the audio loader")
    print("1. 重启 ComfyUI")
    print("1. Restart ComfyUI")
    print("2. 添加 'IndexTTS2 Load Audio File' 节点")
    print("2. Add 'IndexTTS2 Load Audio File' node")
    print("3. 测试文件选择和上传功能")
    print("3. Test file selection and upload functionality")
