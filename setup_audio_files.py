#!/usr/bin/env python3
"""
IndexTTS2 音频文件设置脚本
IndexTTS2 Audio Files Setup Script

This script helps users set up audio files for IndexTTS2 ComfyUI plugin.
"""

import os
import shutil
import sys
from pathlib import Path

def create_audio_directories():
    """创建音频文件目录"""
    
    plugin_dir = Path(__file__).parent
    comfyui_dir = plugin_dir.parent.parent
    
    # 要创建的目录列表
    audio_dirs = [
        plugin_dir / "audio",
        plugin_dir / "examples" / "audio", 
        comfyui_dir / "input" / "audio",
        comfyui_dir / "input" / "audio" / "speakers",
        comfyui_dir / "input" / "audio" / "emotions",
        comfyui_dir / "input" / "audio" / "examples"
    ]
    
    print("创建音频文件目录...")
    print("Creating audio file directories...")
    
    for audio_dir in audio_dirs:
        try:
            audio_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 创建目录: {audio_dir}")
            print(f"✓ Created directory: {audio_dir}")
        except Exception as e:
            print(f"✗ 创建目录失败 {audio_dir}: {e}")
            print(f"✗ Failed to create directory {audio_dir}: {e}")
    
    return audio_dirs

def create_readme_files(audio_dirs):
    """在每个目录中创建说明文件"""
    
    readme_content = {
        "speakers": """# 说话人音频文件 / Speaker Audio Files

请将不同说话人的音频文件放在这里
Please place different speaker audio files here

文件命名建议 / File naming suggestions:
- speaker_female_01.wav
- speaker_male_01.wav
- speaker_child_01.wav

要求 / Requirements:
- 时长: 3-10秒 / Duration: 3-10 seconds
- 质量: 清晰无噪音 / Quality: Clear, noise-free
- 内容: 单一说话人 / Content: Single speaker
""",
        
        "emotions": """# 情感音频文件 / Emotion Audio Files

请将带有明显情感特征的音频文件放在这里
Please place audio files with clear emotional characteristics here

文件命名建议 / File naming suggestions:
- emotion_happy.wav
- emotion_sad.wav
- emotion_angry.wav
- emotion_excited.wav

要求 / Requirements:
- 时长: 2-8秒 / Duration: 2-8 seconds
- 情感: 明显的情感表达 / Emotion: Clear emotional expression
- 质量: 高质量录音 / Quality: High-quality recording
""",
        
        "examples": """# 示例音频文件 / Example Audio Files

请将示例和测试用的音频文件放在这里
Please place example and test audio files here

可以包含 / Can include:
- 测试用音频 / Test audio files
- 演示样本 / Demo samples
- 参考音频 / Reference audio
- 多语言示例 / Multi-language examples
""",
        
        "default": """# 音频文件目录 / Audio Files Directory

请将您的音频文件放在这个目录中
Please place your audio files in this directory

支持的格式 / Supported formats:
- WAV (推荐 / Recommended)
- MP3
- FLAC  
- OGG
- M4A

使用方法 / Usage:
1. 将音频文件复制到此目录
2. 重启ComfyUI
3. 在IndexTTS2节点中选择音频文件

Copy audio files to this directory
Restart ComfyUI
Select audio files in IndexTTS2 nodes
"""
    }
    
    print("\n创建说明文件...")
    print("Creating README files...")
    
    for audio_dir in audio_dirs:
        try:
            # 确定使用哪个说明内容
            dir_name = audio_dir.name
            if dir_name in readme_content:
                content = readme_content[dir_name]
            else:
                content = readme_content["default"]
            
            readme_file = audio_dir / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ 创建说明文件: {readme_file}")
            print(f"✓ Created README: {readme_file}")
            
        except Exception as e:
            print(f"✗ 创建说明文件失败 {audio_dir}: {e}")
            print(f"✗ Failed to create README {audio_dir}: {e}")

def copy_example_files():
    """复制示例文件（如果存在）"""
    
    plugin_dir = Path(__file__).parent
    
    # 检查是否有现有的示例音频文件
    example_sources = [
        plugin_dir / "examples",
        plugin_dir / "audio",
    ]
    
    target_dir = plugin_dir.parent.parent / "input" / "audio" / "examples"
    
    print("\n检查示例音频文件...")
    print("Checking for example audio files...")
    
    copied_files = 0
    for source_dir in example_sources:
        if source_dir.exists():
            for file_path in source_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                    try:
                        target_file = target_dir / file_path.name
                        if not target_file.exists():
                            shutil.copy2(file_path, target_file)
                            print(f"✓ 复制示例文件: {file_path.name}")
                            print(f"✓ Copied example file: {file_path.name}")
                            copied_files += 1
                    except Exception as e:
                        print(f"✗ 复制文件失败 {file_path.name}: {e}")
                        print(f"✗ Failed to copy file {file_path.name}: {e}")
    
    if copied_files == 0:
        print("未找到示例音频文件")
        print("No example audio files found")

def show_usage_instructions():
    """显示使用说明"""
    
    plugin_dir = Path(__file__).parent
    comfyui_dir = plugin_dir.parent.parent
    
    print("\n" + "="*60)
    print("音频文件设置完成！/ Audio files setup completed!")
    print("="*60)
    
    print("\n📁 音频文件目录 / Audio file directories:")
    print(f"1. 主要目录 / Main directory: {comfyui_dir / 'input' / 'audio'}")
    print(f"2. 说话人目录 / Speakers: {comfyui_dir / 'input' / 'audio' / 'speakers'}")
    print(f"3. 情感目录 / Emotions: {comfyui_dir / 'input' / 'audio' / 'emotions'}")
    print(f"4. 示例目录 / Examples: {comfyui_dir / 'input' / 'audio' / 'examples'}")
    
    print("\n📋 下一步操作 / Next steps:")
    print("1. 将您的音频文件复制到上述目录中")
    print("1. Copy your audio files to the directories above")
    print("2. 重启ComfyUI")
    print("2. Restart ComfyUI")
    print("3. 在IndexTTS2节点中选择音频文件")
    print("3. Select audio files in IndexTTS2 nodes")
    
    print("\n💡 提示 / Tips:")
    print("- 使用WAV格式获得最佳质量 / Use WAV format for best quality")
    print("- 音频时长建议3-10秒 / Recommended duration: 3-10 seconds")
    print("- 确保音频清晰无噪音 / Ensure audio is clear and noise-free")
    print("- 单一说话人效果最佳 / Single speaker works best")

def main():
    """主函数"""
    
    print("IndexTTS2 音频文件设置脚本")
    print("IndexTTS2 Audio Files Setup Script")
    print("="*50)
    
    try:
        # 创建目录
        audio_dirs = create_audio_directories()
        
        # 创建说明文件
        create_readme_files(audio_dirs)
        
        # 复制示例文件
        copy_example_files()
        
        # 显示使用说明
        show_usage_instructions()
        
        print("\n🎉 设置完成！")
        print("🎉 Setup completed!")
        
    except Exception as e:
        print(f"\n❌ 设置过程中发生错误: {e}")
        print(f"❌ Error occurred during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
