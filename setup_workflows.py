#!/usr/bin/env python3
"""
IndexTTS2 工作流设置脚本
IndexTTS2 Workflow Setup Script

This script helps set up and validate the IndexTTS2 example workflows.
"""

import os
import json
import shutil
from pathlib import Path

def setup_workflows():
    """设置工作流环境"""
    
    print("IndexTTS2 工作流设置脚本")
    print("IndexTTS2 Workflow Setup Script")
    print("="*50)
    
    # 获取路径
    plugin_dir = Path(__file__).parent
    workflows_dir = plugin_dir / "workflows"
    comfyui_dir = plugin_dir.parent.parent
    
    print(f"插件目录 / Plugin directory: {plugin_dir}")
    print(f"工作流目录 / Workflows directory: {workflows_dir}")
    print(f"ComfyUI目录 / ComfyUI directory: {comfyui_dir}")
    
    # 检查工作流文件
    workflow_files = [
        "01_basic_tts_workflow.json",
        "02_duration_control_workflow.json", 
        "03_emotion_control_workflow.json",
        "04_advanced_control_workflow.json",
        "05_audio_utils_workflow.json",
        "06_comprehensive_demo_workflow.json"
    ]
    
    print("\n检查工作流文件 / Checking workflow files:")
    missing_files = []
    
    for workflow_file in workflow_files:
        workflow_path = workflows_dir / workflow_file
        if workflow_path.exists():
            print(f"✓ {workflow_file}")
            # 验证JSON格式
            try:
                with open(workflow_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                print(f"  └─ JSON格式正确 / JSON format valid")
            except json.JSONDecodeError as e:
                print(f"  └─ ✗ JSON格式错误 / JSON format error: {e}")
        else:
            print(f"✗ {workflow_file} (缺失 / Missing)")
            missing_files.append(workflow_file)
    
    if missing_files:
        print(f"\n⚠️  缺失 {len(missing_files)} 个工作流文件")
        print(f"⚠️  Missing {len(missing_files)} workflow files")
        return False
    
    # 检查音频目录
    print("\n检查音频目录 / Checking audio directories:")
    audio_dirs = [
        comfyui_dir / "input" / "audio",
        comfyui_dir / "input" / "audio" / "speakers",
        comfyui_dir / "input" / "audio" / "emotions",
        plugin_dir / "audio",
        plugin_dir / "examples" / "audio"
    ]
    
    for audio_dir in audio_dirs:
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
            print(f"✓ {audio_dir} ({len(audio_files)} 个音频文件)")
        else:
            print(f"✗ {audio_dir} (不存在 / Does not exist)")
    
    # 创建示例音频文件信息
    create_sample_audio_info()
    
    # 复制工作流到ComfyUI用户目录（如果存在）
    copy_workflows_to_user_dir(comfyui_dir, workflows_dir)
    
    print("\n✅ 工作流设置完成！")
    print("✅ Workflow setup complete!")
    
    return True

def create_sample_audio_info():
    """创建示例音频文件信息"""
    
    plugin_dir = Path(__file__).parent
    info_file = plugin_dir / "workflows" / "SAMPLE_AUDIO_INFO.md"
    
    sample_info = """# 示例音频文件信息 / Sample Audio File Information

## 📁 推荐的音频文件 / Recommended Audio Files

为了获得最佳的工作流演示效果，建议准备以下类型的音频文件：

To get the best workflow demonstration results, it is recommended to prepare the following types of audio files:

### 🎤 基础说话人音频 / Basic Speaker Audio

**文件名建议 / Suggested filenames:**
- `demo_speaker.wav` - 演示用的主要说话人音频
- `female_voice_01.wav` - 女性声音样本
- `male_voice_01.wav` - 男性声音样本
- `premium_speaker.wav` - 高质量说话人样本

**要求 / Requirements:**
- 时长: 3-8秒 / Duration: 3-8 seconds
- 格式: WAV (推荐) / Format: WAV (recommended)
- 质量: 清晰无噪音 / Quality: Clear and noise-free
- 内容: 自然语音 / Content: Natural speech

### 😊 情感音频样本 / Emotional Audio Samples

**文件名建议 / Suggested filenames:**
- `female_voice_happy.wav` - 开心的女性声音
- `male_voice_sad.wav` - 悲伤的男性声音
- `speaker_angry.wav` - 愤怒的语音样本
- `voice_surprised.wav` - 惊讶的语音样本
- `emotion_happy_ref.wav` - 开心情感参考
- `emotion_surprise_ref.wav` - 惊讶情感参考

**要求 / Requirements:**
- 明显的情感特征 / Clear emotional characteristics
- 单一情感表达 / Single emotion expression
- 无背景音乐 / No background music

### 🔧 工具测试音频 / Tool Testing Audio

**文件名建议 / Suggested filenames:**
- `speaker_sample_01.wav` - 分析测试样本
- `noisy_audio_sample.wav` - 带噪音的音频（用于增强测试）
- `emotional_speech.wav` - 情感检测测试
- `speaker_comparison.wav` - 说话人比较测试
- `multilingual_speaker.wav` - 多语言说话人

### 📥 获取示例音频 / Getting Sample Audio

1. **录制自己的声音 / Record your own voice**
   - 使用手机或电脑录音
   - 说一些自然的句子
   - 确保环境安静

2. **使用开源音频 / Use open-source audio**
   - 从开源数据集下载
   - 确保符合使用许可
   - 转换为WAV格式

3. **生成测试音频 / Generate test audio**
   - 使用其他TTS系统生成
   - 作为初始测试样本
   - 逐步替换为真实录音

### 📂 文件放置 / File Placement

将音频文件放置到以下目录：
Place audio files in the following directories:

```
ComfyUI/input/audio/           # 主要音频文件
ComfyUI/input/audio/speakers/  # 说话人音频
ComfyUI/input/audio/emotions/  # 情感音频
ComfyUI/input/audio/examples/  # 示例音频
```

### 🔄 更新工作流 / Update Workflows

添加新音频文件后：
After adding new audio files:

1. 重启ComfyUI / Restart ComfyUI
2. 重新加载工作流 / Reload workflows  
3. 检查下拉菜单中的新文件 / Check new files in dropdown menus
4. 测试工作流功能 / Test workflow functionality

---

**💡 提示**: 高质量的音频文件是获得最佳语音合成效果的关键！
**💡 Tip**: High-quality audio files are key to achieving the best speech synthesis results!
"""
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(sample_info)
    
    print(f"✓ 创建示例音频信息文件: {info_file}")

def copy_workflows_to_user_dir(comfyui_dir, workflows_dir):
    """复制工作流到ComfyUI用户目录"""
    
    # 检查是否存在用户工作流目录
    user_workflows_dir = comfyui_dir / "user" / "default" / "workflows"
    
    if not user_workflows_dir.exists():
        # 尝试其他可能的用户目录
        alt_dirs = [
            comfyui_dir / "workflows",
            comfyui_dir / "user_workflows",
            comfyui_dir / "examples" / "workflows"
        ]
        
        for alt_dir in alt_dirs:
            if alt_dir.exists():
                user_workflows_dir = alt_dir
                break
        else:
            # 创建工作流目录
            user_workflows_dir = comfyui_dir / "workflows"
            user_workflows_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 创建工作流目录: {user_workflows_dir}")
    
    # 复制工作流文件
    print(f"\n复制工作流到用户目录 / Copying workflows to user directory:")
    print(f"目标目录 / Target directory: {user_workflows_dir}")
    
    # 创建IndexTTS2子目录
    indextts2_workflows_dir = user_workflows_dir / "IndexTTS2"
    indextts2_workflows_dir.mkdir(exist_ok=True)
    
    workflow_files = list(workflows_dir.glob("*.json"))
    copied_count = 0
    
    for workflow_file in workflow_files:
        target_file = indextts2_workflows_dir / workflow_file.name
        try:
            shutil.copy2(workflow_file, target_file)
            print(f"✓ 复制: {workflow_file.name}")
            copied_count += 1
        except Exception as e:
            print(f"✗ 复制失败 / Copy failed: {workflow_file.name} - {e}")
    
    # 复制说明文件
    readme_file = workflows_dir / "README_WORKFLOWS.md"
    if readme_file.exists():
        target_readme = indextts2_workflows_dir / "README.md"
        shutil.copy2(readme_file, target_readme)
        print(f"✓ 复制说明文件: README.md")
    
    print(f"\n✅ 成功复制 {copied_count} 个工作流文件到用户目录")
    print(f"✅ Successfully copied {copied_count} workflow files to user directory")

def validate_workflow_json(workflow_path):
    """验证工作流JSON文件"""
    
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        
        # 检查必需的字段
        required_fields = ['nodes', 'links', 'version']
        for field in required_fields:
            if field not in workflow_data:
                return False, f"缺少必需字段: {field}"
        
        # 检查节点
        if not isinstance(workflow_data['nodes'], list):
            return False, "nodes字段必须是列表"
        
        # 检查是否包含IndexTTS2节点
        has_indextts2_node = False
        for node in workflow_data['nodes']:
            if 'type' in node and 'IndexTTS2' in node['type']:
                has_indextts2_node = True
                break
        
        if not has_indextts2_node:
            return False, "工作流中未找到IndexTTS2节点"
        
        return True, "工作流验证通过"
        
    except json.JSONDecodeError as e:
        return False, f"JSON格式错误: {e}"
    except Exception as e:
        return False, f"验证失败: {e}"

def main():
    """主函数"""
    
    try:
        success = setup_workflows()
        
        if success:
            print("\n" + "="*50)
            print("🎉 IndexTTS2 工作流设置成功！")
            print("🎉 IndexTTS2 workflow setup successful!")
            print("\n下一步 / Next steps:")
            print("1. 运行: python setup_audio_files.py")
            print("1. Run: python setup_audio_files.py")
            print("2. 添加音频文件到相应目录")
            print("2. Add audio files to appropriate directories")
            print("3. 重启ComfyUI")
            print("3. Restart ComfyUI")
            print("4. 加载工作流文件")
            print("4. Load workflow files")
            print("5. 开始使用IndexTTS2！")
            print("5. Start using IndexTTS2!")
        else:
            print("\n⚠️  工作流设置遇到问题，请检查上述错误信息")
            print("⚠️  Workflow setup encountered issues, please check the error messages above")
            
    except Exception as e:
        print(f"\n✗ 设置失败: {e}")
        print(f"✗ Setup failed: {e}")

if __name__ == "__main__":
    main()
