#!/usr/bin/env python3
"""
验证节点类型匹配
Verify node type matching
"""

def check_audio_loader_node():
    """检查音频加载节点"""
    print("=== 检查 IndexTTS2LoadAudio 节点 ===")
    
    # 读取文件内容
    with open('nodes/audio_loader_node.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查返回类型
    if 'RETURN_TYPES = ("AUDIO_FILE",)' in content:
        print("✓ LoadAudio节点输出类型: AUDIO_FILE")
        return "AUDIO_FILE"
    else:
        print("✗ LoadAudio节点输出类型不正确")
        return None

def check_tts_nodes():
    """检查TTS节点"""
    print("\n=== 检查 TTS 节点输入类型 ===")
    
    nodes = [
        ('basic_tts_node.py', 'BasicTTS'),
        ('emotion_control_node.py', 'EmotionControl'),
        ('advanced_control_node.py', 'AdvancedControl'),
        ('duration_control_node.py', 'DurationControl')
    ]
    
    all_correct = True
    
    for filename, node_name in nodes:
        with open(f'nodes/{filename}', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if '"speaker_audio": ("AUDIO_FILE"' in content:
            print(f"✓ {node_name}节点输入类型: AUDIO_FILE")
        else:
            print(f"✗ {node_name}节点输入类型不正确")
            all_correct = False
    
    return all_correct

def check_processing_logic():
    """检查处理逻辑"""
    print("\n=== 检查音频文件处理逻辑 ===")
    
    nodes = [
        ('basic_tts_node.py', 'BasicTTS'),
        ('emotion_control_node.py', 'EmotionControl'),
        ('advanced_control_node.py', 'AdvancedControl'),
        ('duration_control_node.py', 'DurationControl')
    ]
    
    all_correct = True
    
    for filename, node_name in nodes:
        with open(f'nodes/{filename}', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否移除了旧的检查条件
        if "isinstance(speaker_audio, dict) and 'audio' in speaker_audio" in content:
            print(f"✗ {node_name}节点仍有旧的处理逻辑")
            all_correct = False
        elif "isinstance(speaker_audio, dict):" in content:
            print(f"✓ {node_name}节点处理逻辑已更新")
        else:
            print(f"? {node_name}节点处理逻辑未找到")
    
    return all_correct

def main():
    """主函数"""
    print("IndexTTS2 节点类型验证")
    print("=" * 40)
    
    # 检查输出类型
    output_type = check_audio_loader_node()
    
    # 检查输入类型
    input_types_correct = check_tts_nodes()
    
    # 检查处理逻辑
    processing_correct = check_processing_logic()
    
    print("\n" + "=" * 40)
    
    if output_type == "AUDIO_FILE" and input_types_correct and processing_correct:
        print("✓ 所有类型检查通过！节点应该可以正常连接。")
        print("\n使用说明:")
        print("1. 添加 'IndexTTS2 Load Audio File' 节点")
        print("2. 选择音频文件")
        print("3. 将 audio_file 输出连接到 TTS 节点的 speaker_audio 输入")
        return True
    else:
        print("✗ 类型检查失败！")
        return False

if __name__ == "__main__":
    main()
