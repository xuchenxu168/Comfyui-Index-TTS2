#!/usr/bin/env python3
"""
验证 TTS-multi-Talk 工作流文件
Validate TTS-multi-Talk workflow files
"""

import json
import os
from pathlib import Path

def validate_workflow_file(file_path):
    """验证单个工作流文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        print(f"✅ {file_path.name}: JSON格式正确")
        
        # 检查基本结构
        required_keys = ['nodes', 'links', 'groups']
        for key in required_keys:
            if key not in workflow:
                print(f"⚠️  {file_path.name}: 缺少 '{key}' 字段")
            else:
                print(f"   - {key}: {len(workflow[key])} 项")
        
        # 检查是否包含 TTS-multi-Talk 节点
        multi_talk_nodes = []
        for node in workflow.get('nodes', []):
            if node.get('type') == 'IndexTTS2_MultiTalk':
                multi_talk_nodes.append(node)
        
        if multi_talk_nodes:
            print(f"   - 找到 {len(multi_talk_nodes)} 个 TTS-multi-Talk 节点")
            for i, node in enumerate(multi_talk_nodes):
                num_speakers = node.get('widgets_values', [''])[0] if node.get('widgets_values') else 'unknown'
                print(f"     节点 {i+1}: {num_speakers} 人对话")
        else:
            print(f"⚠️  {file_path.name}: 未找到 TTS-multi-Talk 节点")
        
        # 检查音频输入节点
        load_audio_nodes = []
        for node in workflow.get('nodes', []):
            if node.get('type') == 'LoadAudio':
                load_audio_nodes.append(node)
        
        print(f"   - 找到 {len(load_audio_nodes)} 个音频输入节点")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ {file_path.name}: JSON格式错误 - {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path.name}: 验证失败 - {e}")
        return False

def main():
    """主函数"""
    print("=== TTS-multi-Talk 工作流验证 ===\n")
    
    # 获取当前目录下的所有 JSON 文件
    current_dir = Path(__file__).parent
    json_files = list(current_dir.glob('*.json'))
    
    if not json_files:
        print("❌ 未找到任何 JSON 工作流文件")
        return
    
    print(f"找到 {len(json_files)} 个工作流文件:\n")
    
    valid_count = 0
    for json_file in sorted(json_files):
        if validate_workflow_file(json_file):
            valid_count += 1
        print()
    
    print("=== 验证总结 ===")
    print(f"总文件数: {len(json_files)}")
    print(f"有效文件: {valid_count}")
    print(f"无效文件: {len(json_files) - valid_count}")
    
    if valid_count == len(json_files):
        print("🎉 所有工作流文件都有效！")
    else:
        print("⚠️  部分工作流文件需要修复")

def check_workflow_completeness():
    """检查工作流完整性"""
    print("\n=== 工作流完整性检查 ===")
    
    expected_workflows = [
        'simple_2speaker_example.json',
        'multi_talk_example.json', 
        'advanced_4speaker_example.json'
    ]
    
    current_dir = Path(__file__).parent
    
    for workflow_name in expected_workflows:
        workflow_path = current_dir / workflow_name
        if workflow_path.exists():
            print(f"✅ {workflow_name}: 存在")
        else:
            print(f"❌ {workflow_name}: 缺失")

def analyze_workflow_features():
    """分析工作流特性"""
    print("\n=== 工作流特性分析 ===")
    
    current_dir = Path(__file__).parent
    json_files = list(current_dir.glob('*.json'))
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            
            print(f"\n📋 {json_file.name}:")
            
            # 分析节点类型
            node_types = {}
            for node in workflow.get('nodes', []):
                node_type = node.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print("   节点类型:")
            for node_type, count in sorted(node_types.items()):
                print(f"     - {node_type}: {count}")
            
            # 分析 TTS-multi-Talk 配置
            for node in workflow.get('nodes', []):
                if node.get('type') == 'IndexTTS2_MultiTalk':
                    widgets = node.get('widgets_values', [])
                    if len(widgets) >= 3:
                        num_speakers = widgets[0]
                        conversation_preview = widgets[1][:100] + "..." if len(widgets[1]) > 100 else widgets[1]
                        output_filename = widgets[2]
                        
                        print(f"   TTS-multi-Talk 配置:")
                        print(f"     - 说话人数: {num_speakers}")
                        print(f"     - 输出文件: {output_filename}")
                        print(f"     - 对话预览: {conversation_preview}")
            
        except Exception as e:
            print(f"❌ 分析 {json_file.name} 失败: {e}")

if __name__ == "__main__":
    main()
    check_workflow_completeness()
    analyze_workflow_features()
    
    print(f"\n🎭 TTS-multi-Talk 工作流验证完成！")
