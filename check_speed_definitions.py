#!/usr/bin/env python3
"""
直接检查speed参数定义
Direct check of speed parameter definitions
"""

import re
import os

def check_speed_in_file(filepath):
    """检查文件中的speed参数定义"""
    print(f"\n=== 检查文件: {filepath} ===")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找speed参数定义
        speed_patterns = [
            r'"speed":\s*\(\s*\[([^\]]+)\]',  # 列表类型定义
            r'"speed":\s*\(\s*"([^"]+)"',     # 字符串类型定义
            r'"speed":\s*\(\s*([^,\)]+)',     # 其他类型定义
        ]
        
        found_issues = False
        
        for i, pattern in enumerate(speed_patterns):
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                matched_text = match.group(0)
                param_type = match.group(1)
                
                print(f"  行 {line_num}: {matched_text}")
                
                if i == 0:  # 列表类型
                    print(f"    ✗ 错误：speed定义为列表选择: {param_type}")
                    found_issues = True
                elif param_type != "FLOAT":
                    print(f"    ✗ 错误：speed类型不是FLOAT: {param_type}")
                    found_issues = True
                else:
                    print(f"    ✓ 正确：speed定义为FLOAT")
        
        # 查找speed_multiplier参数定义
        speed_mult_pattern = r'"speed_multiplier":\s*\(\s*"([^"]+)"'
        matches = re.finditer(speed_mult_pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            matched_text = match.group(0)
            param_type = match.group(1)
            
            print(f"  行 {line_num}: {matched_text}")
            if param_type != "FLOAT":
                print(f"    ✗ 错误：speed_multiplier类型不是FLOAT: {param_type}")
                found_issues = True
            else:
                print(f"    ✓ 正确：speed_multiplier定义为FLOAT")
        
        return not found_issues
        
    except Exception as e:
        print(f"  检查失败: {e}")
        return False

def check_all_node_files():
    """检查所有节点文件"""
    print("检查所有节点文件中的speed参数定义")
    print("=" * 60)
    
    node_files = [
        'nodes/basic_tts_node.py',
        'nodes/basic_tts_node_v2.py',
        'nodes/duration_control_node.py',
        'nodes/advanced_control_node.py',
        'nodes/emotion_control_node.py',
    ]
    
    all_good = True
    
    for filepath in node_files:
        if os.path.exists(filepath):
            if not check_speed_in_file(filepath):
                all_good = False
        else:
            print(f"\n文件不存在: {filepath}")
    
    return all_good

def check_workflow_files():
    """检查工作流文件中的speed参数值"""
    print("\n\n检查工作流文件中的speed参数值")
    print("=" * 60)
    
    workflow_dir = 'workflows'
    if not os.path.exists(workflow_dir):
        print("工作流目录不存在")
        return True
    
    found_issues = False
    
    for filename in os.listdir(workflow_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(workflow_dir, filename)
            print(f"\n=== 检查工作流: {filename} ===")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找可能的speed参数值问题
                if '"auto"' in content and 'speed' in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'speed' in line.lower() and '"auto"' in line:
                            print(f"  行 {i}: {line.strip()}")
                            print(f"    ⚠️ 可能的问题：speed参数值为'auto'")
                            found_issues = True
                
            except Exception as e:
                print(f"  检查失败: {e}")
    
    return not found_issues

def main():
    """主函数"""
    print("IndexTTS2 Speed参数定义检查工具")
    print("=" * 60)
    
    # 检查节点文件
    nodes_ok = check_all_node_files()
    
    # 检查工作流文件
    workflows_ok = check_workflow_files()
    
    print("\n" + "=" * 60)
    print("检查结果:")
    
    if nodes_ok:
        print("✓ 所有节点文件中的speed参数定义正确")
    else:
        print("✗ 发现节点文件中的speed参数定义问题")
    
    if workflows_ok:
        print("✓ 工作流文件中没有发现speed参数问题")
    else:
        print("✗ 工作流文件中可能存在speed参数问题")
    
    if nodes_ok and workflows_ok:
        print("\n建议:")
        print("1. 问题可能在ComfyUI的缓存中")
        print("2. 重启ComfyUI")
        print("3. 清理浏览器缓存")
        print("4. 检查是否有其他插件冲突")
    else:
        print("\n需要修复发现的问题")

if __name__ == "__main__":
    main()
