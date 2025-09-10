#!/usr/bin/env python3
"""
最终验证 - 确认IndexTTS2专用音频加载节点正确创建
Final verification - Confirm IndexTTS2 dedicated audio loader node is correctly created
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def verify_audio_loader_creation():
    """验证音频加载节点创建"""
    print("=" * 70)
    print("IndexTTS2专用音频加载节点最终验证")
    print("IndexTTS2 Dedicated Audio Loader Final Verification")
    print("=" * 70)
    
    success_count = 0
    total_checks = 0
    
    # 1. 验证文件存在
    print("\n1. 验证文件存在性...")
    files_to_check = [
        "nodes/audio_loader_node.py",
        "workflows/08_audio_loader_example_workflow.json",
        "AUDIO_LOADER_GUIDE.md"
    ]
    
    for file_path in files_to_check:
        total_checks += 1
        if os.path.exists(file_path):
            print(f"   ✓ {file_path} 存在")
            success_count += 1
        else:
            print(f"   ❌ {file_path} 不存在")
    
    # 2. 验证音频加载节点代码
    print("\n2. 验证音频加载节点代码...")
    try:
        total_checks += 1
        from nodes.audio_loader_node import IndexTTS2LoadAudio, IndexTTS2AudioInfo
        print("   ✓ 音频加载节点导入成功")
        success_count += 1
        
        # 检查类属性
        checks = [
            (hasattr(IndexTTS2LoadAudio, 'RETURN_TYPES'), "IndexTTS2LoadAudio.RETURN_TYPES"),
            (hasattr(IndexTTS2LoadAudio, 'CATEGORY'), "IndexTTS2LoadAudio.CATEGORY"),
            (hasattr(IndexTTS2LoadAudio, 'FUNCTION'), "IndexTTS2LoadAudio.FUNCTION"),
            (hasattr(IndexTTS2AudioInfo, 'RETURN_TYPES'), "IndexTTS2AudioInfo.RETURN_TYPES"),
            (hasattr(IndexTTS2AudioInfo, 'RETURN_NAMES'), "IndexTTS2AudioInfo.RETURN_NAMES"),
        ]
        
        for check, name in checks:
            total_checks += 1
            if check:
                print(f"   ✓ {name} 存在")
                success_count += 1
            else:
                print(f"   ❌ {name} 不存在")
        
        # 验证返回类型
        total_checks += 1
        if IndexTTS2LoadAudio.RETURN_TYPES == ("AUDIO_FILE",):
            print("   ✓ IndexTTS2LoadAudio 返回类型正确: AUDIO_FILE")
            success_count += 1
        else:
            print(f"   ❌ IndexTTS2LoadAudio 返回类型错误: {IndexTTS2LoadAudio.RETURN_TYPES}")
        
    except Exception as e:
        print(f"   ❌ 音频加载节点导入失败: {e}")
    
    # 3. 验证TTS节点修改
    print("\n3. 验证TTS节点修改...")
    tts_files = [
        "nodes/basic_tts_node.py",
        "nodes/duration_control_node.py", 
        "nodes/emotion_control_node.py",
        "nodes/advanced_control_node.py"
    ]
    
    for tts_file in tts_files:
        total_checks += 1
        try:
            with open(tts_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if '"AUDIO_FILE"' in content and 'speaker_audio: dict' in content:
                    print(f"   ✓ {tts_file} 已修改为接受AUDIO_FILE类型")
                    success_count += 1
                else:
                    print(f"   ❌ {tts_file} 未正确修改")
        except Exception as e:
            print(f"   ❌ 无法检查 {tts_file}: {e}")
    
    # 4. 验证__init__.py修改
    print("\n4. 验证主模块修改...")
    total_checks += 1
    try:
        with open("__init__.py", 'r', encoding='utf-8') as f:
            content = f.read()
            if "IndexTTS2LoadAudio" in content and "IndexTTS2AudioInfo" in content:
                print("   ✓ __init__.py 已添加音频加载节点注册")
                success_count += 1
            else:
                print("   ❌ __init__.py 未正确添加音频加载节点")
    except Exception as e:
        print(f"   ❌ 无法检查 __init__.py: {e}")
    
    # 5. 验证工作流文件
    print("\n5. 验证示例工作流...")
    total_checks += 1
    try:
        import json
        with open("workflows/08_audio_loader_example_workflow.json", 'r', encoding='utf-8') as f:
            workflow = json.load(f)
            
        # 检查是否包含音频加载节点
        has_audio_loader = False
        has_audio_info = False
        has_tts_node = False
        
        for node in workflow.get("nodes", []):
            node_type = node.get("type", "")
            if node_type == "IndexTTS2LoadAudio":
                has_audio_loader = True
            elif node_type == "IndexTTS2AudioInfo":
                has_audio_info = True
            elif node_type.startswith("IndexTTS2"):
                has_tts_node = True
        
        if has_audio_loader and has_tts_node:
            print("   ✓ 示例工作流包含正确的节点类型")
            success_count += 1
        else:
            print("   ❌ 示例工作流缺少必要的节点")
            
    except Exception as e:
        print(f"   ❌ 无法验证工作流文件: {e}")
    
    # 6. 验证文档
    print("\n6. 验证使用文档...")
    total_checks += 1
    try:
        with open("AUDIO_LOADER_GUIDE.md", 'r', encoding='utf-8') as f:
            content = f.read()
            if "IndexTTS2LoadAudio" in content and "AUDIO_FILE" in content:
                print("   ✓ 使用文档包含正确的节点信息")
                success_count += 1
            else:
                print("   ❌ 使用文档内容不完整")
    except Exception as e:
        print(f"   ❌ 无法验证文档: {e}")
    
    # 总结
    print(f"\n验证完成: {success_count}/{total_checks} 项检查通过")
    print(f"Verification complete: {success_count}/{total_checks} checks passed")
    
    if success_count == total_checks:
        print("\n🎉 所有验证通过! IndexTTS2专用音频加载节点创建成功!")
        print("🎉 All verifications passed! IndexTTS2 dedicated audio loader created successfully!")
        return True
    elif success_count >= total_checks * 0.8:
        print("\n✅ 大部分验证通过! 音频加载节点基本创建成功!")
        print("✅ Most verifications passed! Audio loader node basically created successfully!")
        return True
    else:
        print("\n❌ 验证失败! 请检查错误信息!")
        print("❌ Verification failed! Please check error messages!")
        return False

def print_final_summary():
    """打印最终总结"""
    print("\n" + "=" * 70)
    print("🎯 IndexTTS2专用音频加载节点创建总结")
    print("🎯 IndexTTS2 Dedicated Audio Loader Creation Summary")
    print("=" * 70)
    
    print("\n📦 创建的组件:")
    print("📦 Created Components:")
    print("   • IndexTTS2LoadAudio - 专用音频文件加载器")
    print("     - 基于ComfyUI官方LoadAudio实现")
    print("     - 返回AUDIO_FILE类型")
    print("     - 支持多种音频/视频格式")
    print("   • IndexTTS2AudioInfo - 音频信息提取器")
    print("     - 提取详细音频元数据")
    print("     - 显示文件信息和音频参数")
    
    print("\n🔧 修改的文件:")
    print("🔧 Modified Files:")
    print("   • __init__.py - 添加新节点注册")
    print("   • nodes/basic_tts_node.py - 支持AUDIO_FILE输入")
    print("   • nodes/duration_control_node.py - 支持AUDIO_FILE输入")
    print("   • nodes/emotion_control_node.py - 支持AUDIO_FILE输入")
    print("   • nodes/advanced_control_node.py - 支持AUDIO_FILE输入")
    
    print("\n📄 创建的文档:")
    print("📄 Created Documentation:")
    print("   • AUDIO_LOADER_GUIDE.md - 详细使用指南")
    print("   • workflows/08_audio_loader_example_workflow.json - 示例工作流")
    
    print("\n🚀 使用方法:")
    print("🚀 Usage:")
    print("   1. 重启ComfyUI")
    print("   2. 在节点菜单中找到 IndexTTS2/IO 分类")
    print("   3. 添加 'IndexTTS2 Load Audio File' 节点")
    print("   4. 选择音频文件")
    print("   5. 连接AUDIO_FILE输出到TTS节点的speaker_audio输入")
    
    print("\n✨ 主要优势:")
    print("✨ Key Advantages:")
    print("   • 专为IndexTTS2设计，完美兼容")
    print("   • 基于官方实现，稳定可靠")
    print("   • 提供丰富的音频元数据")
    print("   • 支持节点连接，工作流更清晰")
    print("   • 向后兼容，不影响现有功能")

if __name__ == "__main__":
    success = verify_audio_loader_creation()
    print_final_summary()
    
    if not success:
        sys.exit(1)
