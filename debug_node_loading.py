#!/usr/bin/env python3
"""
调试节点加载问题
Debug Node Loading Issues
"""

import os
import sys
import traceback

# 添加当前目录到路径
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

def test_node_import():
    """测试节点导入"""
    print("=" * 60)
    print("调试 IndexTTS2LoadAudio 节点加载问题")
    print("Debugging IndexTTS2LoadAudio Node Loading Issues")
    print("=" * 60)
    
    try:
        print("1. 测试直接导入节点类...")
        from nodes.audio_loader_node import IndexTTS2LoadAudio
        print("   ✓ 成功导入 IndexTTS2LoadAudio")
        
        print("2. 测试节点类属性...")
        print(f"   类名: {IndexTTS2LoadAudio.__name__}")
        print(f"   模块: {IndexTTS2LoadAudio.__module__}")
        
        print("3. 测试 INPUT_TYPES 方法...")
        try:
            input_types = IndexTTS2LoadAudio.INPUT_TYPES()
            print("   ✓ INPUT_TYPES 方法调用成功")
            print(f"   返回类型: {type(input_types)}")
            print(f"   内容: {input_types}")
        except Exception as e:
            print(f"   ✗ INPUT_TYPES 方法调用失败: {e}")
            traceback.print_exc()
        
        print("4. 测试节点实例化...")
        try:
            node = IndexTTS2LoadAudio()
            print("   ✓ 节点实例化成功")
            print(f"   实例类型: {type(node)}")
        except Exception as e:
            print(f"   ✗ 节点实例化失败: {e}")
            traceback.print_exc()
        
        print("5. 测试必需的类属性...")
        required_attrs = ['RETURN_TYPES', 'RETURN_NAMES', 'FUNCTION', 'CATEGORY']
        for attr in required_attrs:
            if hasattr(IndexTTS2LoadAudio, attr):
                value = getattr(IndexTTS2LoadAudio, attr)
                print(f"   ✓ {attr}: {value}")
            else:
                print(f"   ✗ 缺少属性: {attr}")
        
        print("6. 测试方法存在性...")
        required_methods = ['load_audio', 'IS_CHANGED', 'VALIDATE_INPUTS']
        for method in required_methods:
            if hasattr(IndexTTS2LoadAudio, method):
                print(f"   ✓ 方法存在: {method}")
            else:
                print(f"   ✗ 缺少方法: {method}")
        
        return True
        
    except Exception as e:
        print(f"✗ 节点导入失败: {e}")
        traceback.print_exc()
        return False

def test_init_import():
    """测试从__init__.py导入"""
    print("\n" + "=" * 60)
    print("测试从 __init__.py 导入")
    print("Testing Import from __init__.py")
    print("=" * 60)
    
    try:
        print("1. 导入插件模块...")
        import __init__ as plugin
        print("   ✓ 成功导入插件模块")
        
        print("2. 检查 NODE_CLASS_MAPPINGS...")
        if hasattr(plugin, 'NODE_CLASS_MAPPINGS'):
            mappings = plugin.NODE_CLASS_MAPPINGS
            print(f"   ✓ NODE_CLASS_MAPPINGS 存在，包含 {len(mappings)} 个节点")
            
            for key, value in mappings.items():
                print(f"     - {key}: {value}")
                
            if 'IndexTTS2LoadAudio' in mappings:
                print("   ✓ IndexTTS2LoadAudio 在映射中找到")
                node_class = mappings['IndexTTS2LoadAudio']
                print(f"     节点类: {node_class}")
                
                # 测试节点类
                try:
                    input_types = node_class.INPUT_TYPES()
                    print("     ✓ INPUT_TYPES 调用成功")
                except Exception as e:
                    print(f"     ✗ INPUT_TYPES 调用失败: {e}")
                    
            else:
                print("   ✗ IndexTTS2LoadAudio 不在映射中")
        else:
            print("   ✗ NODE_CLASS_MAPPINGS 不存在")
        
        print("3. 检查 NODE_DISPLAY_NAME_MAPPINGS...")
        if hasattr(plugin, 'NODE_DISPLAY_NAME_MAPPINGS'):
            display_mappings = plugin.NODE_DISPLAY_NAME_MAPPINGS
            print(f"   ✓ NODE_DISPLAY_NAME_MAPPINGS 存在，包含 {len(display_mappings)} 个显示名称")
            
            for key, value in display_mappings.items():
                print(f"     - {key}: {value}")
        else:
            print("   ✗ NODE_DISPLAY_NAME_MAPPINGS 不存在")
            
        return True
        
    except Exception as e:
        print(f"✗ 从 __init__.py 导入失败: {e}")
        traceback.print_exc()
        return False

def test_comfyui_compatibility():
    """测试ComfyUI兼容性"""
    print("\n" + "=" * 60)
    print("测试 ComfyUI 兼容性")
    print("Testing ComfyUI Compatibility")
    print("=" * 60)
    
    print("1. 检查 folder_paths 模块...")
    try:
        import folder_paths
        print("   ✓ folder_paths 模块可用")
        print(f"     模块路径: {folder_paths.__file__}")
        
        # 测试关键方法
        try:
            input_dir = folder_paths.get_input_directory()
            print(f"     ✓ get_input_directory(): {input_dir}")
        except Exception as e:
            print(f"     ✗ get_input_directory() 失败: {e}")
            
    except ImportError:
        print("   ⚠️  folder_paths 模块不可用（这在独立测试中是正常的）")
    
    print("2. 检查音频目录...")
    try:
        # 尝试创建音频目录
        audio_dir = "../../input/audio"
        if os.path.exists(audio_dir):
            print(f"   ✓ 音频目录存在: {os.path.abspath(audio_dir)}")
            
            # 列出音频文件
            audio_files = []
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
            for file in os.listdir(audio_dir):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(file)
            
            print(f"     找到 {len(audio_files)} 个音频文件")
            for file in audio_files[:5]:  # 只显示前5个
                print(f"       - {file}")
        else:
            print(f"   ⚠️  音频目录不存在: {os.path.abspath(audio_dir)}")
    except Exception as e:
        print(f"   ✗ 检查音频目录失败: {e}")

if __name__ == "__main__":
    print("IndexTTS2 节点加载调试工具")
    print("IndexTTS2 Node Loading Debug Tool")
    print()
    
    # 运行所有测试
    test1 = test_node_import()
    test2 = test_init_import()
    test_comfyui_compatibility()
    
    print("\n" + "=" * 60)
    print("调试总结 / Debug Summary")
    print("=" * 60)
    
    if test1 and test2:
        print("✅ 节点定义和导入都正常")
        print("✅ Node definition and import are working")
        print("\n可能的问题:")
        print("Possible issues:")
        print("1. ComfyUI 缓存问题 - 尝试重启 ComfyUI")
        print("1. ComfyUI cache issue - try restarting ComfyUI")
        print("2. 插件加载顺序问题 - 检查其他插件冲突")
        print("2. Plugin loading order issue - check for conflicts with other plugins")
        print("3. Web扩展问题 - 检查浏览器控制台错误")
        print("3. Web extension issue - check browser console for errors")
    else:
        print("❌ 发现节点定义或导入问题")
        print("❌ Found issues with node definition or import")
        print("请检查上面的错误信息并修复")
        print("Please check the error messages above and fix them")
