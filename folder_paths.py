"""
模拟ComfyUI的folder_paths模块
Mock ComfyUI folder_paths module for testing
"""

import os

# 获取插件根目录
plugin_root = os.path.dirname(__file__)

def get_input_directory():
    """获取输入目录"""
    # 默认使用ComfyUI的input目录
    comfyui_root = os.path.dirname(os.path.dirname(plugin_root))
    input_dir = os.path.join(comfyui_root, "input")
    
    # 如果不存在，创建一个测试目录
    if not os.path.exists(input_dir):
        input_dir = os.path.join(plugin_root, "test_input")
        os.makedirs(input_dir, exist_ok=True)
    
    return input_dir

def get_output_directory():
    """获取输出目录"""
    # 默认使用ComfyUI的output目录
    comfyui_root = os.path.dirname(os.path.dirname(plugin_root))
    output_dir = os.path.join(comfyui_root, "output")
    
    # 如果不存在，创建一个测试目录
    if not os.path.exists(output_dir):
        output_dir = os.path.join(plugin_root, "test_output")
        os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

# 其他可能需要的属性
base_path = os.path.dirname(os.path.dirname(plugin_root))
input_directory = get_input_directory()
output_directory = get_output_directory()
