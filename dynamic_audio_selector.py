#!/usr/bin/env python3
"""
IndexTTS2 动态音频选择器
IndexTTS2 Dynamic Audio Selector

Provides dynamic audio file selection based on directory choice.
"""

import os
from typing import Dict, List, Any

# 导入目录浏览器
try:
    from .directory_audio_browser import (
        get_audio_directory_choices, 
        get_audio_file_choices, 
        get_full_audio_path
    )
    DIRECTORY_BROWSER_AVAILABLE = True
except ImportError:
    DIRECTORY_BROWSER_AVAILABLE = False

class DynamicAudioSelector:
    """动态音频选择器类"""
    
    @classmethod
    def get_directory_choices(cls):
        """获取目录选择列表"""
        if DIRECTORY_BROWSER_AVAILABLE:
            try:
                return get_audio_directory_choices()
            except Exception as e:
                print(f"获取目录列表出错: {e}")
        
        return [
            "📁 input/audio (推荐)",
            "📁 input", 
            "📁 output",
            "💡 请检查目录浏览器模块"
        ]
    
    @classmethod
    def get_audio_file_choices(cls, directory_choice):
        """根据目录选择获取音频文件列表"""
        if not directory_choice or "请" in directory_choice or "Please" in directory_choice:
            return ["请先选择有效目录 / Please select a valid directory first"]
        
        if DIRECTORY_BROWSER_AVAILABLE:
            try:
                files = get_audio_file_choices(directory_choice)
                if files and len(files) > 0:
                    return files
            except Exception as e:
                print(f"获取音频文件列表出错: {e}")
        
        return ["目录中没有音频文件 / No audio files in directory"]
    
    @classmethod
    def get_full_path(cls, directory_choice, audio_file):
        """获取音频文件的完整路径"""
        if DIRECTORY_BROWSER_AVAILABLE:
            try:
                return get_full_audio_path(directory_choice, audio_file)
            except Exception as e:
                print(f"获取完整路径出错: {e}")
        
        return None

# 全局选择器实例
_audio_selector = DynamicAudioSelector()

def get_directory_choices():
    """获取目录选择列表的便捷函数"""
    return _audio_selector.get_directory_choices()

def get_audio_file_choices_for_directory(directory_choice):
    """获取指定目录的音频文件列表的便捷函数"""
    return _audio_selector.get_audio_file_choices(directory_choice)

def get_audio_full_path(directory_choice, audio_file):
    """获取音频文件完整路径的便捷函数"""
    return _audio_selector.get_full_path(directory_choice, audio_file)

# ComfyUI 自定义输入类型
class AudioDirectoryInput:
    """音频目录输入类型"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": (get_directory_choices(), {
                    "default": get_directory_choices()[0] if get_directory_choices() else "",
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("directory_choice",)
    FUNCTION = "select_directory"
    CATEGORY = "IndexTTS2/Utils"
    
    def select_directory(self, directory):
        return (directory,)

class AudioFileInput:
    """音频文件输入类型"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_choice": ("STRING", {
                    "default": "",
                }),
                "audio_file": (["请先选择目录"], {
                    "default": "请先选择目录",
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("audio_file", "full_path")
    FUNCTION = "select_audio_file"
    CATEGORY = "IndexTTS2/Utils"
    
    def select_audio_file(self, directory_choice, audio_file):
        full_path = get_audio_full_path(directory_choice, audio_file)
        return (audio_file, full_path or "")

# 组合音频选择器
class CombinedAudioSelector:
    """组合音频选择器 - 在一个节点中实现两级选择"""
    
    @classmethod
    def INPUT_TYPES(cls):
        directories = get_directory_choices()
        
        return {
            "required": {
                "audio_directory": (directories, {
                    "default": directories[0] if directories else "",
                    "tooltip": "第一步：选择包含音频文件的目录"
                }),
            },
            "optional": {
                "refresh_files": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "刷新音频文件列表"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("directory", "available_files", "selected_file")
    FUNCTION = "get_audio_info"
    CATEGORY = "IndexTTS2/Utils"
    
    def get_audio_info(self, audio_directory, refresh_files=False):
        if refresh_files:
            # 刷新缓存
            try:
                from .directory_audio_browser import clear_directory_cache
                clear_directory_cache()
            except:
                pass
        
        # 获取该目录下的音频文件
        audio_files = get_audio_file_choices_for_directory(audio_directory)
        
        # 返回信息
        files_info = ", ".join(audio_files[:5])  # 只显示前5个文件
        if len(audio_files) > 5:
            files_info += f" ... (共{len(audio_files)}个文件)"
        
        selected_file = audio_files[0] if audio_files and "请" not in audio_files[0] else ""
        
        return (audio_directory, files_info, selected_file)

# 注册自定义节点类型
NODE_CLASS_MAPPINGS = {
    "IndexTTS2_AudioDirectoryInput": AudioDirectoryInput,
    "IndexTTS2_AudioFileInput": AudioFileInput, 
    "IndexTTS2_CombinedAudioSelector": CombinedAudioSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2_AudioDirectoryInput": "Audio Directory Selector",
    "IndexTTS2_AudioFileInput": "Audio File Selector",
    "IndexTTS2_CombinedAudioSelector": "Combined Audio Selector",
}
