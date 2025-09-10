#!/usr/bin/env python3
"""
IndexTTS2 目录音频浏览器
IndexTTS2 Directory Audio Browser

Two-level audio file selection: first select directory, then select file.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

# 尝试导入 folder_paths，如果失败则使用模拟版本
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    # 创建模拟的 folder_paths 模块用于测试
    class MockFolderPaths:
        def __init__(self):
            # 尝试找到ComfyUI根目录
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / "main.py").exists() or current_dir.name == "ComfyUI":
                    self.base_path = str(current_dir)
                    break
                current_dir = current_dir.parent
            else:
                # 如果找不到，使用当前目录的上级目录
                self.base_path = str(Path(__file__).parent.parent.parent)

        @property
        def base_path(self):
            return self._base_path

        @base_path.setter
        def base_path(self, value):
            self._base_path = value

    folder_paths = MockFolderPaths()
    FOLDER_PATHS_AVAILABLE = False
    print("使用模拟 folder_paths 模块进行测试")

class DirectoryAudioBrowser:
    """目录音频浏览器 - 两级选择模式"""
    
    def __init__(self):
        self.audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
        self.cache_file = os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-index-tts2", "directory_cache.json")
        self._directory_cache = None
        
    def get_audio_directories(self):
        """获取包含音频文件的目录列表"""
        
        # 检查缓存
        if self._directory_cache:
            return self._directory_cache
        
        directories = {}
        
        # 扫描ComfyUI目录结构
        scan_roots = [
            folder_paths.base_path,
        ]
        
        for root in scan_roots:
            self._scan_for_audio_directories(root, directories)
        
        # 排序目录
        sorted_dirs = self._sort_directories(directories)
        
        # 更新缓存
        self._directory_cache = sorted_dirs
        self._save_cache(sorted_dirs)
        
        return sorted_dirs
    
    def _scan_for_audio_directories(self, root_path, directories, max_depth=3, current_depth=0):
        """扫描包含音频文件的目录"""
        
        if current_depth > max_depth or not os.path.exists(root_path):
            return
        
        try:
            for item in os.listdir(root_path):
                if item.startswith('.'):
                    continue
                
                item_path = os.path.join(root_path, item)
                
                if os.path.isdir(item_path):
                    # 跳过不必要的目录
                    skip_dirs = {
                        '.git', '__pycache__', 'node_modules', '.vscode', 
                        'venv', 'env', '.pytest_cache', 'build', 'dist',
                        '.idea', '.vs', 'logs'
                    }
                    if item in skip_dirs:
                        continue
                    
                    # 检查当前目录是否包含音频文件
                    audio_files = self._get_audio_files_in_directory(item_path)
                    
                    if audio_files:
                        # 计算相对路径
                        try:
                            relative_path = os.path.relpath(item_path, folder_paths.base_path)
                        except ValueError:
                            relative_path = item_path
                        
                        directories[relative_path] = {
                            'path': item_path,
                            'relative_path': relative_path,
                            'audio_count': len(audio_files),
                            'audio_files': audio_files,
                            'display_name': self._get_display_name(relative_path)
                        }
                    
                    # 递归扫描子目录
                    if current_depth < max_depth:
                        self._scan_for_audio_directories(item_path, directories, max_depth, current_depth + 1)
        
        except (PermissionError, OSError):
            pass
    
    def _get_audio_files_in_directory(self, directory_path):
        """获取指定目录中的音频文件"""
        audio_files = []
        
        try:
            for file in os.listdir(directory_path):
                if any(file.lower().endswith(ext) for ext in self.audio_extensions):
                    audio_files.append(file)
        except (PermissionError, OSError):
            pass
        
        return sorted(audio_files)
    
    def _get_display_name(self, relative_path):
        """生成目录的显示名称"""
        # 替换路径分隔符为更友好的显示
        display_name = relative_path.replace('\\', ' / ').replace('/', ' / ')
        
        # 添加特殊标记
        if 'input' in relative_path.lower() and 'audio' in relative_path.lower():
            return f"🌟 {display_name}"
        elif 'input' in relative_path.lower():
            return f"📁 {display_name}"
        elif 'audio' in relative_path.lower():
            return f"🎵 {display_name}"
        elif 'output' in relative_path.lower():
            return f"📤 {display_name}"
        elif 'user' in relative_path.lower():
            return f"👤 {display_name}"
        else:
            return f"📂 {display_name}"
    
    def _sort_directories(self, directories):
        """排序目录列表"""
        def get_priority(item):
            path = item[1]['relative_path'].lower()
            
            # 优先级1: input/audio
            if 'input' in path and 'audio' in path:
                return (1, item[1]['display_name'])
            
            # 优先级2: input目录
            elif 'input' in path:
                return (2, item[1]['display_name'])
            
            # 优先级3: 包含audio的目录
            elif 'audio' in path:
                return (3, item[1]['display_name'])
            
            # 优先级4: output目录
            elif 'output' in path:
                return (4, item[1]['display_name'])
            
            # 优先级5: user目录
            elif 'user' in path:
                return (5, item[1]['display_name'])
            
            # 优先级6: 其他目录
            else:
                return (6, item[1]['display_name'])
        
        sorted_items = sorted(directories.items(), key=get_priority)
        return dict(sorted_items)
    
    def get_audio_files_in_directory(self, directory_key):
        """获取指定目录中的音频文件列表"""
        directories = self.get_audio_directories()
        
        if directory_key in directories:
            return directories[directory_key]['audio_files']
        
        return []
    
    def get_directory_choices(self):
        """获取目录选择列表（用于下拉菜单）"""
        directories = self.get_audio_directories()
        
        if not directories:
            return [
                "📁 未找到包含音频文件的目录",
                "📁 No directories with audio files found",
                "💡 请将音频文件放入ComfyUI目录下",
                "💡 Please place audio files in ComfyUI directories"
            ]
        
        choices = []
        for key, info in directories.items():
            display_text = f"{info['display_name']} ({info['audio_count']} 文件)"
            choices.append(display_text)
        
        return choices
    
    def get_directory_key_from_choice(self, choice_text):
        """从选择文本获取目录键"""
        directories = self.get_audio_directories()
        
        for key, info in directories.items():
            expected_text = f"{info['display_name']} ({info['audio_count']} 文件)"
            if choice_text == expected_text:
                return key
        
        return None
    
    def get_audio_file_choices(self, directory_choice):
        """获取指定目录的音频文件选择列表"""
        directory_key = self.get_directory_key_from_choice(directory_choice)
        
        if not directory_key:
            return ["请先选择有效的目录 / Please select a valid directory first"]
        
        audio_files = self.get_audio_files_in_directory(directory_key)
        
        if not audio_files:
            return ["该目录中没有音频文件 / No audio files in this directory"]
        
        return audio_files
    
    def get_full_audio_path(self, directory_choice, audio_file):
        """获取音频文件的完整路径"""
        directory_key = self.get_directory_key_from_choice(directory_choice)
        
        if not directory_key:
            return None
        
        directories = self.get_audio_directories()
        if directory_key in directories:
            directory_path = directories[directory_key]['path']
            return os.path.join(directory_path, audio_file)
        
        return None
    
    def _save_cache(self, directories):
        """保存缓存到文件"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                'directories': directories,
                'version': '1.0'
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"目录缓存保存失败: {e}")
    
    def clear_cache(self):
        """清除缓存"""
        self._directory_cache = None
        
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
        except Exception:
            pass
    
    def refresh_directories(self):
        """刷新目录列表"""
        self.clear_cache()
        return self.get_audio_directories()

# 全局浏览器实例
_directory_browser = DirectoryAudioBrowser()

def get_audio_directory_choices():
    """获取音频目录选择列表的便捷函数"""
    return _directory_browser.get_directory_choices()

def get_audio_file_choices(directory_choice):
    """获取音频文件选择列表的便捷函数"""
    return _directory_browser.get_audio_file_choices(directory_choice)

def get_full_audio_path(directory_choice, audio_file):
    """获取完整音频文件路径的便捷函数"""
    return _directory_browser.get_full_audio_path(directory_choice, audio_file)

def clear_directory_cache():
    """清除目录缓存的便捷函数"""
    _directory_browser.clear_cache()

def refresh_audio_directories():
    """刷新音频目录的便捷函数"""
    return _directory_browser.refresh_directories()
