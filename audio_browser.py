#!/usr/bin/env python3
"""
IndexTTS2 音频文件浏览器
IndexTTS2 Audio File Browser

Advanced audio file browser that can scan and list audio files from any directory.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Set

# 安全导入 folder_paths
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    # 如果在ComfyUI环境外运行，创建模拟版本
    class MockFolderPaths:
        def __init__(self):
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / "main.py").exists() or current_dir.name == "ComfyUI":
                    self.base_path = str(current_dir)
                    break
                current_dir = current_dir.parent
            else:
                self.base_path = str(Path(__file__).parent.parent.parent)

    folder_paths = MockFolderPaths()
    FOLDER_PATHS_AVAILABLE = False

class AudioFileBrowser:
    """高级音频文件浏览器"""
    
    def __init__(self):
        self.audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
        self.cache_file = os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-index-tts2", "audio_cache.json")
        self.cache_duration = 300  # 缓存5分钟
        self._audio_cache = None
        self._cache_timestamp = 0
        
    def get_all_audio_files(self, use_cache=True, max_files=500):
        """获取所有音频文件，支持缓存"""
        
        # 检查缓存
        if use_cache and self._is_cache_valid():
            return self._audio_cache
        
        print("扫描音频文件...")
        audio_files = []
        seen_files = set()
        
        # 定义扫描策略
        scan_strategies = [
            self._scan_priority_directories,
            self._scan_comfyui_structure,
            self._scan_user_directories,
            self._scan_model_directories
        ]
        
        # 执行扫描策略
        for strategy in scan_strategies:
            strategy_files = strategy()
            for file_path in strategy_files:
                if file_path not in seen_files and len(audio_files) < max_files:
                    seen_files.add(file_path)
                    audio_files.append(file_path)
        
        # 排序和优化
        audio_files = self._sort_and_optimize_files(audio_files)
        
        # 更新缓存
        self._update_cache(audio_files)
        
        print(f"找到 {len(audio_files)} 个音频文件")
        return audio_files
    
    def _scan_priority_directories(self):
        """扫描优先级目录"""
        priority_dirs = [
            os.path.join(folder_paths.input_directory, "audio"),
            os.path.join(folder_paths.base_path, "input", "audio"),
            folder_paths.input_directory,
            os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-index-tts2", "audio"),
            os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-index-tts2", "examples", "audio"),
        ]
        
        files = []
        for directory in priority_dirs:
            files.extend(self._scan_directory(directory, max_depth=2))
        return files
    
    def _scan_comfyui_structure(self):
        """扫描ComfyUI标准目录结构"""
        comfyui_dirs = [
            os.path.join(folder_paths.base_path, "input"),
            os.path.join(folder_paths.base_path, "output"),
            os.path.join(folder_paths.base_path, "temp"),
            os.path.join(folder_paths.base_path, "audio"),
        ]
        
        files = []
        for directory in comfyui_dirs:
            files.extend(self._scan_directory(directory, max_depth=3))
        return files
    
    def _scan_user_directories(self):
        """扫描用户目录"""
        user_dirs = [
            os.path.join(folder_paths.base_path, "user"),
            os.path.join(folder_paths.base_path, "workflows"),
            os.path.join(folder_paths.base_path, "examples"),
        ]
        
        files = []
        for directory in user_dirs:
            files.extend(self._scan_directory(directory, max_depth=2))
        return files
    
    def _scan_model_directories(self):
        """扫描模型目录（可能包含音频样本）"""
        model_dirs = [
            os.path.join(folder_paths.base_path, "models"),
            os.path.join(folder_paths.base_path, "checkpoints"),
        ]
        
        files = []
        for directory in model_dirs:
            # 只扫描可能包含音频的子目录
            audio_subdirs = ['audio', 'samples', 'examples', 'demo']
            for subdir in audio_subdirs:
                subdir_path = os.path.join(directory, subdir)
                files.extend(self._scan_directory(subdir_path, max_depth=2))
        return files
    
    def _scan_directory(self, directory, max_depth=2, current_depth=0):
        """递归扫描目录"""
        files = []
        
        if current_depth > max_depth or not os.path.exists(directory):
            return files
        
        try:
            for item in os.listdir(directory):
                if item.startswith('.'):  # 跳过隐藏文件
                    continue
                
                item_path = os.path.join(directory, item)
                
                if os.path.isfile(item_path):
                    # 检查是否是音频文件
                    if self._is_audio_file(item):
                        try:
                            relative_path = os.path.relpath(item_path, folder_paths.base_path)
                            files.append(relative_path)
                        except ValueError:
                            files.append(item_path)
                
                elif os.path.isdir(item_path) and current_depth < max_depth:
                    # 跳过不必要的目录
                    skip_dirs = {
                        '.git', '__pycache__', 'node_modules', '.vscode', 
                        'venv', 'env', '.pytest_cache', 'build', 'dist'
                    }
                    if item not in skip_dirs:
                        files.extend(self._scan_directory(item_path, max_depth, current_depth + 1))
        
        except (PermissionError, OSError):
            pass
        
        return files
    
    def _is_audio_file(self, filename):
        """检查是否是音频文件"""
        return any(filename.lower().endswith(ext) for ext in self.audio_extensions)
    
    def _sort_and_optimize_files(self, files):
        """排序和优化文件列表"""
        def get_sort_priority(file_path):
            """获取文件的排序优先级"""
            path_lower = file_path.lower()
            
            # 优先级1: input/audio目录
            if 'input' in path_lower and 'audio' in path_lower:
                return (1, file_path)
            
            # 优先级2: input目录
            elif 'input' in path_lower:
                return (2, file_path)
            
            # 优先级3: 插件audio目录
            elif 'comfyui-index-tts2' in path_lower and 'audio' in path_lower:
                return (3, file_path)
            
            # 优先级4: 其他audio目录
            elif 'audio' in path_lower:
                return (4, file_path)
            
            # 优先级5: 其他文件
            else:
                return (5, file_path)
        
        # 排序
        sorted_files = sorted(files, key=get_sort_priority)
        
        # 去重
        seen = set()
        unique_files = []
        for file_path in sorted_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)
        
        return unique_files
    
    def _is_cache_valid(self):
        """检查缓存是否有效"""
        if not self._audio_cache:
            return self._load_cache()
        
        return time.time() - self._cache_timestamp < self.cache_duration
    
    def _load_cache(self):
        """从文件加载缓存"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                self._audio_cache = cache_data.get('files', [])
                self._cache_timestamp = cache_data.get('timestamp', 0)
                
                return time.time() - self._cache_timestamp < self.cache_duration
        except Exception:
            pass
        
        return False
    
    def _update_cache(self, files):
        """更新缓存"""
        self._audio_cache = files
        self._cache_timestamp = time.time()
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                'files': files,
                'timestamp': self._cache_timestamp,
                'version': '1.0'
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def clear_cache(self):
        """清除缓存"""
        self._audio_cache = None
        self._cache_timestamp = 0
        
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
        except Exception:
            pass
    
    def get_directory_tree(self, root_dir=None):
        """获取目录树结构（用于调试）"""
        if root_dir is None:
            root_dir = folder_paths.base_path
        
        tree = {}
        
        try:
            for item in os.listdir(root_dir):
                if item.startswith('.'):
                    continue
                
                item_path = os.path.join(root_dir, item)
                
                if os.path.isdir(item_path):
                    # 检查是否包含音频文件
                    audio_count = len(self._scan_directory(item_path, max_depth=1))
                    if audio_count > 0:
                        tree[item] = {
                            'type': 'directory',
                            'audio_files': audio_count,
                            'path': item_path
                        }
                elif self._is_audio_file(item):
                    tree[item] = {
                        'type': 'audio_file',
                        'path': item_path
                    }
        except Exception:
            pass
        
        return tree

# 全局浏览器实例
_audio_browser = AudioFileBrowser()

def get_all_audio_files(use_cache=True, max_files=500):
    """获取所有音频文件的便捷函数"""
    return _audio_browser.get_all_audio_files(use_cache, max_files)

def clear_audio_cache():
    """清除音频文件缓存的便捷函数"""
    _audio_browser.clear_cache()

def get_audio_directory_tree():
    """获取音频目录树的便捷函数"""
    return _audio_browser.get_directory_tree()
