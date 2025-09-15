#!/usr/bin/env python3
"""
IndexTTS2 模型缓存信息节点
Model Cache Info Node - 显示外部模型的缓存位置和状态
"""

import os
import json
from pathlib import Path
from typing import Tuple, Any

class IndexTTS2ModelCacheInfoNode:
    """
    IndexTTS2 模型缓存信息节点
    显示外部模型（w2v-bert-2.0, BigVGAN, MaskGCT, CAMPPlus）的缓存位置和状态
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refresh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "刷新缓存信息"
                }),
                "show_details": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "显示详细信息"
                }),
                "show_file_sizes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "显示文件大小"
                }),
            },
            "optional": {
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "详细输出"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "DICT", "STRING")
    RETURN_NAMES = ("cache_info", "cache_details", "summary")
    FUNCTION = "get_cache_info"
    CATEGORY = "IndexTTS2/Utils"
    DESCRIPTION = "Display IndexTTS2 external model cache information and locations"
    
    def get_cache_info(
        self,
        refresh: bool,
        show_details: bool,
        show_file_sizes: bool,
        verbose: bool = False
    ) -> Tuple[str, dict, str]:
        """
        获取模型缓存信息
        Get model cache information
        """
        try:
            if verbose:
                print("[IndexTTS2 CacheInfo] 获取模型缓存信息...")
            
            # 导入缓存管理器
            from indextts.utils.model_cache_manager import (
                get_comfyui_models_dir, 
                get_indextts2_cache_dir,
                get_model_cache_path
            )
            
            # 获取基本路径信息
            comfyui_models = get_comfyui_models_dir()
            indextts2_cache = get_indextts2_cache_dir()
            
            # 检查各个模型的缓存状态
            models_info = {
                "w2v-bert-2.0": {
                    "description": "Wav2Vec2-BERT语音特征提取器",
                    "cache_path": get_model_cache_path("w2v-bert-2.0"),
                    "hf_repo": "facebook/w2v-bert-2.0"
                },
                "BigVGAN": {
                    "description": "BigVGAN声码器",
                    "cache_path": get_model_cache_path("bigvgan_v2_22khz_80band_256x"),
                    "hf_repo": "nvidia/bigvgan_v2_22khz_80band_256x"
                },
                "MaskGCT": {
                    "description": "MaskGCT语义编解码器",
                    "cache_path": get_model_cache_path("MaskGCT"),
                    "hf_repo": "amphion/MaskGCT"
                },
                "CAMPPlus": {
                    "description": "CAMPPlus说话人编码器",
                    "cache_path": get_model_cache_path("campplus"),
                    "hf_repo": "funasr/campplus"
                }
            }
            
            # 收集详细信息
            cache_details = {
                "comfyui_models_dir": str(comfyui_models) if comfyui_models else None,
                "indextts2_cache_dir": str(indextts2_cache),
                "cache_dir_exists": indextts2_cache.exists(),
                "models": {}
            }
            
            total_size = 0
            cached_count = 0
            
            for model_name, info in models_info.items():
                cache_path = info["cache_path"]
                model_info = {
                    "description": info["description"],
                    "hf_repo": info["hf_repo"],
                    "cache_path": str(cache_path),
                    "exists": cache_path.exists(),
                    "files": [],
                    "total_size": 0
                }
                
                if cache_path.exists():
                    cached_count += 1
                    # 收集文件信息
                    for file_path in cache_path.rglob('*'):
                        if file_path.is_file():
                            file_size = file_path.stat().st_size
                            model_info["files"].append({
                                "name": file_path.name,
                                "path": str(file_path.relative_to(cache_path)),
                                "size": file_size
                            })
                            model_info["total_size"] += file_size
                    
                    total_size += model_info["total_size"]
                
                cache_details["models"][model_name] = model_info
            
            # 生成信息字符串
            info_lines = [
                "=" * 60,
                "IndexTTS2 外部模型缓存信息",
                "IndexTTS2 External Model Cache Information",
                "=" * 60,
                f"ComfyUI Models Directory: {comfyui_models or 'Not detected'}",
                f"IndexTTS2 Cache Directory: {indextts2_cache}",
                f"Cache Directory Exists: {'Yes' if indextts2_cache.exists() else 'No'}",
                "",
                f"Models Status: {cached_count}/4 cached",
                f"Total Cache Size: {self._format_size(total_size)}",
                ""
            ]
            
            # 添加模型详细信息
            for model_name, model_info in cache_details["models"].items():
                status = "✅ Cached" if model_info["exists"] else "❌ Not cached"
                size_info = f" ({self._format_size(model_info['total_size'])})" if model_info["exists"] and show_file_sizes else ""
                
                info_lines.extend([
                    f"{model_name}: {status}{size_info}",
                    f"  Description: {model_info['description']}",
                    f"  HuggingFace Repo: {model_info['hf_repo']}",
                    f"  Cache Path: {model_info['cache_path']}"
                ])
                
                if show_details and model_info["exists"] and model_info["files"]:
                    info_lines.append("  Files:")
                    for file_info in model_info["files"][:5]:  # 只显示前5个文件
                        size_str = f" ({self._format_size(file_info['size'])})" if show_file_sizes else ""
                        info_lines.append(f"    - {file_info['name']}{size_str}")
                    
                    if len(model_info["files"]) > 5:
                        info_lines.append(f"    ... and {len(model_info['files']) - 5} more files")
                
                info_lines.append("")
            
            # 添加使用说明
            info_lines.extend([
                "=" * 60,
                "说明 / Notes:",
                "• 外部模型会在首次使用时自动下载到上述缓存目录",
                "• External models will be automatically downloaded to the cache directory on first use",
                "• 如果ComfyUI Models目录被检测到，模型将下载到ComfyUI/models/tts/IndexTTS-2/external_models/",
                "• If ComfyUI Models directory is detected, models will be downloaded to ComfyUI/models/tts/IndexTTS-2/external_models/",
                "=" * 60
            ])
            
            cache_info = "\n".join(info_lines)
            
            # 生成摘要
            summary = f"Cache: {cached_count}/4 models, {self._format_size(total_size)} total"
            
            if verbose:
                print(f"[IndexTTS2 CacheInfo] 缓存信息获取完成")
                print(f"[IndexTTS2 CacheInfo] 已缓存模型: {cached_count}/4")
                print(f"[IndexTTS2 CacheInfo] 总缓存大小: {self._format_size(total_size)}")
            
            return (cache_info, cache_details, summary)
            
        except Exception as e:
            error_msg = f"Failed to get cache information: {str(e)}"
            print(f"[IndexTTS2 CacheInfo Error] {error_msg}")
            
            error_details = {
                "error": str(e),
                "comfyui_models_dir": None,
                "indextts2_cache_dir": None,
                "models": {}
            }
            
            return (error_msg, error_details, "Error")
    
    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        
        return f"{size_bytes:.1f} TB"

# 节点映射
NODE_CLASS_MAPPINGS = {
    "IndexTTS2ModelCacheInfoNode": IndexTTS2ModelCacheInfoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2ModelCacheInfoNode": "IndexTTS2 Model Cache Info"
}
