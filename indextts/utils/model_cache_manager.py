#!/usr/bin/env python3
"""
IndexTTS2 模型缓存管理器
Model Cache Manager for IndexTTS2 - 统一管理外部模型的下载位置
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union

def get_comfyui_models_dir() -> Optional[Path]:
    """获取ComfyUI的models目录路径"""
    
    # 方法1: 通过当前文件路径推断ComfyUI根目录
    current_file = Path(__file__).resolve()
    
    # 从 custom_nodes/comfyui-Index-TTS2/... 向上查找ComfyUI根目录
    for parent in current_file.parents:
        if parent.name == "ComfyUI" and (parent / "models").exists():
            return parent / "models"
        # 也检查是否在ComfyUI的子目录中
        if (parent / "ComfyUI" / "models").exists():
            return parent / "ComfyUI" / "models"
    
    # 方法2: 检查环境变量
    if "COMFYUI_MODELS_DIR" in os.environ:
        models_dir = Path(os.environ["COMFYUI_MODELS_DIR"])
        if models_dir.exists():
            return models_dir
    
    # 方法3: 检查常见的ComfyUI安装位置
    possible_paths = [
        Path.cwd() / "models",  # 当前工作目录
        Path.home() / "ComfyUI" / "models",  # 用户目录
        Path("C:/ComfyUI/models") if sys.platform == "win32" else Path("/opt/ComfyUI/models"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def get_indextts2_cache_dir() -> Path:
    """获取IndexTTS2专用的缓存目录"""
    
    # 优先使用ComfyUI的models目录
    comfyui_models = get_comfyui_models_dir()
    
    if comfyui_models:
        # 在ComfyUI/models/tts/IndexTTS-2/external_models/下创建缓存
        cache_dir = comfyui_models / "tts" / "IndexTTS-2" / "external_models"
    else:
        # 回退到插件目录下的缓存
        plugin_dir = Path(__file__).resolve().parents[2]  # 回到comfyui-Index-TTS2目录
        cache_dir = plugin_dir / "models_cache"
    
    # 确保目录存在
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[IndexTTS2] Using model cache directory: {cache_dir}")
    return cache_dir

def get_model_cache_path(model_name: str) -> Path:
    """获取特定模型的缓存路径"""
    base_cache = get_indextts2_cache_dir()
    
    # 为不同模型创建子目录
    model_cache_map = {
        "w2v-bert-2.0": "w2v_bert",
        "bigvgan_v2_22khz_80band_256x": "bigvgan",
        "MaskGCT": "maskgct", 
        "campplus": "campplus"
    }
    
    # 标准化模型名称
    for key, folder in model_cache_map.items():
        if key.lower() in model_name.lower():
            return base_cache / folder
    
    # 默认使用模型名称作为文件夹名
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    return base_cache / safe_name

def setup_hf_cache_env():
    """设置HuggingFace缓存环境变量"""
    cache_dir = get_indextts2_cache_dir()
    
    # 设置HuggingFace Hub缓存目录
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "huggingface" / "transformers")
    
    print(f"[IndexTTS2] HuggingFace cache redirected to: {cache_dir / 'huggingface'}")

def get_hf_download_kwargs(model_id: str) -> dict:
    """获取HuggingFace下载的标准参数"""
    cache_dir = get_model_cache_path(model_id)
    
    return {
        "cache_dir": str(cache_dir),
        "local_files_only": False,
        "resume_download": True,
    }

def print_cache_info():
    """打印缓存信息"""
    cache_dir = get_indextts2_cache_dir()
    comfyui_models = get_comfyui_models_dir()
    
    print("=" * 60)
    print("[IndexTTS2] Model Cache Configuration")
    print("=" * 60)
    print(f"ComfyUI Models Directory: {comfyui_models or 'Not found'}")
    print(f"IndexTTS2 Cache Directory: {cache_dir}")
    print(f"Cache Directory Exists: {cache_dir.exists()}")
    
    if cache_dir.exists():
        # 列出已缓存的模型
        cached_models = []
        for item in cache_dir.iterdir():
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                cached_models.append(f"  - {item.name}: {size_mb:.1f} MB")
        
        if cached_models:
            print("Cached Models:")
            print("\n".join(cached_models))
        else:
            print("No cached models found")
    
    print("=" * 60)

# 在模块导入时自动设置缓存环境
setup_hf_cache_env()

# 导出主要接口
__all__ = [
    'get_comfyui_models_dir',
    'get_indextts2_cache_dir', 
    'get_model_cache_path',
    'setup_hf_cache_env',
    'get_hf_download_kwargs',
    'print_cache_info'
]
