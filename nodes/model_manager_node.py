# IndexTTS2 Model Manager Node
# IndexTTS2 模型管理节点

import os
import torch
import gc
from typing import Optional, Tuple, Any, Dict
import threading

class IndexTTS2ModelManagerNode:
    """
    IndexTTS2 模型管理节点
    Model management node for IndexTTS2 with independent loading and caching
    
    Features:
    - Independent model loading and management
    - Memory-efficient model caching
    - Multiple model configuration support
    - GPU/CPU memory optimization
    - Model switching and unloading
    """
    
    # 类级别的模型缓存
    _model_cache = {}
    _cache_lock = threading.Lock()
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {
                    "default": "IndexTTS2-Default",
                    "placeholder": "Model identifier name"
                }),
                "config_path": ("STRING", {
                    "default": "auto",
                    "placeholder": "Path to model config file (auto to detect)"
                }),
                "model_dir": ("STRING", {
                    "default": "auto",
                    "placeholder": "Path to model directory (auto to detect)"
                }),
            },
            "optional": {
                "use_fp16": ("BOOLEAN", {
                    "default": False
                }),
                "use_cuda_kernel": ("BOOLEAN", {
                    "default": False
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
                "cache_model": ("BOOLEAN", {
                    "default": True
                }),
                "force_reload": ("BOOLEAN", {
                    "default": False
                }),
                "memory_optimization": ("BOOLEAN", {
                    "default": True
                }),
                "verbose": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("INDEXTTS2_MODEL", "STRING", "DICT")
    RETURN_NAMES = ("model", "model_info", "model_stats")
    FUNCTION = "load_model"
    CATEGORY = "IndexTTS2/Management"
    DESCRIPTION = "Load and manage IndexTTS2 models with caching and optimization"
    
    def load_model(
        self,
        model_name: str,
        config_path: str,
        model_dir: str,
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        device: str = "auto",
        cache_model: bool = True,
        force_reload: bool = False,
        memory_optimization: bool = True,
        verbose: bool = True
    ) -> Tuple[Any, str, Dict]:
        """
        加载和管理IndexTTS2模型
        Load and manage IndexTTS2 model with caching
        """
        try:
            # 参数验证和自动修正
            config_path, model_dir, device = self._validate_and_fix_parameters(
                config_path, model_dir, device, verbose
            )

            # 生成模型缓存键
            cache_key = self._generate_cache_key(
                model_name, config_path, model_dir, use_fp16, use_cuda_kernel, device
            )

            if verbose:
                print(f"[IndexTTS2 ModelManager] Loading model: {model_name}")
                print(f"[IndexTTS2 ModelManager] Config: {config_path}")
                print(f"[IndexTTS2 ModelManager] Model dir: {model_dir}")
                print(f"[IndexTTS2 ModelManager] Device: {device}")
                print(f"[IndexTTS2 ModelManager] Cache key: {cache_key}")

            # 检查缓存
            with self._cache_lock:
                if not force_reload and cache_model and cache_key in self._model_cache:
                    if verbose:
                        print(f"[IndexTTS2 ModelManager] Using cached model: {cache_key}")

                    cached_model = self._model_cache[cache_key]
                    model_info = self._generate_model_info(cached_model, model_name, "cached")
                    model_stats = self._generate_model_stats(cached_model, model_name)

                    return (cached_model, model_info, model_stats)

            # 验证路径
            plugin_dir = os.path.dirname(os.path.dirname(__file__))
            full_config_path = os.path.join(plugin_dir, config_path)
            full_model_dir = os.path.join(plugin_dir, model_dir)
            
            if not os.path.exists(full_config_path):
                raise FileNotFoundError(f"Config file not found: {full_config_path}")
            
            if not os.path.exists(full_model_dir):
                raise FileNotFoundError(f"Model directory not found: {full_model_dir}")
            
            # 内存优化：清理GPU缓存
            if memory_optimization:
                self._cleanup_memory()
            
            # 加载模型
            if verbose:
                print(f"[IndexTTS2 ModelManager] Loading new model instance...")
            
            model = self._load_model_instance(
                full_config_path, full_model_dir, use_fp16, use_cuda_kernel, device, verbose
            )
            
            # 缓存模型
            if cache_model:
                with self._cache_lock:
                    self._model_cache[cache_key] = model
                    if verbose:
                        print(f"[IndexTTS2 ModelManager] Model cached: {cache_key}")
            
            # 生成模型信息
            model_info = self._generate_model_info(model, model_name, "loaded")
            model_stats = self._generate_model_stats(model, model_name)
            
            if verbose:
                print(f"[IndexTTS2 ModelManager] Model loaded successfully: {model_name}")
            
            return (model, model_info, model_stats)
            
        except Exception as e:
            error_msg = f"IndexTTS2 model loading failed: {str(e)}"
            print(f"[IndexTTS2 ModelManager Error] {error_msg}")
            raise RuntimeError(error_msg)

    def _validate_and_fix_parameters(self, config_path: str, model_dir: str, device: str, verbose: bool):
        """验证和修正参数"""
        original_config = config_path
        original_model_dir = model_dir
        original_device = device

        if verbose:
            print(f"[IndexTTS2 ModelManager] 原始参数:")
            print(f"  config_path: '{config_path}'")
            print(f"  model_dir: '{model_dir}'")
            print(f"  device: '{device}'")

        # 检查是否参数传递错误（device值传给了model_dir）
        if model_dir in ["cuda", "cpu"]:  # 移除 "auto"，因为 "auto" 是合法的 model_dir 值
            if verbose:
                print(f"[IndexTTS2 ModelManager] 检测到参数错误: model_dir='{model_dir}' 看起来像device参数")
            # 修正参数
            device = model_dir
            model_dir = "auto"  # 使用自动检测
            if verbose:
                print(f"[IndexTTS2 ModelManager] 自动修正: model_dir='{model_dir}', device='{device}'")

        # 检查config_path是否也被错误设置
        if config_path in ["cuda", "cpu"]:  # 移除 "auto"，因为 "auto" 是合法的 config_path 值
            if verbose:
                print(f"[IndexTTS2 ModelManager] 检测到参数错误: config_path='{config_path}' 看起来像device参数")
            config_path = "auto"  # 使用自动检测
            if verbose:
                print(f"[IndexTTS2 ModelManager] 自动修正: config_path='{config_path}'")

        # 额外检查：如果config_path看起来像目录而不是文件（但不是 "auto"）
        if config_path != "auto" and not config_path.endswith(('.yaml', '.yml', '.json')):
            if verbose:
                print(f"[IndexTTS2 ModelManager] 检测到config_path='{config_path}'不是配置文件，可能是目录")
            # 如果config_path是目录，尝试在其中找到配置文件
            if config_path in ["checkpoints", "index-tts/checkpoints"]:
                config_path = f"{config_path}/config.yaml"
                if verbose:
                    print(f"[IndexTTS2 ModelManager] 自动修正: config_path='{config_path}'")
            else:
                # 使用自动检测
                config_path = "auto"
                if verbose:
                    print(f"[IndexTTS2 ModelManager] 使用自动检测: config_path='{config_path}'")

        # 智能模型目录检测
        model_dir = self._detect_model_directory(model_dir, verbose)

        # 验证模型目录是否存在且包含必要文件，如果不满足则尝试自动检测
        if not os.path.isabs(model_dir):
            plugin_dir = os.path.dirname(os.path.dirname(__file__))
            full_model_dir = os.path.join(plugin_dir, model_dir)
        else:
            full_model_dir = model_dir

        # 检查目录是否存在且包含必要的模型文件
        if not os.path.exists(full_model_dir) or not self._validate_model_directory(full_model_dir):
            if verbose:
                if not os.path.exists(full_model_dir):
                    print(f"[IndexTTS2 ModelManager] 模型目录不存在: {full_model_dir}")
                else:
                    print(f"[IndexTTS2 ModelManager] 模型目录缺少必要文件: {full_model_dir}")
                print(f"[IndexTTS2 ModelManager] 尝试自动检测...")
            # 如果指定的目录不存在或缺少文件，强制使用自动检测
            model_dir = self._detect_model_directory("auto", verbose)
            if not os.path.isabs(model_dir):
                full_model_dir = os.path.join(plugin_dir, model_dir)
            else:
                full_model_dir = model_dir

            if not os.path.exists(full_model_dir):
                raise FileNotFoundError(f"Model directory not found: {full_model_dir}")

        # 智能配置文件检测
        config_path = self._detect_config_file(config_path, model_dir, verbose)

        # 验证配置文件路径，如果不存在则尝试自动检测
        if not os.path.isabs(config_path):
            plugin_dir = os.path.dirname(os.path.dirname(__file__))
            full_config_path = os.path.join(plugin_dir, config_path)
        else:
            full_config_path = config_path

        if not os.path.exists(full_config_path):
            if verbose:
                print(f"[IndexTTS2 ModelManager] 配置文件不存在: {full_config_path}")
                print(f"[IndexTTS2 ModelManager] 尝试自动检测...")
            # 如果指定的配置文件不存在，强制使用自动检测
            config_path = self._detect_config_file("auto", model_dir, verbose)
            if not os.path.isabs(config_path):
                full_config_path = os.path.join(plugin_dir, config_path)
            else:
                full_config_path = config_path

            if not os.path.exists(full_config_path):
                raise FileNotFoundError(f"Config file not found: {full_config_path}")

        if verbose and (config_path != original_config or model_dir != original_model_dir or device != original_device):
            print(f"[IndexTTS2 ModelManager] 参数修正完成:")
            print(f"  config_path: '{original_config}' -> '{config_path}'")
            print(f"  model_dir: '{original_model_dir}' -> '{model_dir}'")
            print(f"  device: '{original_device}' -> '{device}'")

        return config_path, model_dir, device

    def _detect_model_directory(self, model_dir: str, verbose: bool) -> str:
        """智能检测模型目录"""
        if model_dir != "auto":
            return model_dir

        if verbose:
            print(f"[IndexTTS2 ModelManager] 自动检测模型目录...")

        # 检测顺序：ComfyUI标准目录 -> 插件目录 -> 其他位置
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        # plugin_dir: D:\Ken_ComfyUI_312\ComfyUI\custom_nodes\comfyui-index-tts2
        # custom_nodes_dir: D:\Ken_ComfyUI_312\ComfyUI\custom_nodes
        # comfyui_root: D:\Ken_ComfyUI_312\ComfyUI
        custom_nodes_dir = os.path.dirname(plugin_dir)
        comfyui_root = os.path.dirname(custom_nodes_dir)

        # 1. 检查 ComfyUI 标准模型目录
        comfyui_model_paths = [
            os.path.join(comfyui_root, "models", "TTS", "IndexTTS-2"),
            os.path.join(comfyui_root, "models", "TTS", "Index-TTS"),
            os.path.join(comfyui_root, "models", "TTS", "IndexTTS"),
        ]

        for path in comfyui_model_paths:
            if os.path.exists(path) and self._validate_model_directory(path):
                if verbose:
                    print(f"[IndexTTS2 ModelManager] 找到ComfyUI标准模型目录: {path}")
                return path

        # 2. 检查插件目录下的模型
        plugin_model_paths = [
            os.path.join(plugin_dir, "checkpoints"),
            os.path.join(plugin_dir, "index-tts", "checkpoints"),
            os.path.join(plugin_dir, "indextts"),
        ]

        for path in plugin_model_paths:
            if os.path.exists(path) and self._validate_model_directory(path):
                if verbose:
                    print(f"[IndexTTS2 ModelManager] 找到插件模型目录: {path}")
                return path

        # 3. 默认回退
        default_path = os.path.join(plugin_dir, "checkpoints")
        if verbose:
            print(f"[IndexTTS2 ModelManager] 使用默认模型目录: {default_path}")
        return default_path

    def _validate_model_directory(self, path: str) -> bool:
        """验证模型目录是否包含必要的文件"""
        required_files = ["gpt.pth", "s2mel.pth", "config.yaml"]
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                return False
        return True

    def _detect_config_file(self, config_path: str, model_dir: str, verbose: bool) -> str:
        """智能检测配置文件"""
        if config_path != "auto":
            return config_path

        if verbose:
            print(f"[IndexTTS2 ModelManager] 自动检测配置文件...")

        # 如果模型目录是绝对路径，直接在其中查找配置文件
        if os.path.isabs(model_dir):
            config_in_model_dir = os.path.join(model_dir, "config.yaml")
            if os.path.exists(config_in_model_dir):
                if verbose:
                    print(f"[IndexTTS2 ModelManager] 在模型目录中找到配置文件: {config_in_model_dir}")
                return config_in_model_dir

        # 检查相对路径
        plugin_dir = os.path.dirname(os.path.dirname(__file__))

        # 1. 检查模型目录中的配置文件
        if not os.path.isabs(model_dir):
            model_config_path = os.path.join(plugin_dir, model_dir, "config.yaml")
            if os.path.exists(model_config_path):
                if verbose:
                    print(f"[IndexTTS2 ModelManager] 在相对模型目录中找到配置文件: {os.path.join(model_dir, 'config.yaml')}")
                return os.path.join(model_dir, "config.yaml")

        # 2. 检查标准位置
        standard_configs = [
            "checkpoints/config.yaml",
            "index-tts/checkpoints/config.yaml",
            "config.yaml"
        ]

        for config in standard_configs:
            full_path = os.path.join(plugin_dir, config)
            if os.path.exists(full_path):
                if verbose:
                    print(f"[IndexTTS2 ModelManager] 找到标准配置文件: {config}")
                return config

        # 3. 默认回退
        default_config = "checkpoints/config.yaml"
        if verbose:
            print(f"[IndexTTS2 ModelManager] 使用默认配置文件: {default_config}")
        return default_config

    def _generate_cache_key(self, model_name: str, config_path: str, model_dir: str,
                           use_fp16: bool, use_cuda_kernel: bool, device: str) -> str:
        """生成模型缓存键"""
        key_parts = [
            model_name,
            config_path,
            model_dir,
            str(use_fp16),
            str(use_cuda_kernel),
            device
        ]
        return "|".join(key_parts)
    
    def _load_model_instance(self, config_path: str, model_dir: str, use_fp16: bool,
                            use_cuda_kernel: bool, device: str, verbose: bool):
        """加载模型实例"""
        try:
            from indextts.infer_v2 import IndexTTS2

            # 设备选择
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            if verbose:
                print(f"[IndexTTS2 ModelManager] Using device: {device}")
                print(f"[IndexTTS2 ModelManager] FP16: {use_fp16}")
                print(f"[IndexTTS2 ModelManager] CUDA kernel: {use_cuda_kernel}")

            # 创建模型实例
            model = IndexTTS2(
                cfg_path=config_path,
                model_dir=model_dir,
                is_fp16=use_fp16,
                use_cuda_kernel=use_cuda_kernel
            )

            # 移动到指定设备
            if hasattr(model, 'to'):
                model = model.to(device)

            return model

        except Exception as e:
            error_msg = f"Failed to create IndexTTS2 model instance: {str(e)}"
            # 特别处理DeepSpeed相关错误
            if "deepspeed" in str(e).lower():
                error_msg += "\n[IndexTTS2 ModelManager] DeepSpeed相关错误，但基本功能应该仍然可用"
                error_msg += "\n[IndexTTS2 ModelManager] DeepSpeed-related error, but basic functionality should still work"
            raise RuntimeError(error_msg)
    
    def _cleanup_memory(self):
        """清理内存"""
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"[IndexTTS2 ModelManager] Memory cleanup warning: {e}")
    
    def _generate_model_info(self, model, model_name: str, status: str) -> str:
        """生成模型信息字符串"""
        info_lines = [
            f"=== IndexTTS2 Model Info ===",
            f"Model Name: {model_name}",
            f"Status: {status}",
            f"Model Type: {type(model).__name__}",
        ]
        
        # 添加设备信息
        try:
            if hasattr(model, 'device'):
                info_lines.append(f"Device: {model.device}")
            elif torch.cuda.is_available():
                info_lines.append(f"Device: cuda (available)")
            else:
                info_lines.append(f"Device: cpu")
        except:
            info_lines.append(f"Device: unknown")
        
        # 添加内存信息
        try:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                info_lines.append(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        except:
            pass
        
        return "\n".join(info_lines)
    
    def _generate_model_stats(self, model, model_name: str) -> Dict:
        """生成模型统计信息"""
        stats = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "cached": True,
            "device": "unknown",
            "memory_usage": 0.0,
            "parameters": 0,
        }
        
        try:
            # 设备信息
            if hasattr(model, 'device'):
                stats["device"] = str(model.device)
            elif torch.cuda.is_available():
                stats["device"] = "cuda"
            else:
                stats["device"] = "cpu"
            
            # 内存使用
            if torch.cuda.is_available():
                stats["memory_usage"] = torch.cuda.memory_allocated() / 1024**3  # GB
            
            # 参数数量（如果可获取）
            if hasattr(model, 'parameters'):
                try:
                    total_params = sum(p.numel() for p in model.parameters())
                    stats["parameters"] = total_params
                except:
                    pass
            
        except Exception as e:
            print(f"[IndexTTS2 ModelManager] Stats generation warning: {e}")
        
        return stats
    
    @classmethod
    def clear_cache(cls):
        """清理模型缓存"""
        with cls._cache_lock:
            cls._model_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[IndexTTS2 ModelManager] Model cache cleared")
    
    @classmethod
    def get_cache_info(cls) -> Dict:
        """获取缓存信息"""
        with cls._cache_lock:
            return {
                "cached_models": list(cls._model_cache.keys()),
                "cache_size": len(cls._model_cache),
                "memory_usage": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 如果force_reload为True，总是重新执行
        if kwargs.get("force_reload", False):
            return float("nan")
        # 否则基于模型配置检查
        return hash(str(sorted(kwargs.items())))
