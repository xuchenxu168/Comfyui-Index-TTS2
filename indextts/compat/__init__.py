# -*- coding: utf-8 -*-
"""
IndexTTS2 Transformers Compatibility Layer
IndexTTS2 transformers 兼容层

这个模块提供了一个完整的 transformers 兼容层，确保 IndexTTS2 可以在任何 transformers 版本下运行，
包括不支持 Qwen3 等新模型架构的旧版本。

This module provides a complete transformers compatibility layer, ensuring IndexTTS2 
can run on any transformers version, including older versions that don't support 
Qwen3 and other new model architectures.
"""

import sys
import warnings
from typing import Optional, Any, Dict, Union
import importlib.util

# 版本兼容性检查
def check_transformers_version():
    """检查 transformers 版本并返回兼容性信息"""
    try:
        import transformers
        version = transformers.__version__
        
        # 解析版本号
        from packaging import version as pkg_version
        current_ver = pkg_version.parse(version)
        
        # 完全移除版本限制 - 任何版本都尝试使用
        # Completely remove version restrictions - try to use any version
        recommended = pkg_version.parse("4.36.2")

        # 所有版本都标记为兼容，让实际加载来决定是否可用
        # Mark all versions as compatible, let actual loading decide availability
        compatibility = "compatible"
            
        return {
            "version": version,
            "compatibility": compatibility,
            "recommended": "4.36.2",
            "needs_fallback": compatibility != "compatible"
        }
        
    except ImportError:
        return {
            "version": None,
            "compatibility": "missing",
            "recommended": "4.36.2",
            "needs_fallback": True
        }

# 全局兼容性状态
_compat_info = check_transformers_version()
_use_fallback = _compat_info["needs_fallback"]

def get_compatibility_info():
    """获取兼容性信息"""
    return _compat_info.copy()

def should_use_fallback():
    """是否应该使用回退模式"""
    return _use_fallback

# 兼容性导入函数
def safe_import_transformers_module(module_name: str, fallback_module: str = None):
    """安全导入 transformers 模块，失败时使用回退"""
    try:
        if not _use_fallback:
            # 尝试从系统 transformers 导入
            module = importlib.import_module(f"transformers.{module_name}")
            return module, "system"
        else:
            raise ImportError("Using fallback mode")
    except ImportError:
        if fallback_module:
            # 使用内置回退模块
            try:
                module = importlib.import_module(f"indextts.compat.{fallback_module}")
                return module, "fallback"
            except ImportError:
                raise ImportError(f"Both system and fallback modules failed for {module_name}")
        else:
            raise ImportError(f"No fallback available for {module_name}")

def safe_import_from_transformers(module_name: str, class_names: list, fallback_module: str = None):
    """从 transformers 安全导入类，失败时使用回退"""
    try:
        if not _use_fallback:
            # 尝试从系统 transformers 导入
            module = importlib.import_module(f"transformers.{module_name}")
            result = {}
            for class_name in class_names:
                if hasattr(module, class_name):
                    result[class_name] = getattr(module, class_name)
                else:
                    raise ImportError(f"{class_name} not found in {module_name}")
            return result, "system"
        else:
            raise ImportError("Using fallback mode")
    except ImportError:
        if fallback_module:
            # 使用内置回退模块
            try:
                module = importlib.import_module(f"indextts.compat.{fallback_module}")
                result = {}
                for class_name in class_names:
                    if hasattr(module, class_name):
                        result[class_name] = getattr(module, class_name)
                    else:
                        raise ImportError(f"{class_name} not found in fallback {fallback_module}")
                return result, "fallback"
            except ImportError:
                raise ImportError(f"Both system and fallback modules failed for {module_name}")
        else:
            raise ImportError(f"No fallback available for {module_name}")

# 兼容性警告
def warn_compatibility_issue(issue: str, solution: str = None):
    """发出兼容性警告"""
    message = f"[IndexTTS2 Compatibility] {issue}"
    if solution:
        message += f" Solution: {solution}"
    warnings.warn(message, UserWarning, stacklevel=2)

# 初始化时的兼容性检查
def initialize_compatibility():
    """初始化兼容性层"""
    info = get_compatibility_info()
    
    if info["compatibility"] == "missing":
        warn_compatibility_issue(
            "transformers not installed",
            "pip install transformers==4.36.2"
        )
    elif info["compatibility"] == "too_old":
        warn_compatibility_issue(
            f"transformers {info['version']} is too old",
            f"pip install transformers=={info['recommended']}"
        )
    elif info["compatibility"] == "too_new":
        warn_compatibility_issue(
            f"transformers {info['version']} may have compatibility issues",
            f"pip install transformers=={info['recommended']}"
        )
    
    if _use_fallback:
        print(f"[IndexTTS2] Using compatibility fallback mode for transformers {info.get('version', 'N/A')}")
    else:
        print(f"[IndexTTS2] Using system transformers {info['version']}")

# 自动初始化
initialize_compatibility()

# 导出主要接口
__all__ = [
    'check_transformers_version',
    'get_compatibility_info', 
    'should_use_fallback',
    'safe_import_transformers_module',
    'safe_import_from_transformers',
    'warn_compatibility_issue',
    'initialize_compatibility'
]
