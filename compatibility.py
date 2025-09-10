#!/usr/bin/env python3
"""
IndexTTS2 ComfyUI Plugin - Python 3.12 Compatibility Layer
Python 3.12兼容性层

This module provides compatibility functions and fallbacks for Python 3.12
where some dependencies may not be available.
"""

import sys
import warnings
import importlib.util
from typing import Optional, Any, Dict, List

def check_python_compatibility():
    """检查Python版本兼容性"""
    version = sys.version_info
    if version.major == 3:
        if version.minor >= 12:
            return "python312", f"Python {version.major}.{version.minor}.{version.micro}"
        elif version.minor >= 8:
            return "compatible", f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return "too_old", f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return "unsupported", f"Python {version.major}.{version.minor}.{version.micro}"

def safe_import(module_name: str, fallback_name: Optional[str] = None, required: bool = True):
    """安全导入模块，提供备用方案"""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        if fallback_name:
            try:
                return importlib.import_module(fallback_name)
            except ImportError:
                pass
        
        if required:
            warnings.warn(f"无法导入必需模块 {module_name}: {e}")
            warnings.warn(f"Failed to import required module {module_name}: {e}")
        return None

class CompatibilityManager:
    """兼容性管理器"""
    
    def __init__(self):
        self.compatibility_mode, self.python_version = check_python_compatibility()
        self.available_modules = {}
        self.fallback_functions = {}
        
    def register_fallback(self, module_name: str, fallback_func):
        """注册备用函数"""
        self.fallback_functions[module_name] = fallback_func
    
    def get_module(self, module_name: str, fallback_name: Optional[str] = None):
        """获取模块，如果不可用则返回备用方案"""
        if module_name in self.available_modules:
            return self.available_modules[module_name]
        
        module = safe_import(module_name, fallback_name, required=False)
        if module:
            self.available_modules[module_name] = module
            return module
        
        # 返回备用函数
        if module_name in self.fallback_functions:
            return self.fallback_functions[module_name]
        
        return None
    
    def is_available(self, module_name: str) -> bool:
        """检查模块是否可用"""
        return self.get_module(module_name) is not None

# 全局兼容性管理器
compat = CompatibilityManager()

# 文本处理备用函数
def fallback_text_normalize(text: str) -> str:
    """备用文本规范化函数"""
    # 简单的文本清理
    import re
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 基本的中文数字转换（简化版）
    chinese_numbers = {
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        '十': '10', '百': '100', '千': '1000', '万': '10000'
    }
    
    for cn, num in chinese_numbers.items():
        text = text.replace(cn, num)
    
    return text

def fallback_g2p(text: str) -> List[str]:
    """备用音素转换函数"""
    # 简单的字符级分割
    return list(text.replace(' ', ''))

def fallback_jieba_cut(text: str) -> List[str]:
    """备用中文分词函数"""
    # 简单的字符级分割（备用方案）
    import re
    
    # 按标点符号分割
    words = re.findall(r'[\w]+|[^\w\s]', text)
    return [w for w in words if w.strip()]

# 注册备用函数
compat.register_fallback('WeTextProcessing', type('MockWeTextProcessing', (), {
    'normalize': staticmethod(fallback_text_normalize)
}))

compat.register_fallback('pynini', type('MockPynini', (), {
    'cdrewrite': lambda *args: lambda x: x,
    'acceptor': lambda x: x
}))

compat.register_fallback('g2p_en', type('MockG2P', (), {
    'G2p': lambda: type('G2P', (), {'__call__': fallback_g2p})()
}))

compat.register_fallback('jieba', type('MockJieba', (), {
    'cut': staticmethod(fallback_jieba_cut),
    'lcut': staticmethod(fallback_jieba_cut)
}))

def get_text_processor():
    """获取文本处理器"""
    # 尝试使用WeTextProcessing
    wetext = compat.get_module('WeTextProcessing')
    if wetext:
        return wetext
    
    # 使用备用方案
    return compat.get_module('WeTextProcessing')  # 返回备用实现

def get_g2p_processor():
    """获取音素转换处理器"""
    g2p_en = compat.get_module('g2p_en')
    if g2p_en:
        try:
            return g2p_en.G2p()
        except:
            pass
    
    # 使用备用方案
    fallback = compat.get_module('g2p_en')
    if fallback:
        return fallback.G2p()
    
    return None

def get_jieba_processor():
    """获取中文分词处理器"""
    jieba = compat.get_module('jieba')
    if jieba:
        return jieba
    
    # 使用备用方案
    return compat.get_module('jieba')

def install_compatible_dependencies():
    """安装兼容的依赖 - 已禁用自动安装"""
    print("[Compatibility] 自动依赖安装已禁用")
    print("[Compatibility] Auto dependency installation disabled")
    print("[Compatibility] 请手动安装所需依赖:")
    print("[Compatibility] Please install dependencies manually:")

    if compat.compatibility_mode == "python312":
        print("[Compatibility] Python 3.12环境建议手动安装兼容版本")
        print("[Compatibility] For Python 3.12, manually install compatible versions")
        print("[Compatibility] 参考: pip install -r requirements.txt")
        print("[Compatibility] Reference: pip install -r requirements.txt")
    else:
        print("[Compatibility] 运行: pip install -r requirements.txt")
        print("[Compatibility] Run: pip install -r requirements.txt")

    return False  # 总是返回False，表示未自动安装

def get_compatibility_info() -> Dict[str, Any]:
    """获取兼容性信息"""
    return {
        "python_version": compat.python_version,
        "compatibility_mode": compat.compatibility_mode,
        "available_modules": list(compat.available_modules.keys()),
        "fallback_modules": list(compat.fallback_functions.keys()),
        "is_python312": compat.compatibility_mode == "python312"
    }

def print_compatibility_status():
    """打印兼容性状态"""
    info = get_compatibility_info()
    
    print(f"[Compatibility] Python版本: {info['python_version']}")
    print(f"[Compatibility] Python version: {info['python_version']}")
    print(f"[Compatibility] 兼容性模式: {info['compatibility_mode']}")
    print(f"[Compatibility] Compatibility mode: {info['compatibility_mode']}")
    
    if info['is_python312']:
        print("[Compatibility] ⚠️  Python 3.12模式 - 使用兼容性层")
        print("[Compatibility] ⚠️  Python 3.12 mode - using compatibility layer")
    else:
        print("[Compatibility] ✓ 标准兼容模式")
        print("[Compatibility] ✓ Standard compatibility mode")

if __name__ == "__main__":
    print_compatibility_status()
    
    # 测试备用函数
    print("\n测试备用函数 Testing fallback functions:")
    
    text_processor = get_text_processor()
    if text_processor:
        result = text_processor.normalize("这是一个测试文本")
        print(f"文本规范化: {result}")
    
    jieba_processor = get_jieba_processor()
    if jieba_processor:
        result = jieba_processor.cut("这是一个测试文本")
        print(f"中文分词: {list(result)}")
    
    g2p_processor = get_g2p_processor()
    if g2p_processor:
        result = g2p_processor("hello world")
        print(f"音素转换: {result}")
