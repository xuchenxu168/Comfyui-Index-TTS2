#!/usr/bin/env python3
"""
DeepSpeed兼容性模块
处理不同版本的DeepSpeed和CUDA环境的兼容性问题
Compatibility module for DeepSpeed across different versions and CUDA environments
"""

import warnings
import sys

# 全局DeepSpeed可用性标志
DEEPSPEED_AVAILABLE = False
deepspeed = None

def check_deepspeed_availability():
    """检查DeepSpeed是否可用并返回兼容的模块"""
    global DEEPSPEED_AVAILABLE, deepspeed
    
    try:
        import deepspeed as ds
        
        # 检查关键方法是否存在
        if hasattr(ds, 'init_inference'):
            DEEPSPEED_AVAILABLE = True
            deepspeed = ds
            print("[IndexTTS2] DeepSpeed available and compatible")
            return True, ds
        else:
            print("[IndexTTS2] DeepSpeed found but missing required methods")
            return False, None
            
    except ImportError as e:
        print(f"[IndexTTS2] DeepSpeed not available: {e}")
        return False, None
    except Exception as e:
        print(f"[IndexTTS2] DeepSpeed check failed: {e}")
        # 特别处理CUDA 12.6相关的错误
        if "deepspeed.utils.torch" in str(e):
            print("[IndexTTS2] Detected DeepSpeed CUDA 12.6 compatibility issue")
            print("[IndexTTS2] Recommendation: Update DeepSpeed or use standard inference")
        return False, None

def get_mock_deepspeed():
    """返回一个模拟的DeepSpeed模块，用于兼容性"""
    
    class MockDeepSpeedZero:
        @staticmethod
        def Init(*args, **kwargs):
            class MockContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockContext()
        
        @staticmethod
        def GatheredParameters(*args, **kwargs):
            class MockContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockContext()
    
    class MockDeepSpeed:
        zero = MockDeepSpeedZero()
        
        @staticmethod
        def init_inference(*args, **kwargs):
            raise RuntimeError("DeepSpeed not available - using mock implementation")
    
    return MockDeepSpeed()

def is_deepspeed_zero3_enabled():
    """兼容的is_deepspeed_zero3_enabled实现"""
    if not DEEPSPEED_AVAILABLE:
        return False
    
    try:
        # 尝试不同的导入路径
        try:
            from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled as _is_enabled
            return _is_enabled()
        except ImportError:
            try:
                from transformers.integrations import is_deepspeed_zero3_enabled as _is_enabled
                return _is_enabled()
            except ImportError:
                try:
                    from deepspeed.utils.zero_to_fp32 import is_deepspeed_zero3_enabled as _is_enabled
                    return _is_enabled()
                except ImportError:
                    try:
                        from deepspeed.runtime.zero.stage3 import is_deepspeed_zero3_enabled as _is_enabled
                        return _is_enabled()
                    except ImportError:
                        # 如果所有导入都失败，返回False
                        return False
    except Exception:
        return False

def deepspeed_config():
    """兼容的deepspeed_config实现"""
    if not DEEPSPEED_AVAILABLE:
        return None
    
    try:
        from transformers.integrations import deepspeed_config as _config
        return _config()
    except ImportError:
        try:
            from transformers.integrations.deepspeed import deepspeed_config as _config
            return _config()
        except ImportError:
            return None

def safe_deepspeed_import():
    """安全的DeepSpeed导入，返回可用的模块或模拟模块"""
    available, ds_module = check_deepspeed_availability()
    
    if available:
        return ds_module
    else:
        return get_mock_deepspeed()

# 初始化检查
check_deepspeed_availability()

# 如果DeepSpeed不可用，创建模拟模块
if not DEEPSPEED_AVAILABLE:
    deepspeed = get_mock_deepspeed()

# 导出主要接口
__all__ = [
    'DEEPSPEED_AVAILABLE',
    'deepspeed',
    'is_deepspeed_zero3_enabled',
    'deepspeed_config',
    'safe_deepspeed_import',
    'check_deepspeed_availability'
]
