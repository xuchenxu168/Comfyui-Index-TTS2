# -*- coding: utf-8 -*-
"""
Transformers Imports Compatibility Layer
transformers 导入兼容层

这个模块提供了统一的 transformers 导入接口，自动处理版本兼容性问题。
所有 IndexTTS2 代码都应该通过这个模块导入 transformers 组件。

This module provides a unified transformers import interface that automatically 
handles version compatibility issues. All IndexTTS2 code should import 
transformers components through this module.
"""

# 简化导入，避免循环依赖
import warnings

# =============================================================================
# Core Transformers Components
# =============================================================================

def should_use_fallback():
    """简化的回退检查"""
    try:
        import transformers
        from packaging import version as pkg_version
        current_ver = pkg_version.parse(transformers.__version__)
        max_compatible = pkg_version.parse("4.40.0")
        return current_ver > max_compatible
    except ImportError:
        return True

def get_gpt2_components():
    """获取 GPT2 相关组件"""
    try:
        if not should_use_fallback():
            # 尝试从系统 transformers 导入
            from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
            return GPT2Config, GPT2Model, GPT2PreTrainedModel, "system"
        else:
            raise ImportError("Using fallback")
    except ImportError:
        # 使用内置兼容版本
        from indextts.gpt.transformers_gpt2 import GPT2Config, GPT2Model, GPT2PreTrainedModel
        return GPT2Config, GPT2Model, GPT2PreTrainedModel, "fallback"

def get_generation_components():
    """获取生成相关组件"""
    try:
        if not should_use_fallback():
            from transformers import LogitsProcessorList
            from transformers.generation import GenerationConfig
            from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
            return LogitsProcessorList, GenerationConfig, CausalLMOutputWithCrossAttentions, "system"
        else:
            raise ImportError("Using fallback")
    except ImportError:
        # 使用内置兼容版本
        try:
            from indextts.gpt.transformers_generation_utils import LogitsProcessorList, GenerationConfig
        except ImportError:
            # 如果内置版本不可用，尝试从系统导入部分组件
            from transformers import LogitsProcessorList
            from transformers.generation import GenerationConfig

        # 对于 CausalLMOutputWithCrossAttentions，优先尝试系统版本
        try:
            from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
        except ImportError:
            # 创建一个简单的回退实现
            from dataclasses import dataclass
            from typing import Optional, Tuple
            import torch

            @dataclass
            class CausalLMOutputWithCrossAttentions:
                """兼容版本的 CausalLMOutputWithCrossAttentions"""
                loss: Optional[torch.FloatTensor] = None
                logits: torch.FloatTensor = None
                past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
                hidden_states: Optional[Tuple[torch.FloatTensor]] = None
                attentions: Optional[Tuple[torch.FloatTensor]] = None
                cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

        return LogitsProcessorList, GenerationConfig, CausalLMOutputWithCrossAttentions, "fallback"

def get_tokenizer_components():
    """获取分词器相关组件"""
    try:
        if not should_use_fallback():
            from transformers import AutoTokenizer
            return AutoTokenizer, "system"
        else:
            # 对于分词器，我们总是尝试使用系统版本，因为它们通常更稳定
            from transformers import AutoTokenizer
            return AutoTokenizer, "system"
    except ImportError:
        warn_compatibility_issue("AutoTokenizer not available", "pip install transformers>=4.35.0")
        raise

def get_model_parallel_utils():
    """获取模型并行工具"""
    try:
        if not should_use_fallback():
            from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
            return assert_device_map, get_device_map, "system"
        else:
            raise ImportError("Using fallback")
    except ImportError:
        # 提供简单的回退实现
        def assert_device_map(device_map, num_blocks):
            """简单的设备映射断言"""
            pass
        
        def get_device_map(num_blocks, range_list):
            """简单的设备映射获取"""
            return {i: 0 for i in range(num_blocks)}
        
        return assert_device_map, get_device_map, "fallback"

def get_feature_extractor():
    """获取特征提取器"""
    try:
        from transformers import SeamlessM4TFeatureExtractor
        return SeamlessM4TFeatureExtractor, "system"
    except ImportError:
        warnings.warn("SeamlessM4TFeatureExtractor not available", UserWarning)
        raise

# =============================================================================
# Unified Import Interface
# =============================================================================

class TransformersCompat:
    """Transformers 兼容性接口类"""
    
    def __init__(self):
        self._gpt2_components = None
        self._generation_components = None
        self._tokenizer_components = None
        self._model_parallel_utils = None
        self._feature_extractor = None
    
    @property
    def GPT2Config(self):
        if self._gpt2_components is None:
            self._gpt2_components = get_gpt2_components()
        return self._gpt2_components[0]
    
    @property
    def GPT2Model(self):
        if self._gpt2_components is None:
            self._gpt2_components = get_gpt2_components()
        return self._gpt2_components[1]
    
    @property
    def GPT2PreTrainedModel(self):
        if self._gpt2_components is None:
            self._gpt2_components = get_gpt2_components()
        return self._gpt2_components[2]
    
    @property
    def LogitsProcessorList(self):
        if self._generation_components is None:
            self._generation_components = get_generation_components()
        return self._generation_components[0]
    
    @property
    def GenerationConfig(self):
        if self._generation_components is None:
            self._generation_components = get_generation_components()
        return self._generation_components[1]
    
    @property
    def CausalLMOutputWithCrossAttentions(self):
        if self._generation_components is None:
            self._generation_components = get_generation_components()
        return self._generation_components[2]
    
    @property
    def AutoTokenizer(self):
        if self._tokenizer_components is None:
            self._tokenizer_components = get_tokenizer_components()
        return self._tokenizer_components[0]
    
    @property
    def assert_device_map(self):
        if self._model_parallel_utils is None:
            self._model_parallel_utils = get_model_parallel_utils()
        return self._model_parallel_utils[0]
    
    @property
    def get_device_map(self):
        if self._model_parallel_utils is None:
            self._model_parallel_utils = get_model_parallel_utils()
        return self._model_parallel_utils[1]
    
    @property
    def SeamlessM4TFeatureExtractor(self):
        if self._feature_extractor is None:
            self._feature_extractor = get_feature_extractor()
        return self._feature_extractor[0]
    
    def get_import_status(self):
        """获取导入状态信息"""
        status = {}
        
        if self._gpt2_components:
            status['gpt2'] = self._gpt2_components[3]
        if self._generation_components:
            status['generation'] = self._generation_components[3]
        if self._tokenizer_components:
            status['tokenizer'] = self._tokenizer_components[1]
        if self._model_parallel_utils:
            status['model_parallel'] = self._model_parallel_utils[2]
        if self._feature_extractor:
            status['feature_extractor'] = self._feature_extractor[1]
            
        return status

# 创建全局兼容性实例
compat = TransformersCompat()

# 延迟导出常用组件（向后兼容）- 避免循环导入
def _get_component(name):
    """延迟获取组件，避免循环导入"""
    return getattr(compat, name)

# 使用 __getattr__ 实现延迟导入
def __getattr__(name):
    """模块级别的延迟属性访问"""
    component_map = {
        'GPT2Config': 'GPT2Config',
        'GPT2Model': 'GPT2Model',
        'GPT2PreTrainedModel': 'GPT2PreTrainedModel',
        'LogitsProcessorList': 'LogitsProcessorList',
        'GenerationConfig': 'GenerationConfig',
        'CausalLMOutputWithCrossAttentions': 'CausalLMOutputWithCrossAttentions',
        'AutoTokenizer': 'AutoTokenizer',
        'assert_device_map': 'assert_device_map',
        'get_device_map': 'get_device_map',
        'SeamlessM4TFeatureExtractor': 'SeamlessM4TFeatureExtractor'
    }

    if name in component_map:
        return _get_component(component_map[name])

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 导出主要接口
__all__ = [
    'compat',
    'TransformersCompat',
    'get_gpt2_components',
    'get_generation_components', 
    'get_tokenizer_components',
    'get_model_parallel_utils',
    'get_feature_extractor',
    # 向后兼容导出
    'GPT2Config',
    'GPT2Model',
    'GPT2PreTrainedModel', 
    'LogitsProcessorList',
    'GenerationConfig',
    'CausalLMOutputWithCrossAttentions',
    'AutoTokenizer',
    'assert_device_map',
    'get_device_map',
    'SeamlessM4TFeatureExtractor'
]
