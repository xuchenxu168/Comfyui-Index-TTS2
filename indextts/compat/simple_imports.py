# -*- coding: utf-8 -*-
"""
Simple Transformers Compatibility Layer
简单的 transformers 兼容层

直接解决导入问题，避免复杂的循环依赖。
"""

import warnings

def get_transformers_components():
    """获取所有需要的 transformers 组件 - 支持 4.35+ 所有版本"""
    components = {}

    # 检查 transformers 版本
    try:
        import transformers
        version = transformers.__version__
        print(f"[IndexTTS2 Compat] 检测到 transformers 版本: {version}")

        from packaging import version as pkg_version
        current_ver = pkg_version.parse(version)
        min_supported = pkg_version.parse("4.35.0")

        if current_ver >= min_supported:
            print(f"[IndexTTS2 Compat] ✅ 版本 {version} 受支持 (>= 4.35.0)")
            version_supported = True
        else:
            print(f"[IndexTTS2 Compat] ⚠️ 版本 {version} 过旧，建议升级到 4.35.0+")
            version_supported = False

    except ImportError:
        version = "未安装"
        version_supported = False
        print(f"[IndexTTS2 Compat] ❌ transformers 未安装")

    # 1. GPT2 组件 - 智能选择策略，支持 4.35+ 所有版本
    try:
        # 优先尝试系统 transformers（适用于 4.35+ 所有版本）
        from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel

        # 测试基本功能是否正常
        test_config = GPT2Config(vocab_size=100, n_positions=512, n_embd=768, n_layer=12, n_head=12)

        components['GPT2Config'] = GPT2Config
        components['GPT2Model'] = GPT2Model
        components['GPT2PreTrainedModel'] = GPT2PreTrainedModel
        components['gpt2_source'] = 'system'
        print(f"[IndexTTS2 Compat] ✅ 使用系统 GPT2 组件 (transformers {version})")

    except (ImportError, Exception) as e:
        # 如果系统版本有问题，使用内置兼容版本
        try:
            from indextts.gpt.transformers_gpt2 import GPT2Config, GPT2Model, GPT2PreTrainedModel
            components['GPT2Config'] = GPT2Config
            components['GPT2Model'] = GPT2Model
            components['GPT2PreTrainedModel'] = GPT2PreTrainedModel
            components['gpt2_source'] = 'fallback'
            print(f"[IndexTTS2 Compat] 🔄 使用内置 GPT2 组件 (系统版本问题: {e})")
        except ImportError as fallback_error:
            print(f"[IndexTTS2 Compat] ❌ GPT2 组件加载失败: {fallback_error}")
            raise
    
    # 2. 生成组件 - 4.35+ 版本通常都支持这些组件
    try:
        from transformers import LogitsProcessorList
        from transformers.generation import GenerationConfig
        from transformers import GenerationMixin

        # 测试基本功能
        test_config = GenerationConfig()

        components['LogitsProcessorList'] = LogitsProcessorList
        components['GenerationConfig'] = GenerationConfig
        components['GenerationMixin'] = GenerationMixin
        components['generation_source'] = 'system'
        print(f"[IndexTTS2 Compat] ✅ 使用系统生成组件")

    except (ImportError, Exception) as e:
        try:
            from indextts.gpt.transformers_generation_utils import LogitsProcessorList, GenerationConfig, GenerationMixin
            components['LogitsProcessorList'] = LogitsProcessorList
            components['GenerationConfig'] = GenerationConfig
            components['GenerationMixin'] = GenerationMixin
            components['generation_source'] = 'fallback'
            print(f"[IndexTTS2 Compat] 🔄 使用内置生成组件 (系统版本问题: {e})")
        except ImportError as fallback_error:
            print(f"[IndexTTS2 Compat] ❌ 生成组件加载失败: {fallback_error}")
            raise
    
    # 3. 输出类 - 4.35+ 版本都应该支持
    try:
        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

        # 测试类是否可用
        test_output = CausalLMOutputWithCrossAttentions()

        components['CausalLMOutputWithCrossAttentions'] = CausalLMOutputWithCrossAttentions
        components['output_source'] = 'system'
        print(f"[IndexTTS2 Compat] ✅ 使用系统输出类")

    except (ImportError, Exception) as e:
        # 创建兼容的回退实现
        from dataclasses import dataclass
        from typing import Optional, Tuple
        import torch

        @dataclass
        class CausalLMOutputWithCrossAttentions:
            """兼容版本的 CausalLMOutputWithCrossAttentions - 支持所有 transformers 版本"""
            loss: Optional[torch.FloatTensor] = None
            logits: torch.FloatTensor = None
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
            hidden_states: Optional[Tuple[torch.FloatTensor]] = None
            attentions: Optional[Tuple[torch.FloatTensor]] = None
            cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

        components['CausalLMOutputWithCrossAttentions'] = CausalLMOutputWithCrossAttentions
        components['output_source'] = 'fallback'
        print(f"[IndexTTS2 Compat] 🔄 使用内置输出类 (系统版本问题: {e})")
    
    # 4. 分词器 - 4.35+ 版本都支持 AutoTokenizer
    try:
        from transformers import AutoTokenizer

        # 测试基本功能（不实际加载模型）
        if hasattr(AutoTokenizer, 'from_pretrained'):
            components['AutoTokenizer'] = AutoTokenizer
            components['tokenizer_source'] = 'system'
            print(f"[IndexTTS2 Compat] ✅ 使用系统分词器")
        else:
            raise ImportError("AutoTokenizer 缺少 from_pretrained 方法")

    except (ImportError, Exception) as e:
        print(f"[IndexTTS2 Compat] ❌ AutoTokenizer 不可用: {e}")
        warnings.warn(f"AutoTokenizer not available: {e}", UserWarning)
        raise
    
    # 5. 模型并行工具
    try:
        from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
        components['assert_device_map'] = assert_device_map
        components['get_device_map'] = get_device_map
        components['parallel_source'] = 'system'
    except ImportError:
        # 简单的回退实现
        def assert_device_map(device_map, num_blocks):
            pass
        
        def get_device_map(num_blocks, range_list):
            return {i: 0 for i in range(num_blocks)}
        
        components['assert_device_map'] = assert_device_map
        components['get_device_map'] = get_device_map
        components['parallel_source'] = 'fallback'
    
    # 6. 特征提取器 - 可选组件，某些版本可能不支持
    try:
        from transformers import SeamlessM4TFeatureExtractor

        # 测试类是否可用
        if hasattr(SeamlessM4TFeatureExtractor, '__init__'):
            components['SeamlessM4TFeatureExtractor'] = SeamlessM4TFeatureExtractor
            components['extractor_source'] = 'system'
            print(f"[IndexTTS2 Compat] ✅ 使用系统特征提取器")
        else:
            raise ImportError("SeamlessM4TFeatureExtractor 不完整")

    except (ImportError, Exception) as e:
        print(f"[IndexTTS2 Compat] ⚠️ SeamlessM4TFeatureExtractor 不可用: {e}")
        warnings.warn(f"SeamlessM4TFeatureExtractor not available: {e}", UserWarning)
        # 不设置这个组件，让调用者处理
    
    return components

# 获取所有组件
_components = get_transformers_components()

# 导出组件
GPT2Config = _components['GPT2Config']
GPT2Model = _components['GPT2Model']
GPT2PreTrainedModel = _components['GPT2PreTrainedModel']
LogitsProcessorList = _components['LogitsProcessorList']
GenerationConfig = _components['GenerationConfig']
GenerationMixin = _components['GenerationMixin']
CausalLMOutputWithCrossAttentions = _components['CausalLMOutputWithCrossAttentions']
AutoTokenizer = _components['AutoTokenizer']
assert_device_map = _components['assert_device_map']
get_device_map = _components['get_device_map']

# 可选组件
if 'SeamlessM4TFeatureExtractor' in _components:
    SeamlessM4TFeatureExtractor = _components['SeamlessM4TFeatureExtractor']

# 打印最终加载状态总结
print(f"\n[IndexTTS2 Compat] 🎯 组件加载完成:")
print(f"  • GPT2: {_components['gpt2_source']}")
print(f"  • Generation: {_components['generation_source']}")
print(f"  • Output: {_components['output_source']}")
print(f"  • Tokenizer: {_components['tokenizer_source']}")
print(f"  • Parallel: {_components['parallel_source']}")
if 'extractor_source' in _components:
    print(f"  • Extractor: {_components['extractor_source']}")
else:
    print(f"  • Extractor: 不可用")

# 版本兼容性总结
try:
    import transformers
    version = transformers.__version__
    from packaging import version as pkg_version
    current_ver = pkg_version.parse(version)
    min_supported = pkg_version.parse("4.35.0")

    if current_ver >= min_supported:
        print(f"[IndexTTS2 Compat] 🎉 transformers {version} 完全支持！")
    else:
        print(f"[IndexTTS2 Compat] ⚠️ transformers {version} 版本过旧，建议升级到 4.35.0+")
except ImportError:
    print(f"[IndexTTS2 Compat] ❌ transformers 未安装")

print(f"[IndexTTS2 Compat] 🚀 IndexTTS2 兼容层初始化完成\n")

# 导出列表
__all__ = [
    'GPT2Config',
    'GPT2Model',
    'GPT2PreTrainedModel',
    'LogitsProcessorList',
    'GenerationConfig',
    'GenerationMixin',
    'CausalLMOutputWithCrossAttentions',
    'AutoTokenizer',
    'assert_device_map',
    'get_device_map',
    'SeamlessM4TFeatureExtractor'  # 可能不存在
]
