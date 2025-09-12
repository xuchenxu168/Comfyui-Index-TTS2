#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformers Compatibility Checker for IndexTTS2
检查 transformers 库的兼容性
"""

import sys
import importlib.util

def check_transformers_version():
    """检查 transformers 版本"""
    try:
        import transformers
        version = transformers.__version__
        print(f"✅ transformers 版本: {version}")
        return version
    except ImportError:
        print("❌ transformers 未安装")
        return None

def check_cache_utils():
    """检查 cache_utils 模块的可用性"""
    print("\n🔍 检查 cache_utils 模块...")
    
    try:
        from transformers.cache_utils import Cache, DynamicCache
        print("✅ 基础缓存类可用")
    except ImportError as e:
        print(f"❌ 基础缓存类不可用: {e}")
        return False
    
    # 检查 QuantizedCacheConfig
    try:
        from transformers.cache_utils import QuantizedCacheConfig
        print("✅ QuantizedCacheConfig 可用")
    except ImportError:
        print("⚠️  QuantizedCacheConfig 不可用 (可能是较旧版本)")
    
    return True

def check_generation_config():
    """检查 generation configuration 的可用性"""
    print("\n🔍 检查 generation configuration...")

    try:
        from transformers.generation.configuration_utils import GenerationConfig
        print("✅ GenerationConfig 可用")
    except ImportError as e:
        print(f"❌ GenerationConfig 不可用: {e}")
        return False

    # 检查 GenerationMode
    try:
        from transformers.generation.configuration_utils import GenerationMode
        print("✅ GenerationMode 可用")
    except ImportError:
        print("⚠️  GenerationMode 不可用 (可能是较旧版本)")

    # 检查 QUANT_BACKEND_CLASSES_MAPPING
    try:
        from transformers.generation.configuration_utils import QUANT_BACKEND_CLASSES_MAPPING
        print("✅ QUANT_BACKEND_CLASSES_MAPPING 可用")
    except ImportError:
        print("⚠️  QUANT_BACKEND_CLASSES_MAPPING 不可用 (可能是较旧版本)")

    return True

def check_model_loading():
    """测试模型加载相关的导入"""
    print("\n🔍 检查模型加载相关模块...")
    
    try:
        from transformers.modeling_outputs import CausalLMOutputWithPast
        print("✅ CausalLMOutputWithPast 可用")
    except ImportError as e:
        print(f"❌ CausalLMOutputWithPast 不可用: {e}")
        return False
    
    try:
        from transformers.pytorch_utils import isin_mps_friendly
        print("✅ isin_mps_friendly 可用")
    except ImportError:
        print("⚠️  isin_mps_friendly 不可用 (可能是较旧版本)")
    
    return True

def get_recommended_version():
    """获取推荐的 transformers 版本"""
    return "4.36.0"

def suggest_fixes(version):
    """根据版本问题建议修复方案"""
    print("\n" + "="*60)
    print("🔧 修复建议")
    print("="*60)
    
    if version is None:
        print("📦 安装 transformers:")
        print("   pip install transformers>=4.36.0")
        return
    
    # 解析版本号
    try:
        from packaging import version as pkg_version
        current_version = pkg_version.parse(version)
        recommended_version = pkg_version.parse(get_recommended_version())
        
        if current_version < recommended_version:
            print(f"⬆️  升级 transformers (当前: {version}, 推荐: {get_recommended_version()}):")
            print(f"   pip install transformers>={get_recommended_version()}")
        elif current_version > pkg_version.parse("4.45.0"):
            print(f"⬇️  降级 transformers (当前: {version} 可能太新):")
            print("   pip install transformers==4.36.2")
        else:
            print("✅ transformers 版本看起来合适")
            print("💡 如果仍有问题，尝试重新安装:")
            print("   pip uninstall transformers")
            print("   pip install transformers==4.36.2")
            
    except ImportError:
        print("⚠️  无法解析版本，建议重新安装:")
        print("   pip uninstall transformers")
        print("   pip install transformers==4.36.2")

def main():
    print("🎯 IndexTTS2 Transformers 兼容性检查")
    print("="*50)
    
    # 检查 transformers 版本
    version = check_transformers_version()
    
    # 检查各个模块
    cache_ok = check_cache_utils()
    config_ok = check_generation_config()
    model_ok = check_model_loading()
    
    # 总结结果
    print("\n" + "="*60)
    print("📊 检查结果总结")
    print("="*60)
    
    if cache_ok and config_ok and model_ok:
        print("🎉 所有检查通过！transformers 兼容性良好")
        print("✅ IndexTTS2 应该可以正常工作")
    else:
        print("⚠️  发现兼容性问题")
        suggest_fixes(version)
    
    print("\n💡 其他建议:")
    print("• 确保在正确的 Python 环境中运行")
    print("• 如果使用 conda，尝试: conda install transformers")
    print("• 重启 ComfyUI 以确保更改生效")

if __name__ == "__main__":
    main()
