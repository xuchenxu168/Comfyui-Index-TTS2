#!/usr/bin/env python3
"""
WeTextProcessing Installation Script for IndexTTS2
自动安装 WeTextProcessing 的脚本

WeTextProcessing provides text normalization (tn) functionality
required for IndexTTS2's Chinese and English text processing.

Usage:
    python install_wetextprocessing.py
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """运行命令并返回结果"""
    print(f"🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ {description} - Success")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"   Error: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        print(f"❌ {description} - Exception: {str(e)}")
        return False, str(e)

def test_wetextprocessing_import():
    """测试 WeTextProcessing 是否可以正常导入"""
    try:
        # 测试主要的导入
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer
        print("✅ WeTextProcessing (tn) imported successfully")
        
        # 简单功能测试
        zh_normalizer = ZhNormalizer()
        en_normalizer = EnNormalizer()
        
        # 测试中文文本标准化
        zh_test = zh_normalizer.normalize("我有100元")
        print(f"✅ Chinese normalization test: '我有100元' -> '{zh_test}'")
        
        # 测试英文文本标准化
        en_test = en_normalizer.normalize("I have $100")
        print(f"✅ English normalization test: 'I have $100' -> '{en_test}'")
        
        return True
    except ImportError as e:
        print(f"❌ WeTextProcessing import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ WeTextProcessing test failed: {e}")
        return False

def install_wetextprocessing():
    """安装 WeTextProcessing"""
    print("\n" + "="*60)
    print("📦 Installing WeTextProcessing")
    print("="*60)
    
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "WeTextProcessing>=1.0.3"],
        "Installing WeTextProcessing"
    )
    
    if success:
        return test_wetextprocessing_import()
    return False

def install_with_dependencies():
    """安装 WeTextProcessing 及其依赖"""
    print("\n" + "="*60)
    print("📦 Installing WeTextProcessing with dependencies")
    print("="*60)
    
    # 先安装可能需要的依赖
    dependencies = [
        "pynini",  # 可能需要的依赖
        "protobuf",  # 通常需要的依赖
    ]
    
    for dep in dependencies:
        print(f"\n🔄 Installing dependency: {dep}")
        success, output = run_command(
            [sys.executable, "-m", "pip", "install", dep],
            f"Installing {dep}"
        )
        if not success:
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    # 安装 WeTextProcessing
    return install_wetextprocessing()

def main():
    print("📝 WeTextProcessing Installation Script for IndexTTS2")
    print("=" * 60)
    print("WeTextProcessing provides text normalization (tn) functionality")
    print("required for IndexTTS2's Chinese and English text processing.")
    print("=" * 60)
    
    # 首先检查是否已经安装
    print("🔍 Checking if WeTextProcessing is already installed...")
    if test_wetextprocessing_import():
        print("🎉 WeTextProcessing is already installed and working!")
        return 0
    
    print("📦 WeTextProcessing not found, attempting installation...")
    
    # 尝试直接安装
    if install_wetextprocessing():
        print("\n🎉 Successfully installed WeTextProcessing!")
        return 0
    
    # 尝试安装依赖后再安装
    print("\n🔄 Trying installation with dependencies...")
    if install_with_dependencies():
        print("\n🎉 Successfully installed WeTextProcessing with dependencies!")
        return 0
    
    # 安装失败
    print("\n" + "="*60)
    print("❌ WeTextProcessing installation failed!")
    print("="*60)
    print("\n💡 Manual installation suggestions:")
    print("1. Install pynini first (if not already installed):")
    print("   pip install pynini")
    print("2. Install WeTextProcessing:")
    print("   pip install WeTextProcessing")
    print("3. Check for system dependencies (C++ compiler, etc.)")
    print("4. Try installing in a fresh virtual environment")
    print("\n🔗 For more help, visit:")
    print("   https://github.com/wenet-e2e/WeTextProcessing")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
