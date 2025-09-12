#!/usr/bin/env python3
"""
Advanced WeTextProcessing Installation Script
高级 WeTextProcessing 安装脚本 - 解决 pynini 版本冲突问题

This script addresses the common issue where WeTextProcessing tries to 
recompile pynini even when it's already installed.
"""

import subprocess
import sys
import os

def run_command(cmd, description="", ignore_errors=False):
    """运行命令并返回结果"""
    print(f"🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=not ignore_errors
        )
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True, result.stdout
        else:
            print(f"❌ {description} - Failed")
            print(f"   Error: {result.stderr[:500]}")
            return False, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"   Error: {e.stderr[:500] if e.stderr else str(e)}")
        return False, str(e)
    except Exception as e:
        print(f"❌ {description} - Exception: {str(e)}")
        return False, str(e)

def check_pynini_installed():
    """检查 pynini 是否已安装"""
    try:
        import pynini
        version = getattr(pynini, '__version__', 'unknown')
        print(f"✅ pynini already installed: version {version}")
        return True, version
    except ImportError:
        print("❌ pynini not installed")
        return False, None

def method1_install_specific_pynini_version():
    """方法1: 安装 WeTextProcessing 要求的特定 pynini 版本"""
    print("\n" + "="*60)
    print("📦 Method 1: Install specific pynini version required by WeTextProcessing")
    print("="*60)
    
    # 先安装 WeTextProcessing 要求的 pynini 版本
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "pynini==2.1.6"],
        "Installing pynini==2.1.6 (required by WeTextProcessing)"
    )
    
    if not success:
        return False
    
    # 然后安装 WeTextProcessing
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "WeTextProcessing"],
        "Installing WeTextProcessing"
    )
    
    return success

def method2_install_with_no_deps():
    """方法2: 使用 --no-deps 跳过依赖检查"""
    print("\n" + "="*60)
    print("📦 Method 2: Install WeTextProcessing with --no-deps")
    print("="*60)
    
    # 确保 pynini 已安装
    pynini_installed, version = check_pynini_installed()
    if not pynini_installed:
        print("⚠️ pynini not installed, installing first...")
        success, output = run_command(
            [sys.executable, "-m", "pip", "install", "pynini"],
            "Installing pynini"
        )
        if not success:
            return False
    
    # 使用 --no-deps 安装 WeTextProcessing
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "WeTextProcessing", "--no-deps"],
        "Installing WeTextProcessing without dependency checking"
    )
    
    return success

def method3_install_precompiled():
    """方法3: 尝试安装预编译轮子"""
    print("\n" + "="*60)
    print("📦 Method 3: Install precompiled wheels only")
    print("="*60)
    
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "WeTextProcessing", "--only-binary=all"],
        "Installing WeTextProcessing (precompiled wheels only)",
        ignore_errors=True
    )
    
    return success

def method4_manual_dependency_install():
    """方法4: 手动安装所有依赖"""
    print("\n" + "="*60)
    print("📦 Method 4: Manual dependency installation")
    print("="*60)
    
    # WeTextProcessing 的主要依赖
    dependencies = [
        "pynini==2.1.6",
        "protobuf",
        "six",
        "setuptools",
    ]
    
    for dep in dependencies:
        print(f"\n🔄 Installing dependency: {dep}")
        success, output = run_command(
            [sys.executable, "-m", "pip", "install", dep],
            f"Installing {dep}",
            ignore_errors=True
        )
        if not success:
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    # 最后安装 WeTextProcessing
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "WeTextProcessing", "--no-deps"],
        "Installing WeTextProcessing without deps"
    )
    
    return success

def test_wetextprocessing():
    """测试 WeTextProcessing 是否正常工作"""
    try:
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer
        
        print("✅ WeTextProcessing imported successfully")
        
        # 简单功能测试
        zh_normalizer = ZhNormalizer()
        en_normalizer = EnNormalizer()
        
        zh_test = zh_normalizer.normalize("我有100元")
        en_test = en_normalizer.normalize("I have $100")
        
        print(f"✅ Chinese test: '我有100元' -> '{zh_test}'")
        print(f"✅ English test: 'I have $100' -> '{en_test}'")
        
        return True
    except Exception as e:
        print(f"❌ WeTextProcessing test failed: {e}")
        return False

def main():
    print("🔧 Advanced WeTextProcessing Installation")
    print("=" * 60)
    print("This script addresses pynini version conflicts when installing WeTextProcessing")
    print("=" * 60)
    
    # 检查当前状态
    print("🔍 Checking current installation status...")
    pynini_installed, pynini_version = check_pynini_installed()
    
    try:
        from tn.chinese.normalizer import Normalizer
        print("✅ WeTextProcessing already installed and working!")
        return 0
    except ImportError:
        print("❌ WeTextProcessing not installed or not working")
    
    print(f"\n📋 Installation Plan:")
    print("We will try multiple methods to resolve the pynini conflict:")
    print("1. Install specific pynini version (2.1.6)")
    print("2. Install with --no-deps")
    print("3. Install precompiled wheels only")
    print("4. Manual dependency installation")
    
    # 尝试各种方法
    methods = [
        ("Method 1: Specific pynini version", method1_install_specific_pynini_version),
        ("Method 2: No deps installation", method2_install_with_no_deps),
        ("Method 3: Precompiled wheels", method3_install_precompiled),
        ("Method 4: Manual dependencies", method4_manual_dependency_install),
    ]
    
    for method_name, method_func in methods:
        print(f"\n🚀 Trying {method_name}...")
        
        if method_func():
            print(f"\n🎉 {method_name} succeeded!")
            if test_wetextprocessing():
                print("\n✅ WeTextProcessing installation completed successfully!")
                return 0
            else:
                print(f"\n⚠️ {method_name} installed but not working properly, trying next method...")
        else:
            print(f"\n❌ {method_name} failed, trying next method...")
    
    # 所有方法都失败
    print("\n" + "="*60)
    print("❌ All installation methods failed!")
    print("="*60)
    print("\n💡 Recommendations:")
    print("1. Use IndexTTS2's built-in fallback mechanism (recommended)")
    print("2. Try installing in a fresh virtual environment")
    print("3. Install Visual Studio Build Tools (Windows)")
    print("4. Use Linux environment where pynini compiles more reliably")
    print("\n🔗 IndexTTS2 will work without WeTextProcessing using basic text processing.")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
