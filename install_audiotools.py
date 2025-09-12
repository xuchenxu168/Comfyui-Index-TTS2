#!/usr/bin/env python3
"""
AudioTools Installation Script for IndexTTS2
自动安装 descript-audiotools 的脚本

This script attempts multiple methods to install audiotools:
1. pip install descript-audiotools
2. pip install from GitHub
3. Manual clone and install

Usage:
    python install_audiotools.py
"""

import subprocess
import sys
import os
import tempfile
import shutil

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

def test_audiotools_import():
    """测试 audiotools 是否可以正常导入"""
    try:
        import audiotools
        print("✅ audiotools imported successfully")
        return True
    except ImportError as e:
        print(f"❌ audiotools import failed: {e}")
        return False

def method1_pip_install():
    """方法1: 使用 pip 安装 descript-audiotools"""
    print("\n" + "="*60)
    print("📦 Method 1: Installing descript-audiotools via pip")
    print("="*60)
    
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "descript-audiotools"],
        "Installing descript-audiotools"
    )
    
    if success:
        return test_audiotools_import()
    return False

def method2_github_install():
    """方法2: 从 GitHub 安装"""
    print("\n" + "="*60)
    print("🐙 Method 2: Installing from GitHub")
    print("="*60)
    
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "git+https://github.com/descriptinc/audiotools"],
        "Installing from GitHub"
    )
    
    if success:
        return test_audiotools_import()
    return False

def method3_manual_install():
    """方法3: 手动克隆和安装"""
    print("\n" + "="*60)
    print("🔧 Method 3: Manual clone and install")
    print("="*60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    
    try:
        os.chdir(temp_dir)
        
        # 克隆仓库
        success, output = run_command(
            ["git", "clone", "https://github.com/descriptinc/audiotools.git"],
            "Cloning audiotools repository"
        )
        
        if not success:
            return False
        
        # 进入目录
        os.chdir("audiotools")
        
        # 安装
        success, output = run_command(
            [sys.executable, "-m", "pip", "install", "."],
            "Installing audiotools from source"
        )
        
        if success:
            return test_audiotools_import()
        return False
        
    finally:
        # 清理临时目录
        os.chdir(original_dir)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def main():
    print("🎵 AudioTools Installation Script for IndexTTS2")
    print("=" * 60)
    
    # 首先检查是否已经安装
    print("🔍 Checking if audiotools is already installed...")
    if test_audiotools_import():
        print("🎉 audiotools is already installed and working!")
        return 0
    
    print("📦 audiotools not found, attempting installation...")
    
    # 尝试方法1: pip install
    if method1_pip_install():
        print("\n🎉 Successfully installed audiotools via pip!")
        return 0
    
    # 尝试方法2: GitHub
    if method2_github_install():
        print("\n🎉 Successfully installed audiotools from GitHub!")
        return 0
    
    # 尝试方法3: 手动安装
    if method3_manual_install():
        print("\n🎉 Successfully installed audiotools manually!")
        return 0
    
    # 所有方法都失败
    print("\n" + "="*60)
    print("❌ All installation methods failed!")
    print("="*60)
    print("\n💡 Manual installation suggestions:")
    print("1. Check your internet connection")
    print("2. Ensure git is installed and accessible")
    print("3. Try installing in a fresh virtual environment")
    print("4. Check if you have sufficient permissions")
    print("\n🔗 For more help, visit:")
    print("   https://github.com/descriptinc/audiotools")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
