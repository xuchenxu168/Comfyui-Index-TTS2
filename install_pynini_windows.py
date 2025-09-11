#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows Pynini Installation Helper
为 Windows 用户提供 pynini 安装解决方案
"""

import subprocess
import sys
import os
import platform

def check_conda_available():
    """检查是否有 conda 环境"""
    try:
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def check_pynini_installed():
    """检查 pynini 是否已安装"""
    try:
        import pynini
        print("✅ pynini 已安装，版本:", pynini.__version__)
        return True
    except ImportError:
        print("❌ pynini 未安装")
        return False

def install_with_conda():
    """使用 conda 安装 pynini"""
    print("🔧 尝试使用 conda 安装 pynini...")
    try:
        cmd = ['conda', 'install', '-c', 'conda-forge', 'pynini=2.1.6', '-y']
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ conda 安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ conda 安装失败: {e}")
        return False
    except Exception as e:
        print(f"❌ conda 安装出错: {e}")
        return False

def install_with_wheel():
    """使用项目提供的轮子文件安装 pynini"""
    print("🎯 尝试使用项目提供的轮子文件安装 pynini...")

    # 检测 Python 版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"🐍 检测到 Python 版本: {python_version}")

    # 轮子文件映射
    wheel_files = {
        "3.10": "pynini-wheel/pynini-2.1.6.post1-cp310-cp310-win_amd64.whl",
        "3.11": "pynini-wheel/pynini-2.1.6.post1-cp311-cp311-win_amd64.whl",
        "3.12": "pynini-wheel/pynini-2.1.6.post1-cp312-cp312-win_amd64.whl",
        "3.13": "pynini-wheel/pynini-2.1.6.post1-cp313-cp313-win_amd64.whl"
    }

    if python_version not in wheel_files:
        print(f"❌ 暂不支持 Python {python_version}，目前支持: {', '.join(wheel_files.keys())}")
        return False

    wheel_file = wheel_files[python_version]

    # 检查轮子文件是否存在
    if not os.path.exists(wheel_file):
        print(f"❌ 轮子文件不存在: {wheel_file}")
        return False

    print(f"📦 找到轮子文件: {wheel_file}")

    try:
        cmd = [sys.executable, '-m', 'pip', 'install', wheel_file]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 轮子文件安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 轮子文件安装失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 轮子文件安装出错: {e}")
        return False

def install_with_pip():
    """使用 pip 安装 pynini"""
    print("🔧 尝试使用 pip 安装 pynini...")
    try:
        # 尝试不同的安装方法
        methods = [
            [sys.executable, '-m', 'pip', 'install', 'pynini==2.1.6'],
            [sys.executable, '-m', 'pip', 'install', 'pynini'],
            [sys.executable, '-m', 'pip', 'install', '--only-binary=all', 'pynini'],
        ]

        for i, cmd in enumerate(methods, 1):
            print(f"   尝试方法 {i}: {' '.join(cmd[3:])}")
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("✅ pip 安装成功！")
                return True
            except subprocess.CalledProcessError:
                print(f"   方法 {i} 失败，尝试下一个...")
                continue

        print("❌ 所有 pip 安装方法都失败了")
        return False
    except Exception as e:
        print(f"❌ pip 安装出错: {e}")
        return False

def test_pynini():
    """测试 pynini 功能"""
    try:
        import pynini
        # 简单测试
        rule = pynini.string_map([('$', 'dollar')])
        result = pynini.compose('$5', rule)
        print("✅ pynini 功能测试通过")
        return True
    except Exception as e:
        print(f"❌ pynini 功能测试失败: {e}")
        return False

def show_alternatives():
    """显示替代方案"""
    print("\n" + "="*60)
    print("🔄 pynini 安装失败的替代方案：")
    print("="*60)
    print()
    print("1. 📦 使用 WSL (Windows Subsystem for Linux):")
    print("   - 在 WSL 中安装 Linux 版本的 pynini")
    print("   - WSL 中有预编译的轮子可用")
    print()
    print("2. 🐳 使用 Docker:")
    print("   - 在 Docker 容器中运行 IndexTTS2")
    print("   - 容器中可以轻松安装 pynini")
    print()
    print("3. ⏭️ 跳过 pynini:")
    print("   - IndexTTS2 的基本功能不需要 pynini")
    print("   - 只有复杂文本处理才需要 pynini")
    print("   - 大多数用户可以正常使用")
    print()
    print("4. 🔧 手动编译 (高级用户):")
    print("   - 安装 Visual Studio Build Tools")
    print("   - 编译 OpenFst 和 pynini")
    print("   - 需要 C++ 编译经验")
    print()
    print("推荐：如果您只是普通使用，可以跳过 pynini 安装")

def main():
    print("🎯 Windows Pynini 安装助手")
    print("="*50)
    print()
    
    # 检查系统信息
    print(f"🖥️  操作系统: {platform.system()} {platform.release()}")
    print(f"🐍 Python 版本: {sys.version}")
    print()
    
    # 检查是否已安装
    if check_pynini_installed():
        print("🎉 pynini 已经安装，无需重复安装！")
        if test_pynini():
            print("✅ 安装验证完成")
            return
    
    print("🔍 检查安装环境...")
    
    # 检查 conda
    has_conda = check_conda_available()
    print(f"📦 Conda 可用: {'✅ 是' if has_conda else '❌ 否'}")
    
    # 尝试安装
    success = False

    # 优先尝试轮子文件
    print("\n🎯 首先尝试使用项目提供的轮子文件 (推荐)")
    success = install_with_wheel()
    if success:
        success = check_pynini_installed() and test_pynini()

    if not success and has_conda:
        print("\n🚀 尝试使用 conda 安装")
        choice = input("是否使用 conda 安装? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            success = install_with_conda()
            if success:
                success = check_pynini_installed() and test_pynini()

    if not success:
        print("\n🔄 尝试使用 pip 安装...")
        success = install_with_pip()
        if success:
            success = check_pynini_installed() and test_pynini()
    
    if success:
        print("\n🎉 pynini 安装成功！")
        print("✅ IndexTTS2 现在可以使用高级文本处理功能")
    else:
        print("\n❌ pynini 安装失败")
        show_alternatives()
        print("\n💡 提示：您仍然可以使用 IndexTTS2 的基本功能")

if __name__ == "__main__":
    main()
