#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexTTS2 Requirements Installer
智能安装 IndexTTS2 依赖，自动处理 pynini 安装问题
"""

import subprocess
import sys
import os
import platform

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    try:
        print(f"🔧 {description}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败")
        print(f"   错误: {e.stderr}")
        return False

def install_core_requirements():
    """安装核心依赖"""
    print("📦 安装 IndexTTS2 核心依赖...")
    
    # 核心依赖列表（不包含 pynini）
    core_deps = [
        "librosa>=0.10.1",
        "soundfile>=0.12.1", 
        "jieba>=0.42.1",
        "cn2an>=0.5.22",
        "g2p-en>=2.1.0",
        "omegaconf>=2.3.0",
        "munch>=4.0.0",
        "modelscope>=1.27.0"
    ]
    
    failed_deps = []
    
    for dep in core_deps:
        cmd = [sys.executable, '-m', 'pip', 'install', dep]
        if not run_command(cmd, f"安装 {dep.split('>=')[0]}"):
            failed_deps.append(dep)
    
    if failed_deps:
        print(f"\n⚠️  以下依赖安装失败: {', '.join(failed_deps)}")
        return False
    else:
        print("\n✅ 所有核心依赖安装成功！")
        return True

def check_pynini_needed():
    """询问用户是否需要安装 pynini"""
    print("\n" + "="*60)
    print("🤔 关于 pynini (高级文本处理)")
    print("="*60)
    print()
    print("pynini 是一个可选的高级文本处理库，用于：")
    print("• 📞 电话号码标准化: 123-456-7890 → 一二三四五六七八九零")
    print("• 💰 货币格式处理: $29.99 → 二十九美元九十九美分")
    print("• 📅 日期格式转换: 2024年3月15日 → 二零二四年三月十五日")
    print("• 🔢 数字文本转换: Dr. Smith → Doctor Smith")
    print()
    print("⚠️  注意：")
    print("• pynini 在 Windows 上安装困难")
    print("• 包大小约 150MB")
    print("• 大多数用户不需要这些高级功能")
    print("• IndexTTS2 基本功能不依赖 pynini")
    print()

    while True:
        choice = input("是否尝试安装 pynini? (y/n/skip): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            return True
        elif choice in ['n', 'no', '否', 'skip']:
            return False
        else:
            print("请输入 y (是) 或 n (否)")

def check_deepspeed_needed():
    """询问用户是否需要安装 DeepSpeed"""
    print("\n" + "="*60)
    print("⚡ 关于 DeepSpeed (性能加速)")
    print("="*60)
    print()
    print("DeepSpeed 是一个深度学习优化库，可以：")
    print("• 🚀 显著提升推理速度 (2-5倍)")
    print("• 💾 优化 GPU 内存使用")
    print("• 🔧 自动模型并行和内存管理")
    print("• 🎛️ 支持多种优化策略")
    print()
    print("⚠️  注意：")
    print("• Windows 需要使用社区轮子文件")
    print("• 需要兼容的 CUDA 版本")
    print("• 主要适用于大模型和多GPU环境")
    print("• IndexTTS2 基本功能不依赖 DeepSpeed")
    print()
    print("🔗 Windows 轮子下载: https://github.com/6Morpheus6/deepspeed-windows-wheels/releases")
    print()

    while True:
        choice = input("是否尝试安装 DeepSpeed? (y/n/skip): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            return True
        elif choice in ['n', 'no', '否', 'skip']:
            return False
        else:
            print("请输入 y (是) 或 n (否)")

def install_deepspeed():
    """尝试安装 DeepSpeed"""
    print("\n⚡ 尝试安装 DeepSpeed...")

    # 检查是否已安装
    try:
        import deepspeed
        print("✅ DeepSpeed 已安装")
        return True
    except ImportError:
        pass

    # Windows 系统提示手动安装
    if platform.system() == "Windows":
        print("🪟 检测到 Windows 系统")
        print("💡 DeepSpeed 在 Windows 上需要手动安装轮子文件")
        print("🔗 请访问: https://github.com/6Morpheus6/deepspeed-windows-wheels/releases")
        print("📋 下载适合您 Python 版本的轮子文件，然后使用:")
        print("   pip install [下载的轮子文件名].whl")
        return False

    # Linux/macOS 尝试直接安装
    print("🐧 检测到 Linux/macOS 系统，尝试直接安装...")
    cmd = [sys.executable, '-m', 'pip', 'install', 'deepspeed']
    if run_command(cmd, "安装 DeepSpeed"):
        return True

    print("❌ DeepSpeed 安装失败")
    print("💡 这是正常的，您仍然可以使用 IndexTTS2 的基本功能")
    return False

def install_pynini_with_wheel():
    """使用项目提供的轮子文件安装 pynini"""
    print("🎯 尝试使用项目提供的轮子文件...")

    # 检测 Python 版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # 轮子文件映射
    wheel_files = {
        "3.10": "pynini-wheel/pynini-2.1.6.post1-cp310-cp310-win_amd64.whl",
        "3.11": "pynini-wheel/pynini-2.1.6.post1-cp311-cp311-win_amd64.whl",
        "3.12": "pynini-wheel/pynini-2.1.6.post1-cp312-cp312-win_amd64.whl",
        "3.13": "pynini-wheel/pynini-2.1.6.post1-cp313-cp313-win_amd64.whl"
    }

    if python_version not in wheel_files:
        print(f"⚠️  暂不支持 Python {python_version} 的轮子文件")
        return False

    wheel_file = wheel_files[python_version]

    # 检查轮子文件是否存在
    if not os.path.exists(wheel_file):
        print(f"⚠️  轮子文件不存在: {wheel_file}")
        return False

    print(f"📦 找到 Python {python_version} 轮子文件")
    cmd = [sys.executable, '-m', 'pip', 'install', wheel_file]
    return run_command(cmd, f"安装轮子文件 {wheel_file}")

def install_pynini():
    """尝试安装 pynini"""
    print("\n🔧 尝试安装 pynini...")

    # 检查是否已安装
    try:
        import pynini
        print("✅ pynini 已安装")
        return True
    except ImportError:
        pass

    # Windows 系统优先尝试轮子文件
    if platform.system() == "Windows":
        print("🪟 检测到 Windows 系统，优先尝试轮子文件")
        if install_pynini_with_wheel():
            return True

    # 检查 conda
    try:
        result = subprocess.run(['conda', '--version'],
                              capture_output=True, text=True, timeout=5)
        has_conda = result.returncode == 0
    except:
        has_conda = False

    if has_conda:
        print("📦 检测到 conda，推荐使用 conda 安装")
        choice = input("使用 conda 安装 pynini? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            cmd = ['conda', 'install', '-c', 'conda-forge', 'pynini=2.1.6', '-y']
            if run_command(cmd, "使用 conda 安装 pynini"):
                return True

    # 尝试 pip 安装
    print("🔄 尝试使用 pip 安装...")
    pip_methods = [
        ([sys.executable, '-m', 'pip', 'install', 'pynini==2.1.6'], "pip 安装 pynini 2.1.6"),
        ([sys.executable, '-m', 'pip', 'install', 'pynini'], "pip 安装最新版 pynini"),
    ]

    for cmd, desc in pip_methods:
        if run_command(cmd, desc):
            return True

    print("❌ pynini 安装失败")
    print("💡 这是正常的，您仍然可以使用 IndexTTS2 的基本功能")
    return False

def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    required_modules = [
        'librosa', 'soundfile', 'jieba', 'cn2an', 
        'g2p_en', 'omegaconf', 'munch', 'modelscope'
    ]
    
    failed_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_modules.append(module)
    
    # 检查 pynini (可选)
    try:
        import pynini
        print("✅ pynini (可选)")
    except ImportError:
        print("⚠️  pynini (可选) - 未安装，基本功能不受影响")

    # 检查 DeepSpeed (可选)
    try:
        import deepspeed
        print("✅ DeepSpeed (可选)")
    except ImportError:
        print("⚠️  DeepSpeed (可选) - 未安装，基本功能不受影响")
    
    if failed_modules:
        print(f"\n❌ 安装验证失败，缺少模块: {', '.join(failed_modules)}")
        return False
    else:
        print("\n🎉 安装验证成功！IndexTTS2 已准备就绪")
        return True

def main():
    print("🎯 IndexTTS2 依赖安装器")
    print("="*50)
    print(f"🖥️  系统: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print()
    
    # 安装核心依赖
    if not install_core_requirements():
        print("❌ 核心依赖安装失败，请检查网络连接和权限")
        return
    
    # 询问是否安装 pynini
    if check_pynini_needed():
        install_pynini()
    else:
        print("⏭️  跳过 pynini 安装")

    # 询问是否安装 DeepSpeed
    if check_deepspeed_needed():
        install_deepspeed()
    else:
        print("⏭️  跳过 DeepSpeed 安装")
    
    # 验证安装
    if verify_installation():
        print("\n🚀 安装完成！您现在可以使用 IndexTTS2 了")
        print("📖 查看 README.md 了解使用方法")
    else:
        print("\n⚠️  安装可能存在问题，请检查错误信息")

if __name__ == "__main__":
    main()
