#!/usr/bin/env python3
"""
ComfyUI IndexTTS2 依赖检查工具
Dependency checker for ComfyUI IndexTTS2

使用方法 / Usage:
python check_all_dependencies.py
"""

import sys
import importlib
import subprocess
import platform
from typing import Dict, List, Tuple

def check_python_version() -> bool:
    """检查 Python 版本"""
    version = sys.version_info
    print(f"🐍 Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print("✅ Python 版本兼容")
        return True
    else:
        print("❌ Python 版本不兼容，建议使用 Python 3.8-3.11")
        return False

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, version
    except ImportError:
        return False, "Not installed"

def get_core_dependencies() -> Dict[str, str]:
    """获取核心依赖列表"""
    return {
        'torch': 'torch',
        'torchaudio': 'torchaudio',
        'numpy': 'numpy',
        'transformers': 'transformers',
        'tokenizers': 'tokenizers',
        'accelerate': 'accelerate',
        'sentencepiece': 'sentencepiece',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'scipy': 'scipy',
        'omegaconf': 'omegaconf',
        'einops': 'einops',
        'safetensors': 'safetensors',
        'jieba': 'jieba',
        'cn2an': 'cn2an',
        'g2p_en': 'g2p_en',
        'requests': 'requests',
        'tqdm': 'tqdm',
    }

def get_optional_dependencies() -> Dict[str, str]:
    """获取可选依赖列表"""
    return {
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'gradio': 'gradio',
        'cv2': 'cv2',
        'tensorboard': 'tensorboard',
        'ffmpeg': 'ffmpeg',
        'Cython': 'Cython',
        'numba': 'numba',
        'modelscope': 'modelscope',
    }

def check_platform_specific() -> Dict[str, bool]:
    """检查平台特定依赖"""
    system = platform.system()
    results = {}
    
    if system == "Darwin":  # macOS
        installed, _ = check_package('wetext', 'wetext')
        results['wetext (macOS)'] = installed
    else:  # Linux/Windows
        installed, _ = check_package('WeTextProcessing', 'WeTextProcessing')
        results['WeTextProcessing (Linux/Windows)'] = installed
    
    return results

def check_gpu_support() -> Dict[str, bool]:
    """检查 GPU 支持"""
    results = {}
    
    # 检查 CUDA
    try:
        import torch
        results['CUDA Available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results[f'CUDA Version'] = torch.version.cuda
            results[f'GPU Count'] = torch.cuda.device_count()
            results[f'GPU Name'] = torch.cuda.get_device_name(0)
    except:
        results['CUDA Available'] = False
    
    return results

def install_missing_packages(missing_packages: List[str]) -> None:
    """安装缺失的包"""
    if not missing_packages:
        print("✅ 所有核心依赖都已安装")
        return
    
    print(f"\n📦 发现 {len(missing_packages)} 个缺失的核心依赖:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    
    response = input("\n是否自动安装缺失的依赖? (y/n): ").lower().strip()
    if response == 'y':
        try:
            cmd = [sys.executable, '-m', 'pip', 'install'] + missing_packages
            print(f"\n🔄 执行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("✅ 安装完成!")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败: {e}")
            print("请手动安装缺失的依赖")

def main():
    """主函数"""
    print("=" * 60)
    print("🔍 ComfyUI IndexTTS2 依赖检查工具")
    print("=" * 60)
    
    # 检查 Python 版本
    python_ok = check_python_version()
    print()
    
    # 检查核心依赖
    print("📋 检查核心依赖:")
    core_deps = get_core_dependencies()
    missing_core = []
    
    for package, import_name in core_deps.items():
        installed, version = check_package(package, import_name)
        status = "✅" if installed else "❌"
        print(f"  {status} {package:<20} {version}")
        if not installed:
            missing_core.append(package)
    
    print()
    
    # 检查可选依赖
    print("🎨 检查可选依赖:")
    optional_deps = get_optional_dependencies()
    missing_optional = []
    
    for package, import_name in optional_deps.items():
        installed, version = check_package(package, import_name)
        status = "✅" if installed else "⚠️"
        print(f"  {status} {package:<20} {version}")
        if not installed:
            missing_optional.append(package)
    
    print()
    
    # 检查平台特定依赖
    print("🖥️  检查平台特定依赖:")
    platform_deps = check_platform_specific()
    for name, installed in platform_deps.items():
        status = "✅" if installed else "❌"
        print(f"  {status} {name}")
    
    print()
    
    # 检查 GPU 支持
    print("🚀 检查 GPU 支持:")
    gpu_info = check_gpu_support()
    for name, value in gpu_info.items():
        if isinstance(value, bool):
            status = "✅" if value else "❌"
            print(f"  {status} {name}")
        else:
            print(f"  ℹ️  {name}: {value}")
    
    print()
    
    # 总结
    print("📊 检查总结:")
    print(f"  核心依赖: {len(core_deps) - len(missing_core)}/{len(core_deps)} 已安装")
    print(f"  可选依赖: {len(optional_deps) - len(missing_optional)}/{len(optional_deps)} 已安装")
    
    if missing_core:
        print(f"  ❌ 缺失核心依赖: {len(missing_core)} 个")
        install_missing_packages(missing_core)
    else:
        print("  ✅ 所有核心依赖都已安装")
    
    if missing_optional:
        print(f"  ⚠️  缺失可选依赖: {len(missing_optional)} 个")
        print("     可选依赖不影响基本功能，但可能影响某些高级特性")
    
    print()
    
    # 安装建议
    if missing_core or missing_optional:
        print("💡 安装建议:")
        print("  最小安装: pip install -r requirements_minimal.txt")
        print("  标准安装: pip install -r requirements.txt")
        print("  完整安装: pip install -r requirements_full.txt")
    
    print("\n" + "=" * 60)
    print("检查完成!")

if __name__ == "__main__":
    main()
