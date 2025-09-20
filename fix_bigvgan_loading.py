#!/usr/bin/env python3
"""
BigVGAN加载问题快速修复脚本
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def check_environment():
    """检查环境设置"""
    print("检查环境设置...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    required_packages = [
        "torch",
        "huggingface_hub",
        "transformers",
        "requests"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少必要的包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def setup_huggingface_mirror():
    """设置HuggingFace镜像"""
    print("\n设置HuggingFace镜像...")
    
    # 设置环境变量
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("✓ 设置HF_ENDPOINT=https://hf-mirror.com")
    
    # 创建.bashrc或.bat文件
    home_dir = Path.home()
    
    if os.name == 'nt':  # Windows
        batch_file = home_dir / "set_hf_mirror.bat"
        with open(batch_file, 'w') as f:
            f.write("@echo off\n")
            f.write("set HF_ENDPOINT=https://hf-mirror.com\n")
            f.write("echo HuggingFace镜像已设置\n")
        print(f"✓ 创建批处理文件: {batch_file}")
    else:  # Linux/Mac
        bashrc_file = home_dir / ".bashrc"
        with open(bashrc_file, 'a') as f:
            f.write("\n# HuggingFace镜像设置\n")
            f.write("export HF_ENDPOINT=https://hf-mirror.com\n")
        print(f"✓ 更新.bashrc文件: {bashrc_file}")

def check_local_bigvgan():
    """检查本地BigVGAN模型"""
    print("\n检查本地BigVGAN模型...")
    
    # 查找ComfyUI模型目录
    from pathlib import Path
    
    # 通过当前文件路径推断ComfyUI根目录
    current_file = Path(__file__).resolve()
    comfyui_models = None
    
    for parent in current_file.parents:
        if parent.name == "ComfyUI" and (parent / "models").exists():
            comfyui_models = parent / "models"
            break
        if (parent / "ComfyUI" / "models").exists():
            comfyui_models = parent / "ComfyUI" / "models"
            break
    
    if not comfyui_models:
        print("✗ 未找到ComfyUI模型目录")
        return False
    
    print(f"✓ ComfyUI模型目录: {comfyui_models}")
    
    # 检查可能的BigVGAN位置
    possible_locations = [
        comfyui_models / "tts" / "IndexTTS-2" / "external_models",
        comfyui_models / "tts" / "IndexTTS-2" / "external_models" / "bigvgan",
    ]
    
    for location in possible_locations:
        if location.exists():
            config_file = location / "config.json"
            model_file = location / "bigvgan_generator.pt"
            
            if config_file.exists() and model_file.exists():
                print(f"✓ 找到本地BigVGAN模型: {location}")
                print(f"  - config.json: {config_file}")
                print(f"  - bigvgan_generator.pt: {model_file}")
                
                # 检查文件大小
                config_size = config_file.stat().st_size / 1024  # KB
                model_size = model_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  - config.json大小: {config_size:.2f} KB")
                print(f"  - bigvgan_generator.pt大小: {model_size:.2f} MB")
                
                return True
    
    print("✗ 未找到本地BigVGAN模型文件")
    return False

def test_bigvgan_download():
    """测试BigVGAN下载"""
    print("\n测试BigVGAN模型下载...")
    
    # 先检查本地是否有模型
    if check_local_bigvgan():
        print("✓ 本地已有BigVGAN模型，跳过下载测试")
        return True
    
    try:
        from huggingface_hub import hf_hub_download
        
        model_id = "nvidia/bigvgan_v2_22khz_80band_256x"
        cache_dir = "./test_bigvgan_cache"
        
        print(f"下载模型: {model_id}")
        print(f"缓存目录: {cache_dir}")
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 下载配置文件
        print("下载配置文件...")
        config_file = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        print(f"✓ 配置文件: {config_file}")
        
        # 下载模型文件
        print("下载模型文件...")
        model_file = hf_hub_download(
            repo_id=model_id,
            filename="bigvgan_generator.pt",
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        print(f"✓ 模型文件: {model_file}")
        
        # 检查文件大小
        file_size = os.path.getsize(model_file) / (1024 * 1024)
        print(f"✓ 文件大小: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

def create_manual_download_script():
    """创建手动下载脚本"""
    print("\n创建手动下载脚本...")
    
    script_content = '''#!/bin/bash
# BigVGAN手动下载脚本

echo "开始手动下载BigVGAN模型..."

# 创建目录
mkdir -p ./bigvgan_model

# 下载配置文件
echo "下载配置文件..."
wget -O ./bigvgan_model/config.json https://hf-mirror.com/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/config.json

# 下载模型文件
echo "下载模型文件..."
wget -O ./bigvgan_model/bigvgan_generator.pt https://hf-mirror.com/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/bigvgan_generator.pt

echo "下载完成！"
echo "模型文件位置: ./bigvgan_model/"
'''
    
    with open("download_bigvgan.sh", "w") as f:
        f.write(script_content)
    
    # 创建Windows批处理文件
    bat_content = '''@echo off
echo 开始手动下载BigVGAN模型...

REM 创建目录
mkdir bigvgan_model 2>nul

REM 下载配置文件
echo 下载配置文件...
curl -L -o bigvgan_model\\config.json https://hf-mirror.com/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/config.json

REM 下载模型文件
echo 下载模型文件...
curl -L -o bigvgan_model\\bigvgan_generator.pt https://hf-mirror.com/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/bigvgan_generator.pt

echo 下载完成！
echo 模型文件位置: bigvgan_model\\
pause
'''
    
    with open("download_bigvgan.bat", "w") as f:
        f.write(bat_content)
    
    print("✓ 创建下载脚本:")
    print("  - download_bigvgan.sh (Linux/Mac)")
    print("  - download_bigvgan.bat (Windows)")

def main():
    """主函数"""
    print("BigVGAN加载问题快速修复脚本")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("\n请先安装必要的依赖包")
        return
    
    # 设置HuggingFace镜像
    setup_huggingface_mirror()
    
    # 测试下载
    if test_bigvgan_download():
        print("\n✓ BigVGAN模型下载测试成功！")
        print("现在可以重新运行ComfyUI了")
    else:
        print("\n✗ 自动下载失败，创建手动下载脚本...")
        create_manual_download_script()
        print("\n请运行手动下载脚本，或检查网络连接")
    
    print("\n修复完成！")

if __name__ == "__main__":
    main()
