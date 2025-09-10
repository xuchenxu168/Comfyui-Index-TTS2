#!/usr/bin/env python3
# IndexTTS2 模型下载脚本
# Model download script for IndexTTS2

import os
import sys
import subprocess
import argparse
import json
import requests
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    try:
        import huggingface_hub
        return True, "huggingface_hub"
    except ImportError:
        try:
            import modelscope
            return True, "modelscope"
        except ImportError:
            return False, None

def download_with_requests(checkpoints_dir):
    """使用requests直接下载模型文件"""
    # 权重文件列表
    MODEL_FILES = [
        "bpe.model",
        "config.yaml",
        "feat1.pt",
        "feat2.pt",
        "gpt.pth",
        "qwen0.6bemo4-merge",
        "s2mel.pth",
        "wav2vec2bert_stats.pt"
    ]

    BASE_URL = "https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/"

    print("使用直接下载方式...")
    print("Using direct download method...")

    success_count = 0
    total_files = len(MODEL_FILES)

    for fname in MODEL_FILES:
        url = BASE_URL + fname
        save_path = checkpoints_dir / fname

        if save_path.exists():
            print(f"[跳过] 已存在: {fname}")
            print(f"[Skip] Exists: {fname}")
            success_count += 1
            continue

        print(f"[下载] {fname} ...")
        print(f"[Download] {fname} ...")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r进度: {percent:.1f}% ({downloaded}/{total_size})", end="")

            print(f"\n[完成] {fname}")
            print(f"[Done] {fname}")
            success_count += 1

        except Exception as e:
            print(f"\n[失败] {fname}: {e}")
            print(f"[Failed] {fname}: {e}")

    return success_count == total_files

def download_with_huggingface(checkpoints_dir):
    """使用HuggingFace Hub下载模型"""
    try:
        from huggingface_hub import snapshot_download

        print("使用HuggingFace Hub下载模型...")
        print("Downloading model with HuggingFace Hub...")

        snapshot_download(
            repo_id="IndexTeam/IndexTTS-2",
            local_dir=str(checkpoints_dir),
            local_dir_use_symlinks=False
        )

        return True

    except Exception as e:
        print(f"HuggingFace下载失败: {e}")
        print(f"HuggingFace download failed: {e}")
        return False

def download_with_modelscope(checkpoints_dir):
    """使用ModelScope下载模型"""
    try:
        from modelscope import snapshot_download

        print("使用ModelScope下载模型...")
        print("Downloading model with ModelScope...")

        snapshot_download(
            model_id="IndexTeam/IndexTTS-2",
            cache_dir=str(checkpoints_dir.parent),
            local_dir=str(checkpoints_dir)
        )

        return True

    except Exception as e:
        print(f"ModelScope下载失败: {e}")
        print(f"ModelScope download failed: {e}")
        return False

def download_with_cli(checkpoints_dir, method="huggingface"):
    """使用命令行工具下载"""
    try:
        if method == "huggingface":
            cmd = [
                "huggingface-cli", "download",
                "IndexTeam/IndexTTS-2",
                "--local-dir", str(checkpoints_dir)
            ]
        else:  # modelscope
            cmd = [
                "modelscope", "download",
                "--model", "IndexTeam/IndexTTS-2",
                "--local_dir", str(checkpoints_dir)
            ]

        print(f"执行命令: {' '.join(cmd)}")
        print(f"Executing: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"命令行下载失败: {e}")
        print(f"CLI download failed: {e}")
        return False
    except FileNotFoundError:
        print(f"未找到{method}命令行工具")
        print(f"{method} CLI tool not found")
        return False

def verify_download(checkpoints_dir):
    """验证下载的模型文件"""
    required_files = [
        "config.yaml",
        "gpt.pth",
        "s2mel.pth"
    ]

    missing_files = []
    for file_name in required_files:
        file_path = checkpoints_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        print(f"缺少关键文件: {missing_files}")
        print(f"Missing critical files: {missing_files}")
        return False

    print("✓ 模型文件验证通过")
    print("✓ Model files verification passed")
    return True

def download_models(force=False):
    """下载IndexTTS2模型文件"""

    # 获取插件目录
    plugin_dir = Path(__file__).parent
    checkpoints_dir = plugin_dir / "checkpoints"

    # 检查是否已经下载
    if not force and checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
        config_file = checkpoints_dir / "config.yaml"
        if config_file.exists():
            print("模型文件已存在且完整")
            print("Model files already exist and complete")
            return True

    # 创建checkpoints目录
    checkpoints_dir.mkdir(exist_ok=True)

    print("开始下载IndexTTS2模型...")
    print("Starting IndexTTS2 model download...")

    # 检查依赖
    has_deps, dep_type = check_dependencies()

    # 尝试多种下载方法
    download_methods = []

    # 总是首先尝试直接下载（最可靠）
    download_methods.append(("Direct Download", lambda: download_with_requests(checkpoints_dir)))

    if has_deps:
        if dep_type == "huggingface_hub":
            download_methods.extend([
                ("HuggingFace Hub API", lambda: download_with_huggingface(checkpoints_dir)),
                ("HuggingFace CLI", lambda: download_with_cli(checkpoints_dir, "huggingface")),
            ])
        elif dep_type == "modelscope":
            download_methods.extend([
                ("ModelScope API", lambda: download_with_modelscope(checkpoints_dir)),
                ("ModelScope CLI", lambda: download_with_cli(checkpoints_dir, "modelscope")),
            ])

    # 添加CLI方法作为备用
    download_methods.extend([
        ("HuggingFace CLI", lambda: download_with_cli(checkpoints_dir, "huggingface")),
        ("ModelScope CLI", lambda: download_with_cli(checkpoints_dir, "modelscope"))
    ])

    # 尝试下载
    for method_name, method_func in download_methods:
        print(f"\n尝试使用 {method_name} 下载...")
        print(f"\nTrying {method_name} download...")

        if method_func():
            print(f"✓ {method_name} 下载成功")
            print(f"✓ {method_name} download successful")

            # 验证下载
            if verify_download(checkpoints_dir):
                return True
            else:
                print(f"⚠ {method_name} 下载不完整，尝试下一种方法")
                print(f"⚠ {method_name} download incomplete, trying next method")
                continue
        else:
            print(f"✗ {method_name} 下载失败")
            print(f"✗ {method_name} download failed")

    print("\n所有下载方法都失败了")
    print("\nAll download methods failed")
    return False

def install_dependencies():
    """安装必要的依赖"""
    print("安装下载依赖...")
    print("Installing download dependencies...")

    try:
        # 尝试安装huggingface_hub
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        print("✓ huggingface_hub 安装成功")
        print("✓ huggingface_hub installed successfully")
        return True
    except subprocess.CalledProcessError:
        try:
            # 备用：安装modelscope
            subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], check=True)
            print("✓ modelscope 安装成功")
            print("✓ modelscope installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ 依赖安装失败")
            print("✗ Dependency installation failed")
            return False

def main():
    parser = argparse.ArgumentParser(description="下载IndexTTS2模型 / Download IndexTTS2 models")
    parser.add_argument("--force", action="store_true", help="强制重新下载 / Force re-download")
    parser.add_argument("--install-deps", action="store_true", help="安装下载依赖 / Install download dependencies")

    args = parser.parse_args()

    if args.install_deps:
        if install_dependencies():
            print("依赖安装完成，现在可以下载模型了")
            print("Dependencies installed, you can now download models")
        else:
            print("依赖安装失败")
            print("Dependency installation failed")
            return

    success = download_models(force=args.force)

    if success:
        print("\n✓ 模型下载成功！现在可以使用IndexTTS2插件了。")
        print("\n✓ Model download successful! You can now use the IndexTTS2 plugin.")

        # 显示下载的文件信息
        plugin_dir = Path(__file__).parent
        checkpoints_dir = plugin_dir / "checkpoints"

        print(f"\n模型文件位置: {checkpoints_dir}")
        print(f"Model files location: {checkpoints_dir}")

        if checkpoints_dir.exists():
            files = list(checkpoints_dir.iterdir())
            print(f"下载的文件数量: {len(files)}")
            print(f"Downloaded files count: {len(files)}")
    else:
        print("\n✗ 模型下载失败，请手动下载模型文件。")
        print("\n✗ Model download failed, please download model files manually.")
        print("\n手动下载方法 / Manual download methods:")
        print("1. HuggingFace: https://huggingface.co/IndexTeam/IndexTTS-2")
        print("2. ModelScope: https://modelscope.cn/models/IndexTeam/IndexTTS-2")
        print("\n或者尝试安装依赖后重试:")
        print("Or try installing dependencies and retry:")
        print("python download_models.py --install-deps")

if __name__ == "__main__":
    main()
