#!/usr/bin/env python3
"""
检查本地BigVGAN模型文件位置和结构
"""

import os
import sys
from pathlib import Path

def find_comfyui_models_dir():
    """查找ComfyUI模型目录"""
    print("查找ComfyUI模型目录...")
    
    # 方法1: 通过当前文件路径推断
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if parent.name == "ComfyUI" and (parent / "models").exists():
            return parent / "models"
        if (parent / "ComfyUI" / "models").exists():
            return parent / "ComfyUI" / "models"
    
    # 方法2: 检查环境变量
    if "COMFYUI_MODELS_DIR" in os.environ:
        models_dir = Path(os.environ["COMFYUI_MODELS_DIR"])
        if models_dir.exists():
            return models_dir
    
    # 方法3: 常见位置
    possible_paths = [
        Path.cwd() / "models",
        Path.home() / "ComfyUI" / "models",
        Path("C:/ComfyUI/models") if sys.platform == "win32" else Path("/opt/ComfyUI/models"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def check_bigvgan_files(base_dir):
    """检查BigVGAN模型文件"""
    print(f"\n检查BigVGAN模型文件 (基础目录: {base_dir})")
    
    # 可能的BigVGAN文件位置
    possible_locations = [
        base_dir / "tts" / "IndexTTS-2" / "external_models",
        base_dir / "tts" / "IndexTTS-2" / "external_models" / "bigvgan",
        base_dir / "tts" / "IndexTTS-2" / "external_models" / "nvidia_bigvgan_v2_22khz_80band_256x",
        base_dir / "tts" / "IndexTTS-2" / "external_models" / "bigvgan_v2_22khz_80band_256x",
    ]
    
    # 检查HuggingFace缓存格式
    hf_cache_paths = [
        base_dir / "tts" / "IndexTTS-2" / "external_models" / "bigvgan" / "models--nvidia--bigvgan_v2_22khz_80band_256x",
        base_dir / "tts" / "IndexTTS-2" / "external_models" / "bigvgan" / "nvidia_bigvgan_v2_22khz_80band_256x",
    ]
    
    # 查找HuggingFace缓存中的snapshots目录
    for hf_path in hf_cache_paths:
        if hf_path.exists():
            snapshots_dir = hf_path / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir():
                        possible_locations.append(snapshot)
    
    found_locations = []
    
    for location in possible_locations:
        if location.exists():
            print(f"\n✓ 发现目录: {location}")
            
            # 检查必需的文件
            config_file = location / "config.json"
            model_file = location / "bigvgan_generator.pt"
            
            files_status = {
                "config.json": config_file.exists(),
                "bigvgan_generator.pt": model_file.exists(),
            }
            
            for filename, exists in files_status.items():
                status = "✓" if exists else "✗"
                print(f"  {status} {filename}")
                
                if exists:
                    file_size = location / filename
                    if file_size.exists():
                        size_mb = file_size.stat().st_size / (1024 * 1024)
                        print(f"    文件大小: {size_mb:.2f} MB")
            
            # 检查是否完整
            if all(files_status.values()):
                found_locations.append(location)
                print(f"  ✓ 完整的BigVGAN模型")
            else:
                print(f"  ✗ 不完整的BigVGAN模型")
        else:
            print(f"✗ 目录不存在: {location}")
    
    return found_locations

def suggest_fix(locations):
    """建议修复方案"""
    print("\n" + "="*60)
    print("修复建议:")
    
    if locations:
        print(f"✓ 找到 {len(locations)} 个完整的BigVGAN模型位置:")
        for i, location in enumerate(locations, 1):
            print(f"  {i}. {location}")
        
        print("\n建议操作:")
        print("1. 确保模型文件在以下位置之一:")
        for location in locations:
            print(f"   - {location}")
        print("2. 重新运行ComfyUI")
        print("3. 如果仍然有问题，检查文件权限")
        
    else:
        print("✗ 未找到完整的BigVGAN模型文件")
        print("\n建议操作:")
        print("1. 下载BigVGAN模型文件:")
        print("   - config.json")
        print("   - bigvgan_generator.pt")
        print("2. 将文件放置在以下位置之一:")
        print("   - ComfyUI/models/tts/IndexTTS-2/external_models/")
        print("   - ComfyUI/models/tts/IndexTTS-2/external_models/bigvgan/")
        print("3. 运行 fix_bigvgan_loading.py 进行自动下载")

def main():
    """主函数"""
    print("BigVGAN本地模型文件检查工具")
    print("="*60)
    
    # 查找ComfyUI模型目录
    models_dir = find_comfyui_models_dir()
    if not models_dir:
        print("✗ 未找到ComfyUI模型目录")
        print("请确保ComfyUI已正确安装")
        return
    
    print(f"✓ 找到ComfyUI模型目录: {models_dir}")
    
    # 检查BigVGAN文件
    locations = check_bigvgan_files(models_dir)
    
    # 提供修复建议
    suggest_fix(locations)
    
    print("\n" + "="*60)
    print("检查完成")

if __name__ == "__main__":
    main()
