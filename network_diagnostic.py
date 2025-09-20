#!/usr/bin/env python3
"""
IndexTTS2 网络诊断工具
用于检查HuggingFace Hub连接和BigVGAN模型下载问题
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, login
from huggingface_hub.utils import HfHubHTTPError

def check_internet_connection():
    """检查基本网络连接"""
    print("=" * 60)
    print("检查基本网络连接...")
    
    test_urls = [
        "https://www.google.com",
        "https://huggingface.co",
        "https://hf-mirror.com"  # 中国镜像
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✓ 成功连接到: {url}")
                return True
        except Exception as e:
            print(f"✗ 连接失败: {url} - {e}")
    
    return False

def check_huggingface_connection():
    """检查HuggingFace Hub连接"""
    print("=" * 60)
    print("检查HuggingFace Hub连接...")
    
    try:
        # 测试连接HuggingFace API
        response = requests.get("https://huggingface.co/api/whoami", timeout=10)
        if response.status_code == 200:
            print("✓ HuggingFace Hub API连接正常")
            return True
        else:
            print(f"✗ HuggingFace Hub API响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ HuggingFace Hub连接失败: {e}")
        return False

def test_bigvgan_download():
    """测试BigVGAN模型下载"""
    print("=" * 60)
    print("测试BigVGAN模型下载...")
    
    model_id = "nvidia/bigvgan_v2_22khz_80band_256x"
    cache_dir = "./test_cache"
    
    try:
        print(f"开始下载模型: {model_id}")
        print(f"缓存目录: {cache_dir}")
        
        # 创建测试缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 设置超时
        start_time = time.time()
        
        # 尝试下载配置文件
        config_file = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        
        config_time = time.time() - start_time
        print(f"✓ 配置文件下载成功: {config_file}")
        print(f"  下载时间: {config_time:.2f}秒")
        
        # 尝试下载模型文件
        start_time = time.time()
        model_file = hf_hub_download(
            repo_id=model_id,
            filename="bigvgan_generator.pt",
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        
        model_time = time.time() - start_time
        print(f"✓ 模型文件下载成功: {model_file}")
        print(f"  下载时间: {model_time:.2f}秒")
        
        # 检查文件大小
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        print(f"  文件大小: {file_size:.2f} MB")
        
        return True
        
    except HfHubHTTPError as e:
        print(f"✗ HuggingFace Hub HTTP错误: {e}")
        if "401" in str(e):
            print("  建议: 检查HuggingFace访问权限")
        elif "403" in str(e):
            print("  建议: 检查模型访问权限")
        elif "404" in str(e):
            print("  建议: 检查模型ID是否正确")
        return False
        
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        return False

def check_proxy_settings():
    """检查代理设置"""
    print("=" * 60)
    print("检查代理设置...")
    
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    proxy_found = False
    
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ 发现代理设置: {var}={value}")
            proxy_found = True
    
    if not proxy_found:
        print("✗ 未发现代理设置")
        print("  如果在中国大陆，建议设置HuggingFace镜像:")
        print("  export HF_ENDPOINT=https://hf-mirror.com")
    
    return proxy_found

def suggest_solutions():
    """提供解决方案建议"""
    print("=" * 60)
    print("解决方案建议:")
    print()
    
    print("1. 网络连接问题:")
    print("   - 检查网络连接是否稳定")
    print("   - 尝试使用VPN或代理")
    print("   - 检查防火墙设置")
    print()
    
    print("2. HuggingFace访问问题:")
    print("   - 设置HuggingFace镜像 (中国大陆用户):")
    print("     export HF_ENDPOINT=https://hf-mirror.com")
    print("   - 或者使用HuggingFace CLI登录:")
    print("     huggingface-cli login")
    print()
    
    print("3. 手动下载模型:")
    print("   - 访问: https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x")
    print("   - 手动下载 bigvgan_generator.pt 和 config.json")
    print("   - 放置到ComfyUI模型目录")
    print()
    
    print("4. 使用本地模型:")
    print("   - 将模型文件放在本地目录")
    print("   - 修改配置文件中的模型路径")
    print()

def main():
    """主函数"""
    print("IndexTTS2 网络诊断工具")
    print("=" * 60)
    
    # 检查基本网络连接
    internet_ok = check_internet_connection()
    
    # 检查代理设置
    proxy_ok = check_proxy_settings()
    
    # 检查HuggingFace连接
    hf_ok = check_huggingface_connection()
    
    # 测试BigVGAN下载
    if internet_ok and hf_ok:
        download_ok = test_bigvgan_download()
    else:
        download_ok = False
        print("跳过模型下载测试 (网络连接问题)")
    
    # 总结结果
    print("=" * 60)
    print("诊断结果总结:")
    print(f"网络连接: {'✓' if internet_ok else '✗'}")
    print(f"代理设置: {'✓' if proxy_ok else '✗'}")
    print(f"HuggingFace连接: {'✓' if hf_ok else '✗'}")
    print(f"模型下载: {'✓' if download_ok else '✗'}")
    
    if not download_ok:
        suggest_solutions()
    
    # 清理测试文件
    try:
        import shutil
        if os.path.exists("./test_cache"):
            shutil.rmtree("./test_cache")
            print("\n✓ 清理测试文件完成")
    except:
        pass

if __name__ == "__main__":
    main()
