#!/usr/bin/env python3
"""
Download WeTextProcessing Windows Wheel
下载 WeTextProcessing Windows 轮子的脚本

This script provides direct download links and methods for WeTextProcessing wheels.
"""

import subprocess
import sys
import os
import urllib.request
import json

def get_pypi_download_links():
    """获取 PyPI 上的下载链接"""
    print("🔍 Getting WeTextProcessing download links from PyPI...")
    
    try:
        # 获取 PyPI API 信息
        url = "https://pypi.org/pypi/WeTextProcessing/json"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        
        print("📦 Available WeTextProcessing files:")
        print("-" * 50)
        
        # 获取最新版本的文件
        latest_version = data['info']['version']
        files = data['releases'][latest_version]
        
        wheel_files = []
        source_files = []
        
        for file_info in files:
            filename = file_info['filename']
            download_url = file_info['url']
            size = file_info['size']
            upload_date = file_info['upload_time']
            
            print(f"📄 {filename}")
            print(f"   📥 Download: {download_url}")
            print(f"   📊 Size: {size:,} bytes ({size/1024/1024:.1f} MB)")
            print(f"   📅 Upload: {upload_date}")
            print()
            
            if filename.endswith('.whl'):
                wheel_files.append((filename, download_url, size))
            else:
                source_files.append((filename, download_url, size))
        
        return wheel_files, source_files
        
    except Exception as e:
        print(f"❌ Error getting PyPI info: {e}")
        return [], []

def download_wheel_file(url, filename):
    """下载轮子文件"""
    print(f"📥 Downloading {filename}...")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\r   Progress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Downloaded: {filename}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def install_from_wheel(wheel_file):
    """从轮子文件安装"""
    print(f"🔧 Installing from {wheel_file}...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", wheel_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Installation successful!")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def test_installation():
    """测试安装是否成功"""
    print("🧪 Testing WeTextProcessing installation...")
    
    try:
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer
        
        print("✅ WeTextProcessing imported successfully")
        
        # 简单测试
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
    print("📦 WeTextProcessing Windows Wheel Downloader")
    print("=" * 60)
    
    # 获取下载链接
    wheel_files, source_files = get_pypi_download_links()
    
    if not wheel_files:
        print("❌ No wheel files found!")
        return 1
    
    print("🎯 Direct Download Links:")
    print("=" * 60)
    
    # 显示直接下载链接
    for filename, url, size in wheel_files:
        print(f"📄 {filename}")
        print(f"📥 Direct Link: {url}")
        print(f"📊 Size: {size:,} bytes ({size/1024/1024:.1f} MB)")
        print()
        
        # 如果是 Windows 通用轮子
        if 'py3-none-any' in filename:
            print("🌟 This is the Windows-compatible wheel!")
            print("💡 You can download and install it with:")
            print(f"   wget {url}")
            print(f"   pip install {filename}")
            print()
            
            # 询问是否要下载
            try:
                choice = input("🤔 Do you want to download and install this wheel now? (y/n): ").lower()
                if choice in ['y', 'yes']:
                    if download_wheel_file(url, filename):
                        if install_from_wheel(filename):
                            test_installation()
                            # 清理下载的文件
                            try:
                                os.remove(filename)
                                print(f"🗑️ Cleaned up {filename}")
                            except:
                                pass
                        return 0
                    else:
                        return 1
            except KeyboardInterrupt:
                print("\n⏹️ Cancelled by user")
                return 1
    
    print("🔗 Alternative installation methods:")
    print("=" * 60)
    print("1. Direct pip install (may trigger pynini compilation):")
    print("   pip install WeTextProcessing")
    print()
    print("2. Force wheel-only installation:")
    print("   pip install WeTextProcessing --only-binary=all")
    print()
    print("3. Download manually and install:")
    for filename, url, size in wheel_files:
        if 'py3-none-any' in filename:
            print(f"   curl -O {url}")
            print(f"   pip install {filename}")
            break
    print()
    print("4. Use wetext instead (no pynini dependency):")
    print("   pip install wetext")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
