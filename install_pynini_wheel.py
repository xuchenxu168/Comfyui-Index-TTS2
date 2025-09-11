#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Pynini Wheel Installer for Windows
Windows pynini 轮子文件快速安装器
"""

import subprocess
import sys
import os
import platform

def main():
    print("🎯 Pynini Windows 轮子文件安装器")
    print("="*50)
    
    # 检查操作系统
    if platform.system() != "Windows":
        print("❌ 此脚本仅适用于 Windows 系统")
        print("💡 其他系统请使用: conda install -c conda-forge pynini=2.1.6")
        return
    
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
    
    # 检查是否支持当前 Python 版本
    if python_version not in wheel_files:
        print(f"❌ 暂不支持 Python {python_version}")
        print(f"✅ 支持的版本: {', '.join(wheel_files.keys())}")
        print("💡 请使用支持的 Python 版本或尝试其他安装方法")
        return
    
    wheel_file = wheel_files[python_version]
    
    # 检查轮子文件是否存在
    if not os.path.exists(wheel_file):
        print(f"❌ 轮子文件不存在: {wheel_file}")
        print("💡 请确保您在正确的目录中运行此脚本")
        return
    
    # 检查是否已安装
    try:
        import pynini
        print("✅ pynini 已安装")
        print(f"📦 版本: {pynini.__version__}")
        
        choice = input("是否重新安装? (y/n): ").lower().strip()
        if choice not in ['y', 'yes', '是']:
            print("⏭️  跳过安装")
            return
    except ImportError:
        print("📦 pynini 未安装，开始安装...")
    
    # 安装轮子文件
    print(f"🔧 安装轮子文件: {wheel_file}")
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', wheel_file]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 安装成功！")
        
        # 验证安装
        try:
            import pynini
            print(f"🎉 验证成功！pynini 版本: {pynini.__version__}")
            
            # 简单功能测试
            rule = pynini.string_map([('$', 'dollar')])
            print("✅ 功能测试通过")
            
        except Exception as e:
            print(f"⚠️  安装成功但验证失败: {e}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        print("💡 请尝试以管理员权限运行或使用其他安装方法")
    except Exception as e:
        print(f"❌ 安装出错: {e}")

if __name__ == "__main__":
    main()
