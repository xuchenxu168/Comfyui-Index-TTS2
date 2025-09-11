#!/usr/bin/env python3
"""
IndexTTS2 ComfyUI Plugin - Python 3.12 Quick Fix
Python 3.12 快速修复脚本

This script provides an immediate fix for Python 3.12 compatibility issues.
Run this script to quickly resolve the import errors.
"""

import os
import sys
from pathlib import Path

def create_minimal_indextts():
    """创建最小化的indextts模块以解决导入错误"""
    
    plugin_dir = Path(__file__).parent
    indextts_dir = plugin_dir / "indextts"
    
    print(f"[Quick Fix] 创建最小化indextts模块: {indextts_dir}")
    print(f"[Quick Fix] Creating minimal indextts module: {indextts_dir}")
    
    # 创建目录
    indextts_dir.mkdir(exist_ok=True)
    
    # 创建 __init__.py
    init_file = indextts_dir / "__init__.py"
    init_content = '''"""
IndexTTS2 最小化模块 - 快速修复版本
Minimal IndexTTS2 module - Quick fix version
"""

__version__ = "2.0.0-quickfix"

print("[IndexTTS2] 最小化模块已加载 - 快速修复模式")
print("[IndexTTS2] Minimal module loaded - Quick fix mode")
print("[IndexTTS2] 请运行 python install_py312.py 进行完整安装")
print("[IndexTTS2] Please run python install_py312.py for full installation")
'''
    
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    # 创建 infer_v2.py
    infer_file = indextts_dir / "infer_v2.py"
    infer_content = '''"""
IndexTTS2 推理模块 - 快速修复版本
IndexTTS2 inference module - Quick fix version
"""

class IndexTTS2:
    """IndexTTS2 占位符类 - 快速修复版本"""
    
    def __init__(self, *args, **kwargs):
        print("[IndexTTS2] 快速修复模式 - 功能有限")
        print("[IndexTTS2] Quick fix mode - Limited functionality")
        print("[IndexTTS2] 请运行完整安装: python install_py312.py")
        print("[IndexTTS2] Please run full installation: python install_py312.py")
    
    def __call__(self, *args, **kwargs):
        return {
            "success": False,
            "error": "Quick fix mode - Please run full installation",
            "message": "请运行 python install_py312.py 进行完整安装"
        }

# 兼容性别名
IndexTTSInference = IndexTTS2
'''
    
    with open(infer_file, 'w', encoding='utf-8') as f:
        f.write(infer_content)
    
    print("[Quick Fix] ✓ 最小化模块创建完成")
    print("[Quick Fix] ✓ Minimal module created")
    
    return True

def update_init_file():
    """更新__init__.py以处理导入错误"""
    
    plugin_dir = Path(__file__).parent
    init_file = plugin_dir / "__init__.py"
    
    print(f"[Quick Fix] 检查 {init_file}")
    print(f"[Quick Fix] Checking {init_file}")
    
    if init_file.exists():
        # 读取现有内容
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经有快速修复标记
        if "QUICK_FIX_APPLIED" in content:
            print("[Quick Fix] 快速修复已应用")
            print("[Quick Fix] Quick fix already applied")
            return True
        
        # 在文件末尾添加快速修复代码
        quick_fix_code = '''

# ===== QUICK FIX FOR PYTHON 3.12 =====
# Python 3.12 快速修复代码

QUICK_FIX_APPLIED = True

# 确保indextts模块可以被导入
import sys
import os
plugin_dir = os.path.dirname(__file__)
indextts_path = os.path.join(plugin_dir, "indextts")

if os.path.exists(indextts_path) and indextts_path not in sys.path:
    sys.path.insert(0, plugin_dir)
    print("[Quick Fix] indextts路径已添加到sys.path")
    print("[Quick Fix] indextts path added to sys.path")

# 测试导入
try:
    import indextts
    print("[Quick Fix] ✓ indextts导入成功")
    print("[Quick Fix] ✓ indextts import successful")
except ImportError as e:
    print(f"[Quick Fix] ⚠️  indextts导入失败: {e}")
    print(f"[Quick Fix] ⚠️  indextts import failed: {e}")
    print("[Quick Fix] 请运行: python quick_fix_py312.py")
    print("[Quick Fix] Please run: python quick_fix_py312.py")

# ===== END QUICK FIX =====
'''
        
        # 写入更新的内容
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(content + quick_fix_code)
        
        print("[Quick Fix] ✓ __init__.py 已更新")
        print("[Quick Fix] ✓ __init__.py updated")
        return True
    
    else:
        print("[Quick Fix] ✗ __init__.py 不存在")
        print("[Quick Fix] ✗ __init__.py not found")
        return False

def main():
    """主修复流程"""
    print("=" * 50)
    print("IndexTTS2 Python 3.12 快速修复")
    print("IndexTTS2 Python 3.12 Quick Fix")
    print("=" * 50)
    
    # 检查Python版本
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 12:
        print("✓ 检测到Python 3.12+，应用快速修复")
        print("✓ Python 3.12+ detected, applying quick fix")
    else:
        print("⚠️  非Python 3.12环境，快速修复可能不必要")
        print("⚠️  Not Python 3.12 environment, quick fix may not be necessary")
    
    # 创建最小化模块
    if not create_minimal_indextts():
        print("✗ 最小化模块创建失败")
        print("✗ Minimal module creation failed")
        return False
    
    # 更新__init__.py
    if not update_init_file():
        print("✗ __init__.py 更新失败")
        print("✗ __init__.py update failed")
        return False
    
    print("\n" + "=" * 50)
    print("✓ 快速修复完成！")
    print("✓ Quick fix completed!")
    print("=" * 50)
    
    print("\n下一步 / Next steps:")
    print("1. 重启ComfyUI")
    print("1. Restart ComfyUI")
    print("2. 运行完整安装: python install_py312.py")
    print("2. Run full installation: python install_py312.py")
    print("3. 下载模型: python download_models.py")
    print("3. Download models: python download_models.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 快速修复成功！")
            print("🎉 Quick fix successful!")
        else:
            print("\n❌ 快速修复失败")
            print("❌ Quick fix failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 快速修复过程中发生错误: {e}")
        print(f"❌ Error occurred during quick fix: {e}")
        sys.exit(1)
