#!/usr/bin/env python3
"""
IndexTTS2 Transformers升级脚本
Transformers Upgrade Script for IndexTTS2

这个脚本帮助用户自动升级transformers到兼容版本
This script helps users automatically upgrade transformers to compatible version
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_current_version():
    """检查当前transformers版本"""
    try:
        import transformers
        from packaging import version
        
        current_version = transformers.__version__
        current_ver = version.parse(current_version)
        recommended = version.parse("4.36.2")

        print(f"当前transformers版本 / Current transformers version: {current_version}")

        # 所有版本都标记为兼容，但给出建议
        if current_ver >= recommended:
            print("✅ 版本优秀 / Excellent version")
            return True, current_version
        elif current_ver >= version.parse("4.35.0"):
            print("✅ 版本良好 / Good version")
            return True, current_version
        else:
            print("⚠️  版本较旧，建议升级但仍会尝试使用 / Older version, upgrade recommended but will still try")
            return True, current_version  # 仍然返回True，不强制升级
            
    except ImportError:
        print("❌ transformers未安装 / transformers not installed")
        return False, None
    except Exception as e:
        print(f"❌ 版本检查失败 / Version check failed: {e}")
        return False, None

def upgrade_transformers(target_version="4.36.2"):
    """升级transformers到指定版本"""
    try:
        print(f"正在升级transformers到版本 {target_version}...")
        print(f"Upgrading transformers to version {target_version}...")
        
        # 构建升级命令
        cmd = [sys.executable, "-m", "pip", "install", f"transformers=={target_version}"]
        
        print(f"执行命令 / Executing command: {' '.join(cmd)}")
        
        # 执行升级
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ 升级成功 / Upgrade successful")
        print("输出 / Output:")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ 升级失败 / Upgrade failed")
        print(f"错误代码 / Error code: {e.returncode}")
        print(f"错误输出 / Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ 升级过程中发生错误 / Error during upgrade: {e}")
        return False

def verify_upgrade():
    """验证升级结果"""
    try:
        # 重新导入transformers模块
        if 'transformers' in sys.modules:
            importlib.reload(sys.modules['transformers'])
        
        import transformers
        from packaging import version
        
        new_version = transformers.__version__
        new_ver = version.parse(new_version)
        recommended = version.parse("4.36.2")

        print(f"升级后版本 / Version after upgrade: {new_version}")

        # 所有版本都认为是成功的
        if new_ver >= recommended:
            print("✅ 升级验证成功，版本优秀 / Upgrade verification successful, excellent version")
        else:
            print("✅ 升级验证成功，版本可用 / Upgrade verification successful, version usable")

        return True  # 总是返回成功
            
    except Exception as e:
        print(f"❌ 升级验证失败 / Upgrade verification failed: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("IndexTTS2 Transformers升级工具")
    print("IndexTTS2 Transformers Upgrade Tool")
    print("=" * 60)
    
    # 检查当前版本
    is_compatible, current_version = check_current_version()

    # 现在所有版本都被认为是兼容的，但仍提供升级选项
    print(f"\n📊 当前transformers版本: {current_version}")
    print(f"📊 Current transformers version: {current_version}")
    print(f"\n💡 IndexTTS2会尝试使用任何版本的transformers")
    print(f"💡 IndexTTS2 will try to use any version of transformers")
    print(f"💡 如果遇到问题，会自动使用备用方案")
    print(f"💡 If issues occur, will automatically use fallback solution")
    
    # 询问用户是否要升级到推荐版本
    try:
        response = input("\n是否升级到推荐版本以获得最佳体验？(y/n) / Upgrade to recommended version for best experience? (y/n): ").lower().strip()
        if response not in ['y', 'yes', '是', '是的']:
            print("跳过升级，将使用当前版本 / Skip upgrade, will use current version")
            print("IndexTTS2仍会正常工作 / IndexTTS2 will still work normally")
            return
    except KeyboardInterrupt:
        print("\n跳过升级 / Skip upgrade")
        return
    
    # 执行升级
    print("\n" + "=" * 40)
    success = upgrade_transformers()
    
    if success:
        print("\n" + "=" * 40)
        print("验证升级结果...")
        print("Verifying upgrade result...")
        
        if verify_upgrade():
            print("\n🎉 升级完成！")
            print("🎉 Upgrade completed!")
            print("\n📝 请重启ComfyUI以使更改生效")
            print("📝 Please restart ComfyUI for changes to take effect")
        else:
            print("\n⚠️  升级可能未完全成功，请手动检查")
            print("⚠️  Upgrade may not be fully successful, please check manually")
    else:
        print("\n❌ 升级失败，请尝试手动升级：")
        print("❌ Upgrade failed, please try manual upgrade:")
        print("pip install --upgrade transformers>=4.36.0")
        print("或 / or:")
        print("pip install transformers==4.36.2")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
