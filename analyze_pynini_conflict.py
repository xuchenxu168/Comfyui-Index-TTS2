#!/usr/bin/env python3
"""
Analyze pynini version conflict with WeTextProcessing
分析 pynini 版本冲突问题
"""

import subprocess
import sys
import pkg_resources

def check_installed_pynini():
    """检查已安装的 pynini 版本"""
    try:
        import pynini
        version = getattr(pynini, '__version__', 'unknown')
        print(f"✅ pynini installed: version {version}")
        return version
    except ImportError:
        print("❌ pynini not installed")
        return None

def check_wetextprocessing_requirements():
    """检查 WeTextProcessing 的依赖要求"""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'show', 'WeTextProcessing'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ WeTextProcessing installed:")
            for line in result.stdout.split('\n'):
                if 'Requires:' in line or 'Version:' in line:
                    print(f"   {line}")
        else:
            print("❌ WeTextProcessing not installed")
    except Exception as e:
        print(f"❌ Error checking WeTextProcessing: {e}")

def get_wetextprocessing_dependencies():
    """获取 WeTextProcessing 的依赖信息"""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--dry-run', '--report', '-', 'WeTextProcessing'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("📋 WeTextProcessing dependency report:")
            print(result.stdout[:1000])  # 限制输出长度
        else:
            print("⚠️ Could not get dependency report")
            print("Error:", result.stderr[:500])
    except Exception as e:
        print(f"❌ Error getting dependencies: {e}")

def check_pynini_versions_available():
    """检查可用的 pynini 版本"""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'index', 'versions', 'pynini'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("📦 Available pynini versions:")
            print(result.stdout)
        else:
            # 尝试另一种方法
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'pynini==999.999.999'
            ], capture_output=True, text=True)
            
            if "Could not find a version" in result.stderr:
                # 从错误信息中提取可用版本
                lines = result.stderr.split('\n')
                for line in lines:
                    if 'from versions:' in line:
                        print("📦 Available pynini versions:")
                        print(line)
                        break
    except Exception as e:
        print(f"❌ Error checking pynini versions: {e}")

def analyze_conflict():
    """分析版本冲突的原因"""
    print("\n🔍 Conflict Analysis:")
    print("=" * 50)
    
    reasons = [
        "1. 版本锁定 (Version Pinning):",
        "   WeTextProcessing 可能要求特定版本的 pynini (如 pynini==2.1.6)",
        "   即使您安装了其他版本，pip 也会尝试重新安装指定版本",
        "",
        "2. 依赖解析 (Dependency Resolution):",
        "   pip 在安装时会检查所有依赖的兼容性",
        "   如果发现版本不匹配，会尝试重新安装兼容版本",
        "",
        "3. 构建依赖 (Build Dependencies):",
        "   WeTextProcessing 可能在 setup.py 中指定了构建时依赖",
        "   这会导致即使 pynini 已安装，也会重新编译",
        "",
        "4. 平台兼容性 (Platform Compatibility):",
        "   您安装的 pynini 轮子可能与 WeTextProcessing 要求的平台不匹配",
    ]
    
    for reason in reasons:
        print(reason)

def suggest_solutions():
    """提供解决方案"""
    print("\n💡 Suggested Solutions:")
    print("=" * 50)
    
    solutions = [
        "1. 使用 --no-deps 安装:",
        "   pip install WeTextProcessing --no-deps",
        "   (跳过依赖检查，但可能导致兼容性问题)",
        "",
        "2. 强制使用已安装的 pynini:",
        "   pip install WeTextProcessing --force-reinstall --no-deps",
        "",
        "3. 安装特定版本的 pynini:",
        "   pip install pynini==2.1.6  # WeTextProcessing 要求的版本",
        "   pip install WeTextProcessing",
        "",
        "4. 使用预编译轮子 (如果可用):",
        "   pip install WeTextProcessing --only-binary=all",
        "",
        "5. 使用我们的回退机制 (推荐):",
        "   不安装 WeTextProcessing，使用 IndexTTS2 的内置回退",
    ]
    
    for solution in solutions:
        print(solution)

def main():
    print("🔍 Pynini Version Conflict Analysis")
    print("=" * 60)
    
    # 检查当前状态
    pynini_version = check_installed_pynini()
    check_wetextprocessing_requirements()
    
    print("\n" + "=" * 60)
    get_wetextprocessing_dependencies()
    
    print("\n" + "=" * 60)
    check_pynini_versions_available()
    
    # 分析和建议
    analyze_conflict()
    suggest_solutions()
    
    print("\n🎯 Recommendation:")
    print("=" * 60)
    print("由于 WeTextProcessing 的依赖复杂性，建议使用 IndexTTS2 的")
    print("内置回退机制，这样可以避免所有依赖冲突问题。")
    print("\n如果您确实需要 WeTextProcessing，请尝试:")
    print("pip install pynini==2.1.6")
    print("pip install WeTextProcessing --no-deps")

if __name__ == "__main__":
    main()
