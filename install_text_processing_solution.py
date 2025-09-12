#!/usr/bin/env python3
"""
Complete Text Processing Solution for IndexTTS2
完整的文本处理解决方案

This script provides multiple options for text normalization:
1. wetext - No pynini dependency (RECOMMENDED for Windows)
2. WeTextProcessing - Full features but requires pynini
3. Built-in fallback - Basic text processing
"""

import subprocess
import sys
import os

def run_command(cmd, description="", ignore_errors=False):
    """运行命令并返回结果"""
    print(f"🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=not ignore_errors
        )
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True, result.stdout
        else:
            print(f"❌ {description} - Failed")
            print(f"   Error: {result.stderr[:500]}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ {description} - Exception: {str(e)}")
        return False, str(e)

def test_wetext():
    """测试 wetext 是否正常工作"""
    try:
        from wetext import Normalizer
        
        print("✅ wetext imported successfully")
        
        # 测试中文文本标准化
        zh_normalizer = Normalizer(lang="zh", operator="tn", remove_erhua=True)
        zh_test = zh_normalizer.normalize("我有100元，简直666")
        print(f"✅ Chinese TN test: '我有100元，简直666' -> '{zh_test}'")
        
        # 测试英文文本标准化
        en_normalizer = Normalizer(lang="en", operator="tn")
        en_test = en_normalizer.normalize("I have $100")
        print(f"✅ English TN test: 'I have $100' -> '{en_test}'")
        
        return True
    except Exception as e:
        print(f"❌ wetext test failed: {e}")
        return False

def test_wetextprocessing():
    """测试 WeTextProcessing 是否正常工作"""
    try:
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer
        
        print("✅ WeTextProcessing imported successfully")
        
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

def install_wetext():
    """安装 wetext (推荐方案)"""
    print("\n" + "="*60)
    print("📦 Installing wetext (No pynini dependency - RECOMMENDED)")
    print("="*60)
    
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "wetext"],
        "Installing wetext"
    )
    
    if success:
        return test_wetext()
    return False

def install_wetextprocessing_with_wheels():
    """尝试安装 WeTextProcessing 使用轮子"""
    print("\n" + "="*60)
    print("📦 Installing WeTextProcessing (with wheels)")
    print("="*60)
    
    # WeTextProcessing 有 py3-none-any.whl，应该可以直接安装
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "WeTextProcessing", "--only-binary=all"],
        "Installing WeTextProcessing (wheels only)",
        ignore_errors=True
    )
    
    if success:
        return test_wetextprocessing()
    
    # 如果轮子安装失败，尝试普通安装
    print("\n🔄 Trying normal installation...")
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "WeTextProcessing"],
        "Installing WeTextProcessing (normal)",
        ignore_errors=True
    )
    
    if success:
        return test_wetextprocessing()
    
    return False

def modify_indexTTS2_for_wetext():
    """修改 IndexTTS2 代码以支持 wetext"""
    print("\n🔧 Modifying IndexTTS2 to support wetext...")
    
    # 检查是否需要修改代码
    front_py_path = "indextts/utils/front.py"
    if not os.path.exists(front_py_path):
        print(f"❌ {front_py_path} not found")
        return False
    
    try:
        with open(front_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经支持 wetext
        if 'from wetext import Normalizer' in content:
            print("✅ IndexTTS2 already supports wetext")
            return True
        
        # 添加 wetext 支持
        wetext_support_code = '''
        # Try wetext first (no pynini dependency)
        try:
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(lang="zh", operator="tn", remove_erhua=False)
            self.en_normalizer = Normalizer(lang="en", operator="tn")
            print("✅ Using wetext for text normalization")
            return
        except ImportError:
            pass
        '''
        
        # 在现有的 try 块之前插入 wetext 支持
        if 'if platform.system() == "Darwin":' in content:
            content = content.replace(
                'if platform.system() == "Darwin":',
                wetext_support_code + '\n        if platform.system() == "Darwin":'
            )
            
            with open(front_py_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Added wetext support to IndexTTS2")
            return True
        else:
            print("⚠️ Could not modify IndexTTS2 code automatically")
            return False
            
    except Exception as e:
        print(f"❌ Error modifying IndexTTS2: {e}")
        return False

def check_current_status():
    """检查当前文本处理状态"""
    print("🔍 Checking current text processing status...")
    
    # 检查 wetext
    try:
        import wetext
        print("✅ wetext is installed")
        wetext_available = True
    except ImportError:
        print("❌ wetext not installed")
        wetext_available = False
    
    # 检查 WeTextProcessing
    try:
        import tn.chinese.normalizer
        print("✅ WeTextProcessing is installed")
        wetextprocessing_available = True
    except ImportError:
        print("❌ WeTextProcessing not installed")
        wetextprocessing_available = False
    
    return wetext_available, wetextprocessing_available

def main():
    print("🎯 Complete Text Processing Solution for IndexTTS2")
    print("=" * 60)
    print("This script provides the best text normalization solution for your system.")
    print("=" * 60)
    
    # 检查当前状态
    wetext_available, wetextprocessing_available = check_current_status()
    
    if wetext_available and test_wetext():
        print("\n🎉 wetext is already working perfectly!")
        print("✅ This is the best solution for Windows - no pynini required!")
        return 0
    
    if wetextprocessing_available and test_wetextprocessing():
        print("\n🎉 WeTextProcessing is already working!")
        return 0
    
    print(f"\n📋 Available Solutions:")
    print("1. 🌟 wetext - No pynini dependency (RECOMMENDED for Windows)")
    print("2. 🔧 WeTextProcessing - Full features but requires pynini")
    print("3. 🛡️ Built-in fallback - Basic text processing (already available)")
    
    # 尝试安装 wetext (推荐)
    print(f"\n🚀 Trying Solution 1: wetext (RECOMMENDED)")
    if install_wetext():
        print("\n🎉 wetext installation successful!")
        print("✅ This provides excellent text normalization without pynini!")
        
        # 尝试修改 IndexTTS2 以优先使用 wetext
        modify_indexTTS2_for_wetext()
        return 0
    
    # 如果 wetext 失败，尝试 WeTextProcessing
    print(f"\n🚀 Trying Solution 2: WeTextProcessing")
    if install_wetextprocessing_with_wheels():
        print("\n🎉 WeTextProcessing installation successful!")
        return 0
    
    # 所有方案都失败
    print("\n" + "="*60)
    print("⚠️ Advanced text normalization installation failed")
    print("="*60)
    print("✅ Don't worry! IndexTTS2 will use built-in fallback text processing.")
    print("✅ The plugin will work normally with basic text normalization.")
    print("\n💡 Summary of what we found:")
    print("📦 WeTextProcessing: Has Windows wheels (py3-none-any.whl)")
    print("📦 wetext: Has Windows wheels and no pynini dependency")
    print("🔧 Both should work on Windows, but installation may fail due to dependencies")
    print("\n🎯 Recommendation: Use IndexTTS2's built-in fallback - it's reliable!")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
