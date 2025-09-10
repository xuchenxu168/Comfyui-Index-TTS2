#!/usr/bin/env python3
"""
Pynini Installation Script for IndexTTS2
========================================

This script helps install Pynini (advanced text normalization library) 
with platform-specific optimizations.

Pynini provides professional-grade text normalization for TTS:
- Phone numbers: "123-456-7890" → "one two three four five six seven eight nine zero"
- Currency: "$29.99" → "twenty nine dollars and ninety nine cents"
- Dates: "2024年3月15日" → "二零二四年三月十五日"
- Abbreviations: "Dr. Smith" → "Doctor Smith"

Usage:
    python install_pynini.py [--force] [--test]
    
Options:
    --force    Force installation even if already installed
    --test     Test installation after installing
"""

import sys
import platform
import subprocess
import importlib.util
import argparse
from pathlib import Path

def check_pynini_installed():
    """Check if Pynini is already installed."""
    try:
        import pynini
        return True, pynini.__version__
    except ImportError:
        return False, None

def get_platform_info():
    """Get detailed platform information."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    return {
        'system': system,
        'machine': machine,
        'python_version': python_version,
        'is_linux_x64': system == 'linux' and 'x86_64' in machine,
        'is_macos': system == 'darwin',
        'is_windows': system == 'windows'
    }

def check_conda_available():
    """Check if conda is available."""
    try:
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def install_pynini_pip(version="2.1.6"):
    """Install Pynini using pip."""
    print(f"📦 Installing Pynini {version} using pip...")
    print("⚠️  This may take a while and require compilation on some platforms.")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", f"pynini=={version}"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Pynini installed successfully using pip!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Pynini using pip:")
        print(f"   Error: {e.stderr}")
        return False

def install_pynini_conda(version="2.1.6"):
    """Install Pynini using conda-forge."""
    print(f"📦 Installing Pynini {version} using conda-forge...")
    
    try:
        cmd = ["conda", "install", "-c", "conda-forge", f"pynini={version}", "-y"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Pynini installed successfully using conda!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Pynini using conda:")
        print(f"   Error: {e.stderr}")
        return False

def test_pynini_installation():
    """Test Pynini installation with a simple example."""
    print("\n🧪 Testing Pynini installation...")
    
    try:
        import pynini
        print(f"✅ Pynini version: {pynini.__version__}")
        
        # Simple test
        rule = pynini.string_map([
            ('$', 'dollar'),
            ('123', 'one two three'),
            ('Dr.', 'Doctor')
        ])
        
        # Test the rule
        test_cases = ['$', '123', 'Dr.']
        for test in test_cases:
            try:
                result = pynini.compose(test, rule).string()
                print(f"   '{test}' → '{result}'")
            except:
                print(f"   '{test}' → (no match)")
        
        print("✅ Pynini test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pynini test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Install Pynini for IndexTTS2")
    parser.add_argument('--force', action='store_true', 
                       help='Force installation even if already installed')
    parser.add_argument('--test', action='store_true',
                       help='Test installation after installing')
    parser.add_argument('--version', default='2.1.6',
                       help='Pynini version to install (default: 2.1.6)')
    
    args = parser.parse_args()
    
    print("🔤 Pynini Installation Script for IndexTTS2")
    print("=" * 50)
    
    # Check if already installed
    is_installed, current_version = check_pynini_installed()
    if is_installed and not args.force:
        print(f"✅ Pynini {current_version} is already installed!")
        if args.test:
            test_pynini_installation()
        return
    
    # Get platform info
    platform_info = get_platform_info()
    print(f"🖥️  Platform: {platform_info['system']} {platform_info['machine']}")
    print(f"🐍 Python: {platform_info['python_version']}")
    
    # Determine installation method
    conda_available = check_conda_available()
    
    print("\n📋 Installation Strategy:")
    
    if platform_info['is_linux_x64']:
        print("🐧 Linux x86_64 detected - using pip (pre-compiled wheels available)")
        success = install_pynini_pip(args.version)
    elif conda_available:
        print(f"📦 Conda available - using conda-forge (recommended for {platform_info['system']})")
        success = install_pynini_conda(args.version)
    else:
        print("⚠️  Conda not available - trying pip (may require compilation)")
        if platform_info['is_windows']:
            print("💡 Tip: Consider installing conda/miniconda for easier installation on Windows")
        success = install_pynini_pip(args.version)
    
    if success:
        print("\n🎉 Pynini installation completed!")
        
        if args.test:
            test_pynini_installation()
        
        print("\n📚 What's Next?")
        print("- Restart ComfyUI to use enhanced text processing")
        print("- Pynini will automatically handle complex text formats")
        print("- Check the README for usage examples")
        
    else:
        print("\n❌ Installation failed!")
        print("\n🔧 Troubleshooting:")
        print("1. Try using conda: conda install -c conda-forge pynini=2.1.6")
        print("2. Check if you have a C++ compiler installed")
        print("3. On Windows, consider using WSL (Windows Subsystem for Linux)")
        print("4. Visit: https://github.com/kylebgorman/pynini for more help")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
