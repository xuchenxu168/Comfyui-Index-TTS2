# comfyui-index-tts2 依赖自动检测脚本
# Dependency auto-check script for comfyui-index-tts2
# 用法 Usage: python check_dependencies.py

import importlib
import sys

REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "soundfile",
    "scipy",
    "requests"
]

missing = []
for pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print("缺少依赖包 Missing packages:")
    for pkg in missing:
        print(f"  {pkg}")
    print("请使用 pip 安装上述依赖\nPlease install above packages via pip.")
    sys.exit(1)
else:
    print("所有依赖已安装 All dependencies are installed.")
