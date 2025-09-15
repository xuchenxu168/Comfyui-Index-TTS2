"""
IndexTTS2 ComfyUI Plugin Version Information
"""

__version__ = "2.2.0"
__version_info__ = (2, 2, 0)

# Version history
VERSION_HISTORY = {
    "2.2.0": {
        "date": "2025-01-15",
        "changes": [
            "🎯 完全修复 Transformers 4.56.1+ 兼容性问题",
            "🤖 新增智能 Qwen 模型选择系统 (支持 Qwen3/2.5/2/1.5)",
            "📊 新增 Qwen 模型状态显示节点",
            "🔧 优化模型信息直观显示功能",
            "⚡ 移除有问题的 QwenEmotion 实例创建",
            "🛡️ 增强错误处理和兼容性检查",
            "📝 完善版本管理和更新日志系统",
            "🧪 新增完整的测试脚本和验证工具"
        ],
        "compatibility": {
            "transformers": ">=4.35.0 (推荐 4.56.1+)",
            "python": "3.8-3.12",
            "comfyui": "最新版本"
        },
        "new_features": [
            "IndexTTS2 Qwen Model Display 节点",
            "IndexTTS2 Qwen Model Status 节点", 
            "智能 Qwen 模型兼容性检查",
            "直观的模型信息显示系统"
        ]
    },
    "2.1.0": {
        "date": "2024-12-XX",
        "changes": [
            "基础 TTS 功能实现",
            "多说话人对话系统",
            "情感控制功能",
            "音频增强功能"
        ]
    }
}

def get_version():
    """获取当前版本"""
    return __version__

def get_version_info():
    """获取版本信息元组"""
    return __version_info__

def get_latest_changes():
    """获取最新版本的更新内容"""
    return VERSION_HISTORY.get(__version__, {}).get("changes", [])

def print_version_info():
    """打印版本信息"""
    print(f"IndexTTS2 ComfyUI Plugin v{__version__}")
    print("=" * 50)
    
    latest = VERSION_HISTORY.get(__version__, {})
    if latest:
        print(f"发布日期: {latest.get('date', 'Unknown')}")
        print(f"Release Date: {latest.get('date', 'Unknown')}")
        
        changes = latest.get("changes", [])
        if changes:
            print("\n🆕 新功能和改进:")
            print("🆕 New Features and Improvements:")
            for change in changes:
                print(f"  {change}")
        
        compatibility = latest.get("compatibility", {})
        if compatibility:
            print("\n🔧 兼容性要求:")
            print("🔧 Compatibility Requirements:")
            for key, value in compatibility.items():
                print(f"  {key}: {value}")
        
        new_features = latest.get("new_features", [])
        if new_features:
            print("\n✨ 新增节点:")
            print("✨ New Nodes:")
            for feature in new_features:
                print(f"  • {feature}")

if __name__ == "__main__":
    print_version_info()
