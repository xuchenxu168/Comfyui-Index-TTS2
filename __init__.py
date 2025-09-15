# comfyui-index-tts2 插件入口
# Entry point for comfyui-index-tts2 plugin

import os
import sys
import subprocess

# 导入版本信息
# Import version information
try:
    from .version import __version__, print_version_info
    print(f"[IndexTTS2] 🎙️ IndexTTS2 ComfyUI Plugin v{__version__} 加载中...")
    print(f"[IndexTTS2] 🎙️ IndexTTS2 ComfyUI Plugin v{__version__} Loading...")
except ImportError:
    __version__ = "2.2.0"
    print(f"[IndexTTS2] 🎙️ IndexTTS2 ComfyUI Plugin v{__version__} 加载中...")
    print(f"[IndexTTS2] 🎙️ IndexTTS2 ComfyUI Plugin v{__version__} Loading...")

# 添加插件目录到Python路径
# Add plugin directory to Python path
plugin_dir = os.path.dirname(__file__)
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

# 导入兼容性层
# Import compatibility layer
try:
    from .compatibility import compat, install_compatible_dependencies, print_compatibility_status
    COMPATIBILITY_AVAILABLE = True
except ImportError:
    COMPATIBILITY_AVAILABLE = False
    print("[IndexTTS2] 兼容性层不可用，使用基础模式")
    print("[IndexTTS2] Compatibility layer unavailable, using basic mode")

# 插件环境准备：检测并安装 indextts 包
# Plugin environment setup: detect and install indextts package

def check_python_version():
    """检查Python版本兼容性"""
    import sys
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        return False, f"Python {version.major}.{version.minor}.{version.micro}"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"

def auto_setup():
    plugin_dir = os.path.dirname(__file__)

    # 打印兼容性状态
    if COMPATIBILITY_AVAILABLE:
        print_compatibility_status()

    # 检查Python版本兼容性
    is_compatible, python_version = check_python_version()
    if not is_compatible:
        print(f"[IndexTTS2] ⚠️  Python版本兼容性警告")
        print(f"[IndexTTS2] ⚠️  Python Version Compatibility Warning")
        print(f"[IndexTTS2] 当前版本: {python_version}")
        print(f"[IndexTTS2] Current version: {python_version}")
        print(f"[IndexTTS2] IndexTTS2依赖需要Python 3.8-3.11，当前版本可能不兼容")
        print(f"[IndexTTS2] IndexTTS2 dependencies require Python 3.8-3.11, current version may be incompatible")
        print(f"[IndexTTS2] 建议使用Python 3.10或3.11环境")
        print(f"[IndexTTS2] Recommend using Python 3.10 or 3.11 environment")

        # 自动依赖安装已禁用 - Auto dependency installation disabled
        print("[IndexTTS2] 自动依赖安装已禁用，请手动安装所需依赖")
        print("[IndexTTS2] Auto dependency installation disabled, please install dependencies manually")

    # 检查是否已安装 indextts 包
    # Check if indextts package is already installed
    try:
        import indextts
        print("[IndexTTS2] ✓ indextts 包已安装")
        print(f"[IndexTTS2] ✓ indextts package installed at: {indextts.__file__}")
        return  # 成功导入，直接返回
    except ImportError:
        print("[IndexTTS2] indextts 包未安装，尝试安装...")
        print("[IndexTTS2] indextts package not installed, attempting installation...")

        # 如果Python版本不兼容，跳过自动安装
        if not is_compatible:
            print("[IndexTTS2] 由于Python版本不兼容，跳过自动安装")
            print("[IndexTTS2] Skipping auto-installation due to Python version incompatibility")
            print("[IndexTTS2] 请手动安装兼容版本的依赖或使用兼容的Python版本")
            print("[IndexTTS2] Please manually install compatible dependencies or use compatible Python version")

            # 尝试添加本地路径作为备用方案
            local_indextts = os.path.join(plugin_dir, "indextts")
            if os.path.exists(local_indextts):
                print(f"[IndexTTS2] 尝试使用本地 indextts 路径: {local_indextts}")
                print(f"[IndexTTS2] Trying to use local indextts path: {local_indextts}")
                if plugin_dir not in sys.path:
                    sys.path.insert(0, plugin_dir)
                try:
                    import indextts
                    print("[IndexTTS2] ✓ 本地 indextts 导入成功")
                    print("[IndexTTS2] ✓ Local indextts import successful")
                    return
                except ImportError as e:
                    print(f"[IndexTTS2] ✗ 本地 indextts 导入失败: {e}")
                    print(f"[IndexTTS2] ✗ Local indextts import failed: {e}")

            print("[IndexTTS2] 插件将以有限功能模式运行")
            print("[IndexTTS2] Plugin will run in limited functionality mode")
            return

        # 尝试从 index-tts 目录安装
        # Try to install from index-tts directory
        index_tts_dir = os.path.join(plugin_dir, "index-tts")
        if os.path.exists(index_tts_dir) and os.path.exists(os.path.join(index_tts_dir, "setup.py")):
            print(f"[IndexTTS2] 发现 indextts 源码目录: {index_tts_dir}")
            print(f"[IndexTTS2] Found indextts source directory: {index_tts_dir}")
            print("[IndexTTS2] 自动安装已禁用，请手动运行: pip install -e index-tts/")
            print("[IndexTTS2] Auto installation disabled, please run manually: pip install -e index-tts/")
            # 自动安装已禁用
            # try:
            #     subprocess.run([sys.executable, "-m", "pip", "install", "-e", index_tts_dir],
            #                  check=True, capture_output=True, text=True)
            #     print("[IndexTTS2] ✓ indextts 包安装成功！")
            #     print("[IndexTTS2] ✓ indextts package installation successful!")
            #     # 验证安装
            #     import indextts
            #     print("[IndexTTS2] ✓ 安装验证成功")
            #     print("[IndexTTS2] ✓ Installation verification successful")
            #     return
            # except subprocess.CalledProcessError as e:
            #     print(f"[IndexTTS2] ✗ 安装失败: {e}")
            #     print(f"[IndexTTS2] ✗ Installation failed: {e}")
            #     if e.stderr:
            #         print(f"[IndexTTS2] 错误详情: {e.stderr[:500]}...")
            #         print(f"[IndexTTS2] Error details: {e.stderr[:500]}...")
            #     # 安装失败，尝试本地路径方案
            #     print("[IndexTTS2] 尝试使用本地路径方案...")
            #     print("[IndexTTS2] Trying local path solution...")

        # 如果没有 index-tts 目录，或安装失败，尝试添加本地 indextts 到路径
        # If no index-tts directory or installation failed, try adding local indextts to path
        local_indextts = os.path.join(plugin_dir, "indextts")
        if os.path.exists(local_indextts):
            print(f"[IndexTTS2] 添加本地 indextts 路径: {local_indextts}")
            print(f"[IndexTTS2] Adding local indextts path: {local_indextts}")
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            # 验证导入
            try:
                import indextts
                print("[IndexTTS2] ✓ 本地 indextts 导入成功")
                print("[IndexTTS2] ✓ Local indextts import successful")
                return
            except ImportError as e:
                print(f"[IndexTTS2] ✗ 本地 indextts 导入失败: {e}")
                print(f"[IndexTTS2] ✗ Local indextts import failed: {e}")
                print("[IndexTTS2] 请参考 INSTALL.md 进行手动安装")
                print("[IndexTTS2] Please refer to INSTALL.md for manual installation")

        # 所有方法都失败了
        print("[IndexTTS2] ⚠️  无法自动设置 indextts 环境")
        print("[IndexTTS2] ⚠️  Unable to automatically setup indextts environment")
        print("[IndexTTS2] 请手动安装:")
        print("[IndexTTS2] Please install manually:")
        print("[IndexTTS2] 1. 确保Python版本为3.8-3.11")
        print("[IndexTTS2] 1. Ensure Python version is 3.8-3.11")
        print("[IndexTTS2] 2. 运行: pip install -e index-tts/")
        print("[IndexTTS2] 2. Run: pip install -e index-tts/")
        print("[IndexTTS2] 3. 或参考 INSTALL.md 详细说明")
        print("[IndexTTS2] 3. Or refer to INSTALL.md for detailed instructions")

# 插件初始化时自动加载节点和环境
# Automatically load nodes and setup environment on plugin init
auto_setup()

# 导入所有节点类
# Import all node classes
from .nodes.basic_tts_node import IndexTTS2BasicNode
from .nodes.duration_control_node import IndexTTS2DurationNode
from .nodes.emotion_control_node import IndexTTS2EmotionNode
from .nodes.advanced_control_node import IndexTTS2AdvancedNode
from .nodes.model_manager_node import IndexTTS2ModelManagerNode
from .nodes.audio_utils_node import IndexTTS2AudioUtilsNode
from .nodes.audio_enhancement_node import AudioEnhancementNode
from .nodes.multi_talk_node import IndexTTS2MultiTalkNode
from .nodes.speaker_emotion_config_node import IndexTTS2SpeakerEmotionConfigNode
from .nodes.emotion_voice_multi_talk_node import IndexTTS2EmotionVoiceMultiTalkNode
from .nodes.model_cache_info_node import IndexTTS2ModelCacheInfoNode
from .nodes.transformers_compatibility_check_node import IndexTTS2TransformersCompatibilityCheckNode
from .nodes.qwen_model_status_node import IndexTTS2QwenModelStatusNode
from .nodes.qwen_model_display_node import IndexTTS2QwenModelDisplayNode
# 移除自定义音频加载节点，直接使用ComfyUI官方的"加载音频"节点



# 节点类映射
# Node class mappings
NODE_CLASS_MAPPINGS = {
    "IndexTTS2_Basic": IndexTTS2BasicNode,
    "IndexTTS2_Duration": IndexTTS2DurationNode,
    "IndexTTS2_Emotion": IndexTTS2EmotionNode,
    "IndexTTS2_Advanced": IndexTTS2AdvancedNode,
    "IndexTTS2_ModelManager": IndexTTS2ModelManagerNode,
    "IndexTTS2_AudioUtils": IndexTTS2AudioUtilsNode,
    "IndexTTS2_AudioEnhancement": AudioEnhancementNode,
    "IndexTTS2_MultiTalk": IndexTTS2MultiTalkNode,
    "IndexTTS2_SpeakerEmotionConfig": IndexTTS2SpeakerEmotionConfigNode,
    "IndexTTS2_EmotionVoiceMultiTalk": IndexTTS2EmotionVoiceMultiTalkNode,
    "IndexTTS2_ModelCacheInfo": IndexTTS2ModelCacheInfoNode,
    "IndexTTS2_TransformersCompatCheck": IndexTTS2TransformersCompatibilityCheckNode,
    "IndexTTS2_QwenModelStatus": IndexTTS2QwenModelStatusNode,
    "IndexTTS2_QwenModelDisplay": IndexTTS2QwenModelDisplayNode,
    # 移除自定义音频加载节点
}

# 节点显示名称映射
# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2_Basic": "IndexTTS2 Basic TTS",
    "IndexTTS2_Duration": "IndexTTS2 Duration Control",
    "IndexTTS2_Emotion": "IndexTTS2 Emotion Control",
    "IndexTTS2_Advanced": "IndexTTS2 Advanced Control",
    "IndexTTS2_ModelManager": "IndexTTS2 Model Manager",
    "IndexTTS2_AudioUtils": "IndexTTS2 Audio Utils",
    "IndexTTS2_AudioEnhancement": "IndexTTS2 Audio Enhancement",
    "IndexTTS2_MultiTalk": "IndexTTS2 Multi-Talk with Emotion Control",
    "IndexTTS2_SpeakerEmotionConfig": "IndexTTS2 Speaker Emotion Config",
    "IndexTTS2_EmotionVoiceMultiTalk": "IndexTTS2 Emotion Voice Multi-Talk",
    "IndexTTS2_ModelCacheInfo": "IndexTTS2 Model Cache Info",
    "IndexTTS2_TransformersCompatCheck": "IndexTTS2 Transformers Compatibility Check",
    "IndexTTS2_QwenModelStatus": "IndexTTS2 Qwen Model Status",
    "IndexTTS2_QwenModelDisplay": "IndexTTS2 Qwen Model Display",
    # 移除自定义音频加载节点
}

# 导出节点映射供ComfyUI使用
# Export node mappings for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 打印加载状态和版本信息
print("[IndexTTS2] ✓ 所有节点加载完成，包括音质增强节点和情绪语音多人对话节点")
print("[IndexTTS2] ✓ All nodes loaded, including audio enhancement node and emotion voice multi-talk node")

# 打印版本更新信息
print(f"[IndexTTS2] 🎉 版本 {__version__} 新功能:")
print(f"[IndexTTS2] 🎉 Version {__version__} New Features:")
print("[IndexTTS2]   🤖 智能 Qwen 模型兼容性系统 (支持 Transformers 4.56.1+)")
print("[IndexTTS2]   🤖 Smart Qwen Model Compatibility System (Supports Transformers 4.56.1+)")
print("[IndexTTS2]   📊 新增 Qwen 模型状态显示节点")
print("[IndexTTS2]   📊 New Qwen Model Status Display Nodes")
print("[IndexTTS2]   🛡️ 完全修复兼容性问题，零配置自动适配")
print("[IndexTTS2]   🛡️ Complete compatibility fix with zero-configuration auto-adaptation")
print("[IndexTTS2] 🚀 IndexTTS2 准备就绪！Ready to use!")
print("[IndexTTS2] 📝 查看 CHANGELOG.md 了解详细更新内容")
print("[IndexTTS2] 📝 Check CHANGELOG.md for detailed update information")
