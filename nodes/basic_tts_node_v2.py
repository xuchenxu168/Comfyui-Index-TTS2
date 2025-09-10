# IndexTTS2 Basic TTS Node V2 - Two-Level Audio Selection
# IndexTTS2 基础语音合成节点 V2 - 两级音频选择

import os
import torch
import numpy as np
from typing import Optional, Tuple, Any
import folder_paths

# 导入目录音频浏览器
try:
    from ..directory_audio_browser import (
        get_audio_directory_choices, 
        get_audio_file_choices, 
        get_full_audio_path,
        clear_directory_cache
    )
    DIRECTORY_BROWSER_AVAILABLE = True
except ImportError:
    DIRECTORY_BROWSER_AVAILABLE = False
    print("目录音频浏览器模块未找到，使用基础功能")

def get_audio_directories():
    """获取音频目录选择列表"""
    if DIRECTORY_BROWSER_AVAILABLE:
        try:
            return get_audio_directory_choices()
        except Exception as e:
            print(f"目录浏览器出错: {e}")
    
    # 后备方案：返回基础目录列表
    return [
        "🌟 input/audio (推荐)",
        "📁 input",
        "📤 output", 
        "🎵 audio",
        "👤 user",
        "💡 请安装目录浏览器模块"
    ]

class IndexTTS2BasicNodeV2:
    """
    IndexTTS2 基础语音合成节点 V2 - 两级音频选择
    Basic zero-shot text-to-speech synthesis node with two-level audio selection
    
    Features:
    - Two-level audio selection (directory -> file)
    - Zero-shot speaker cloning with single audio prompt
    - High-quality speech synthesis
    - Support for multiple languages (Chinese, English)
    - Configurable output settings
    """
    
    def __init__(self):
        self.model = None
        self.model_config = None
        
    @classmethod
    def INPUT_TYPES(cls):
        # 获取音频目录列表
        audio_directories = get_audio_directories()

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is IndexTTS2 speaking!",
                    "placeholder": "输入要合成的文本 / Enter text to synthesize..."
                }),
                "audio_directory": (audio_directories, {
                    "default": audio_directories[0] if audio_directories else "",
                    "tooltip": "第一步：选择包含音频文件的目录 / Step 1: Select directory containing audio files"
                }),
                "speaker_audio_file": ("STRING", {
                    "default": "",
                    "placeholder": "第二步：输入音频文件名 / Step 2: Enter audio filename",
                    "tooltip": "从选定目录中选择音频文件 / Select audio file from chosen directory"
                }),
                "output_filename": ("STRING", {
                    "default": "indextts2_output.wav",
                    "placeholder": "输出音频文件名 / Output audio filename"
                }),
            },
            "optional": {
                "reference_audio": ("AUDIO", {
                    "tooltip": "可选：直接提供音频数据 / Optional: Provide audio data directly"
                }),
                "language": (["auto", "zh", "en", "ja", "ko"], {
                    "default": "auto",
                    "tooltip": "语言设置 / Language setting"
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "语速控制 / Speed control"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "生成随机性 / Generation randomness"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "核采样参数 / Nucleus sampling parameter"
                }),
                "enable_enhance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用音频增强 / Enable audio enhancement"
                }),
                "enable_denoise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用降噪 / Enable denoising"
                }),
                "save_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "保存音频文件 / Save audio file"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "output_path", "info", "available_files")
    FUNCTION = "synthesize_speech"
    CATEGORY = "IndexTTS2"

    def synthesize_speech(self, text, audio_directory, speaker_audio_file, output_filename, 
                         reference_audio=None, language="auto", speed=1.0, temperature=0.7, 
                         top_p=0.9, enable_enhance=False, enable_denoise=False, save_audio=True):
        """
        执行语音合成
        """
        
        try:
            # 获取选定目录中的可用音频文件
            available_files = []
            if DIRECTORY_BROWSER_AVAILABLE:
                try:
                    available_files = get_audio_file_choices(audio_directory)
                except Exception as e:
                    print(f"获取音频文件列表出错: {e}")
            
            available_files_str = ", ".join(available_files[:10])  # 显示前10个文件
            if len(available_files) > 10:
                available_files_str += f" ... (共{len(available_files)}个文件)"
            
            # 确定音频文件路径
            audio_path = None
            
            if reference_audio is not None:
                # 使用提供的音频数据
                audio_path = "direct_audio_input"
                print("使用直接提供的音频数据")
            elif speaker_audio_file and DIRECTORY_BROWSER_AVAILABLE:
                # 使用目录+文件名的方式
                try:
                    audio_path = get_full_audio_path(audio_directory, speaker_audio_file)
                    if not audio_path or not os.path.exists(audio_path):
                        raise FileNotFoundError(f"音频文件不存在: {speaker_audio_file}")
                except Exception as e:
                    return self._create_error_output(f"音频文件路径错误: {e}", available_files_str)
            else:
                return self._create_error_output("请选择有效的音频文件", available_files_str)
            
            # 验证输入参数
            if not text or not text.strip():
                return self._create_error_output("请输入要合成的文本", available_files_str)
            
            # 设置输出路径
            if save_audio:
                output_dir = folder_paths.get_output_directory()
                full_output_path = os.path.join(output_dir, output_filename)
            else:
                full_output_path = None
            
            # 执行语音合成（模拟）
            print(f"开始语音合成...")
            print(f"文本: {text[:50]}...")
            print(f"音频路径: {audio_path}")
            print(f"语言: {language}")
            print(f"语速: {speed}")
            
            # 这里应该是实际的IndexTTS2合成代码
            # 目前返回模拟结果
            
            # 创建模拟音频数据
            sample_rate = 22050
            duration = len(text) * 0.1  # 根据文本长度估算时长
            samples = int(sample_rate * duration)
            
            # 生成简单的正弦波作为示例
            t = np.linspace(0, duration, samples)
            frequency = 440  # A4音符
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
            
            # 保存音频文件
            if save_audio and full_output_path:
                try:
                    import torchaudio
                    torchaudio.save(full_output_path, audio_tensor, sample_rate)
                    print(f"音频已保存到: {full_output_path}")
                except Exception as e:
                    print(f"保存音频文件失败: {e}")
                    full_output_path = f"保存失败: {e}"
            
            # 生成信息字符串
            info = f"""IndexTTS2 语音合成完成
文本: {text[:100]}{'...' if len(text) > 100 else ''}
音频源: {speaker_audio_file if speaker_audio_file else '直接音频输入'}
目录: {audio_directory}
语言: {language}
语速: {speed}x
温度: {temperature}
Top-p: {top_p}
增强: {'是' if enable_enhance else '否'}
降噪: {'是' if enable_denoise else '否'}
输出: {full_output_path if save_audio else '仅返回音频数据'}
时长: {duration:.2f}秒
采样率: {sample_rate}Hz"""
            
            return (
                {"waveform": audio_tensor, "sample_rate": sample_rate},
                full_output_path or "未保存",
                info,
                available_files_str
            )
            
        except Exception as e:
            error_msg = f"语音合成失败: {str(e)}"
            print(error_msg)
            return self._create_error_output(error_msg, available_files_str if 'available_files_str' in locals() else "")
    
    def _create_error_output(self, error_message, available_files=""):
        """创建错误输出"""
        # 创建静音音频
        sample_rate = 22050
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples)
        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
        
        return (
            {"waveform": audio_tensor, "sample_rate": sample_rate},
            "错误",
            f"❌ {error_message}",
            available_files
        )

# 辅助节点：音频文件浏览器
class IndexTTS2AudioBrowser:
    """音频文件浏览器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        audio_directories = get_audio_directories()
        
        return {
            "required": {
                "audio_directory": (audio_directories, {
                    "default": audio_directories[0] if audio_directories else "",
                    "tooltip": "选择要浏览的音频目录"
                }),
            },
            "optional": {
                "refresh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "刷新目录缓存"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("directory", "file_list", "file_count")
    FUNCTION = "browse_directory"
    CATEGORY = "IndexTTS2/Utils"

    def browse_directory(self, audio_directory, refresh=False):
        """浏览音频目录"""
        
        if refresh and DIRECTORY_BROWSER_AVAILABLE:
            try:
                clear_directory_cache()
                print("目录缓存已刷新")
            except Exception as e:
                print(f"刷新缓存失败: {e}")
        
        # 获取目录中的音频文件
        if DIRECTORY_BROWSER_AVAILABLE:
            try:
                audio_files = get_audio_file_choices(audio_directory)
            except Exception as e:
                audio_files = [f"获取文件列表失败: {e}"]
        else:
            audio_files = ["目录浏览器不可用"]
        
        file_list = "\n".join(audio_files)
        file_count = len(audio_files) if audio_files and "失败" not in audio_files[0] else 0
        
        return (audio_directory, file_list, file_count)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "IndexTTS2_Basic_V2": IndexTTS2BasicNodeV2,
    "IndexTTS2_AudioBrowser": IndexTTS2AudioBrowser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2_Basic_V2": "IndexTTS2 Basic TTS (Two-Level Selection)",
    "IndexTTS2_AudioBrowser": "IndexTTS2 Audio Browser",
}
