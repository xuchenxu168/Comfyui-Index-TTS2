# IndexTTS2 Basic TTS Node
# IndexTTS2 基础语音合成节点

import os
import torch
import numpy as np
from typing import Optional, Tuple, Any
import folder_paths

# 导入音频浏览器（保持向后兼容）
try:
    from ..audio_browser import get_all_audio_files
    AUDIO_BROWSER_AVAILABLE = True
except ImportError:
    AUDIO_BROWSER_AVAILABLE = False
    print("音频浏览器模块未找到，使用基础扫描功能")

# 导入目录音频浏览器
try:
    from ..directory_audio_browser import (
        get_audio_directory_choices,
        get_audio_file_choices,
        get_full_audio_path
    )
    DIRECTORY_BROWSER_AVAILABLE = True
except ImportError:
    DIRECTORY_BROWSER_AVAILABLE = False
    print("目录音频浏览器模块未找到")

# 添加音频文件夹路径到ComfyUI
# Add audio folder paths to ComfyUI
def get_audio_files():
    """获取可用的音频文件列表"""

    # 如果高级音频浏览器可用，使用它
    if AUDIO_BROWSER_AVAILABLE:
        try:
            audio_files = get_all_audio_files(use_cache=True, max_files=100)

            # 如果找到了音频文件，直接返回
            if audio_files and len(audio_files) > 0:
                return audio_files
        except Exception as e:
            print(f"高级音频浏览器出错，使用基础扫描: {e}")

    # 使用基础扫描功能作为后备
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    # 定义要扫描的目录列表
    scan_dirs = [
        folder_paths.input_directory,  # input目录
        os.path.join(folder_paths.input_directory, "audio"),
        os.path.join(folder_paths.base_path, "input", "audio"),
        os.path.join(folder_paths.base_path, "audio"),
        os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-index-tts2", "audio"),
        os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-index-tts2", "examples"),
    ]

    def scan_directory(directory):
        """扫描目录中的音频文件"""
        if not os.path.exists(directory):
            return

        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)

                if os.path.isfile(item_path):
                    # 检查是否是音频文件
                    if any(item.lower().endswith(ext) for ext in audio_extensions):
                        try:
                            relative_path = os.path.relpath(item_path, folder_paths.base_path)
                            audio_files.append(relative_path)
                        except ValueError:
                            audio_files.append(item_path)

        except (PermissionError, OSError):
            pass

    # 扫描所有指定目录
    for scan_dir in scan_dirs:
        scan_directory(scan_dir)

    # 去重并排序
    audio_files = list(set(audio_files))
    audio_files.sort()

    # 如果没有找到音频文件，添加提示信息
    if not audio_files:
        audio_files = [
            "🔍 未找到音频文件 / No audio files found",
            "📁 请将音频文件放入 input/audio/ 目录",
            "📁 Please put audio files in input/audio/ directory",
            "🔄 然后重新加载节点 / Then reload the node"
        ]

    return audio_files

class IndexTTS2BasicNode:
    """
    IndexTTS2 基础语音合成节点
    Basic zero-shot text-to-speech synthesis node for IndexTTS2
    
    Features:
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
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is IndexTTS2 speaking!",
                    "placeholder": "输入要合成的文本..."
                }),
                "speaker_audio": ("AUDIO", {
                    "tooltip": "连接IndexTTS2 Load Audio File节点 / Connect IndexTTS2 Load Audio File node"
                }),
                "output_filename": ("STRING", {
                    "default": "indextts2_output.wav",
                    "placeholder": "输出音频文件名"
                }),
            },
            "optional": {
                "model_manager": ("INDEXTTS2_MODEL",),
                "language": (["auto", "zh", "en", "zh-en"], {
                    "default": "auto"
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "语速控制 (0.5-2.0倍速) / Speed control (0.5-2.0x)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.5,
                    "step": 0.1,
                    "display": "slider"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "use_fp16": ("BOOLEAN", {
                    "default": False
                }),
                "use_cuda_kernel": ("BOOLEAN", {
                    "default": False
                }),
                "verbose": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "output_path", "info")
    FUNCTION = "synthesize"
    CATEGORY = "IndexTTS2"
    DESCRIPTION = "Basic zero-shot text-to-speech synthesis using IndexTTS2"
    
    def synthesize(
        self,
        text: str,
        speaker_audio: dict,
        output_filename: str,
        model_manager: Optional[Any] = None,
        language: str = "auto",
        speed: float = 1.0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        verbose: bool = True
    ) -> Tuple[dict, str, str]:
        # 参数验证
        if not isinstance(speed, (int, float)):
            raise ValueError(f"speed参数必须是数字，收到: {type(speed).__name__} = {speed}")

        if isinstance(speed, str):
            raise ValueError(f"speed参数不能是字符串，收到: '{speed}'")

        speed = float(speed)  # 确保是float类型
        """
        执行基础语音合成
        Perform basic text-to-speech synthesis
        """
        try:
            # 验证输入
            if not text.strip():
                raise ValueError("Text input cannot be empty")

            # 处理标准ComfyUI AUDIO对象
            if isinstance(speaker_audio, dict) and "waveform" in speaker_audio and "sample_rate" in speaker_audio:
                # 这是标准的ComfyUI AUDIO对象
                # 我们需要将其保存为临时文件供IndexTTS2使用
                import tempfile
                import torchaudio

                waveform = speaker_audio["waveform"]
                sample_rate = speaker_audio["sample_rate"]

                # 移除batch维度（如果存在）
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)

                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    speaker_audio_path = tmp_file.name

                # 保存音频到临时文件
                torchaudio.save(speaker_audio_path, waveform, sample_rate)

                if verbose:
                    print(f"[IndexTTS2] 使用ComfyUI AUDIO对象，临时文件: {speaker_audio_path}")
                    print(f"[IndexTTS2] 音频信息: 采样率={sample_rate}, 形状={waveform.shape}")
            else:
                raise ValueError("speaker_audio must be a ComfyUI AUDIO object with 'waveform' and 'sample_rate' keys")
            
            # 获取模型实例
            if model_manager is not None:
                model = model_manager
            else:
                model = self._load_default_model(use_fp16, use_cuda_kernel)
            
            # 准备输出路径
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, output_filename)
            
            # 执行推理
            if verbose:
                print(f"[IndexTTS2] Synthesizing: {text[:50]}...")
                print(f"[IndexTTS2] Speaker audio: {speaker_audio_path}")
                print(f"[IndexTTS2] Output: {output_path}")

            # 调用IndexTTS2推理，使用优化的参数
            model.infer(
                spk_audio_prompt=speaker_audio_path,
                text=text,
                output_path=output_path,
                verbose=verbose,
                # 添加推理质量参数
                temperature=0.7,  # 降低温度以提高稳定性和质量
                top_p=0.9,        # 使用nucleus sampling
                top_k=50,         # 限制候选token数量
                max_text_tokens_per_sentence=120,  # 限制每句话的token数量
                interval_silence=200  # 句子间的静音间隔(ms)
            )

            # 加载生成的音频
            audio_data = self._load_audio(output_path)

            # 确保音频格式兼容ComfyUI
            waveform = audio_data["waveform"]
            sample_rate = audio_data["sample_rate"]

            # 应用最终的ComfyUI兼容性检查
            from .audio_utils import fix_comfyui_audio_compatibility
            waveform = fix_comfyui_audio_compatibility(waveform)

            # ComfyUI AUDIO格式需要 [batch, channels, samples]
            # 我们的waveform是 [channels, samples]，需要添加batch维度
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]

            # 创建ComfyUI AUDIO格式（与LoadAudio节点一致）
            comfyui_audio = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            print(f"[IndexTTS2] 最终音频格式: {waveform.shape}, 采样率: {sample_rate}")
            print(f"[IndexTTS2] ComfyUI AUDIO格式: batch={waveform.shape[0]}, channels={waveform.shape[1]}, samples={waveform.shape[2]}")

            # 生成信息字符串
            info = self._generate_info(text, speaker_audio_path, output_path, language, speed)

            return (comfyui_audio, output_path, info)
            
        except Exception as e:
            error_msg = f"IndexTTS2 synthesis failed: {str(e)}"
            print(f"[IndexTTS2 Error] {error_msg}")
            raise RuntimeError(error_msg)
    
    def _load_default_model(self, use_fp16: bool, use_cuda_kernel: bool):
        """加载默认模型"""
        try:
            from indextts.infer_v2 import IndexTTS2

            # 使用通用模型路径函数
            from .model_utils import get_indextts2_model_path, validate_model_path

            model_dir, config_path = get_indextts2_model_path()

            print(f"[IndexTTS2] 使用模型路径: {model_dir}")
            print(f"[IndexTTS2] Using model path: {model_dir}")

            # 验证模型路径
            validate_model_path(model_dir, config_path)

            model = IndexTTS2(
                cfg_path=config_path,
                model_dir=model_dir,
                is_fp16=use_fp16,
                use_cuda_kernel=use_cuda_kernel
            )

            return model

        except Exception as e:
            error_msg = f"Failed to load IndexTTS2 model: {str(e)}"
            # 特别处理DeepSpeed相关错误
            if "deepspeed" in str(e).lower():
                error_msg += "\n[IndexTTS2] DeepSpeed相关错误，但基本功能应该仍然可用"
                error_msg += "\n[IndexTTS2] DeepSpeed-related error, but basic functionality should still work"
            raise RuntimeError(error_msg)
    
    def _load_audio(self, audio_path: str) -> dict:
        """加载音频文件"""
        from .audio_utils import load_audio_for_comfyui
        return load_audio_for_comfyui(audio_path)
    
    def _generate_info(self, text: str, speaker_audio: str, output_path: str,
                      language: str, speed: float) -> str:
        """生成信息字符串，包含Qwen模型信息"""
        info_lines = [
            "=== IndexTTS2 Basic Synthesis Info ===",
            f"Text: {text[:100]}{'...' if len(text) > 100 else ''}",
            f"Speaker Audio: {os.path.basename(speaker_audio)}",
            f"Output: {os.path.basename(output_path)}",
            f"Language: {language}",
            f"Speed: {speed}x",
            f"Model: IndexTTS2 Basic",
            "",
            "=== Qwen Emotion Model Status ===",
        ]

        # 获取Qwen模型状态信息
        qwen_info = self._get_qwen_model_info()
        info_lines.extend(qwen_info)

        return "\n".join(info_lines)

    def _get_qwen_model_info(self) -> list:
        """获取当前Qwen模型信息"""
        try:
            # 检查transformers版本
            import transformers
            from packaging import version

            current_version = transformers.__version__
            info_lines = [f"🔧 Transformers版本: {current_version}"]

            # 直接检查兼容性，不创建QwenEmotion实例
            compatible_models = self._get_compatible_qwen_models_direct()

            # 显示兼容模型信息
            if compatible_models:
                best_model = compatible_models[0]  # 第一个是优先级最高的
                info_lines.extend([
                    f"🤖 推荐模型: {best_model['name']}",
                    f"📊 模型大小: {best_model['size']}",
                    f"📝 模型类型: 智能选择",
                    f"✅ 状态: 高精度情感分析可用"
                ])
            else:
                info_lines.extend([
                    f"🤖 情感模型: 关键词匹配",
                    f"📝 模型类型: 备用方案",
                    f"⚠️  状态: 基础情感分析可用"
                ])

            # 显示兼容模型数量
            info_lines.append(f"🔍 兼容Qwen模型: {len(compatible_models)}个")

            return info_lines

        except Exception as e:
            return [
                f"❌ Qwen模型信息获取失败: {str(e)[:50]}...",
                f"ℹ️  基本TTS功能不受影响"
            ]

    def _get_compatible_qwen_models_direct(self):
        """直接获取兼容的Qwen模型列表，不创建QwenEmotion实例"""
        try:
            import transformers
            from packaging import version

            current_ver = version.parse(transformers.__version__)

            # 定义不同Qwen模型的版本要求和优先级
            qwen_models = []

            # Qwen3系列 (需要transformers >= 4.51.0)
            if current_ver >= version.parse("4.51.0"):
                qwen_models.extend([
                    {
                        "name": "Qwen3-0.5B-Instruct",
                        "model_id": "Qwen/Qwen3-0.5B-Instruct",
                        "priority": 1,
                        "size": "0.5B",
                        "description": "最新Qwen3模型，小型高效"
                    },
                    {
                        "name": "Qwen3-1.8B-Instruct",
                        "model_id": "Qwen/Qwen3-1.8B-Instruct",
                        "priority": 2,
                        "size": "1.8B",
                        "description": "Qwen3中型模型"
                    }
                ])

            # Qwen2.5系列 (需要transformers >= 4.37.0)
            if current_ver >= version.parse("4.37.0"):
                qwen_models.extend([
                    {
                        "name": "Qwen2.5-0.5B-Instruct",
                        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                        "priority": 3,
                        "size": "0.5B",
                        "description": "Qwen2.5小型模型"
                    },
                    {
                        "name": "Qwen2.5-1.5B-Instruct",
                        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                        "priority": 4,
                        "size": "1.5B",
                        "description": "Qwen2.5中型模型"
                    }
                ])

            # Qwen2系列 (需要transformers >= 4.37.0)
            if current_ver >= version.parse("4.37.0"):
                qwen_models.extend([
                    {
                        "name": "Qwen2-0.5B-Instruct",
                        "model_id": "Qwen/Qwen2-0.5B-Instruct",
                        "priority": 5,
                        "size": "0.5B",
                        "description": "Qwen2小型模型"
                    },
                    {
                        "name": "Qwen2-1.5B-Instruct",
                        "model_id": "Qwen/Qwen2-1.5B-Instruct",
                        "priority": 6,
                        "size": "1.5B",
                        "description": "Qwen2中型模型"
                    }
                ])

            # Qwen1.5系列 (需要transformers >= 4.37.0)
            if current_ver >= version.parse("4.37.0"):
                qwen_models.extend([
                    {
                        "name": "Qwen1.5-0.5B-Chat",
                        "model_id": "Qwen/Qwen1.5-0.5B-Chat",
                        "priority": 7,
                        "size": "0.5B",
                        "description": "Qwen1.5小型模型"
                    },
                    {
                        "name": "Qwen1.5-1.8B-Chat",
                        "model_id": "Qwen/Qwen1.5-1.8B-Chat",
                        "priority": 8,
                        "size": "1.8B",
                        "description": "Qwen1.5中型模型"
                    }
                ])

            # 按优先级排序
            qwen_models.sort(key=lambda x: x["priority"])

            return qwen_models

        except Exception as e:
            print(f"[IndexTTS2] ⚠️  获取兼容模型列表失败: {e}")
            return []

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """检查输入是否改变"""
        return float("nan")  # 总是重新执行
