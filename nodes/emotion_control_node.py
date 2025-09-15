# IndexTTS2 Emotion Control Node
# IndexTTS2 情感控制节点

import os
import torch
import numpy as np
from typing import Optional, Tuple, Any, List
import folder_paths

# 导入音频文件获取函数
from .basic_tts_node import get_audio_files

# 导入高级音频浏览器
try:
    from ..audio_browser import get_all_audio_files, clear_audio_cache
    AUDIO_BROWSER_AVAILABLE = True
except ImportError:
    AUDIO_BROWSER_AVAILABLE = False

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

class IndexTTS2EmotionNode:
    """
    IndexTTS2 情感控制节点
    Emotion-controlled speech synthesis node for IndexTTS2
    
    Features:
    - Emotion control via audio prompts
    - Emotion vector control (8-dimensional)
    - Text-based emotion description
    - Speaker-emotion disentanglement
    - Independent timbre and emotion control
    """
    
    def __init__(self):
        self.model = None
        
    @classmethod
    def INPUT_TYPES(cls):
        # 简化为文件选择器模式

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "This speech will have controlled emotional expression.",
                    "placeholder": "输入要合成的文本（带情感控制）..."
                }),
                "speaker_audio": ("AUDIO", {
                    "tooltip": "连接IndexTTS2 Load Audio File节点 / Connect IndexTTS2 Load Audio File node"
                }),
                "emotion_mode": (["audio_prompt", "emotion_vector", "text_description", "auto"], {
                    "default": "audio_prompt"
                }),
                "output_filename": ("STRING", {
                    "default": "indextts2_emotion_output.wav",
                    "placeholder": "Output audio filename"
                }),
            },
            "optional": {
                "model_manager": ("INDEXTTS2_MODEL",),
                "emotion_audio": ("AUDIO", {
                    "tooltip": "连接加载音频节点提供情绪参考音频 / Connect Load Audio node for emotion reference"
                }),
                "emotion_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "forceInput": False
                }),
                # 8-dimensional emotion vector: Happy, Angry, Sad, Fear, Hate, Low, Surprise, Neutral
                "happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "sad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "fear": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "hate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "surprise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "neutral": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "emotion_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the desired emotion in text..."
                }),
                "use_random": ("BOOLEAN", {
                    "default": False
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
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "output_path", "info", "emotion_analysis")
    FUNCTION = "synthesize_with_emotion"
    CATEGORY = "IndexTTS2/Control"
    DESCRIPTION = "Emotion-controlled speech synthesis with multiple control methods"
    
    def synthesize_with_emotion(
        self,
        text: str,
        speaker_audio: dict,
        emotion_mode: str,
        output_filename: str,
        model_manager: Optional[Any] = None,
        emotion_audio: Optional[dict] = None,
        emotion_alpha: float = 1.0,
        happy: float = 0.0,
        angry: float = 0.0,
        sad: float = 0.0,
        fear: float = 0.0,
        hate: float = 0.0,
        low: float = 0.0,
        surprise: float = 0.0,
        neutral: float = 1.0,
        emotion_text: str = "",
        use_random: bool = False,
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        verbose: bool = True
    ) -> Tuple[dict, str, str, str]:
        """
        执行情感控制的语音合成
        Perform emotion-controlled text-to-speech synthesis
        """
        try:
            # 初始化临时文件列表用于清理
            temp_files_to_cleanup = []

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
            
            if verbose:
                print(f"[IndexTTS2 Emotion] Mode: {emotion_mode}")
                print(f"[IndexTTS2 Emotion] Text: {text[:50]}...")
                print(f"[IndexTTS2 Emotion] Speaker: {os.path.basename(speaker_audio_path)}")

            # 根据模式执行不同的推理策略
            emotion_analysis = ""

            if emotion_mode == "audio_prompt":
                # 处理情绪音频输入
                emotion_audio_path = ""
                if emotion_audio is not None:
                    emotion_audio_path = self._prepare_emotion_audio(emotion_audio, verbose)
                    if emotion_audio_path:
                        temp_files_to_cleanup.append(emotion_audio_path)

                emotion_analysis = self._synthesize_audio_emotion(
                    model, text, speaker_audio_path, emotion_audio_path, emotion_alpha,
                    output_path, use_random, verbose
                )
            elif emotion_mode == "emotion_vector":
                emotion_vector = [happy, angry, sad, fear, hate, low, surprise, neutral]
                emotion_analysis = self._synthesize_vector_emotion(
                    model, text, speaker_audio_path, emotion_vector,
                    output_path, use_random, verbose
                )
            elif emotion_mode == "text_description":
                emotion_analysis = self._synthesize_text_emotion(
                    model, text, speaker_audio_path, emotion_text,
                    output_path, use_random, verbose
                )
            else:  # auto mode
                emotion_analysis = self._synthesize_auto_emotion(
                    model, text, speaker_audio_path, output_path, use_random, verbose
                )
            
            # 加载生成的音频
            audio_data = self._load_audio(output_path)
            
            # 生成信息字符串
            # 对于audio_prompt模式，传递转换后的文件路径
            emotion_audio_for_info = ""
            if emotion_mode == "audio_prompt" and 'emotion_audio_path' in locals():
                emotion_audio_for_info = emotion_audio_path

            info = self._generate_info(
                text, speaker_audio_path, output_path, emotion_mode,
                emotion_audio_for_info, emotion_alpha, [happy, angry, sad, fear, hate, low, surprise, neutral],
                emotion_text
            )
            
            # 确保音频格式兼容ComfyUI
            waveform = audio_data["waveform"]
            sample_rate = audio_data["sample_rate"]

            # 应用ComfyUI兼容性修复
            from .audio_utils import fix_comfyui_audio_compatibility
            waveform = fix_comfyui_audio_compatibility(waveform)

            # ComfyUI AUDIO格式需要 [batch, channels, samples]
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]

            # 创建ComfyUI AUDIO格式
            comfyui_audio = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            # 清理临时文件
            self._cleanup_temp_files(temp_files_to_cleanup)

            return (comfyui_audio, output_path, info, emotion_analysis)

        except Exception as e:
            # 异常时也要清理临时文件
            if 'temp_files_to_cleanup' in locals():
                self._cleanup_temp_files(temp_files_to_cleanup)

            error_msg = f"IndexTTS2 emotion synthesis failed: {str(e)}"
            print(f"[IndexTTS2 Emotion Error] {error_msg}")
            raise RuntimeError(error_msg)
    
    def _synthesize_audio_emotion(self, model, text: str, speaker_audio: str, 
                                 emotion_audio: str, emotion_alpha: float,
                                 output_path: str, use_random: bool, verbose: bool) -> str:
        """音频情感控制模式合成"""
        if emotion_audio and os.path.exists(emotion_audio):
            if verbose:
                print(f"[IndexTTS2] Using emotion audio: {os.path.basename(emotion_audio)}")
                print(f"[IndexTTS2] Emotion alpha: {emotion_alpha}")
            
            model.infer(
                spk_audio_prompt=speaker_audio,
                text=text,
                output_path=output_path,
                emo_audio_prompt=emotion_audio,
                emo_alpha=emotion_alpha,
                use_random=use_random,
                verbose=verbose
            )
            
            return f"Audio emotion control with {os.path.basename(emotion_audio)} (alpha: {emotion_alpha})"
        else:
            # 回退到基础推理
            model.infer(
                spk_audio_prompt=speaker_audio,
                text=text,
                output_path=output_path,
                use_random=use_random,
                verbose=verbose
            )
            
            return "No emotion audio provided, using speaker audio emotion"
    
    def _synthesize_vector_emotion(self, model, text: str, speaker_audio: str,
                                  emotion_vector: List[float], output_path: str,
                                  use_random: bool, verbose: bool) -> str:
        """情感向量控制模式合成"""
        emotion_names = ["Happy", "Angry", "Sad", "Fear", "Hate", "Low", "Surprise", "Neutral"]

        # 检查情感向量是否全为零
        max_emotion_value = max(emotion_vector)
        if max_emotion_value == 0.0:
            if verbose:
                print(f"[IndexTTS2] All emotion values are zero, using neutral emotion")
            # 当所有情感值都为0时，设置一个小的中性情感值以避免模型问题
            emotion_vector = emotion_vector.copy()  # 避免修改原始向量
            emotion_vector[7] = 0.1  # 设置Neutral为0.1

        if verbose:
            active_emotions = []
            for i, (name, value) in enumerate(zip(emotion_names, emotion_vector)):
                if value > 0.05:  # 降低阈值以显示更多信息
                    active_emotions.append(f"{name}: {value:.2f}")
            if active_emotions:
                print(f"[IndexTTS2] Emotion vector: {', '.join(active_emotions)}")
            else:
                print(f"[IndexTTS2] Emotion vector: All neutral")

        model.infer(
            spk_audio_prompt=speaker_audio,
            text=text,
            output_path=output_path,
            emo_vector=emotion_vector,
            use_random=use_random,
            verbose=verbose
        )
        
        # 分析主要情感
        max_value = max(emotion_vector)
        if max_value == 0.0:
            # 如果所有情感值都是0，使用中性情感
            dominant_emotion = "Neutral"
            dominant_value = 0.0
            return f"Vector emotion control - All emotions neutral (0.00)"
        else:
            max_idx = emotion_vector.index(max_value)
            dominant_emotion = emotion_names[max_idx]
            dominant_value = emotion_vector[max_idx]
            return f"Vector emotion control - Dominant: {dominant_emotion} ({dominant_value:.2f})"
    
    def _synthesize_text_emotion(self, model, text: str, speaker_audio: str,
                                emotion_text: str, output_path: str,
                                use_random: bool, verbose: bool) -> str:
        """文本情感控制模式合成"""
        if emotion_text.strip():
            if verbose:
                print(f"[IndexTTS2] Emotion description: {emotion_text[:50]}...")
            
            model.infer(
                spk_audio_prompt=speaker_audio,
                text=text,
                output_path=output_path,
                use_emo_text=True,
                emo_text=emotion_text,
                use_random=use_random,
                verbose=verbose
            )
            
            return f"Text emotion control: {emotion_text[:100]}{'...' if len(emotion_text) > 100 else ''}"
        else:
            # 从合成文本推断情感
            model.infer(
                spk_audio_prompt=speaker_audio,
                text=text,
                output_path=output_path,
                use_emo_text=True,
                use_random=use_random,
                verbose=verbose
            )
            
            return "Text emotion control: Inferred from synthesis text"
    
    def _synthesize_auto_emotion(self, model, text: str, speaker_audio: str,
                                output_path: str, use_random: bool, verbose: bool) -> str:
        """自动情感模式合成"""
        model.infer(
            spk_audio_prompt=speaker_audio,
            text=text,
            output_path=output_path,
            use_random=use_random,
            verbose=verbose
        )
        
        return "Auto emotion mode: Using speaker audio emotion characteristics"
    
    def _load_default_model(self, use_fp16: bool, use_cuda_kernel: bool):
        """加载默认模型"""
        try:
            from indextts.infer_v2 import IndexTTS2

            # 使用通用模型路径函数
            from .model_utils import get_indextts2_model_path, validate_model_path

            model_dir, config_path = get_indextts2_model_path()
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
        try:
            from .audio_utils import load_audio_for_comfyui
            return load_audio_for_comfyui(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")
    
    def _generate_info(self, text: str, speaker_audio: str, output_path: str,
                      emotion_mode: str, emotion_audio: str, emotion_alpha: float,
                      emotion_vector: List[float], emotion_text: str) -> str:
        """生成信息字符串"""
        info_lines = [
            "=== IndexTTS2 Emotion Control Info ===",
            f"Text: {text[:100]}{'...' if len(text) > 100 else ''}",
            f"Speaker Audio: {os.path.basename(speaker_audio)}",
            f"Output: {os.path.basename(output_path)}",
            f"Emotion Mode: {emotion_mode}",
        ]
        
        if emotion_mode == "audio_prompt" and emotion_audio:
            info_lines.append(f"Emotion Audio: {os.path.basename(emotion_audio)}")
            info_lines.append(f"Emotion Alpha: {emotion_alpha}")
        elif emotion_mode == "emotion_vector":
            emotion_names = ["Happy", "Angry", "Sad", "Fear", "Hate", "Low", "Surprise", "Neutral"]
            active_emotions = [f"{name}: {val:.2f}" for name, val in zip(emotion_names, emotion_vector) if val > 0.1]
            info_lines.append(f"Emotion Vector: {', '.join(active_emotions)}")
        elif emotion_mode == "text_description" and emotion_text:
            info_lines.append(f"Emotion Text: {emotion_text[:50]}{'...' if len(emotion_text) > 50 else ''}")
        
        return "\n".join(info_lines)

    def _prepare_emotion_audio(self, emotion_audio: dict, verbose: bool) -> str:
        """将AUDIO对象转换为临时文件路径"""
        import tempfile
        import torchaudio

        if not isinstance(emotion_audio, dict) or "waveform" not in emotion_audio or "sample_rate" not in emotion_audio:
            if verbose:
                print("[IndexTTS2] Invalid emotion audio object, skipping emotion control")
            return ""

        waveform = emotion_audio["waveform"]
        sample_rate = emotion_audio["sample_rate"]

        # 移除batch维度（如果存在）
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        # 保存音频到临时文件
        torchaudio.save(temp_path, waveform, sample_rate)

        if verbose:
            print(f"[IndexTTS2] Emotion audio saved to temporary file: {temp_path}")
            print(f"[IndexTTS2] Emotion audio shape: {waveform.shape}, sample rate: {sample_rate}")

        return temp_path

    def _cleanup_temp_files(self, temp_files: list):
        """清理临时文件"""
        import os

        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                # 清理失败不应该影响主流程
                print(f"[IndexTTS2] Warning: Failed to cleanup temp file {temp_file}: {e}")
                pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
