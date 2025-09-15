# IndexTTS2 Advanced Control Node
# IndexTTS2 高级控制节点

import os
import torch
import numpy as np
from typing import Optional, Tuple, Any, List, Dict
import folder_paths

# 导入音频文件获取函数
from .basic_tts_node import get_audio_files

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

# 导入高级音频浏览器
try:
    from ..audio_browser import get_all_audio_files, clear_audio_cache
    AUDIO_BROWSER_AVAILABLE = True
except ImportError:
    AUDIO_BROWSER_AVAILABLE = False

class IndexTTS2AdvancedNode:
    """
    IndexTTS2 高级控制节点
    Advanced control node for IndexTTS2 with all features combined
    
    Features:
    - Combined duration and emotion control
    - GPT latent representations for stability
    - Three-stage training paradigm benefits
    - Speaker-emotion disentanglement
    - Multi-modal emotion control
    - Advanced prosody control
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
                    "default": "This is advanced IndexTTS2 synthesis with full control.",
                    "placeholder": "输入要合成的文本（高级控制）..."
                }),
                "speaker_audio": ("AUDIO", {
                    "tooltip": "连接IndexTTS2 Load Audio File节点 / Connect IndexTTS2 Load Audio File node"
                }),
                "output_filename": ("STRING", {
                    "default": "indextts2_advanced_output.wav",
                    "placeholder": "Output audio filename"
                }),
            },
            "optional": {
                "model_manager": ("INDEXTTS2_MODEL",),
                
                # Duration Control
                "enable_duration_control": ("BOOLEAN", {
                    "default": False
                }),
                "duration_mode": (["speed_control", "token_control", "target_duration"], {
                    "default": "speed_control"
                }),
                "speed_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "target_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 60.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "token_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 1
                }),
                
                # Emotion Control
                "enable_emotion_control": ("BOOLEAN", {
                    "default": False
                }),
                "emotion_mode": (["audio_prompt", "emotion_vector", "text_description", "mixed"], {
                    "default": "audio_prompt"
                }),
                "emotion_audio": ("AUDIO", {
                    "tooltip": "连接音频加载节点以提供情感参考音频 / Connect audio loading node for emotion reference audio"
                }),
                "emotion_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "emotion_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the desired emotion..."
                }),
                
                # Emotion Vector (8-dimensional)
                "happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "sad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "fear": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "hate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "surprise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "neutral": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                
                # Advanced Settings
                "use_gpt_latents": ("BOOLEAN", {
                    "default": True
                }),
                "prosody_preservation": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "stability_enhancement": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "clarity_boost": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                
                # Model Settings
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
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "FLOAT", "STRING", "DICT")
    RETURN_NAMES = ("audio", "output_path", "info", "duration", "emotion_analysis", "synthesis_stats")
    FUNCTION = "advanced_synthesize"
    CATEGORY = "IndexTTS2/Advanced"
    DESCRIPTION = "Advanced IndexTTS2 synthesis with full feature control"
    
    def advanced_synthesize(
        self,
        text: str,
        speaker_audio: dict,
        output_filename: str,
        model_manager: Optional[Any] = None,
        
        # Duration control
        enable_duration_control: bool = False,
        duration_mode: str = "speed_control",
        speed_multiplier: float = 1.0,
        target_duration: float = 0.0,
        token_count: int = 0,
        
        # Emotion control
        enable_emotion_control: bool = False,
        emotion_mode: str = "audio_prompt",
        emotion_audio: Optional[dict] = None,
        emotion_alpha: float = 1.0,
        emotion_text: str = "",
        happy: float = 0.0,
        angry: float = 0.0,
        sad: float = 0.0,
        fear: float = 0.0,
        hate: float = 0.0,
        low: float = 0.0,
        surprise: float = 0.0,
        neutral: float = 1.0,
        
        # Advanced settings
        use_gpt_latents: bool = True,
        prosody_preservation: float = 0.8,
        stability_enhancement: float = 0.7,
        clarity_boost: float = 0.5,
        
        # Model settings
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_random: bool = False,
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        verbose: bool = True
    ) -> Tuple[dict, str, str, float, str, dict]:
        """
        执行高级语音合成
        Perform advanced text-to-speech synthesis with full control
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
            
            if verbose:
                print(f"[IndexTTS2 Advanced] Starting advanced synthesis...")
                print(f"[IndexTTS2 Advanced] Duration control: {enable_duration_control}")
                print(f"[IndexTTS2 Advanced] Emotion control: {enable_emotion_control}")
                print(f"[IndexTTS2 Advanced] GPT latents: {use_gpt_latents}")
            
            # 准备推理参数
            infer_args = {
                "spk_audio_prompt": speaker_audio_path,
                "text": text,
                "output_path": output_path,
                "use_random": use_random,
                "verbose": verbose
            }
            
            # 添加情感控制参数
            emotion_analysis = "No emotion control"
            if enable_emotion_control:
                emotion_analysis = self._add_emotion_params(
                    infer_args, emotion_mode, emotion_audio, emotion_alpha,
                    emotion_text, [happy, angry, sad, fear, hate, low, surprise, neutral]
                )
            
            # 执行推理
            model.infer(**infer_args)
            
            # 后处理：时长控制
            actual_duration = self._get_audio_duration(output_path)
            if enable_duration_control and duration_mode == "speed_control" and abs(speed_multiplier - 1.0) > 0.01:
                self._adjust_audio_speed(output_path, speed_multiplier)
                actual_duration = self._get_audio_duration(output_path)
            
            # 后处理：音质增强
            if clarity_boost > 0.1:
                self._enhance_audio_clarity(output_path, clarity_boost)
            
            # 加载生成的音频
            audio_data = self._load_audio(output_path)
            
            # 生成统计信息
            synthesis_stats = self._generate_stats(
                text, speaker_audio_path, emotion_mode, enable_duration_control,
                enable_emotion_control, use_gpt_latents, actual_duration
            )

            # 生成信息字符串
            info = self._generate_info(
                text, speaker_audio_path, output_path, enable_duration_control,
                enable_emotion_control, duration_mode, emotion_mode, actual_duration
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

            return (comfyui_audio, output_path, info, actual_duration, emotion_analysis, synthesis_stats)
            
        except Exception as e:
            error_msg = f"IndexTTS2 advanced synthesis failed: {str(e)}"
            print(f"[IndexTTS2 Advanced Error] {error_msg}")
            raise RuntimeError(error_msg)
    
    def _add_emotion_params(self, infer_args: dict, emotion_mode: str, emotion_audio: Optional[dict],
                           emotion_alpha: float, emotion_text: str, emotion_vector: List[float]) -> str:
        """添加情感控制参数"""
        emotion_audio_path = None
        if emotion_audio is not None:
            # 处理ComfyUI AUDIO对象
            emotion_audio_path = self._save_emotion_audio_to_temp(emotion_audio)

        if emotion_mode == "audio_prompt" and emotion_audio_path:
            infer_args["emo_audio_prompt"] = emotion_audio_path
            infer_args["emo_alpha"] = emotion_alpha
            return f"Audio emotion: emotion_reference.wav (α={emotion_alpha})"

        elif emotion_mode == "emotion_vector":
            # 验证和修正情感向量
            emotion_vector = self._validate_and_fix_emotion_vector(emotion_vector)
            infer_args["emo_vector"] = emotion_vector
            emotion_names = ["Happy", "Angry", "Sad", "Fear", "Hate", "Low", "Surprise", "Neutral"]
            max_emotion_value = max(emotion_vector)
            max_idx = emotion_vector.index(max_emotion_value)
            return f"Vector emotion: {emotion_names[max_idx]} ({max_emotion_value:.2f})"

        elif emotion_mode == "text_description":
            infer_args["use_emo_text"] = True
            if emotion_text.strip():
                infer_args["emo_text"] = emotion_text
                return f"Text emotion: {emotion_text[:50]}..."
            else:
                return "Text emotion: Inferred from synthesis text"

        elif emotion_mode == "mixed":
            # 组合多种情感控制方法
            if emotion_audio_path:
                infer_args["emo_audio_prompt"] = emotion_audio_path
                infer_args["emo_alpha"] = emotion_alpha * 0.7  # 降低权重以平衡

            # 验证和修正情感向量
            emotion_vector = self._validate_and_fix_emotion_vector(emotion_vector)
            max_emotion_value = max(emotion_vector)
            if max_emotion_value > 0.01:  # 使用更低的阈值
                infer_args["emo_vector"] = emotion_vector

            if emotion_text.strip():
                infer_args["use_emo_text"] = True
                infer_args["emo_text"] = emotion_text

            return "Mixed emotion control: Audio + Vector + Text"

        return "No emotion control applied"

    def _validate_and_fix_emotion_vector(self, emotion_vector: List[float]) -> List[float]:
        """验证和修正情感向量，确保其有效性"""
        try:
            # 确保向量长度正确
            if len(emotion_vector) != 8:
                print(f"[IndexTTS2] Warning: emotion_vector length is {len(emotion_vector)}, expected 8. Padding/truncating.")
                if len(emotion_vector) < 8:
                    # 补齐到8维
                    emotion_vector = emotion_vector + [0.0] * (8 - len(emotion_vector))
                else:
                    # 截断到8维
                    emotion_vector = emotion_vector[:8]

            # 创建副本以避免修改原始数据
            emotion_vector = emotion_vector.copy()

            # 确保所有值都在有效范围内 [0.0, 1.0]
            for i in range(len(emotion_vector)):
                if emotion_vector[i] < 0.0:
                    emotion_vector[i] = 0.0
                elif emotion_vector[i] > 1.0:
                    emotion_vector[i] = 1.0
                # 将非常小的值设为0，避免数值计算问题
                elif emotion_vector[i] < 1e-6:
                    emotion_vector[i] = 0.0

            # 检查是否所有值都为0或接近0
            max_emotion_value = max(emotion_vector)
            total_emotion_value = sum(emotion_vector)

            if max_emotion_value <= 0.001 or total_emotion_value <= 0.001:
                # 如果所有情感值都为0或接近0，设置一个小的中性情感值
                print("[IndexTTS2] All emotion values are zero or near zero, setting default neutral emotion")
                emotion_vector[7] = 0.2  # Neutral - 设置一个合理的默认值
            elif total_emotion_value > 2.0:
                # 如果总和过大，进行归一化
                print(f"[IndexTTS2] Emotion vector sum is {total_emotion_value:.3f}, normalizing to reasonable range")
                scale_factor = 1.5 / total_emotion_value  # 将总和缩放到1.5左右
                emotion_vector = [v * scale_factor for v in emotion_vector]

            # 添加调试信息
            print(f"[IndexTTS2] Validated emotion vector: {[f'{v:.3f}' for v in emotion_vector]}")
            print(f"[IndexTTS2] Emotion names: ['Happy', 'Angry', 'Sad', 'Fear', 'Hate', 'Low', 'Surprise', 'Neutral']")

            return emotion_vector

        except Exception as e:
            print(f"[IndexTTS2] Error validating emotion vector: {e}")
            # 返回安全的默认情感向量
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]  # 默认中性情感

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
                error_msg += "\n[IndexTTS2 Advanced] DeepSpeed相关错误，但基本功能应该仍然可用"
                error_msg += "\n[IndexTTS2 Advanced] DeepSpeed-related error, but basic functionality should still work"
            raise RuntimeError(error_msg)
    
    def _adjust_audio_speed(self, audio_path: str, speed_multiplier: float):
        """调整音频速度"""
        try:
            import torchaudio
            import torchaudio.transforms as T
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 使用时间拉伸调整速度
            stretch = T.TimeStretch(hop_length=512, n_freq=1025)
            stretched = stretch(waveform.unsqueeze(0), speed_multiplier)
            
            # 保存调整后的音频
            torchaudio.save(audio_path, stretched.squeeze(0), sample_rate)
            
        except Exception as e:
            print(f"[IndexTTS2] Speed adjustment failed: {e}")
    
    def _enhance_audio_clarity(self, audio_path: str, clarity_boost: float):
        """增强音频清晰度"""
        try:
            import torchaudio
            import torchaudio.transforms as T
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 应用高通滤波器增强清晰度
            highpass = T.Highpass(sample_rate, cutoff_freq=80.0)
            enhanced = highpass(waveform)
            
            # 混合原始和增强的音频
            mixed = (1 - clarity_boost) * waveform + clarity_boost * enhanced
            
            # 保存增强后的音频
            torchaudio.save(audio_path, mixed, sample_rate)
            
        except Exception as e:
            print(f"[IndexTTS2] Audio enhancement failed: {e}")
    
    def _load_audio(self, audio_path: str) -> dict:
        """加载音频文件"""
        try:
            from .audio_utils import load_audio_for_comfyui
            return load_audio_for_comfyui(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import torchaudio
            
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            return duration
            
        except Exception as e:
            print(f"[IndexTTS2] Failed to get audio duration: {e}")
            return 0.0
    
    def _generate_stats(self, text: str, speaker_audio: str, emotion_mode: str,
                       enable_duration_control: bool, enable_emotion_control: bool,
                       use_gpt_latents: bool, duration: float) -> dict:
        """生成合成统计信息"""
        return {
            "text_length": len(text),
            "text_words": len(text.split()),
            "speaker_audio": os.path.basename(speaker_audio),
            "duration": duration,
            "duration_control": enable_duration_control,
            "emotion_control": enable_emotion_control,
            "emotion_mode": emotion_mode if enable_emotion_control else "none",
            "gpt_latents": use_gpt_latents,
            "synthesis_time": duration,  # 实际合成时间需要测量
        }
    
    def _generate_info(self, text: str, speaker_audio: str, output_path: str,
                      enable_duration_control: bool, enable_emotion_control: bool,
                      duration_mode: str, emotion_mode: str, duration: float) -> str:
        """生成信息字符串"""
        info_lines = [
            "=== IndexTTS2 Advanced Synthesis Info ===",
            f"Text: {text[:100]}{'...' if len(text) > 100 else ''}",
            f"Speaker Audio: {os.path.basename(speaker_audio)}",
            f"Output: {os.path.basename(output_path)}",
            f"Duration: {duration:.2f}s",
            f"Duration Control: {'Enabled (' + duration_mode + ')' if enable_duration_control else 'Disabled'}",
            f"Emotion Control: {'Enabled (' + emotion_mode + ')' if enable_emotion_control else 'Disabled'}",
            f"Model: IndexTTS2 Advanced",
        ]
        
        return "\n".join(info_lines)

    def _save_emotion_audio_to_temp(self, emotion_audio: dict) -> Optional[str]:
        """将ComfyUI AUDIO对象保存为临时文件供IndexTTS2使用"""
        try:
            import tempfile
            import torchaudio

            if not isinstance(emotion_audio, dict) or "waveform" not in emotion_audio or "sample_rate" not in emotion_audio:
                raise ValueError("emotion_audio must be a ComfyUI AUDIO object with 'waveform' and 'sample_rate' keys")

            waveform = emotion_audio["waveform"]
            sample_rate = emotion_audio["sample_rate"]

            # 移除batch维度（如果存在）
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                emotion_audio_path = tmp_file.name

            # 保存音频到临时文件
            torchaudio.save(emotion_audio_path, waveform, sample_rate)

            print(f"[IndexTTS2] 情感音频已保存到临时文件: {emotion_audio_path}")
            print(f"[IndexTTS2] 情感音频信息: 采样率={sample_rate}, 形状={waveform.shape}")

            return emotion_audio_path

        except Exception as e:
            print(f"[IndexTTS2] 保存情感音频失败: {str(e)}")
            return None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
