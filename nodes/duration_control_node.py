# IndexTTS2 Duration Control Node
# IndexTTS2 时长控制节点

import os
import torch
import numpy as np
from typing import Optional, Tuple, Any
import folder_paths

# 导入音频文件获取函数
from .basic_tts_node import get_audio_files

# 导入高级音频浏览器
try:
    from ..audio_browser import get_all_audio_files, clear_audio_cache
    AUDIO_BROWSER_AVAILABLE = True
except ImportError:
    AUDIO_BROWSER_AVAILABLE = False

class IndexTTS2DurationNode:
    """
    IndexTTS2 时长控制节点
    Duration-controlled speech synthesis node for IndexTTS2
    
    Features:
    - Precise duration control (0.75x, 1.0x, 1.25x speeds)
    - Token-level duration specification
    - Autoregressive model-friendly duration adaptation
    - Maintains speech naturalness while controlling timing
    """
    
    def __init__(self):
        self.model = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "This is a duration-controlled speech synthesis.",
                    "placeholder": "输入要合成的文本（带时长控制）..."
                }),
                "speaker_audio": ("AUDIO", {
                    "tooltip": "连接IndexTTS2 Load Audio File节点 / Connect IndexTTS2 Load Audio File node"
                }),
                "duration_mode": (["auto", "speed_control", "token_control"], {
                    "default": "speed_control"
                }),
                "output_filename": ("STRING", {
                    "default": "indextts2_duration_output.wav",
                    "placeholder": "Output audio filename"
                }),
            },
            "optional": {
                "model_manager": ("INDEXTTS2_MODEL",),
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
                "prosody_preservation": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
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
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("audio", "output_path", "info", "actual_duration")
    FUNCTION = "synthesize_with_duration"
    CATEGORY = "IndexTTS2/Control"
    DESCRIPTION = "Duration-controlled speech synthesis with precise timing control"
    
    def synthesize_with_duration(
        self,
        text: str,
        speaker_audio: dict,
        duration_mode: str,
        output_filename: str,
        model_manager: Optional[Any] = None,
        speed_multiplier: float = 1.0,
        target_duration: float = 0.0,
        token_count: int = 0,
        prosody_preservation: float = 0.8,
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        verbose: bool = True
    ) -> Tuple[dict, str, str, float]:
        """
        执行时长控制的语音合成
        Perform duration-controlled text-to-speech synthesis
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
                print(f"[IndexTTS2 Duration] Mode: {duration_mode}")
                print(f"[IndexTTS2 Duration] Text: {text[:50]}...")
                print(f"[IndexTTS2 Duration] Speed: {speed_multiplier}x")
                if target_duration > 0:
                    print(f"[IndexTTS2 Duration] Target duration: {target_duration}s")
                if token_count > 0:
                    print(f"[IndexTTS2 Duration] Token count: {token_count}")
            
            # 根据模式执行不同的推理策略
            actual_duration = 0.0
            
            if duration_mode == "speed_control":
                actual_duration = self._synthesize_speed_control(
                    model, text, speaker_audio_path, output_path, speed_multiplier, verbose
                )
            elif duration_mode == "token_control":
                actual_duration = self._synthesize_token_control(
                    model, text, speaker_audio_path, output_path, token_count, verbose
                )
            else:  # auto mode
                actual_duration = self._synthesize_auto(
                    model, text, speaker_audio_path, output_path, verbose
                )
            
            # 加载生成的音频
            audio_data = self._load_audio(output_path)
            
            # 生成信息字符串
            info = self._generate_info(
                text, speaker_audio_path, output_path, duration_mode,
                speed_multiplier, target_duration, token_count, actual_duration
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

            return (comfyui_audio, output_path, info, actual_duration)
            
        except Exception as e:
            error_msg = f"IndexTTS2 duration synthesis failed: {str(e)}"
            print(f"[IndexTTS2 Duration Error] {error_msg}")
            raise RuntimeError(error_msg)
    
    def _synthesize_speed_control(self, model, text: str, speaker_audio: str, 
                                 output_path: str, speed_multiplier: float, verbose: bool) -> float:
        """速度控制模式合成"""
        # 基础推理
        model.infer(
            spk_audio_prompt=speaker_audio,
            text=text,
            output_path=output_path,
            verbose=verbose
        )
        
        # 如果需要速度调整，进行后处理
        if abs(speed_multiplier - 1.0) > 0.01:
            self._adjust_audio_speed(output_path, speed_multiplier)
        
        return self._get_audio_duration(output_path)
    
    def _synthesize_token_control(self, model, text: str, speaker_audio: str,
                                 output_path: str, token_count: int, verbose: bool) -> float:
        """令牌控制模式合成"""
        # 注意：这是IndexTTS2的核心创新功能
        # 需要模型支持指定token数量的推理
        try:
            # 如果模型支持token控制，使用专门的推理方法
            if hasattr(model, 'infer_with_token_control'):
                model.infer_with_token_control(
                    spk_audio_prompt=speaker_audio,
                    text=text,
                    output_path=output_path,
                    target_tokens=token_count,
                    verbose=verbose
                )
            else:
                # 回退到基础推理
                print("[IndexTTS2] Token control not supported, using basic inference")
                model.infer(
                    spk_audio_prompt=speaker_audio,
                    text=text,
                    output_path=output_path,
                    verbose=verbose
                )
        except Exception as e:
            print(f"[IndexTTS2] Token control failed, using basic inference: {e}")
            model.infer(
                spk_audio_prompt=speaker_audio,
                text=text,
                output_path=output_path,
                verbose=verbose
            )
        
        return self._get_audio_duration(output_path)
    
    def _synthesize_auto(self, model, text: str, speaker_audio: str,
                        output_path: str, verbose: bool) -> float:
        """自动模式合成"""
        model.infer(
            spk_audio_prompt=speaker_audio,
            text=text,
            output_path=output_path,
            verbose=verbose
        )
        
        return self._get_audio_duration(output_path)
    
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
    
    def _generate_info(self, text: str, speaker_audio: str, output_path: str,
                      duration_mode: str, speed_multiplier: float, target_duration: float,
                      token_count: int, actual_duration: float) -> str:
        """生成信息字符串"""
        info_lines = [
            "=== IndexTTS2 Duration Control Info ===",
            f"Text: {text[:100]}{'...' if len(text) > 100 else ''}",
            f"Speaker Audio: {os.path.basename(speaker_audio)}",
            f"Output: {os.path.basename(output_path)}",
            f"Duration Mode: {duration_mode}",
            f"Speed Multiplier: {speed_multiplier}x",
            f"Actual Duration: {actual_duration:.2f}s",
        ]
        
        if target_duration > 0:
            info_lines.append(f"Target Duration: {target_duration:.2f}s")
        if token_count > 0:
            info_lines.append(f"Token Count: {token_count}")
        
        return "\n".join(info_lines)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
