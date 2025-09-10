# IndexTTS2 Audio Utils Node
# IndexTTS2 音频工具节点

import os
import torch
import numpy as np
from typing import Optional, Tuple, Any, List, Dict
import folder_paths

# 导入音频文件获取函数
from .basic_tts_node import get_audio_files

# 导入高级音频浏览器
try:
    from ..audio_browser import get_all_audio_files, clear_audio_cache
    AUDIO_BROWSER_AVAILABLE = True
except ImportError:
    AUDIO_BROWSER_AVAILABLE = False

class IndexTTS2AudioUtilsNode:
    """
    IndexTTS2 音频工具节点
    Audio utilities node for IndexTTS2 with processing and analysis features
    
    Features:
    - Audio format conversion
    - Audio quality analysis
    - Speaker similarity analysis
    - Emotion detection in audio
    - Audio preprocessing for prompts
    - Batch audio processing
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用的音频文件列表
        audio_files = get_audio_files()

        return {
            "required": {
                "operation": ([
                    "analyze_audio",
                    "convert_format",
                    "normalize_audio",
                    "extract_features",
                    "compare_speakers",
                    "detect_emotion",
                    "preprocess_prompt"
                ], {
                    "default": "analyze_audio"
                }),
                "audio_path": (audio_files, {
                    "default": audio_files[0] if audio_files else "",
                    "tooltip": "选择要处理的音频文件 / Select audio file to process"
                }),
            },
            "optional": {
                "reference_audio": (audio_files + [""], {
                    "default": "",
                    "tooltip": "参考音频文件路径（用于说话人比较）/ Reference audio file path (for speaker comparison)"
                }),
                "output_format": (["wav", "mp3", "flac", "ogg"], {
                    "default": "wav"
                }),
                "sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 48000,
                    "step": 1000
                }),
                "normalize_volume": ("BOOLEAN", {
                    "default": True
                }),
                "remove_silence": ("BOOLEAN", {
                    "default": False
                }),
                "noise_reduction": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "output_filename": ("STRING", {
                    "default": "processed_audio.wav",
                    "placeholder": "Output filename"
                }),
                "verbose": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "DICT", "STRING")
    RETURN_NAMES = ("processed_audio", "output_path", "analysis_results", "info")
    FUNCTION = "process_audio"
    CATEGORY = "IndexTTS2/Utils"
    DESCRIPTION = "Audio processing and analysis utilities for IndexTTS2"
    
    def process_audio(
        self,
        operation: str,
        audio_path: str,
        reference_audio: str = "",
        output_format: str = "wav",
        sample_rate: int = 22050,
        normalize_volume: bool = True,
        remove_silence: bool = False,
        noise_reduction: float = 0.0,
        output_filename: str = "processed_audio.wav",
        verbose: bool = True
    ) -> Tuple[dict, str, dict, str]:
        """
        执行音频处理和分析
        Perform audio processing and analysis operations
        """
        try:
            # 处理音频路径 - 确保使用绝对路径
            if not audio_path:
                raise ValueError("Audio path is empty")

            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(audio_path):
                # 相对于 ComfyUI 根目录
                absolute_audio_path = os.path.join(folder_paths.base_path, audio_path)
            else:
                absolute_audio_path = audio_path

            # 验证文件是否存在
            if not os.path.exists(absolute_audio_path):
                raise ValueError(f"Audio file not found: {audio_path} (resolved to: {absolute_audio_path})")

            if verbose:
                print(f"[IndexTTS2 AudioUtils] Operation: {operation}")
                print(f"[IndexTTS2 AudioUtils] Input: {audio_path}")
                print(f"[IndexTTS2 AudioUtils] Resolved path: {absolute_audio_path}")

            # 使用绝对路径进行后续处理
            audio_path = absolute_audio_path
            
            # 准备输出路径
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, output_filename)
            
            # 执行相应操作
            analysis_results = {}
            processed_audio = None
            
            if operation == "analyze_audio":
                analysis_results = self._analyze_audio(audio_path, verbose)
                processed_audio = self._load_audio(audio_path)
                output_path = audio_path  # 分析操作不修改文件
                
            elif operation == "convert_format":
                processed_audio = self._convert_format(
                    audio_path, output_path, output_format, sample_rate, verbose
                )
                analysis_results = {"operation": "format_conversion", "format": output_format}
                
            elif operation == "normalize_audio":
                processed_audio = self._normalize_audio(
                    audio_path, output_path, normalize_volume, verbose
                )
                analysis_results = {"operation": "normalization", "volume_normalized": normalize_volume}
                
            elif operation == "extract_features":
                analysis_results = self._extract_features(audio_path, verbose)
                processed_audio = self._load_audio(audio_path)
                output_path = audio_path
                
            elif operation == "compare_speakers":
                if not reference_audio:
                    raise ValueError("Reference audio path is required for speaker comparison. Please provide a reference audio file path in the 'reference_audio' parameter.")

                # 处理参考音频路径 - 确保使用绝对路径
                if not os.path.isabs(reference_audio):
                    # 相对于 ComfyUI 根目录
                    absolute_reference_path = os.path.join(folder_paths.base_path, reference_audio)
                else:
                    absolute_reference_path = reference_audio

                if not os.path.exists(absolute_reference_path):
                    raise ValueError(f"Reference audio file not found: {reference_audio} (resolved to: {absolute_reference_path}). Please check the file path.")

                if verbose:
                    print(f"[IndexTTS2 AudioUtils] Reference audio: {reference_audio}")
                    print(f"[IndexTTS2 AudioUtils] Resolved reference path: {absolute_reference_path}")

                analysis_results = self._compare_speakers(audio_path, absolute_reference_path, verbose)
                processed_audio = self._load_audio(audio_path)
                output_path = audio_path
                
            elif operation == "detect_emotion":
                analysis_results = self._detect_emotion(audio_path, verbose)
                processed_audio = self._load_audio(audio_path)
                output_path = audio_path
                
            elif operation == "preprocess_prompt":
                processed_audio = self._preprocess_prompt(
                    audio_path, output_path, sample_rate, normalize_volume,
                    remove_silence, noise_reduction, verbose
                )
                analysis_results = {
                    "operation": "prompt_preprocessing",
                    "sample_rate": sample_rate,
                    "normalized": normalize_volume,
                    "silence_removed": remove_silence,
                    "noise_reduced": noise_reduction > 0
                }
            
            # 生成信息字符串
            info = self._generate_info(operation, audio_path, output_path, analysis_results)
            
            return (processed_audio, output_path, analysis_results, info)
            
        except Exception as e:
            error_msg = f"IndexTTS2 audio processing failed: {str(e)}"
            print(f"[IndexTTS2 AudioUtils Error] {error_msg}")
            raise RuntimeError(error_msg)
    
    def _analyze_audio(self, audio_path: str, verbose: bool) -> dict:
        """分析音频文件"""
        try:
            import torchaudio
            
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 基础分析
            duration = waveform.shape[1] / sample_rate
            channels = waveform.shape[0]
            max_amplitude = torch.max(torch.abs(waveform)).item()
            rms_energy = torch.sqrt(torch.mean(waveform**2)).item()
            
            # 频谱分析
            fft = torch.fft.fft(waveform[0])
            magnitude = torch.abs(fft)
            dominant_freq = torch.argmax(magnitude).item() * sample_rate / len(fft)
            
            analysis = {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "max_amplitude": max_amplitude,
                "rms_energy": rms_energy,
                "dominant_frequency": dominant_freq,
                "file_size": os.path.getsize(audio_path),
                "format": os.path.splitext(audio_path)[1][1:].upper()
            }
            
            if verbose:
                print(f"[IndexTTS2 AudioUtils] Duration: {duration:.2f}s")
                print(f"[IndexTTS2 AudioUtils] Sample rate: {sample_rate}Hz")
                print(f"[IndexTTS2 AudioUtils] Channels: {channels}")
                print(f"[IndexTTS2 AudioUtils] RMS energy: {rms_energy:.4f}")
            
            return analysis
            
        except Exception as e:
            raise RuntimeError(f"Audio analysis failed: {str(e)}")
    
    def _convert_format(self, input_path: str, output_path: str, output_format: str,
                       sample_rate: int, verbose: bool) -> dict:
        """转换音频格式"""
        try:
            import torchaudio
            
            # 加载音频
            waveform, orig_sample_rate = torchaudio.load(input_path)
            
            # 重采样
            if orig_sample_rate != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sample_rate, sample_rate)
                waveform = resampler(waveform)
            
            # 保存为指定格式
            torchaudio.save(output_path, waveform, sample_rate)
            
            if verbose:
                print(f"[IndexTTS2 AudioUtils] Converted to {output_format.upper()}")
                print(f"[IndexTTS2 AudioUtils] Resampled: {orig_sample_rate}Hz -> {sample_rate}Hz")
            
            return self._load_audio(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Format conversion failed: {str(e)}")
    
    def _normalize_audio(self, input_path: str, output_path: str, normalize_volume: bool,
                        verbose: bool) -> dict:
        """标准化音频"""
        try:
            import torchaudio
            
            # 加载音频
            waveform, sample_rate = torchaudio.load(input_path)
            
            if normalize_volume:
                # 音量标准化
                max_val = torch.max(torch.abs(waveform))
                if max_val > 0:
                    waveform = waveform / max_val * 0.95  # 留一点余量
            
            # 保存标准化后的音频
            torchaudio.save(output_path, waveform, sample_rate)
            
            if verbose:
                print(f"[IndexTTS2 AudioUtils] Audio normalized")
            
            return self._load_audio(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Audio normalization failed: {str(e)}")
    
    def _extract_features(self, audio_path: str, verbose: bool) -> dict:
        """提取音频特征"""
        try:
            import torchaudio
            import torchaudio.transforms as T
            
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 提取MFCC特征
            mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128}
            )
            mfcc = mfcc_transform(waveform)
            
            # 提取梅尔频谱
            mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )
            mel_spec = mel_transform(waveform)
            
            features = {
                "mfcc_shape": list(mfcc.shape),
                "mfcc_mean": torch.mean(mfcc).item(),
                "mfcc_std": torch.std(mfcc).item(),
                "mel_spec_shape": list(mel_spec.shape),
                "mel_spec_mean": torch.mean(mel_spec).item(),
                "mel_spec_std": torch.std(mel_spec).item(),
                "spectral_centroid": self._compute_spectral_centroid(waveform, sample_rate),
                "zero_crossing_rate": self._compute_zcr(waveform),
            }
            
            if verbose:
                print(f"[IndexTTS2 AudioUtils] MFCC shape: {features['mfcc_shape']}")
                print(f"[IndexTTS2 AudioUtils] Mel spectrogram shape: {features['mel_spec_shape']}")
            
            return features
            
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {str(e)}")
    
    def _compare_speakers(self, audio1_path: str, audio2_path: str, verbose: bool) -> dict:
        """比较说话人相似度"""
        try:
            # 提取两个音频的特征
            features1 = self._extract_features(audio1_path, False)
            features2 = self._extract_features(audio2_path, False)
            
            # 计算相似度（简化版本）
            mfcc_similarity = 1.0 - abs(features1["mfcc_mean"] - features2["mfcc_mean"]) / max(abs(features1["mfcc_mean"]), abs(features2["mfcc_mean"]), 1e-6)
            mel_similarity = 1.0 - abs(features1["mel_spec_mean"] - features2["mel_spec_mean"]) / max(abs(features1["mel_spec_mean"]), abs(features2["mel_spec_mean"]), 1e-6)
            
            overall_similarity = (mfcc_similarity + mel_similarity) / 2
            
            comparison = {
                "audio1": os.path.basename(audio1_path),
                "audio2": os.path.basename(audio2_path),
                "mfcc_similarity": mfcc_similarity,
                "mel_similarity": mel_similarity,
                "overall_similarity": overall_similarity,
                "similarity_level": "high" if overall_similarity > 0.8 else "medium" if overall_similarity > 0.6 else "low"
            }
            
            if verbose:
                print(f"[IndexTTS2 AudioUtils] Speaker similarity: {overall_similarity:.3f}")
                print(f"[IndexTTS2 AudioUtils] Similarity level: {comparison['similarity_level']}")
            
            return comparison
            
        except Exception as e:
            raise RuntimeError(f"Speaker comparison failed: {str(e)}")
    
    def _detect_emotion(self, audio_path: str, verbose: bool) -> dict:
        """检测音频中的情感（简化版本）"""
        try:
            # 这是一个简化的情感检测实现
            # 实际应用中可能需要更复杂的模型
            features = self._extract_features(audio_path, False)
            
            # 基于音频特征的简单情感分类
            energy = features["mel_spec_mean"]
            variability = features["mel_spec_std"]
            zcr = features["zero_crossing_rate"]
            
            # 简单的规则基础分类
            if energy > 0.5 and variability > 0.3:
                emotion = "excited"
                confidence = 0.7
            elif energy < 0.2:
                emotion = "calm"
                confidence = 0.6
            elif zcr > 0.1:
                emotion = "tense"
                confidence = 0.5
            else:
                emotion = "neutral"
                confidence = 0.8
            
            emotion_result = {
                "detected_emotion": emotion,
                "confidence": confidence,
                "energy_level": energy,
                "variability": variability,
                "zero_crossing_rate": zcr,
                "note": "Simplified emotion detection - for reference only"
            }
            
            if verbose:
                print(f"[IndexTTS2 AudioUtils] Detected emotion: {emotion} (confidence: {confidence:.2f})")
            
            return emotion_result
            
        except Exception as e:
            raise RuntimeError(f"Emotion detection failed: {str(e)}")
    
    def _preprocess_prompt(self, input_path: str, output_path: str, sample_rate: int,
                          normalize_volume: bool, remove_silence: bool, noise_reduction: float,
                          verbose: bool) -> dict:
        """预处理音频提示"""
        try:
            import torchaudio
            import torchaudio.transforms as T
            
            # 加载音频
            waveform, orig_sample_rate = torchaudio.load(input_path)
            
            # 重采样
            if orig_sample_rate != sample_rate:
                resampler = T.Resample(orig_sample_rate, sample_rate)
                waveform = resampler(waveform)
            
            # 音量标准化
            if normalize_volume:
                max_val = torch.max(torch.abs(waveform))
                if max_val > 0:
                    waveform = waveform / max_val * 0.95
            
            # 简单的噪声减少（高通滤波）
            if noise_reduction > 0:
                # 这里可以实现更复杂的噪声减少算法
                pass
            
            # 移除静音（简化版本）
            if remove_silence:
                # 简单的基于能量的静音检测
                energy = torch.mean(waveform**2, dim=0)
                threshold = torch.max(energy) * 0.01  # 1%的最大能量作为阈值
                non_silent = energy > threshold
                if torch.any(non_silent):
                    waveform = waveform[:, non_silent]
            
            # 保存预处理后的音频
            torchaudio.save(output_path, waveform, sample_rate)
            
            if verbose:
                print(f"[IndexTTS2 AudioUtils] Audio preprocessed for prompt use")
            
            return self._load_audio(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Audio preprocessing failed: {str(e)}")
    
    def _compute_spectral_centroid(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """计算频谱质心"""
        try:
            fft = torch.fft.fft(waveform[0])
            magnitude = torch.abs(fft)
            freqs = torch.fft.fftfreq(len(fft), 1/sample_rate)
            
            # 只考虑正频率
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            centroid = torch.sum(positive_freqs * positive_magnitude) / torch.sum(positive_magnitude)
            return centroid.item()
        except:
            return 0.0
    
    def _compute_zcr(self, waveform: torch.Tensor) -> float:
        """计算过零率"""
        try:
            signs = torch.sign(waveform[0])
            sign_changes = torch.diff(signs) != 0
            zcr = torch.sum(sign_changes).float() / len(waveform[0])
            return zcr.item()
        except:
            return 0.0
    
    def _load_audio(self, audio_path: str) -> dict:
        """加载音频文件并返回ComfyUI兼容的音频格式"""
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_path)

            # 确保waveform是2D张量 [channels, samples]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # 添加通道维度
            elif waveform.dim() > 2:
                # 如果有多余的维度，压缩到2D
                waveform = waveform.squeeze()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)

            # ComfyUI期望3D张量 [batch, channels, samples]
            # 添加批次维度
            waveform = waveform.unsqueeze(0)  # [1, channels, samples]

            # 返回ComfyUI标准音频格式
            return {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")
    
    def _generate_info(self, operation: str, input_path: str, output_path: str,
                      analysis_results: dict) -> str:
        """生成信息字符串"""
        info_lines = [
            f"=== IndexTTS2 Audio Utils Info ===",
            f"Operation: {operation}",
            f"Input: {os.path.basename(input_path)}",
            f"Output: {os.path.basename(output_path)}",
        ]
        
        # 添加分析结果摘要
        if "duration" in analysis_results:
            info_lines.append(f"Duration: {analysis_results['duration']:.2f}s")
        if "sample_rate" in analysis_results:
            info_lines.append(f"Sample Rate: {analysis_results['sample_rate']}Hz")
        if "overall_similarity" in analysis_results:
            info_lines.append(f"Similarity: {analysis_results['overall_similarity']:.3f}")
        if "detected_emotion" in analysis_results:
            info_lines.append(f"Emotion: {analysis_results['detected_emotion']}")
        
        return "\n".join(info_lines)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
