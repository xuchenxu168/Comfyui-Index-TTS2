# IndexTTS2 Multi-Talk Node with Emotion Control
# IndexTTS2 多人对话语音合成节点（带情感控制）

import os
import torch
import numpy as np
import tempfile
import torchaudio
from typing import Optional, Tuple, Any, List, Dict
import folder_paths
import torch.nn.functional as F
import sys

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入高级音频系统
try:
    from advanced_audio_systems import SpeakerEmbeddingCache, VoiceConsistencyController, AdaptiveQualityMonitor
    ADVANCED_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"[MultiTalkNode] 高级音频系统导入失败: {e}")
    ADVANCED_SYSTEMS_AVAILABLE = False

# 智能音频预处理器
class IntelligentAudioPreprocessor:
    """智能音频预处理器"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.noise_gate_threshold = -40  # dB
        self.compressor_threshold = -12  # dB
        self.compressor_ratio = 4.0

    def apply_noise_gate(self, audio: torch.Tensor, threshold_db: float = -40) -> torch.Tensor:
        """噪声门限处理"""
        # 转换为dB
        audio_db = 20 * torch.log10(torch.abs(audio) + 1e-8)

        # 创建门限掩码
        gate_mask = audio_db > threshold_db

        # 应用软门限（避免突变）
        soft_mask = torch.sigmoid((audio_db - threshold_db) * 0.5)

        return audio * soft_mask

    def apply_dynamic_compression(self, audio: torch.Tensor,
                                threshold_db: float = -12,
                                ratio: float = 4.0) -> torch.Tensor:
        """动态范围压缩"""
        # 计算音频包络
        envelope = torch.abs(audio)
        envelope_db = 20 * torch.log10(envelope + 1e-8)

        # 计算增益减少
        gain_reduction = torch.zeros_like(envelope_db)
        over_threshold = envelope_db > threshold_db
        gain_reduction[over_threshold] = (envelope_db[over_threshold] - threshold_db) * (1 - 1/ratio)

        # 应用增益减少
        gain_linear = torch.pow(10, -gain_reduction / 20)

        return audio * gain_linear

    def apply_spectral_enhancement(self, audio: torch.Tensor,
                                 enhancement_strength: float = 0.3) -> torch.Tensor:
        """频谱增强"""
        if audio.shape[-1] < 1024:
            return audio

        # 高频增强滤波器
        kernel = torch.tensor([[-0.05, -0.1, 0.7, -0.1, -0.05]], dtype=audio.dtype, device=audio.device)
        kernel = kernel.unsqueeze(0)

        enhanced_channels = []
        for ch in range(audio.shape[0]):
            ch_data = audio[ch:ch+1].unsqueeze(0)
            enhanced = F.conv1d(ch_data, kernel, padding=2)

            # 混合原始和增强信号
            mixed = ch_data * (1 - enhancement_strength) + enhanced * enhancement_strength
            enhanced_channels.append(mixed.squeeze(0))

        return torch.cat(enhanced_channels, dim=0)

    def normalize_loudness(self, audio: torch.Tensor, target_lufs: float = -23.0) -> torch.Tensor:
        """响度标准化（简化版LUFS）"""
        # 计算RMS
        rms = torch.sqrt(torch.mean(audio ** 2))

        if rms > 0:
            # 简化的LUFS到RMS转换
            target_rms = 10 ** ((target_lufs + 3.01) / 20)  # 近似转换
            gain = target_rms / rms

            # 限制增益范围
            gain = torch.clamp(gain, 0.1, 3.0)
            audio = audio * gain

        return audio

    def process(self, audio: torch.Tensor,
                noise_gate: bool = True,
                compression: bool = True,
                spectral_enhancement: bool = True,
                loudness_normalization: bool = True) -> torch.Tensor:
        """完整的音频预处理流程"""
        processed_audio = audio.clone()

        if noise_gate:
            processed_audio = self.apply_noise_gate(processed_audio)

        if compression:
            processed_audio = self.apply_dynamic_compression(processed_audio)

        if spectral_enhancement:
            processed_audio = self.apply_spectral_enhancement(processed_audio)

        if loudness_normalization:
            processed_audio = self.normalize_loudness(processed_audio)

        # 最终限幅
        processed_audio = torch.clamp(processed_audio, -0.95, 0.95)

        return processed_audio

class IndexTTS2MultiTalkNode:
    """
    IndexTTS2 多人对话语音合成节点（带情感控制）
    Multi-speaker conversation text-to-speech synthesis node for IndexTTS2 with emotion control

    Features:
    - Support 1-4 speakers: 1=voice cloning, 2-4=conversation
    - Individual speaker voice cloning
    - Individual emotion control for each speaker
    - Multiple emotion control modes (audio, vector, text, auto)
    - Automatic conversation flow
    - Configurable silence intervals
    - High-quality multi-speaker synthesis with emotions
    """

    def __init__(self):
        self.model = None
        self.model_config = None
        # 智能音频预处理器
        self.audio_preprocessor = IntelligentAudioPreprocessor()

        # 高级音频系统（第二阶段改进）
        if ADVANCED_SYSTEMS_AVAILABLE:
            self.speaker_embedding_cache = SpeakerEmbeddingCache(
                cache_size=100,  # 多人对话节点使用较小的缓存
                similarity_threshold=0.92,
                enable_multi_sample_fusion=True
            )
            self.voice_consistency_controller = VoiceConsistencyController(
                consistency_threshold=0.75,  # 多人对话允许更多变化
                adaptation_rate=0.15
            )
            self.quality_monitor = AdaptiveQualityMonitor()
            print("[MultiTalkNode] ✓ 高级音频系统初始化完成")
        else:
            self.speaker_embedding_cache = None
            self.voice_consistency_controller = None
            self.quality_monitor = None
            print("[MultiTalkNode] ⚠️ 高级音频系统不可用，使用基础功能")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_speakers": (["1", "2", "3", "4"], {
                    "default": "2",
                    "tooltip": "对话人数 / Number of speakers (1=纯语音克隆)"
                }),
                "conversation_text": ("STRING", {
                    "multiline": True,
                    "default": "Speaker1: Hello, how are you today!\nSpeaker2: I'm doing great, thank you for asking!",
                    "placeholder": "单人模式：直接输入文本\\n多人模式：Speaker1: 文本\\nSpeaker2: 文本..."
                }),
                "speaker1_audio": ("AUDIO", {
                    "tooltip": "说话人1的音频样本 / Speaker 1 audio sample"
                }),
                "output_filename": ("STRING", {
                    "default": "multi_talk_emotion_output.wav",
                    "placeholder": "输出音频文件名"
                }),
            },
            "optional": {
                "speaker2_audio": ("AUDIO", {
                    "tooltip": "说话人2的音频样本 / Speaker 2 audio sample (多人模式必需)"
                }),
                "speaker3_audio": ("AUDIO", {
                    "tooltip": "说话人3的音频样本（3-4人对话时需要）/ Speaker 3 audio sample (required for 3-4 speakers)"
                }),
                "speaker4_audio": ("AUDIO", {
                    "tooltip": "说话人4的音频样本（4人对话时需要）/ Speaker 4 audio sample (required for 4 speakers)"
                }),
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
                    "tooltip": "语速控制 / Speed control"
                }),
                "silence_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "全局默认静音时长（秒）/ Global default silence duration between speakers (seconds)"
                }),
                "speaker1_pause": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "说话人1说完后的停顿时间（秒）/ Pause duration after Speaker 1 (seconds)"
                }),
                "speaker2_pause": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "说话人2说完后的停顿时间（秒）/ Pause duration after Speaker 2 (seconds)"
                }),
                "speaker3_pause": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "说话人3说完后的停顿时间（秒）/ Pause duration after Speaker 3 (seconds)"
                }),
                "speaker4_pause": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "说话人4说完后的停顿时间（秒）/ Pause duration after Speaker 4 (seconds)"
                }),
                "voice_consistency": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "声音一致性强度（越高越接近参考音频）/ Voice consistency strength (higher = closer to reference audio)"
                }),
                "reference_boost": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用参考音频增强（提高声音相似度）/ Enable reference audio enhancement (improves voice similarity)"
                }),
                # 情感控制输入端口
                "speaker1_emotion_config": ("SPEAKER_EMOTION_CONFIG", {
                    "tooltip": "说话人1情感配置 / Speaker 1 emotion config"
                }),
                "speaker2_emotion_config": ("SPEAKER_EMOTION_CONFIG", {
                    "tooltip": "说话人2情感配置 / Speaker 2 emotion config"
                }),
                "speaker3_emotion_config": ("SPEAKER_EMOTION_CONFIG", {
                    "tooltip": "说话人3情感配置（3-4人对话时需要）/ Speaker 3 emotion config (required for 3-4 speakers)"
                }),
                "speaker4_emotion_config": ("SPEAKER_EMOTION_CONFIG", {
                    "tooltip": "说话人4情感配置（4人对话时需要）/ Speaker 4 emotion config (required for 4 speakers)"
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
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "output_path", "info", "emotion_analysis")
    FUNCTION = "synthesize_conversation"
    CATEGORY = "IndexTTS2/Advanced"
    DESCRIPTION = "1-4 speaker conversation synthesis with individual emotion control using IndexTTS2 (1=voice cloning, 2-4=conversation)"
    
    def synthesize_conversation(
        self,
        num_speakers: str,
        conversation_text: str,
        speaker1_audio: dict,
        output_filename: str,
        speaker2_audio: Optional[dict] = None,
        speaker3_audio: Optional[dict] = None,
        speaker4_audio: Optional[dict] = None,
        model_manager: Optional[Any] = None,
        language: str = "auto",
        speed: float = 1.0,
        silence_duration: float = 0.5,
        speaker1_pause: float = 0.5,
        speaker2_pause: float = 0.5,
        speaker3_pause: float = 0.5,
        speaker4_pause: float = 0.5,
        voice_consistency: float = 1.0,
        reference_boost: bool = True,
        speaker1_emotion_config: Optional[dict] = None,
        speaker2_emotion_config: Optional[dict] = None,
        speaker3_emotion_config: Optional[dict] = None,
        speaker4_emotion_config: Optional[dict] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        verbose: bool = True
    ) -> Tuple[dict, str, str, str]:
        """
        执行多人对话语音合成（带情感控制）
        Perform multi-speaker conversation text-to-speech synthesis with emotion control
        """
        try:
            # 验证输入
            if not conversation_text.strip():
                raise ValueError("Conversation text cannot be empty")

            num_speakers_int = int(num_speakers)

            # 检查是否为单人模式
            if num_speakers_int == 1:
                # 单人模式：纯语音克隆
                return self._synthesize_single_speaker(
                    conversation_text, speaker1_audio, speaker1_emotion_config,
                    output_filename, model_manager, language, speed, temperature,
                    top_p, use_fp16, use_cuda_kernel, verbose
                )

            # 多人模式：原有逻辑
            # 验证说话人音频
            if speaker2_audio is None:
                raise ValueError("Speaker 2 audio is required for 2+ speakers conversation")

            speaker_audios = [speaker1_audio, speaker2_audio]
            if num_speakers_int >= 3:
                if speaker3_audio is None:
                    raise ValueError("Speaker 3 audio is required for 3+ speakers conversation")
                speaker_audios.append(speaker3_audio)
            if num_speakers_int >= 4:
                if speaker4_audio is None:
                    raise ValueError("Speaker 4 audio is required for 4 speakers conversation")
                speaker_audios.append(speaker4_audio)

            # 解析对话文本
            conversation_lines = self._parse_conversation(conversation_text, num_speakers_int, verbose)

            # 准备说话人音频文件（带一致性增强）
            speaker_audio_paths = self._prepare_speaker_audios(
                speaker_audios[:num_speakers_int], verbose, voice_consistency, reference_boost
            )

            # 准备情感控制参数
            emotion_configs = self._prepare_emotion_configs_from_inputs(
                num_speakers_int,
                [speaker1_emotion_config, speaker2_emotion_config, speaker3_emotion_config, speaker4_emotion_config],
                verbose
            )

            # 获取模型实例
            if model_manager is not None:
                model = model_manager
            else:
                model = self._load_default_model(use_fp16, use_cuda_kernel)
            
            # 合成每个对话片段（带情感控制）
            audio_segments = []
            emotion_analysis_list = []

            for line_info in conversation_lines:
                speaker_idx = line_info["speaker_idx"]
                text = line_info["text"]
                speaker_audio_path = speaker_audio_paths[speaker_idx]
                emotion_config = emotion_configs[speaker_idx]

                if verbose:
                    print(f"[MultiTalk] Synthesizing Speaker{speaker_idx + 1}: {text[:50]}...")
                    if emotion_config["mode"] != "none":
                        print(f"[MultiTalk] Emotion mode: {emotion_config['mode']}")

                # 创建临时输出文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_output = tmp_file.name

                # 处理语言参数
                processed_language = language if language != "auto" else "zh"

                # 执行单个片段的情感合成（带一致性控制）
                emotion_analysis = self._synthesize_with_emotion(
                    model, text, speaker_audio_path, emotion_config,
                    temp_output, temperature, top_p, verbose, voice_consistency, processed_language
                )
                emotion_analysis_list.append(f"Speaker{speaker_idx + 1}: {emotion_analysis}")

                # 加载合成的音频
                segment_audio = self._load_audio(temp_output)
                audio_segments.append(segment_audio)

                # 清理临时文件
                try:
                    os.unlink(temp_output)
                except:
                    pass
            
            # 准备个性化停顿时间配置
            speaker_pauses = [speaker1_pause, speaker2_pause, speaker3_pause, speaker4_pause]

            # 合并音频片段（使用个性化停顿时间）
            final_audio = self._merge_audio_segments_with_custom_pauses(
                audio_segments, conversation_lines, speaker_pauses, silence_duration, verbose
            )
            
            # 准备输出路径
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存最终音频
            torchaudio.save(output_path, final_audio["waveform"], final_audio["sample_rate"])

            # 质量监控（第二阶段改进）
            if self.quality_monitor is not None:
                try:
                    # 对最终音频进行质量评估
                    quality_assessment = self.quality_monitor.assess_quality(
                        final_audio["waveform"].float(), final_audio["sample_rate"]
                    )

                    if verbose:
                        print(f"[MultiTalk] 🎵 多人对话音频质量评估:")
                        print(f"  - 综合质量分数: {quality_assessment['overall_quality']:.3f}")
                        print(f"  - SNR: {quality_assessment['metrics']['snr']:.1f} dB")
                        print(f"  - THD: {quality_assessment['metrics']['thd']:.3f}")
                        print(f"  - 动态范围: {quality_assessment['metrics']['dynamic_range']:.1f} dB")
                        print(f"  - 峰值电平: {quality_assessment['metrics']['peak_level']:.1f} dB")

                        if quality_assessment['violations'] > 0:
                            print(f"  ⚠️  检测到 {quality_assessment['violations']} 项质量问题")

                            # 自动改进功能已禁用，使用原始音频
                            # if quality_assessment['improvement_applied'] and quality_assessment['improved_audio'] is not None:
                            #     print(f"  🔧 自动质量改进已应用")
                            #     final_audio["waveform"] = quality_assessment['improved_audio']
                            #
                            #     # 重新评估改进后的音频
                            #     improved_assessment = self.quality_monitor.assess_quality(
                            #         final_audio["waveform"].float(), final_audio["sample_rate"]
                            #     )
                            #     print(f"  📈 改进后质量分数: {improved_assessment['overall_quality']:.3f}")
                            #     print(f"  📈 改进后违规项: {improved_assessment['violations']}")
                            print(f"  ℹ️ 自动改进功能已禁用，使用原始音频")
                        else:
                            print(f"  ✅ 多人对话音频质量良好")

                except Exception as e:
                    if verbose:
                        print(f"[MultiTalk] ⚠️ 质量监控失败: {e}")

            # 确保音频格式兼容ComfyUI
            waveform = final_audio["waveform"]
            sample_rate = final_audio["sample_rate"]
            
            # 应用ComfyUI兼容性检查
            from .audio_utils import fix_comfyui_audio_compatibility
            waveform = fix_comfyui_audio_compatibility(waveform)
            
            # ComfyUI AUDIO格式需要 [batch, channels, samples]
            if waveform.dim() == 1:
                # [samples] -> [1, 1, samples]
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.dim() == 2:
                # [channels, samples] -> [1, channels, samples]
                waveform = waveform.unsqueeze(0)
            
            # 创建ComfyUI AUDIO格式
            comfyui_audio = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
            if verbose:
                print(f"[MultiTalk] 对话合成完成: {len(conversation_lines)} 个片段")
                print(f"[MultiTalk] 最终音频格式: {waveform.shape}, 采样率: {sample_rate}")

            # 生成信息字符串（包含个性化停顿时间）
            info = self._generate_info_with_emotion_and_pauses(
                conversation_lines, num_speakers_int, output_path, language, speed,
                silence_duration, speaker_pauses, emotion_configs
            )

            # 生成情感分析字符串
            emotion_analysis = "\n".join(emotion_analysis_list)

            # 清理临时文件
            for path in speaker_audio_paths:
                try:
                    os.unlink(path)
                except:
                    pass

            return (comfyui_audio, output_path, info, emotion_analysis)
            
        except Exception as e:
            error_msg = f"IndexTTS2 multi-talk synthesis failed: {str(e)}"
            print(f"[MultiTalk Error] {error_msg}")
            raise RuntimeError(error_msg)

    def _synthesize_single_speaker(
        self,
        text: str,
        speaker_audio: dict,
        emotion_config: Optional[dict],
        output_filename: str,
        model_manager: Optional[Any],
        language: str,
        speed: float,
        temperature: float,
        top_p: float,
        use_fp16: bool,
        use_cuda_kernel: bool,
        verbose: bool
    ) -> Tuple[dict, str, str, str]:
        """
        单人模式合成
        Single speaker mode synthesis
        """
        try:
            if verbose:
                print(f"[MultiTalk] 单人模式合成 / Single speaker mode synthesis")
                print(f"[MultiTalk] 文本长度: {len(text)} 字符 / Text length: {len(text)} characters")

            # 获取模型实例
            if model_manager is not None:
                model = model_manager
            else:
                model = self._load_default_model(use_fp16, use_cuda_kernel)

            # 准备说话人音频
            speaker_audio_path = self._prepare_speaker_audios([speaker_audio], verbose, 1.0, True)[0]

            # 准备情感控制参数
            if emotion_config is None:
                emotion_config = {"mode": "none"}

            # 创建临时输出文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_output_path = temp_file.name

            # 处理语言参数
            processed_language = language if language != "auto" else "zh"

            # 合成音频
            emotion_analysis = self._synthesize_with_emotion(
                model, text, speaker_audio_path, emotion_config, temp_output_path,
                temperature, top_p, verbose, language=processed_language
            )

            # 从临时文件加载音频
            if os.path.exists(temp_output_path):
                audio_tensor, sample_rate = torchaudio.load(temp_output_path)
                # 确保采样率正确
                if sample_rate != 24000:
                    import torchaudio.transforms as T
                    resampler = T.Resample(sample_rate, 24000)
                    audio_tensor = resampler(audio_tensor)
            else:
                raise RuntimeError("合成失败，临时音频文件不存在")

            # 清理临时文件
            try:
                os.unlink(temp_output_path)
            except:
                pass

            # 保存音频
            output_path = self._save_audio(audio_tensor, output_filename)

            # 生成信息
            info = f"单人语音合成完成\n文本长度: {len(text)} 字符\n音频长度: {len(audio_tensor[0])/24000:.2f} 秒"
            if emotion_config["mode"] != "none":
                info += f"\n情绪模式: {emotion_config['mode']}"
                info += f"\n情感分析: {emotion_analysis}"

            # 清理临时文件
            try:
                os.unlink(speaker_audio_path)
            except:
                pass

            # 确保音频格式符合 ComfyUI 标准 [batch, channels, samples]
            if audio_tensor.dim() == 1:
                # [samples] -> [1, 1, samples]
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                # [channels, samples] -> [1, channels, samples]
                audio_tensor = audio_tensor.unsqueeze(0)

            if verbose:
                print(f"[MultiTalk SingleSpeaker] 最终音频格式: {audio_tensor.shape}")
                print(f"[MultiTalk SingleSpeaker] ComfyUI AUDIO格式: batch={audio_tensor.shape[0]}, channels={audio_tensor.shape[1]}, samples={audio_tensor.shape[2]}")

            # 返回ComfyUI格式的音频
            comfyui_audio = {"waveform": audio_tensor, "sample_rate": 24000}

            return (comfyui_audio, output_path, info, emotion_analysis)

        except Exception as e:
            error_msg = f"单人模式合成失败: {str(e)}"
            print(f"[SingleSpeaker Error] {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

    def _extract_pause_from_text(self, text: str) -> tuple:
        """从文本中提取停顿时间标记

        支持格式：-0.8s-、-1.2s-、-0.5s- 等
        返回：(清理后的文本, 停顿时间或None)
        """
        import re

        # 匹配停顿时间标记的正则表达式
        # 支持格式：-0.8s-、-1.2s-、-0.5s-、-2s-、-0.1s- 等
        pause_pattern = r'-(\d+(?:\.\d+)?)s-'

        # 查找所有停顿标记
        matches = re.findall(pause_pattern, text)

        if matches:
            # 取最后一个停顿标记作为该句话的停顿时间
            pause_time = float(matches[-1])

            # 从文本中移除所有停顿标记
            clean_text = re.sub(pause_pattern, '', text).strip()

            return clean_text, pause_time

        return text, None

    def _parse_conversation(self, conversation_text: str, num_speakers: int, verbose: bool) -> List[Dict]:
        """解析对话文本 - 支持自定义说话人名称和内嵌停顿时间标记"""
        lines = conversation_text.strip().split('\n')
        conversation_lines = []

        # 首先扫描所有行，提取所有说话人名称
        speaker_names = []
        for line in lines:
            line = line.strip()
            if ':' in line:
                potential_speaker = line.split(':', 1)[0].strip()
                if potential_speaker and potential_speaker not in speaker_names:
                    speaker_names.append(potential_speaker)

        # 限制说话人数量
        speaker_names = speaker_names[:num_speakers]

        if verbose:
            print(f"[MultiTalk] 检测到说话人: {speaker_names}")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 查找说话人标识
            speaker_found = False

            # 首先尝试匹配检测到的说话人名称
            for i, speaker_name in enumerate(speaker_names):
                if line.startswith(f"{speaker_name}:"):
                    text = line[len(f"{speaker_name}:"):].strip()
                    if text:
                        # 提取停顿时间标记
                        clean_text, pause_time = self._extract_pause_from_text(text)

                        conversation_lines.append({
                            "speaker_idx": i,  # 0-based index
                            "speaker_name": speaker_name,
                            "text": clean_text,
                            "custom_pause": pause_time  # 自定义停顿时间
                        })
                        speaker_found = True
                        break

            # 如果没有找到，尝试标准格式
            if not speaker_found:
                for i in range(1, num_speakers + 1):
                    speaker_patterns = [f"Speaker{i}:", f"speaker{i}:", f"说话人{i}:", f"S{i}:"]
                    for pattern in speaker_patterns:
                        if line.startswith(pattern):
                            text = line[len(pattern):].strip()
                            if text:
                                # 提取停顿时间标记
                                clean_text, pause_time = self._extract_pause_from_text(text)

                                conversation_lines.append({
                                    "speaker_idx": i - 1,  # 0-based index
                                    "speaker_name": f"Speaker{i}",
                                    "text": clean_text,
                                    "custom_pause": pause_time  # 自定义停顿时间
                                })
                                speaker_found = True
                                break
                    if speaker_found:
                        break

            if not speaker_found and line:
                # 如果没有找到说话人标识，默认分配给第一个说话人
                # 提取停顿时间标记
                clean_text, pause_time = self._extract_pause_from_text(line)

                conversation_lines.append({
                    "speaker_idx": 0,
                    "speaker_name": speaker_names[0] if speaker_names else "Speaker1",
                    "text": clean_text,
                    "custom_pause": pause_time  # 自定义停顿时间
                })
                if verbose:
                    print(f"[MultiTalk] 未识别说话人，分配给{speaker_names[0] if speaker_names else 'Speaker1'}: {clean_text[:30]}...")

        if not conversation_lines:
            raise ValueError("No valid conversation lines found. Please use format: 'Speaker1: text' or 'YourName: text'")

        if verbose:
            print(f"[MultiTalk] 解析到 {len(conversation_lines)} 个对话片段")
            for i, line in enumerate(conversation_lines):
                print(f"  {i+1}. {line['speaker_name']}: {line['text'][:50]}...")

        return conversation_lines

    def _prepare_emotion_configs_from_inputs(self, num_speakers: int,
                                            emotion_config_inputs: List[Optional[Dict]],
                                            verbose: bool) -> List[Dict]:
        """从输入的情感配置对象准备情感控制配置"""
        emotion_configs = []

        for i in range(num_speakers):
            emotion_input = emotion_config_inputs[i] if i < len(emotion_config_inputs) else None

            if emotion_input is None or not emotion_input.get("enabled", True):
                # 如果没有提供情感配置或被禁用，使用默认配置
                emotion_config = {"mode": "none"}
                if verbose:
                    print(f"[MultiTalk] Speaker{i+1}: No emotion control (default)")
            else:
                # 使用提供的情感配置
                emotion_config = {
                    "mode": emotion_input.get("mode", "none"),
                    "audio": emotion_input.get("audio", ""),
                    "alpha": emotion_input.get("alpha", 1.0),
                    "vector": emotion_input.get("vector", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                    "text": emotion_input.get("text", "")
                }

                if verbose:
                    mode = emotion_config["mode"]
                    speaker_name = emotion_input.get("speaker_name", f"Speaker{i+1}")
                    print(f"[MultiTalk] {speaker_name} emotion mode: {mode}")

                    if mode == "emotion_vector":
                        vector = emotion_config["vector"]
                        emotion_names = ["Happy", "Angry", "Sad", "Fear", "Hate", "Low", "Surprise", "Neutral"]
                        active_emotions = [f"{name}: {val:.2f}" for name, val in zip(emotion_names, vector) if val > 0.05]
                        if active_emotions:
                            print(f"[MultiTalk] {speaker_name} emotions: {', '.join(active_emotions)}")

            emotion_configs.append(emotion_config)

        return emotion_configs

    def _save_audio(self, audio_tensor, filename):
        """保存音频文件"""
        output_dir = folder_paths.get_output_directory()

        # 确保文件名有正确的扩展名
        if not filename.lower().endswith(('.wav', '.mp3', '.flac')):
            filename = filename + ".wav"

        output_path = os.path.join(output_dir, filename)

        # 确保音频是正确的形状
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # 保存音频
        torchaudio.save(output_path, audio_tensor, 24000)
        print(f"💾 Multi-talk conversation saved to: {output_path}")

        return output_path

    def _synthesize_with_emotion(self, model, text: str, speaker_audio_path: str,
                                emotion_config: Dict, output_path: str,
                                temperature: float, top_p: float, verbose: bool,
                                voice_consistency: float = 1.0, language: str = "zh") -> str:
        """执行带情感控制的语音合成"""
        emotion_mode = emotion_config["mode"]

        # 应用声音一致性参数
        consistency_temp = max(0.1, temperature / voice_consistency)
        consistency_top_p = min(0.99, top_p * voice_consistency)

        if emotion_mode == "none":
            # 无情感控制，使用基础合成
            model.infer(
                spk_audio_prompt=speaker_audio_path,
                text=text,
                output_path=output_path,
                verbose=False,
                temperature=consistency_temp,
                top_p=consistency_top_p,
                top_k=50,
                max_text_tokens_per_sentence=120,
                interval_silence=200
            )
            return "No emotion control"

        elif emotion_mode == "audio_prompt":
            emotion_audio_info = emotion_config["audio"]
            emotion_alpha = emotion_config["alpha"]

            if emotion_audio_info and isinstance(emotion_audio_info, dict) and "audio_object" in emotion_audio_info:
                # 将AUDIO对象保存为临时文件
                emotion_audio_path = self._save_emotion_audio_to_temp(emotion_audio_info["audio_object"])
                if emotion_audio_path:
                    try:
                        model.infer(
                            spk_audio_prompt=speaker_audio_path,
                            text=text,
                            output_path=output_path,
                            emo_audio_prompt=emotion_audio_path,
                            emo_alpha=emotion_alpha,
                            verbose=False,
                            temperature=consistency_temp,
                            top_p=consistency_top_p,
                            top_k=50,
                            max_text_tokens_per_sentence=120,
                            interval_silence=200
                        )
                    finally:
                        # 清理临时文件
                        try:
                            os.unlink(emotion_audio_path)
                        except:
                            pass
                else:
                    # 回退到无情感控制
                    model.infer(
                        spk_audio_prompt=speaker_audio_path,
                        text=text,
                        output_path=output_path,
                        verbose=False,
                        temperature=consistency_temp,
                        top_p=consistency_top_p,
                        top_k=50,
                        max_text_tokens_per_sentence=120,
                        interval_silence=200
                    )
                return f"Audio emotion ({os.path.basename(emotion_audio)}, α={emotion_alpha})"
            else:
                # 回退到基础合成
                model.infer(
                    spk_audio_prompt=speaker_audio_path,
                    text=text,
                    output_path=output_path,
                    verbose=False,
                    temperature=consistency_temp,
                    top_p=consistency_top_p,
                    top_k=50,
                    max_text_tokens_per_sentence=120,
                    interval_silence=200
                )
                return "No emotion audio provided"

        elif emotion_mode == "emotion_vector":
            emotion_vector = emotion_config["vector"]

            # 检查情感向量是否全为零
            max_emotion_value = max(emotion_vector)
            if max_emotion_value == 0.0:
                # 设置一个小的中性情感值
                emotion_vector = emotion_vector.copy()
                emotion_vector[7] = 0.1  # Neutral

            model.infer(
                spk_audio_prompt=speaker_audio_path,
                text=text,
                output_path=output_path,
                emo_vector=emotion_vector,
                verbose=False,
                temperature=consistency_temp,
                top_p=consistency_top_p,
                top_k=50,
                max_text_tokens_per_sentence=120,
                interval_silence=200
            )

            # 分析主要情感
            emotion_names = ["Happy", "Angry", "Sad", "Fear", "Hate", "Low", "Surprise", "Neutral"]
            max_value = max(emotion_vector)
            if max_value > 0:
                max_idx = emotion_vector.index(max_value)
                dominant_emotion = emotion_names[max_idx]
                return f"Vector emotion ({dominant_emotion}: {max_value:.2f})"
            else:
                return "Vector emotion (Neutral)"

        elif emotion_mode == "text_description":
            emotion_text = emotion_config["text"]

            if emotion_text.strip():
                model.infer(
                    spk_audio_prompt=speaker_audio_path,
                    text=text,
                    output_path=output_path,
                    use_emo_text=True,
                    emo_text=emotion_text,
                    verbose=False,
                    temperature=consistency_temp,
                    top_p=consistency_top_p,
                    top_k=50,
                    max_text_tokens_per_sentence=120,
                    interval_silence=200
                )
                return f"Text emotion ({emotion_text[:30]}...)"
            else:
                # 从合成文本推断情感
                model.infer(
                    spk_audio_prompt=speaker_audio_path,
                    text=text,
                    output_path=output_path,
                    use_emo_text=True,
                    verbose=False,
                    temperature=consistency_temp,
                    top_p=consistency_top_p,
                    top_k=50,
                    max_text_tokens_per_sentence=120,
                    interval_silence=200
                )
                return "Text emotion (inferred)"

        else:  # auto mode
            model.infer(
                spk_audio_prompt=speaker_audio_path,
                text=text,
                output_path=output_path,
                verbose=False,
                temperature=consistency_temp,
                top_p=consistency_top_p,
                top_k=50,
                max_text_tokens_per_sentence=120,
                interval_silence=200
            )
            return "Auto emotion"

    def _prepare_speaker_audios(self, speaker_audios: List[dict], verbose: bool,
                               voice_consistency: float = 1.0, reference_boost: bool = True) -> List[str]:
        """准备说话人音频文件（带一致性增强）"""
        speaker_audio_paths = []

        for i, speaker_audio in enumerate(speaker_audios):
            if not isinstance(speaker_audio, dict) or "waveform" not in speaker_audio or "sample_rate" not in speaker_audio:
                raise ValueError(f"Speaker {i+1} audio must be a ComfyUI AUDIO object")

            waveform = speaker_audio["waveform"]
            sample_rate = speaker_audio["sample_rate"]

            # 移除batch维度（如果存在）
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)

            # 应用参考音频增强
            if reference_boost and voice_consistency > 1.0:
                waveform = self._enhance_reference_audio(waveform, voice_consistency)

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=f"_speaker{i+1}.wav", delete=False) as tmp_file:
                speaker_audio_path = tmp_file.name

            # 保存音频到临时文件
            torchaudio.save(speaker_audio_path, waveform, sample_rate)
            speaker_audio_paths.append(speaker_audio_path)

            if verbose:
                print(f"[MultiTalk] Speaker{i+1} 音频: 采样率={sample_rate}, 形状={waveform.shape}")
                if reference_boost and voice_consistency > 1.0:
                    print(f"[MultiTalk] Speaker{i+1} 应用了参考音频增强 (一致性={voice_consistency})")

        return speaker_audio_paths

    def _smooth_audio_transition(self, audio1: torch.Tensor, audio2: torch.Tensor,
                               fade_samples: int = 1024) -> torch.Tensor:
        """在两个音频片段间添加平滑过渡"""
        if audio1.shape[-1] < fade_samples or audio2.shape[-1] < fade_samples:
            return torch.cat([audio1, audio2], dim=-1)

        # 创建淡入淡出窗口
        fade_out = torch.linspace(1.0, 0.0, fade_samples, device=audio1.device, dtype=audio1.dtype)
        fade_in = torch.linspace(0.0, 1.0, fade_samples, device=audio2.device, dtype=audio2.dtype)

        # 确保维度匹配
        if audio1.dim() == 2:  # [channels, samples]
            fade_out = fade_out.unsqueeze(0).expand(audio1.shape[0], -1)
            fade_in = fade_in.unsqueeze(0).expand(audio2.shape[0], -1)

        # 应用交叉淡化
        audio1_end = audio1[..., -fade_samples:] * fade_out
        audio2_start = audio2[..., :fade_samples] * fade_in

        # 混合重叠部分
        mixed_section = audio1_end + audio2_start

        # 拼接最终音频
        result = torch.cat([
            audio1[..., :-fade_samples],
            mixed_section,
            audio2[..., fade_samples:]
        ], dim=-1)

        return result

    def _enhance_reference_audio(self, waveform: torch.Tensor, voice_consistency: float) -> torch.Tensor:
        """增强版参考音频处理 - 使用智能预处理器"""
        try:
            # 1. 音频长度检查和处理
            min_length = max(16000, int(0.5 * 22050))  # 至少0.5秒
            if waveform.shape[-1] < min_length:
                repeat_times = int(min_length / waveform.shape[-1]) + 1
                waveform = waveform.repeat(1, repeat_times)[:, :min_length]

            # 2. 根据voice_consistency参数决定处理强度
            if voice_consistency <= 1.0:
                # 基础处理：仅音量标准化
                processed_audio = self.audio_preprocessor.normalize_loudness(waveform)
            elif voice_consistency <= 1.5:
                # 中等处理：降噪 + 标准化
                processed_audio = self.audio_preprocessor.process(
                    waveform,
                    noise_gate=True,
                    compression=False,
                    spectral_enhancement=False,
                    loudness_normalization=True
                )
            else:
                # 完整处理：全套智能预处理
                enhancement_strength = min((voice_consistency - 1.0) * 0.3, 0.5)

                # 自定义处理参数
                processed_audio = waveform.clone()

                # 噪声门限
                processed_audio = self.audio_preprocessor.apply_noise_gate(processed_audio, threshold_db=-35)

                # 动态压缩（轻微）
                processed_audio = self.audio_preprocessor.apply_dynamic_compression(
                    processed_audio, threshold_db=-15, ratio=2.0
                )

                # 频谱增强
                processed_audio = self.audio_preprocessor.apply_spectral_enhancement(
                    processed_audio, enhancement_strength=enhancement_strength
                )

                # 响度标准化
                processed_audio = self.audio_preprocessor.normalize_loudness(processed_audio)

            # 3. 最终限幅处理
            processed_audio = torch.clamp(processed_audio, -0.95, 0.95)

            return processed_audio

        except Exception as e:
            print(f"[MultiTalk] 智能音频预处理失败，使用原始音频: {e}")
            return waveform

    def _merge_audio_segments(self, audio_segments: List[dict], silence_duration: float, verbose: bool) -> dict:
        """合并音频片段"""
        if not audio_segments:
            raise ValueError("No audio segments to merge")

        # 获取第一个片段的采样率
        sample_rate = audio_segments[0]["sample_rate"]

        # 确保所有片段的采样率一致
        for i, segment in enumerate(audio_segments):
            if segment["sample_rate"] != sample_rate:
                if verbose:
                    print(f"[MultiTalk] 重采样片段 {i+1}: {segment['sample_rate']} -> {sample_rate}")
                # 重采样到统一采样率
                resampler = torchaudio.transforms.Resample(segment["sample_rate"], sample_rate)
                segment["waveform"] = resampler(segment["waveform"])
                segment["sample_rate"] = sample_rate

        # 计算静音片段
        silence_samples = int(silence_duration * sample_rate)
        silence_waveform = torch.zeros(audio_segments[0]["waveform"].shape[0], silence_samples)

        # 合并所有片段
        merged_waveforms = []
        for i, segment in enumerate(audio_segments):
            waveform = segment["waveform"]

            # 确保是2D张量 [channels, samples]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            merged_waveforms.append(waveform)

            # 在片段之间添加静音（除了最后一个片段）
            if i < len(audio_segments) - 1 and silence_duration > 0:
                merged_waveforms.append(silence_waveform)

        # 连接所有波形
        final_waveform = torch.cat(merged_waveforms, dim=1)

        if verbose:
            total_duration = final_waveform.shape[1] / sample_rate
            print(f"[MultiTalk] 合并完成: {len(audio_segments)} 个片段, 总时长: {total_duration:.2f}秒")

        return {
            "waveform": final_waveform,
            "sample_rate": sample_rate
        }

    def _merge_audio_segments_with_custom_pauses(self, audio_segments: List[dict],
                                               conversation_lines: List[Dict],
                                               speaker_pauses: List[float],
                                               default_silence: float,
                                               verbose: bool) -> dict:
        """合并音频片段（支持每个说话人的个性化停顿时间）"""
        if not audio_segments:
            raise ValueError("No audio segments to merge")

        # 获取第一个片段的采样率
        sample_rate = audio_segments[0]["sample_rate"]

        # 确保所有片段的采样率一致
        for i, segment in enumerate(audio_segments):
            if segment["sample_rate"] != sample_rate:
                if verbose:
                    print(f"[MultiTalk] 重采样片段 {i+1}: {segment['sample_rate']} -> {sample_rate}")
                # 重采样到统一采样率
                resampler = torchaudio.transforms.Resample(segment["sample_rate"], sample_rate)
                segment["waveform"] = resampler(segment["waveform"])
                segment["sample_rate"] = sample_rate

        # 使用平滑过渡合并所有片段
        total_pause_time = 0.0
        fade_samples = min(512, sample_rate // 50)  # 约20ms的淡化时间

        # 处理第一个片段
        first_segment = audio_segments[0]
        current_waveform = first_segment["waveform"]

        # 确保是2D张量 [channels, samples]
        if current_waveform.dim() == 3:
            current_waveform = current_waveform.squeeze(0)
        elif current_waveform.dim() == 1:
            current_waveform = current_waveform.unsqueeze(0)

        for i in range(1, len(audio_segments)):
            # 获取当前片段
            segment = audio_segments[i]
            next_waveform = segment["waveform"]

            # 确保是2D张量 [channels, samples]
            if next_waveform.dim() == 3:
                next_waveform = next_waveform.squeeze(0)
            elif next_waveform.dim() == 1:
                next_waveform = next_waveform.unsqueeze(0)

            # 添加个性化停顿时间
            current_line = conversation_lines[i-1]  # 前一个说话人的停顿
            current_speaker_idx = current_line["speaker_idx"]

            # 检查是否有自定义停顿时间
            custom_pause = current_line.get("custom_pause")
            if custom_pause is not None:
                pause_duration = custom_pause
                pause_source = "文本标记"
            else:
                # 使用说话人设置的停顿时间
                pause_duration = speaker_pauses[current_speaker_idx] if current_speaker_idx < len(speaker_pauses) else default_silence
                pause_source = "说话人设置"

            if pause_duration > 0:
                pause_samples = int(pause_duration * sample_rate)
                pause_waveform = torch.zeros(current_waveform.shape[0], pause_samples, device=current_waveform.device, dtype=current_waveform.dtype)

                # 使用平滑过渡连接：当前音频 -> 停顿 -> 下一个音频
                current_waveform = self._smooth_audio_transition(current_waveform, pause_waveform, fade_samples)
                current_waveform = self._smooth_audio_transition(current_waveform, next_waveform, fade_samples)

                total_pause_time += pause_duration

                if verbose:
                    print(f"[MultiTalk] Speaker{current_speaker_idx + 1} 停顿时间: {pause_duration:.2f}秒 ({pause_source}) [平滑过渡]")
            else:
                # 直接使用平滑过渡连接
                current_waveform = self._smooth_audio_transition(current_waveform, next_waveform, fade_samples)

                if verbose:
                    print(f"[MultiTalk] Speaker{current_speaker_idx + 1} -> Speaker{conversation_lines[i]['speaker_idx'] + 1} [平滑过渡]")

        final_waveform = current_waveform

        if verbose:
            total_duration = final_waveform.shape[1] / sample_rate
            audio_duration = total_duration - total_pause_time
            print(f"[MultiTalk] 个性化停顿合并完成:")
            print(f"  - 音频片段: {len(audio_segments)} 个")
            print(f"  - 纯音频时长: {audio_duration:.2f}秒")
            print(f"  - 总停顿时长: {total_pause_time:.2f}秒")
            print(f"  - 最终总时长: {total_duration:.2f}秒")

        return {
            "waveform": final_waveform,
            "sample_rate": sample_rate
        }

    def _load_default_model(self, use_fp16: bool, use_cuda_kernel: bool):
        """加载默认模型（带缓存机制）"""
        try:
            # 创建缓存键
            cache_key = f"fp16_{use_fp16}_cuda_{use_cuda_kernel}"

            # 检查是否已有缓存的模型实例
            if not hasattr(self, '_model_cache'):
                self._model_cache = {}



            if cache_key in self._model_cache:
                cached_model = self._model_cache[cache_key]
                # 验证缓存的模型是否有效
                if (hasattr(cached_model, 'spk_matrix') and
                    cached_model.spk_matrix is not None and
                    isinstance(cached_model.spk_matrix, (list, tuple)) and
                    len(cached_model.spk_matrix) > 0):
                    print(f"[MultiTalk] ✓ 使用缓存的模型实例")
                    return cached_model
                else:
                    print(f"[MultiTalk] ⚠️ 缓存的模型实例无效，重新创建")
                    if cache_key in self._model_cache:
                        del self._model_cache[cache_key]

            # 统一使用标准导入路径
            from indextts.infer_v2 import IndexTTS2

            # 使用通用模型路径函数
            from .model_utils import get_indextts2_model_path, validate_model_path

            model_dir, config_path = get_indextts2_model_path()

            print(f"[MultiTalk] 创建模型实例，路径: {model_dir}")

            # 验证模型路径
            validate_model_path(model_dir, config_path)

            model = IndexTTS2(
                cfg_path=config_path,
                model_dir=model_dir,
                is_fp16=use_fp16,
                use_cuda_kernel=use_cuda_kernel
            )

            # 验证模型初始化是否成功
            if (hasattr(model, 'spk_matrix') and
                model.spk_matrix is not None and
                isinstance(model.spk_matrix, (list, tuple)) and
                len(model.spk_matrix) > 0):
                print(f"[MultiTalk] ✓ 模型初始化成功，缓存实例")
                self._model_cache[cache_key] = model
            else:
                print(f"[MultiTalk] ⚠️ 模型初始化不完整，不缓存此实例")

            return model

        except Exception as e:
            error_msg = f"Failed to load IndexTTS2 model: {str(e)}"
            # 特别处理DeepSpeed相关错误
            if "deepspeed" in str(e).lower():
                error_msg += "\n[MultiTalk] DeepSpeed相关错误，但基本功能应该仍然可用"
                error_msg += "\n[MultiTalk] DeepSpeed-related error, but basic functionality should still work"
            raise RuntimeError(error_msg)

    def _load_audio(self, audio_path: str) -> dict:
        """加载音频文件"""
        from .audio_utils import load_audio_for_comfyui
        return load_audio_for_comfyui(audio_path)

    def _generate_info_with_emotion(self, conversation_lines: List[Dict], num_speakers: int,
                                   output_path: str, language: str, speed: float, silence_duration: float,
                                   emotion_configs: List[Dict]) -> str:
        """生成包含情感信息的信息字符串"""
        info_lines = [
            "=== IndexTTS2 Multi-Talk Synthesis with Emotion Control ===",
            f"Speakers: {num_speakers}",
            f"Conversation Lines: {len(conversation_lines)}",
            f"Language: {language}",
            f"Speed: {speed}x",
            f"Silence Duration: {silence_duration}s",
            f"Output: {os.path.basename(output_path)}",
            "",
            "=== Speaker Emotion Settings ===",
        ]

        # 添加每个说话人的情感设置
        for i, emotion_config in enumerate(emotion_configs):
            if i < num_speakers:
                mode = emotion_config.get("mode", "none")
                info_lines.append(f"Speaker{i+1}: {mode}")

                if mode == "emotion_vector":
                    vector = emotion_config.get("vector", [])
                    emotion_names = ["Happy", "Angry", "Sad", "Fear", "Hate", "Low", "Surprise", "Neutral"]
                    active_emotions = [f"{name}: {val:.2f}" for name, val in zip(emotion_names, vector) if val > 0.1]
                    if active_emotions:
                        info_lines.append(f"  Emotions: {', '.join(active_emotions)}")
                elif mode == "audio_prompt":
                    audio_info = emotion_config.get("audio", None)
                    alpha = emotion_config.get("alpha", 1.0)
                    if audio_info and isinstance(audio_info, dict):
                        duration = audio_info.get("duration", 0)
                        info_lines.append(f"  Audio: {duration:.2f}s (α={alpha})")
                elif mode == "text_description":
                    text = emotion_config.get("text", "")
                    if text:
                        info_lines.append(f"  Description: {text[:50]}...")

        info_lines.extend([
            "",
            "=== Conversation Preview ===",
        ])

        # 添加对话预览（最多显示前5行）
        for i, line in enumerate(conversation_lines[:5]):
            preview_text = line["text"][:60] + "..." if len(line["text"]) > 60 else line["text"]
            info_lines.append(f"{line['speaker_name']}: {preview_text}")

        if len(conversation_lines) > 5:
            info_lines.append(f"... and {len(conversation_lines) - 5} more lines")

        # 添加Qwen模型信息
        info_lines.extend([
            "",
            "=== Qwen Emotion Model Status ===",
        ])

        qwen_info = self._get_qwen_model_info()
        info_lines.extend(qwen_info)

        return "\n".join(info_lines)

    def _generate_info_with_emotion_and_pauses(self, conversation_lines: List[Dict], num_speakers: int,
                                             output_path: str, language: str, speed: float,
                                             silence_duration: float, speaker_pauses: List[float],
                                             emotion_configs: List[Dict]) -> str:
        """生成包含情感信息和个性化停顿时间的信息字符串"""
        info_lines = [
            "=== IndexTTS2 Multi-Talk Synthesis with Emotion Control & Custom Pauses ===",
            f"Speakers: {num_speakers}",
            f"Conversation Lines: {len(conversation_lines)}",
            f"Language: {language}",
            f"Speed: {speed}x",
            f"Default Silence Duration: {silence_duration}s",
            f"Output: {os.path.basename(output_path)}",
            "",
            "=== Individual Speaker Pause Settings ===",
        ]

        # 添加每个说话人的停顿时间设置
        for i in range(num_speakers):
            pause_time = speaker_pauses[i] if i < len(speaker_pauses) else silence_duration
            info_lines.append(f"Speaker{i+1} Pause: {pause_time:.2f}s")

        info_lines.extend([
            "",
            "=== Speaker Emotion Settings ===",
        ])

        # 添加每个说话人的情感设置
        for i, emotion_config in enumerate(emotion_configs):
            if i < num_speakers:
                mode = emotion_config.get("mode", "none")
                pause_time = speaker_pauses[i] if i < len(speaker_pauses) else silence_duration
                info_lines.append(f"Speaker{i+1}: {mode} (Pause: {pause_time:.2f}s)")

                if mode == "emotion_vector":
                    vector = emotion_config.get("vector", [])
                    emotion_names = ["Happy", "Angry", "Sad", "Fear", "Hate", "Low", "Surprise", "Neutral"]
                    active_emotions = [f"{name}: {val:.2f}" for name, val in zip(emotion_names, vector) if val > 0.1]
                    if active_emotions:
                        info_lines.append(f"  Emotions: {', '.join(active_emotions)}")
                elif mode == "audio_prompt":
                    audio_info = emotion_config.get("audio", None)
                    alpha = emotion_config.get("alpha", 1.0)
                    if audio_info and isinstance(audio_info, dict):
                        duration = audio_info.get("duration", 0)
                        info_lines.append(f"  Audio: {duration:.2f}s (α={alpha})")
                elif mode == "text_description":
                    text = emotion_config.get("text", "")
                    if text:
                        info_lines.append(f"  Description: {text[:50]}...")

        info_lines.extend([
            "",
            "=== Conversation Preview ===",
        ])

        # 添加对话预览（最多显示前5行）
        for i, line in enumerate(conversation_lines[:5]):
            preview_text = line["text"][:60] + "..." if len(line["text"]) > 60 else line["text"]
            speaker_idx = line["speaker_idx"]

            # 优先显示文本中的自定义停顿时间
            custom_pause = line.get("custom_pause")
            if custom_pause is not None:
                pause_time = custom_pause
                pause_source = "文本"
            else:
                pause_time = speaker_pauses[speaker_idx] if speaker_idx < len(speaker_pauses) else silence_duration
                pause_source = "设置"

            info_lines.append(f"{line['speaker_name']}: {preview_text} [Pause: {pause_time:.2f}s ({pause_source})]")

        if len(conversation_lines) > 5:
            info_lines.append(f"... and {len(conversation_lines) - 5} more lines")

        # 添加Qwen模型信息
        info_lines.extend([
            "",
            "=== Qwen Emotion Model Status ===",
        ])

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

    def _save_emotion_audio_to_temp(self, emotion_audio: dict) -> Optional[str]:
        """将ComfyUI AUDIO对象保存为临时文件供IndexTTS2使用"""
        try:
            import tempfile
            import torchaudio

            if not isinstance(emotion_audio, dict) or "waveform" not in emotion_audio or "sample_rate" not in emotion_audio:
                print("[MultiTalk] Invalid emotion audio object")
                return None

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

            return emotion_audio_path

        except Exception as e:
            print(f"[MultiTalk] Failed to save emotion audio: {str(e)}")
            return None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """检查输入是否改变"""
        return float("nan")  # 总是重新执行
