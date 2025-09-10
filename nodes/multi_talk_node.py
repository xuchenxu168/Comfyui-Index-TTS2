# IndexTTS2 Multi-Talk Node with Emotion Control
# IndexTTS2 多人对话语音合成节点（带情感控制）

import os
import torch
import numpy as np
import tempfile
import torchaudio
from typing import Optional, Tuple, Any, List, Dict
import folder_paths

class IndexTTS2MultiTalkNode:
    """
    IndexTTS2 多人对话语音合成节点（带情感控制）
    Multi-speaker conversation text-to-speech synthesis node for IndexTTS2 with emotion control

    Features:
    - Support 2-4 speakers conversation
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
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_speakers": (["2", "3", "4"], {
                    "default": "2",
                    "tooltip": "对话人数 / Number of speakers"
                }),
                "conversation_text": ("STRING", {
                    "multiline": True,
                    "default": "Speaker1: Hello, how are you today?\nSpeaker2: I'm doing great, thank you for asking!",
                    "placeholder": "输入对话文本，格式：Speaker1: 文本\\nSpeaker2: 文本..."
                }),
                "speaker1_audio": ("AUDIO", {
                    "tooltip": "说话人1的音频样本 / Speaker 1 audio sample"
                }),
                "speaker2_audio": ("AUDIO", {
                    "tooltip": "说话人2的音频样本 / Speaker 2 audio sample"
                }),
                "output_filename": ("STRING", {
                    "default": "multi_talk_emotion_output.wav",
                    "placeholder": "输出音频文件名"
                }),
            },
            "optional": {
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
                    "tooltip": "说话人之间的静音时长（秒）/ Silence duration between speakers (seconds)"
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
    DESCRIPTION = "Multi-speaker conversation synthesis with individual emotion control using IndexTTS2 (2-4 speakers)"
    
    def synthesize_conversation(
        self,
        num_speakers: str,
        conversation_text: str,
        speaker1_audio: dict,
        speaker2_audio: dict,
        output_filename: str,
        speaker3_audio: Optional[dict] = None,
        speaker4_audio: Optional[dict] = None,
        model_manager: Optional[Any] = None,
        language: str = "auto",
        speed: float = 1.0,
        silence_duration: float = 0.5,
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

            # 验证说话人音频
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

                # 执行单个片段的情感合成（带一致性控制）
                emotion_analysis = self._synthesize_with_emotion(
                    model, text, speaker_audio_path, emotion_config,
                    temp_output, temperature, top_p, verbose, voice_consistency
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
            
            # 合并音频片段
            final_audio = self._merge_audio_segments(audio_segments, silence_duration, verbose)
            
            # 准备输出路径
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存最终音频
            torchaudio.save(output_path, final_audio["waveform"], final_audio["sample_rate"])
            
            # 确保音频格式兼容ComfyUI
            waveform = final_audio["waveform"]
            sample_rate = final_audio["sample_rate"]
            
            # 应用ComfyUI兼容性检查
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
            
            if verbose:
                print(f"[MultiTalk] 对话合成完成: {len(conversation_lines)} 个片段")
                print(f"[MultiTalk] 最终音频格式: {waveform.shape}, 采样率: {sample_rate}")

            # 生成信息字符串
            info = self._generate_info_with_emotion(
                conversation_lines, num_speakers_int, output_path, language, speed,
                silence_duration, emotion_configs
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

    def _parse_conversation(self, conversation_text: str, num_speakers: int, verbose: bool) -> List[Dict]:
        """解析对话文本 - 支持自定义说话人名称"""
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
                        conversation_lines.append({
                            "speaker_idx": i,  # 0-based index
                            "speaker_name": speaker_name,
                            "text": text
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
                                conversation_lines.append({
                                    "speaker_idx": i - 1,  # 0-based index
                                    "speaker_name": f"Speaker{i}",
                                    "text": text
                                })
                                speaker_found = True
                                break
                    if speaker_found:
                        break

            if not speaker_found and line:
                # 如果没有找到说话人标识，默认分配给第一个说话人
                conversation_lines.append({
                    "speaker_idx": 0,
                    "speaker_name": speaker_names[0] if speaker_names else "Speaker1",
                    "text": line
                })
                if verbose:
                    print(f"[MultiTalk] 未识别说话人，分配给{speaker_names[0] if speaker_names else 'Speaker1'}: {line[:30]}...")

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

    def _synthesize_with_emotion(self, model, text: str, speaker_audio_path: str,
                                emotion_config: Dict, output_path: str,
                                temperature: float, top_p: float, verbose: bool,
                                voice_consistency: float = 1.0) -> str:
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
            emotion_audio = emotion_config["audio"]
            emotion_alpha = emotion_config["alpha"]

            if emotion_audio and os.path.exists(emotion_audio):
                model.infer(
                    spk_audio_prompt=speaker_audio_path,
                    text=text,
                    output_path=output_path,
                    emo_audio_prompt=emotion_audio,
                    emo_alpha=emotion_alpha,
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

    def _enhance_reference_audio(self, waveform: torch.Tensor, voice_consistency: float) -> torch.Tensor:
        """增强参考音频以提高声音一致性"""
        try:
            import torch.nn.functional as F

            # 确保音频长度足够
            if waveform.shape[-1] < 1000:
                # 如果音频太短，进行重复
                repeat_times = int(1000 / waveform.shape[-1]) + 1
                waveform = waveform.repeat(1, repeat_times)[:, :1000]

            # 应用轻微的音频增强
            if voice_consistency > 1.0:
                # 增强音频的清晰度
                enhancement_factor = min(voice_consistency, 2.0)

                # 轻微的高频增强（提高清晰度）
                if waveform.shape[-1] > 512:
                    # 简单的高通滤波效果
                    kernel = torch.tensor([[-0.1, -0.1, 0.8, -0.1, -0.1]], dtype=waveform.dtype)
                    kernel = kernel.unsqueeze(0)  # [1, 1, 5]

                    # 对每个声道分别处理
                    enhanced_waveform = []
                    for ch in range(waveform.shape[0]):
                        ch_data = waveform[ch:ch+1].unsqueeze(0)  # [1, 1, length]
                        # 应用卷积
                        enhanced = F.conv1d(ch_data, kernel, padding=2)
                        # 混合原始和增强的信号
                        mix_ratio = (enhancement_factor - 1.0) * 0.3  # 限制增强强度
                        enhanced = ch_data * (1 - mix_ratio) + enhanced * mix_ratio
                        enhanced_waveform.append(enhanced.squeeze(0))

                    waveform = torch.cat(enhanced_waveform, dim=0)

                # 轻微的音量标准化
                max_val = torch.max(torch.abs(waveform))
                if max_val > 0:
                    target_level = 0.7  # 目标音量级别
                    waveform = waveform * (target_level / max_val)

            return waveform

        except Exception as e:
            print(f"[MultiTalk] 音频增强失败，使用原始音频: {e}")
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

    def _load_default_model(self, use_fp16: bool, use_cuda_kernel: bool):
        """加载默认模型"""
        try:
            from ..indextts.infer_v2 import IndexTTS2

            # 使用通用模型路径函数
            from .model_utils import get_indextts2_model_path, validate_model_path

            model_dir, config_path = get_indextts2_model_path()

            print(f"[MultiTalk] 使用模型路径: {model_dir}")

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
            raise RuntimeError(f"Failed to load IndexTTS2 model: {str(e)}")

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
                    audio = emotion_config.get("audio", "")
                    alpha = emotion_config.get("alpha", 1.0)
                    if audio:
                        info_lines.append(f"  Audio: {os.path.basename(audio)} (α={alpha})")
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

        return "\n".join(info_lines)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """检查输入是否改变"""
        return float("nan")  # 总是重新执行
