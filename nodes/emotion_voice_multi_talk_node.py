# IndexTTS2 Emotion Voice Multi-Talk Node
# IndexTTS2 情绪语音多人对话节点

import os
import torch
import numpy as np
import tempfile
import torchaudio
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

class IndexTTS2EmotionVoiceMultiTalkNode:
    """
    IndexTTS2 情绪语音多人对话节点
    Emotion Voice Multi-Speaker Conversation Node for IndexTTS2
    
    Features:
    - Support 2-4 speakers conversation
    - Individual speaker voice cloning
    - Emotion control via emotion voice samples for each speaker
    - Multiple emotion modes (emotion_voice, emotion_vector, auto)
    - Automatic conversation flow with emotional expressions
    - Configurable silence intervals
    - High-quality multi-speaker synthesis with emotion voice control
    """

    def __init__(self):
        self.model = None
        self.model_config = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_speakers": (["1", "2", "3", "4"], {
                    "default": "2",
                    "tooltip": "对话人数 / Number of speakers (1-4, 1=纯语音克隆)"
                }),
                "conversation_text": ("STRING", {
                    "multiline": True,
                    "default": "Speaker1: [Happy] Hello, how are you today!\nSpeaker2: [Excited] I'm doing great, thank you for asking!",
                    "placeholder": "单人模式：直接输入文本\\n多人模式：Speaker1: [情绪] 文本\\nSpeaker2: [情绪] 文本..."
                }),
                "speaker1_voice": ("AUDIO", {
                    "tooltip": "说话人1的音色样本 / Speaker 1 voice sample"
                }),
                "output_filename": ("STRING", {
                    "default": "emotion_voice_conversation.wav",
                    "placeholder": "输出音频文件名 / Output audio filename"
                }),
            },
            "optional": {
                "speaker2_voice": ("AUDIO", {
                    "tooltip": "说话人2的音色样本 / Speaker 2 voice sample (多人模式必需)"
                }),
                "speaker3_voice": ("AUDIO", {
                    "tooltip": "说话人3的音色样本 / Speaker 3 voice sample"
                }),
                "speaker4_voice": ("AUDIO", {
                    "tooltip": "说话人4的音色样本 / Speaker 4 voice sample"
                }),
                "model_manager": ("INDEXTTS2_MODEL",),
                "speaker1_emotion_voice": ("AUDIO", {
                    "tooltip": "说话人1的情绪语音样本 / Speaker 1 emotion voice sample"
                }),
                "speaker1_emotion_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "说话人1情绪强度 / Speaker 1 emotion intensity"
                }),
                "speaker2_emotion_voice": ("AUDIO", {
                    "tooltip": "说话人2的情绪语音样本 / Speaker 2 emotion voice sample"
                }),
                "speaker2_emotion_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "说话人2情绪强度 / Speaker 2 emotion intensity"
                }),
                "speaker3_emotion_voice": ("AUDIO", {
                    "tooltip": "说话人3的情绪语音样本 / Speaker 3 emotion voice sample"
                }),
                "speaker3_emotion_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "说话人3情绪强度 / Speaker 3 emotion intensity"
                }),
                "speaker4_emotion_voice": ("AUDIO", {
                    "tooltip": "说话人4的情绪语音样本 / Speaker 4 emotion voice sample"
                }),
                "speaker4_emotion_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "说话人4情绪强度 / Speaker 4 emotion intensity"
                }),
                "silence_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "说话间隔时间(秒) / Silence duration between speakers (seconds)"
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "语速控制 / Speech speed control"
                }),
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "使用FP16加速 / Use FP16 acceleration"
                }),
                "use_cuda_kernel": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用CUDA内核加速 / Use CUDA kernel acceleration"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "conversation_info")
    FUNCTION = "synthesize_emotion_voice_conversation"
    CATEGORY = "IndexTTS2/Advanced"
    DESCRIPTION = "1-4 speaker conversation synthesis with emotion voice control (1=voice cloning, 2-4=conversation)"

    def synthesize_emotion_voice_conversation(
        self,
        num_speakers,
        conversation_text,
        speaker1_voice,
        output_filename,
        speaker2_voice=None,
        speaker3_voice=None,
        speaker4_voice=None,
        model_manager=None,
        speaker1_emotion_voice=None,
        speaker1_emotion_alpha=1.0,
        speaker2_emotion_voice=None,
        speaker2_emotion_alpha=1.0,
        speaker3_emotion_voice=None,
        speaker3_emotion_alpha=1.0,
        speaker4_emotion_voice=None,
        speaker4_emotion_alpha=1.0,
        silence_duration=0.5,
        speed=1.0,
        use_fp16=True,
        use_cuda_kernel=False,
    ):
        """
        合成带情绪语音控制的多人对话
        Synthesize multi-speaker conversation with emotion voice control
        """
        try:
            # 加载模型
            model = self._load_default_model(use_fp16, use_cuda_kernel)

            # 检查是否为单人模式
            if int(num_speakers) == 1:
                # 单人模式：纯语音克隆
                return self._synthesize_single_speaker(
                    model, conversation_text, speaker1_voice, speaker1_emotion_voice,
                    speaker1_emotion_alpha, output_filename, speed
                )
            else:
                # 多人模式：原有逻辑
                # 验证多人模式必需的说话人音频
                if speaker2_voice is None:
                    raise ValueError("Speaker 2 voice is required for 2+ speakers conversation")

                # 解析对话文本
                conversation_lines = self._parse_conversation_text(conversation_text)

                # 配置说话人
                speakers_config = self._configure_speakers(
                    num_speakers,
                    speaker1_voice, speaker1_emotion_voice, speaker1_emotion_alpha,
                    speaker2_voice, speaker2_emotion_voice, speaker2_emotion_alpha,
                    speaker3_voice, speaker3_emotion_voice, speaker3_emotion_alpha,
                    speaker4_voice, speaker4_emotion_voice, speaker4_emotion_alpha
                )

                # 合成对话
                conversation_audio = self._synthesize_with_emotion_voice(
                    model, conversation_lines, speakers_config, silence_duration, speed
                )

                # 保存音频
                output_path = self._save_audio(conversation_audio, output_filename)

                # 生成对话信息
                conversation_info = self._generate_conversation_info(conversation_lines, speakers_config)

                # 确保音频格式符合 ComfyUI 标准 [batch, channels, samples]
                if conversation_audio.dim() == 2:
                    # [channels, samples] -> [1, channels, samples]
                    conversation_audio = conversation_audio.unsqueeze(0)
                elif conversation_audio.dim() == 1:
                    # [samples] -> [1, 1, samples]
                    conversation_audio = conversation_audio.unsqueeze(0).unsqueeze(0)

                print(f"[EmotionVoiceMultiTalk] 最终音频格式: {conversation_audio.shape}")
                print(f"[EmotionVoiceMultiTalk] ComfyUI AUDIO格式: batch={conversation_audio.shape[0]}, channels={conversation_audio.shape[1]}, samples={conversation_audio.shape[2]}")

                return ({"waveform": conversation_audio, "sample_rate": 24000}, conversation_info)
            
        except Exception as e:
            error_msg = f"IndexTTS2 emotion voice multi-talk synthesis failed: {str(e)}"
            print(f"[EmotionVoiceMultiTalk Error] {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

    def _synthesize_single_speaker(self, model, text, voice_audio, emotion_voice_audio, emotion_alpha, output_filename, speed):
        """
        单人模式合成
        Single speaker mode synthesis
        """
        try:
            # 使用与多人模式相同的合成方法
            speaker_config = {
                'voice_audio': voice_audio,
                'emotion_mode': 'emotion_voice' if emotion_voice_audio is not None else 'none',
                'emotion_voice_audio': emotion_voice_audio,
                'emotion_alpha': emotion_alpha
            }

            # 合成音频
            audio_tensor = self._synthesize_single_line(model, text, speaker_config, None, speed)

            if audio_tensor is None:
                raise RuntimeError("合成失败，返回空音频")

            # 保存音频
            output_path = self._save_audio(audio_tensor, output_filename)

            # 确保音频格式符合 ComfyUI 标准 [batch, channels, samples]
            if audio_tensor.dim() == 1:
                # [samples] -> [1, 1, samples]
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                # [channels, samples] -> [1, channels, samples]
                audio_tensor = audio_tensor.unsqueeze(0)

            print(f"[SingleSpeaker] 最终音频格式: {audio_tensor.shape}")

            # 生成信息
            info = f"单人语音克隆完成\n文本长度: {len(text)} 字符\n音频长度: {audio_tensor.shape[-1]/24000:.2f} 秒"
            if emotion_voice_audio is not None:
                info += f"\n情绪强度: {emotion_alpha}"

            return ({"waveform": audio_tensor, "sample_rate": 24000}, info)

        except Exception as e:
            error_msg = f"单人模式合成失败: {str(e)}"
            print(f"[SingleSpeaker Error] {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

    def _load_default_model(self, use_fp16=True, use_cuda_kernel=False):
        """加载默认模型"""
        try:
            from ..indextts.infer_v2 import IndexTTS2

            if self.model is None:
                print("🔄 Loading IndexTTS2 model for emotion voice multi-talk...")

                from indextts.infer_v2 import IndexTTS2

                # 使用通用模型路径函数
                from .model_utils import get_indextts2_model_path, validate_model_path

                model_dir, config_path = get_indextts2_model_path()

                print(f"[IndexTTS2] 使用模型路径: {model_dir}")
                print(f"[IndexTTS2] Using model path: {model_dir}")

                # 验证模型路径
                validate_model_path(model_dir, config_path)

                # 初始化模型
                self.model = IndexTTS2(
                    cfg_path=config_path,
                    model_dir=model_dir,
                    is_fp16=use_fp16,
                    use_cuda_kernel=use_cuda_kernel
                )

                print("✅ IndexTTS2 model loaded successfully for emotion voice multi-talk")

            return self.model

        except Exception as e:
            raise RuntimeError(f"Failed to load IndexTTS2 model: {str(e)}")

    def _parse_conversation_text(self, conversation_text):
        """解析对话文本，提取说话人、情绪和文本"""
        lines = []
        for line in conversation_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # 解析格式: Speaker1: [Emotion] Text 或 Speaker1: Text
            if ':' in line:
                speaker_part, text_part = line.split(':', 1)
                speaker = speaker_part.strip()
                text_part = text_part.strip()

                # 检查是否有情绪标记 [Emotion]
                emotion = None
                if text_part.startswith('[') and ']' in text_part:
                    end_bracket = text_part.find(']')
                    emotion = text_part[1:end_bracket].strip()
                    text = text_part[end_bracket + 1:].strip()
                else:
                    text = text_part

                lines.append({
                    'speaker': speaker,
                    'emotion': emotion,
                    'text': text
                })

        return lines

    def _configure_speakers(self, num_speakers, *speaker_configs):
        """配置说话人信息"""
        speakers = {}

        # 解析说话人配置参数 (voice, emotion_voice, emotion_alpha)
        config_groups = []
        for i in range(0, len(speaker_configs), 3):
            if i + 2 < len(speaker_configs):
                config_groups.append(speaker_configs[i:i+3])

        speaker_names = ["Speaker1", "Speaker2", "Speaker3", "Speaker4"]

        for i, (voice, emotion_voice, emotion_alpha) in enumerate(config_groups):
            if i >= int(num_speakers):
                break

            speaker_name = speaker_names[i]

            # 只有当voice不为None时才配置说话人
            if voice is not None:
                speakers[speaker_name] = {
                    'voice_audio': voice,
                    'emotion_mode': 'emotion_voice',  # 默认使用情绪语音模式
                    'emotion_voice_audio': emotion_voice,  # 现在是AUDIO对象
                    'emotion_alpha': emotion_alpha
                }

        return speakers

    def _synthesize_with_emotion_voice(self, model, conversation_lines, speakers_config, silence_duration, speed):
        """使用情绪语音合成对话"""
        audio_segments = []

        for line in conversation_lines:
            speaker = line['speaker']
            text = line['text']
            emotion = line['emotion']

            if speaker not in speakers_config:
                print(f"⚠️ Speaker {speaker} not configured, skipping...")
                continue

            speaker_config = speakers_config[speaker]

            # 合成单句话
            audio_segment = self._synthesize_single_line(
                model, text, speaker_config, emotion, speed
            )

            if audio_segment is not None:
                audio_segments.append(audio_segment)

                # 添加静音间隔
                if silence_duration > 0:
                    silence_samples = int(24000 * silence_duration)  # 24kHz采样率
                    silence = torch.zeros(silence_samples)
                    audio_segments.append(silence)

        # 合并所有音频段
        if audio_segments:
            # 移除最后的静音
            if len(audio_segments) > 1 and silence_duration > 0:
                audio_segments = audio_segments[:-1]

            # 确保所有音频段都是1D张量
            normalized_segments = []
            for segment in audio_segments:
                if segment.dim() > 1:
                    segment = segment.squeeze()  # 移除多余维度
                if segment.dim() == 0:
                    segment = segment.unsqueeze(0)  # 确保至少是1D
                normalized_segments.append(segment)

            conversation_audio = torch.cat(normalized_segments, dim=0)

            # 确保返回的是2D张量 [channels, samples]
            if conversation_audio.dim() == 1:
                conversation_audio = conversation_audio.unsqueeze(0)  # [samples] -> [1, samples]

            return conversation_audio
        else:
            # 返回空音频 [1, samples]
            return torch.zeros(1, 1000)

    def _synthesize_single_line(self, model, text, speaker_config, emotion, speed):
        """合成单句话"""
        try:
            # 获取说话人音频
            voice_audio = speaker_config['voice_audio']
            emotion_mode = speaker_config['emotion_mode']
            emotion_voice_audio = speaker_config['emotion_voice_audio']
            emotion_alpha = speaker_config['emotion_alpha']

            # 准备说话人音频文件
            import tempfile
            import torchaudio

            # 保存说话人音频到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_speaker_file:
                speaker_audio_path = temp_speaker_file.name

                # 确保音频张量是2D格式 [channels, samples]
                speaker_waveform = voice_audio['waveform']
                if speaker_waveform.dim() == 3:
                    speaker_waveform = speaker_waveform.squeeze(0)  # 移除batch维度
                elif speaker_waveform.dim() == 1:
                    speaker_waveform = speaker_waveform.unsqueeze(0)  # 添加channel维度

                torchaudio.save(
                    speaker_audio_path,
                    speaker_waveform,
                    voice_audio['sample_rate']
                )

            # 创建输出临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output_file:
                output_path = temp_output_file.name

            try:
                # 根据情绪模式调用不同的合成方法
                if emotion_mode == "emotion_voice" and emotion_voice_audio is not None:
                    # 保存情绪语音到临时文件
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_emotion_file:
                        emotion_audio_path = temp_emotion_file.name

                        # 确保情绪音频张量是2D格式 [channels, samples]
                        emotion_waveform = emotion_voice_audio['waveform']
                        if emotion_waveform.dim() == 3:
                            emotion_waveform = emotion_waveform.squeeze(0)  # 移除batch维度
                        elif emotion_waveform.dim() == 1:
                            emotion_waveform = emotion_waveform.unsqueeze(0)  # 添加channel维度

                        torchaudio.save(
                            emotion_audio_path,
                            emotion_waveform,
                            emotion_voice_audio['sample_rate']
                        )

                    try:
                        # 使用情绪语音控制
                        model.infer(
                            spk_audio_prompt=speaker_audio_path,
                            text=text,
                            output_path=output_path,
                            emo_audio_prompt=emotion_audio_path,
                            emo_alpha=emotion_alpha,
                            verbose=False,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=50,
                            max_text_tokens_per_sentence=120,
                            interval_silence=200
                        )
                    finally:
                        # 清理情绪音频临时文件
                        try:
                            os.unlink(emotion_audio_path)
                        except:
                            pass
                else:
                    # 基础合成（无情绪控制）
                    model.infer(
                        spk_audio_prompt=speaker_audio_path,
                        text=text,
                        output_path=output_path,
                        verbose=False,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        max_text_tokens_per_sentence=120,
                        interval_silence=200
                    )

                # 加载生成的音频
                audio_data, sample_rate = torchaudio.load(output_path)

                # 返回音频tensor
                return audio_data.squeeze()

            finally:
                # 清理临时文件
                try:
                    os.unlink(speaker_audio_path)
                    os.unlink(output_path)
                except:
                    pass

        except Exception as e:
            print(f"❌ Failed to synthesize line: {text}, error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

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
        print(f"💾 Emotion voice conversation saved to: {output_path}")

        return output_path

    def _generate_conversation_info(self, conversation_lines, speakers_config):
        """生成对话信息"""
        info_lines = []
        info_lines.append("🎭 Emotion Voice Multi-Talk Conversation Info")
        info_lines.append("=" * 50)

        # 说话人配置信息
        info_lines.append("\n👥 Speakers Configuration:")
        for speaker, config in speakers_config.items():
            info_lines.append(f"  {speaker}:")
            info_lines.append(f"    - Emotion Mode: {config['emotion_mode']}")
            info_lines.append(f"    - Emotion Alpha: {config['emotion_alpha']}")
            if config.get('emotion_voice_audio') is not None:
                info_lines.append(f"    - Emotion Voice: Connected")
            else:
                info_lines.append(f"    - Emotion Voice: None")

        # 对话内容信息
        info_lines.append(f"\n💬 Conversation Lines: {len(conversation_lines)}")
        for i, line in enumerate(conversation_lines, 1):
            emotion_info = f" [{line['emotion']}]" if line['emotion'] else ""
            info_lines.append(f"  {i}. {line['speaker']}{emotion_info}: {line['text'][:50]}...")

        return "\n".join(info_lines)
