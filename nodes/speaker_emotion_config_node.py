# IndexTTS2 Speaker Emotion Config Node
# IndexTTS2 说话人情感配置节点

import os
from typing import Optional, Tuple, Any, List, Dict

class IndexTTS2SpeakerEmotionConfigNode:
    """
    IndexTTS2 说话人情感配置节点
    Speaker emotion configuration node for IndexTTS2
    
    Features:
    - Individual speaker emotion configuration
    - Multiple emotion control modes
    - 8-dimensional emotion vector control
    - Audio and text emotion prompts
    - Reusable emotion configurations
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "speaker_name": ("STRING", {
                    "default": "Speaker1",
                    "placeholder": "说话人名称 / Speaker name"
                }),
                "emotion_mode": (["inherit", "audio_prompt", "emotion_vector", "text_description", "auto"], {
                    "default": "emotion_vector",
                    "tooltip": "情感控制模式 / Emotion control mode"
                }),
            },
            "optional": {
                "emotion_audio": ("STRING", {
                    "default": "",
                    "placeholder": "情感音频文件路径 / Emotion audio file path"
                }),
                "emotion_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "情感强度 / Emotion intensity"
                }),
                # 8-dimensional emotion vector: Happy, Angry, Sad, Fear, Hate, Low, Surprise, Neutral
                "happy": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "快乐情感 / Happy emotion"
                }),
                "angry": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "愤怒情感 / Angry emotion"
                }),
                "sad": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "悲伤情感 / Sad emotion"
                }),
                "fear": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "恐惧情感 / Fear emotion"
                }),
                "hate": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "厌恶情感 / Hate emotion"
                }),
                "low": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "低沉情感 / Low emotion"
                }),
                "surprise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "惊讶情感 / Surprise emotion"
                }),
                "neutral": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "中性情感 / Neutral emotion"
                }),
                "emotion_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "情感文本描述 / Emotion text description"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用此情感配置 / Enable this emotion config"
                }),
            }
        }
    
    RETURN_TYPES = ("SPEAKER_EMOTION_CONFIG", "STRING")
    RETURN_NAMES = ("emotion_config", "info")
    FUNCTION = "create_emotion_config"
    CATEGORY = "IndexTTS2/Config"
    DESCRIPTION = "Create emotion configuration for individual speakers in multi-talk synthesis"
    
    def create_emotion_config(
        self,
        speaker_name: str,
        emotion_mode: str,
        emotion_audio: str = "",
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
        enabled: bool = True
    ) -> Tuple[dict, str]:
        """
        创建说话人情感配置
        Create speaker emotion configuration
        """
        try:
            # 创建情感向量
            emotion_vector = [happy, angry, sad, fear, hate, low, surprise, neutral]
            
            # 检查情感向量是否全为零
            max_emotion_value = max(emotion_vector)
            if max_emotion_value == 0.0 and emotion_mode == "emotion_vector":
                # 当所有情感值都为0时，设置一个小的中性情感值
                emotion_vector[7] = 0.1  # 设置Neutral为0.1
            
            # 创建情感配置字典
            emotion_config = {
                "speaker_name": speaker_name.strip() or "Speaker1",
                "mode": emotion_mode,
                "audio": emotion_audio.strip(),
                "alpha": emotion_alpha,
                "vector": emotion_vector,
                "text": emotion_text.strip(),
                "enabled": enabled
            }
            
            # 生成信息字符串
            info = self._generate_info(emotion_config)
            
            return (emotion_config, info)
            
        except Exception as e:
            error_msg = f"Failed to create emotion config: {str(e)}"
            print(f"[SpeakerEmotionConfig Error] {error_msg}")
            
            # 返回默认配置
            default_config = {
                "speaker_name": speaker_name or "Speaker1",
                "mode": "inherit",
                "audio": "",
                "alpha": 1.0,
                "vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "text": "",
                "enabled": False
            }
            
            return (default_config, f"Error: {error_msg}")
    
    def _generate_info(self, emotion_config: dict) -> str:
        """生成情感配置信息字符串"""
        info_lines = [
            f"=== {emotion_config['speaker_name']} Emotion Config ===",
            f"Mode: {emotion_config['mode']}",
            f"Enabled: {'Yes' if emotion_config['enabled'] else 'No'}",
        ]
        
        if not emotion_config['enabled']:
            info_lines.append("Status: Disabled - will use default/inherit settings")
            return "\n".join(info_lines)
        
        mode = emotion_config['mode']
        
        if mode == "audio_prompt":
            audio = emotion_config['audio']
            alpha = emotion_config['alpha']
            if audio:
                info_lines.append(f"Audio: {os.path.basename(audio)}")
                info_lines.append(f"Alpha: {alpha}")
            else:
                info_lines.append("Audio: Not specified")
                
        elif mode == "emotion_vector":
            vector = emotion_config['vector']
            emotion_names = ["Happy", "Angry", "Sad", "Fear", "Hate", "Low", "Surprise", "Neutral"]
            active_emotions = []
            
            for name, value in zip(emotion_names, vector):
                if value > 0.05:  # 只显示有意义的情感值
                    active_emotions.append(f"{name}: {value:.2f}")
            
            if active_emotions:
                info_lines.append("Emotions:")
                for emotion in active_emotions:
                    info_lines.append(f"  {emotion}")
            else:
                info_lines.append("Emotions: All neutral")
                
        elif mode == "text_description":
            text = emotion_config['text']
            if text:
                preview = text[:50] + "..." if len(text) > 50 else text
                info_lines.append(f"Description: {preview}")
            else:
                info_lines.append("Description: Will infer from synthesis text")
                
        elif mode == "inherit":
            info_lines.append("Will inherit global emotion settings")
            
        elif mode == "auto":
            info_lines.append("Will use automatic emotion detection")
        
        return "\n".join(info_lines)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """检查输入是否改变"""
        return float("nan")  # 总是重新执行
