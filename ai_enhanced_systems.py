#!/usr/bin/env python3
"""
IndexTTS2 AI增强系统
AI Enhanced Systems for IndexTTS2

包含智能参数学习、自适应优化、用户偏好学习等高级功能
"""

import torch
import numpy as np
import json
import os
import time
import pickle
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import threading
from datetime import datetime, timedelta

@dataclass
class SpeakerProfile:
    """说话人档案"""
    speaker_id: str
    embedding_history: List[torch.Tensor]
    quality_scores: List[float]
    preferred_params: Dict[str, float]
    usage_count: int
    last_updated: datetime
    voice_characteristics: Dict[str, float]
    optimal_settings: Dict[str, Any]

@dataclass
class UserPreference:
    """用户偏好"""
    user_id: str
    quality_preference: float  # 0.0-1.0, 质量vs速度偏好
    style_preference: Dict[str, float]  # 说话风格偏好
    pause_preference: Dict[str, float]  # 停顿偏好
    enhancement_preference: Dict[str, bool]  # 增强功能偏好
    feedback_history: List[Dict[str, Any]]
    learning_rate: float

class IntelligentParameterLearner:
    """智能参数学习系统"""
    
    def __init__(self, data_dir: str = "ai_learning_data"):
        self.data_dir = data_dir
        self.ensure_data_dir()
        
        # 说话人档案管理
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}
        self.load_speaker_profiles()
        
        # 用户偏好管理
        self.user_preferences: Dict[str, UserPreference] = {}
        self.load_user_preferences()
        
        # 学习历史
        self.learning_history = deque(maxlen=10000)
        self.parameter_trends = defaultdict(list)
        
        # 学习配置
        self.learning_config = {
            'min_samples_for_learning': 5,
            'learning_rate': 0.1,
            'decay_factor': 0.95,
            'quality_weight': 0.7,
            'consistency_weight': 0.3,
            'update_frequency': 10  # 每10次使用更新一次
        }
        
        # 统计信息
        self.stats = {
            'total_learning_sessions': 0,
            'successful_optimizations': 0,
            'parameter_updates': 0,
            'quality_improvements': 0
        }
        
        # 线程安全
        self.lock = threading.Lock()
        
        print("[IntelligentParameterLearner] 智能参数学习系统初始化完成")
    
    def ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def load_speaker_profiles(self):
        """加载说话人档案"""
        profiles_file = os.path.join(self.data_dir, "speaker_profiles.pkl")
        if os.path.exists(profiles_file):
            try:
                with open(profiles_file, 'rb') as f:
                    self.speaker_profiles = pickle.load(f)
                print(f"[ParameterLearner] 加载了 {len(self.speaker_profiles)} 个说话人档案")
            except Exception as e:
                print(f"[ParameterLearner] 加载说话人档案失败: {e}")
                self.speaker_profiles = {}
    
    def save_speaker_profiles(self):
        """保存说话人档案"""
        profiles_file = os.path.join(self.data_dir, "speaker_profiles.pkl")
        try:
            with open(profiles_file, 'wb') as f:
                pickle.dump(self.speaker_profiles, f)
        except Exception as e:
            print(f"[ParameterLearner] 保存说话人档案失败: {e}")
    
    def load_user_preferences(self):
        """加载用户偏好"""
        prefs_file = os.path.join(self.data_dir, "user_preferences.json")
        if os.path.exists(prefs_file):
            try:
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id, pref_data in data.items():
                        # 转换datetime字符串
                        if 'feedback_history' in pref_data:
                            for feedback in pref_data['feedback_history']:
                                if 'timestamp' in feedback:
                                    feedback['timestamp'] = datetime.fromisoformat(feedback['timestamp'])
                        
                        self.user_preferences[user_id] = UserPreference(**pref_data)
                print(f"[ParameterLearner] 加载了 {len(self.user_preferences)} 个用户偏好")
            except Exception as e:
                print(f"[ParameterLearner] 加载用户偏好失败: {e}")
                self.user_preferences = {}
    
    def save_user_preferences(self):
        """保存用户偏好"""
        prefs_file = os.path.join(self.data_dir, "user_preferences.json")
        try:
            data = {}
            for user_id, pref in self.user_preferences.items():
                pref_dict = asdict(pref)
                # 转换datetime为字符串
                if 'feedback_history' in pref_dict:
                    for feedback in pref_dict['feedback_history']:
                        if 'timestamp' in feedback and isinstance(feedback['timestamp'], datetime):
                            feedback['timestamp'] = feedback['timestamp'].isoformat()
                data[user_id] = pref_dict
            
            with open(prefs_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ParameterLearner] 保存用户偏好失败: {e}")
    
    def record_synthesis_session(self, speaker_id: str, embedding: torch.Tensor, 
                                params: Dict[str, Any], quality_score: float,
                                user_feedback: Optional[Dict[str, Any]] = None):
        """记录合成会话数据"""
        with self.lock:
            try:
                # 更新说话人档案
                if speaker_id not in self.speaker_profiles:
                    self.speaker_profiles[speaker_id] = SpeakerProfile(
                        speaker_id=speaker_id,
                        embedding_history=[],
                        quality_scores=[],
                        preferred_params={},
                        usage_count=0,
                        last_updated=datetime.now(),
                        voice_characteristics={},
                        optimal_settings={}
                    )
                
                profile = self.speaker_profiles[speaker_id]
                profile.embedding_history.append(embedding.detach().cpu())
                profile.quality_scores.append(quality_score)
                profile.usage_count += 1
                profile.last_updated = datetime.now()
                
                # 限制历史记录长度
                if len(profile.embedding_history) > 50:
                    profile.embedding_history = profile.embedding_history[-50:]
                    profile.quality_scores = profile.quality_scores[-50:]
                
                # 记录学习历史
                session_data = {
                    'timestamp': datetime.now(),
                    'speaker_id': speaker_id,
                    'params': params.copy(),
                    'quality_score': quality_score,
                    'user_feedback': user_feedback
                }
                self.learning_history.append(session_data)
                
                # 定期学习和优化
                if profile.usage_count % self.learning_config['update_frequency'] == 0:
                    self._learn_speaker_preferences(speaker_id)
                
                self.stats['total_learning_sessions'] += 1
                
            except Exception as e:
                print(f"[ParameterLearner] 记录会话数据失败: {e}")
    
    def _learn_speaker_preferences(self, speaker_id: str):
        """学习说话人偏好"""
        try:
            profile = self.speaker_profiles[speaker_id]
            
            if len(profile.quality_scores) < self.learning_config['min_samples_for_learning']:
                return
            
            # 分析质量趋势
            recent_scores = profile.quality_scores[-20:]  # 最近20次
            avg_quality = np.mean(recent_scores)
            quality_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            # 分析嵌入特征
            if len(profile.embedding_history) >= 3:
                embeddings = torch.stack(profile.embedding_history[-10:])  # 最近10个嵌入
                
                # 计算嵌入特征统计
                mean_embedding = torch.mean(embeddings, dim=0)
                std_embedding = torch.std(embeddings, dim=0)
                
                # 更新声音特征
                profile.voice_characteristics = {
                    'embedding_stability': float(torch.mean(std_embedding)),
                    'average_quality': float(avg_quality),
                    'quality_trend': float(quality_trend),
                    'consistency_score': float(1.0 / (1.0 + torch.mean(std_embedding)))
                }
            
            # 学习最优参数
            self._optimize_speaker_parameters(speaker_id)
            
            self.stats['successful_optimizations'] += 1
            
        except Exception as e:
            print(f"[ParameterLearner] 学习说话人偏好失败: {e}")
    
    def _optimize_speaker_parameters(self, speaker_id: str):
        """优化说话人参数"""
        try:
            profile = self.speaker_profiles[speaker_id]
            
            # 分析历史会话数据
            speaker_sessions = [
                session for session in self.learning_history 
                if session['speaker_id'] == speaker_id
            ]
            
            if len(speaker_sessions) < self.learning_config['min_samples_for_learning']:
                return
            
            # 找到高质量会话的参数
            high_quality_sessions = [
                session for session in speaker_sessions[-20:]  # 最近20次
                if session['quality_score'] > np.percentile([s['quality_score'] for s in speaker_sessions], 75)
            ]
            
            if not high_quality_sessions:
                return
            
            # 计算最优参数
            optimal_params = {}
            param_keys = set()
            for session in high_quality_sessions:
                param_keys.update(session['params'].keys())
            
            for param_key in param_keys:
                values = [
                    session['params'].get(param_key, 0) 
                    for session in high_quality_sessions 
                    if param_key in session['params']
                ]
                if values:
                    if isinstance(values[0], (int, float)):
                        optimal_params[param_key] = float(np.mean(values))
                    else:
                        # 对于非数值参数，选择最常见的值
                        from collections import Counter
                        optimal_params[param_key] = Counter(values).most_common(1)[0][0]
            
            # 更新档案
            profile.preferred_params = optimal_params
            profile.optimal_settings = {
                'voice_consistency': profile.voice_characteristics.get('consistency_score', 0.8),
                'quality_target': profile.voice_characteristics.get('average_quality', 0.7),
                'stability_preference': min(1.0, profile.voice_characteristics.get('embedding_stability', 0.1) * 10)
            }
            
            self.stats['parameter_updates'] += 1
            
            print(f"[ParameterLearner] 为说话人 {speaker_id} 优化了参数")
            
        except Exception as e:
            print(f"[ParameterLearner] 优化说话人参数失败: {e}")
    
    def get_recommended_parameters(self, speaker_id: str, 
                                 current_params: Dict[str, Any]) -> Dict[str, Any]:
        """获取推荐参数"""
        try:
            if speaker_id not in self.speaker_profiles:
                return current_params
            
            profile = self.speaker_profiles[speaker_id]
            
            if not profile.preferred_params:
                return current_params
            
            # 合并推荐参数
            recommended_params = current_params.copy()
            
            # 应用学习到的偏好参数
            for param_key, param_value in profile.preferred_params.items():
                if param_key in recommended_params:
                    # 使用学习率平滑过渡
                    current_value = recommended_params[param_key]
                    if isinstance(current_value, (int, float)) and isinstance(param_value, (int, float)):
                        learning_rate = self.learning_config['learning_rate']
                        recommended_params[param_key] = (
                            current_value * (1 - learning_rate) + 
                            param_value * learning_rate
                        )
                    else:
                        recommended_params[param_key] = param_value
            
            # 应用最优设置
            if profile.optimal_settings:
                if 'voice_consistency' in profile.optimal_settings:
                    recommended_params['voice_consistency'] = profile.optimal_settings['voice_consistency']
            
            return recommended_params
            
        except Exception as e:
            print(f"[ParameterLearner] 获取推荐参数失败: {e}")
            return current_params
    
    def record_user_feedback(self, user_id: str, session_data: Dict[str, Any], 
                           feedback: Dict[str, Any]):
        """记录用户反馈"""
        with self.lock:
            try:
                if user_id not in self.user_preferences:
                    self.user_preferences[user_id] = UserPreference(
                        user_id=user_id,
                        quality_preference=0.7,
                        style_preference={},
                        pause_preference={},
                        enhancement_preference={},
                        feedback_history=[],
                        learning_rate=0.1
                    )
                
                user_pref = self.user_preferences[user_id]
                
                # 记录反馈
                feedback_entry = {
                    'timestamp': datetime.now(),
                    'session_data': session_data,
                    'feedback': feedback
                }
                user_pref.feedback_history.append(feedback_entry)
                
                # 限制历史长度
                if len(user_pref.feedback_history) > 100:
                    user_pref.feedback_history = user_pref.feedback_history[-100:]
                
                # 学习用户偏好
                self._learn_user_preferences(user_id)
                
            except Exception as e:
                print(f"[ParameterLearner] 记录用户反馈失败: {e}")
    
    def _learn_user_preferences(self, user_id: str):
        """学习用户偏好"""
        try:
            user_pref = self.user_preferences[user_id]
            
            if len(user_pref.feedback_history) < 3:
                return
            
            # 分析反馈模式
            recent_feedback = user_pref.feedback_history[-10:]  # 最近10次反馈
            
            # 学习质量偏好
            quality_ratings = [
                fb['feedback'].get('quality_rating', 0.5) 
                for fb in recent_feedback 
                if 'quality_rating' in fb['feedback']
            ]
            if quality_ratings:
                avg_quality_pref = np.mean(quality_ratings)
                user_pref.quality_preference = (
                    user_pref.quality_preference * 0.8 + avg_quality_pref * 0.2
                )
            
            # 学习风格偏好
            for fb in recent_feedback:
                if 'style_preference' in fb['feedback']:
                    for style, rating in fb['feedback']['style_preference'].items():
                        if style not in user_pref.style_preference:
                            user_pref.style_preference[style] = rating
                        else:
                            user_pref.style_preference[style] = (
                                user_pref.style_preference[style] * 0.9 + rating * 0.1
                            )
            
            print(f"[ParameterLearner] 更新了用户 {user_id} 的偏好")
            
        except Exception as e:
            print(f"[ParameterLearner] 学习用户偏好失败: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        return {
            'stats': self.stats.copy(),
            'speaker_count': len(self.speaker_profiles),
            'user_count': len(self.user_preferences),
            'learning_sessions': len(self.learning_history),
            'top_speakers': [
                {
                    'speaker_id': speaker_id,
                    'usage_count': profile.usage_count,
                    'avg_quality': np.mean(profile.quality_scores) if profile.quality_scores else 0,
                    'last_updated': profile.last_updated.isoformat()
                }
                for speaker_id, profile in sorted(
                    self.speaker_profiles.items(), 
                    key=lambda x: x[1].usage_count, 
                    reverse=True
                )[:5]
            ]
        }
    
    def save_all_data(self):
        """保存所有学习数据"""
        try:
            self.save_speaker_profiles()
            self.save_user_preferences()
            
            # 保存统计信息
            stats_file = os.path.join(self.data_dir, "learning_stats.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.get_learning_stats(), f, ensure_ascii=False, indent=2, default=str)
            
            print("[ParameterLearner] 所有学习数据已保存")
            
        except Exception as e:
            print(f"[ParameterLearner] 保存学习数据失败: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """清理旧数据"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # 清理说话人档案中的旧数据
            for profile in self.speaker_profiles.values():
                if profile.last_updated < cutoff_date:
                    # 保留最近的一些数据
                    if len(profile.embedding_history) > 10:
                        profile.embedding_history = profile.embedding_history[-10:]
                        profile.quality_scores = profile.quality_scores[-10:]
            
            # 清理用户偏好中的旧反馈
            for user_pref in self.user_preferences.values():
                user_pref.feedback_history = [
                    fb for fb in user_pref.feedback_history
                    if fb['timestamp'] > cutoff_date
                ]
            
            # 清理学习历史
            self.learning_history = deque([
                session for session in self.learning_history
                if session['timestamp'] > cutoff_date
            ], maxlen=10000)
            
            print(f"[ParameterLearner] 清理了 {days_to_keep} 天前的旧数据")
            
        except Exception as e:
            print(f"[ParameterLearner] 清理旧数据失败: {e}")


class AdaptiveAudioEnhancer:
    """自适应音频增强系统"""

    def __init__(self, parameter_learner: IntelligentParameterLearner):
        self.parameter_learner = parameter_learner

        # 情感分析模型（简化版）
        self.emotion_keywords = {
            'happy': ['开心', '高兴', '快乐', '兴奋', '愉快', '欢乐'],
            'sad': ['伤心', '难过', '悲伤', '沮丧', '失落', '痛苦'],
            'angry': ['生气', '愤怒', '恼火', '气愤', '暴怒', '愤慨'],
            'calm': ['平静', '冷静', '安静', '宁静', '淡定', '从容'],
            'excited': ['激动', '兴奋', '热情', '狂热', '亢奋', '振奋']
        }

        # 内容类型检测
        self.content_patterns = {
            'news': ['新闻', '报道', '消息', '通知', '公告'],
            'story': ['故事', '小说', '传说', '童话', '寓言'],
            'dialogue': ['对话', '聊天', '交谈', '讨论', '会话'],
            'presentation': ['演讲', '报告', '介绍', '展示', '讲解'],
            'casual': ['随便', '聊天', '闲聊', '日常', '普通']
        }

        # 增强策略
        self.enhancement_strategies = {
            'emotion_based': {
                'happy': {'voice_consistency': 0.7, 'energy_boost': 1.1},
                'sad': {'voice_consistency': 0.9, 'energy_boost': 0.8},
                'angry': {'voice_consistency': 0.6, 'energy_boost': 1.3},
                'calm': {'voice_consistency': 0.95, 'energy_boost': 0.9},
                'excited': {'voice_consistency': 0.5, 'energy_boost': 1.4}
            },
            'content_based': {
                'news': {'clarity_boost': 1.2, 'pace_adjustment': 0.9},
                'story': {'expressiveness': 1.3, 'pace_variation': 1.1},
                'dialogue': {'naturalness': 1.2, 'pause_optimization': 1.1},
                'presentation': {'authority': 1.1, 'clarity_boost': 1.3},
                'casual': {'naturalness': 1.3, 'relaxation': 1.1}
            }
        }

        # 统计信息
        self.enhancement_stats = {
            'total_enhancements': 0,
            'emotion_detections': defaultdict(int),
            'content_detections': defaultdict(int),
            'enhancement_effectiveness': defaultdict(list)
        }

        print("[AdaptiveAudioEnhancer] 自适应音频增强系统初始化完成")

    def analyze_text_emotion(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        try:
            text_lower = text.lower()
            emotion_scores = {}

            for emotion, keywords in self.emotion_keywords.items():
                score = 0
                for keyword in keywords:
                    score += text_lower.count(keyword)

                # 标准化分数
                emotion_scores[emotion] = min(1.0, score / len(text.split()) * 10)

            # 如果没有明显情感，设为中性
            if all(score < 0.1 for score in emotion_scores.values()):
                emotion_scores['calm'] = 0.5

            return emotion_scores

        except Exception as e:
            print(f"[AdaptiveEnhancer] 情感分析失败: {e}")
            return {'calm': 0.5}

    def detect_content_type(self, text: str) -> Dict[str, float]:
        """检测内容类型"""
        try:
            text_lower = text.lower()
            content_scores = {}

            for content_type, patterns in self.content_patterns.items():
                score = 0
                for pattern in patterns:
                    score += text_lower.count(pattern)

                content_scores[content_type] = min(1.0, score / len(text.split()) * 5)

            # 如果没有明显类型，设为casual
            if all(score < 0.1 for score in content_scores.values()):
                content_scores['casual'] = 0.5

            return content_scores

        except Exception as e:
            print(f"[AdaptiveEnhancer] 内容类型检测失败: {e}")
            return {'casual': 0.5}

    def generate_enhancement_parameters(self, text: str, speaker_id: str,
                                      base_params: Dict[str, Any]) -> Dict[str, Any]:
        """生成增强参数"""
        try:
            # 分析文本
            emotion_scores = self.analyze_text_emotion(text)
            content_scores = self.detect_content_type(text)

            # 获取主要情感和内容类型
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            primary_content = max(content_scores.items(), key=lambda x: x[1])[0]

            # 记录检测结果
            self.enhancement_stats['emotion_detections'][primary_emotion] += 1
            self.enhancement_stats['content_detections'][primary_content] += 1

            # 生成增强参数
            enhanced_params = base_params.copy()

            # 应用情感增强
            if primary_emotion in self.enhancement_strategies['emotion_based']:
                emotion_strategy = self.enhancement_strategies['emotion_based'][primary_emotion]
                for param, multiplier in emotion_strategy.items():
                    if param == 'voice_consistency' and param in enhanced_params:
                        enhanced_params[param] = min(1.0, enhanced_params[param] * multiplier)
                    elif param == 'energy_boost':
                        enhanced_params['energy_level'] = enhanced_params.get('energy_level', 1.0) * multiplier

            # 应用内容类型增强
            if primary_content in self.enhancement_strategies['content_based']:
                content_strategy = self.enhancement_strategies['content_based'][primary_content]
                for param, multiplier in content_strategy.items():
                    if param == 'clarity_boost':
                        enhanced_params['clarity_factor'] = enhanced_params.get('clarity_factor', 1.0) * multiplier
                    elif param == 'pace_adjustment':
                        enhanced_params['pace_factor'] = enhanced_params.get('pace_factor', 1.0) * multiplier
                    elif param == 'expressiveness':
                        enhanced_params['expression_level'] = enhanced_params.get('expression_level', 1.0) * multiplier
                    elif param == 'naturalness':
                        enhanced_params['naturalness_factor'] = enhanced_params.get('naturalness_factor', 1.0) * multiplier

            # 获取说话人特定的增强
            if speaker_id in self.parameter_learner.speaker_profiles:
                profile = self.parameter_learner.speaker_profiles[speaker_id]
                if profile.voice_characteristics:
                    # 根据说话人特征调整
                    consistency_score = profile.voice_characteristics.get('consistency_score', 0.8)
                    if consistency_score < 0.7:  # 一致性较低的说话人
                        enhanced_params['voice_consistency'] = min(1.0, enhanced_params.get('voice_consistency', 0.8) * 1.1)

            # 添加增强元数据
            enhanced_params['enhancement_metadata'] = {
                'primary_emotion': primary_emotion,
                'emotion_confidence': emotion_scores[primary_emotion],
                'primary_content': primary_content,
                'content_confidence': content_scores[primary_content],
                'enhancement_applied': True
            }

            self.enhancement_stats['total_enhancements'] += 1

            return enhanced_params

        except Exception as e:
            print(f"[AdaptiveEnhancer] 生成增强参数失败: {e}")
            return base_params

    def evaluate_enhancement_effectiveness(self, original_quality: float,
                                         enhanced_quality: float,
                                         enhancement_metadata: Dict[str, Any]):
        """评估增强效果"""
        try:
            improvement = enhanced_quality - original_quality

            if 'primary_emotion' in enhancement_metadata:
                emotion = enhancement_metadata['primary_emotion']
                self.enhancement_stats['enhancement_effectiveness'][emotion].append(improvement)

            # 限制历史长度
            for emotion in self.enhancement_stats['enhancement_effectiveness']:
                if len(self.enhancement_stats['enhancement_effectiveness'][emotion]) > 100:
                    self.enhancement_stats['enhancement_effectiveness'][emotion] = \
                        self.enhancement_stats['enhancement_effectiveness'][emotion][-100:]

        except Exception as e:
            print(f"[AdaptiveEnhancer] 评估增强效果失败: {e}")

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """获取增强统计信息"""
        try:
            # 计算平均改进效果
            avg_improvements = {}
            for emotion, improvements in self.enhancement_stats['enhancement_effectiveness'].items():
                if improvements:
                    avg_improvements[emotion] = {
                        'avg_improvement': float(np.mean(improvements)),
                        'success_rate': float(np.mean([i > 0 for i in improvements])),
                        'sample_count': len(improvements)
                    }

            return {
                'total_enhancements': self.enhancement_stats['total_enhancements'],
                'emotion_distribution': dict(self.enhancement_stats['emotion_detections']),
                'content_distribution': dict(self.enhancement_stats['content_detections']),
                'effectiveness_by_emotion': avg_improvements
            }

        except Exception as e:
            print(f"[AdaptiveEnhancer] 获取增强统计失败: {e}")
            return {}


class IntelligentQualityPredictor:
    """智能质量预测系统"""

    def __init__(self, parameter_learner: IntelligentParameterLearner):
        self.parameter_learner = parameter_learner

        # 质量预测模型（简化的机器学习模型）
        self.quality_features = [
            'text_length', 'speaker_consistency', 'parameter_complexity',
            'historical_quality', 'embedding_stability', 'content_difficulty'
        ]

        # 预测历史
        self.prediction_history = deque(maxlen=1000)
        self.prediction_accuracy = deque(maxlen=100)

        # 质量阈值
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5,
            'poor': 0.3
        }

        # 预测模型权重（简化的线性模型）
        self.model_weights = {
            'text_length': -0.1,      # 文本越长，质量可能下降
            'speaker_consistency': 0.3,  # 说话人一致性越高，质量越好
            'parameter_complexity': -0.05,  # 参数越复杂，质量可能不稳定
            'historical_quality': 0.4,   # 历史质量是重要指标
            'embedding_stability': 0.2,  # 嵌入稳定性影响质量
            'content_difficulty': -0.15  # 内容难度影响质量
        }

        # 统计信息
        self.prediction_stats = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'quality_improvements_suggested': 0,
            'preventive_adjustments': 0
        }

        print("[IntelligentQualityPredictor] 智能质量预测系统初始化完成")

    def extract_quality_features(self, text: str, speaker_id: str,
                                params: Dict[str, Any]) -> Dict[str, float]:
        """提取质量预测特征"""
        try:
            features = {}

            # 文本长度特征
            features['text_length'] = min(1.0, len(text) / 1000)  # 标准化到0-1

            # 说话人一致性特征
            if speaker_id in self.parameter_learner.speaker_profiles:
                profile = self.parameter_learner.speaker_profiles[speaker_id]
                features['speaker_consistency'] = profile.voice_characteristics.get('consistency_score', 0.5)
                features['historical_quality'] = profile.voice_characteristics.get('average_quality', 0.5)
                features['embedding_stability'] = 1.0 - min(1.0, profile.voice_characteristics.get('embedding_stability', 0.1))
            else:
                features['speaker_consistency'] = 0.5
                features['historical_quality'] = 0.5
                features['embedding_stability'] = 0.5

            # 参数复杂度特征
            param_count = len([v for v in params.values() if isinstance(v, (int, float))])
            features['parameter_complexity'] = min(1.0, param_count / 20)

            # 内容难度特征（基于文本复杂度）
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            features['content_difficulty'] = min(1.0, avg_word_length / 10)

            return features

        except Exception as e:
            print(f"[QualityPredictor] 提取特征失败: {e}")
            return {feature: 0.5 for feature in self.quality_features}

    def predict_quality(self, text: str, speaker_id: str,
                       params: Dict[str, Any]) -> Dict[str, Any]:
        """预测音频质量"""
        try:
            # 提取特征
            features = self.extract_quality_features(text, speaker_id, params)

            # 简化的线性预测模型
            predicted_score = 0.5  # 基础分数

            for feature_name, feature_value in features.items():
                if feature_name in self.model_weights:
                    predicted_score += self.model_weights[feature_name] * feature_value

            # 限制预测分数范围
            predicted_score = max(0.0, min(1.0, predicted_score))

            # 确定质量等级
            quality_level = 'poor'
            for level, threshold in sorted(self.quality_thresholds.items(),
                                         key=lambda x: x[1], reverse=True):
                if predicted_score >= threshold:
                    quality_level = level
                    break

            # 生成预测结果
            prediction = {
                'predicted_score': predicted_score,
                'quality_level': quality_level,
                'confidence': self._calculate_confidence(features),
                'features': features,
                'suggestions': self._generate_suggestions(predicted_score, features),
                'risk_factors': self._identify_risk_factors(features)
            }

            # 记录预测
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'speaker_id': speaker_id,
                'prediction': prediction,
                'actual_quality': None  # 将在实际合成后更新
            })

            self.prediction_stats['total_predictions'] += 1

            return prediction

        except Exception as e:
            print(f"[QualityPredictor] 质量预测失败: {e}")
            return {
                'predicted_score': 0.5,
                'quality_level': 'acceptable',
                'confidence': 0.5,
                'features': {},
                'suggestions': [],
                'risk_factors': []
            }

    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """计算预测置信度"""
        try:
            # 基于特征的可靠性计算置信度
            confidence_factors = []

            # 历史数据可靠性
            if features.get('historical_quality', 0) > 0.1:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)

            # 说话人一致性可靠性
            consistency = features.get('speaker_consistency', 0.5)
            confidence_factors.append(consistency)

            # 嵌入稳定性可靠性
            stability = features.get('embedding_stability', 0.5)
            confidence_factors.append(stability)

            return float(np.mean(confidence_factors))

        except Exception as e:
            print(f"[QualityPredictor] 计算置信度失败: {e}")
            return 0.5

    def _generate_suggestions(self, predicted_score: float,
                            features: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        suggestions = []

        try:
            if predicted_score < 0.6:
                # 低质量预测，提供改进建议

                if features.get('text_length', 0) > 0.8:
                    suggestions.append("文本较长，建议分段处理以提高质量")

                if features.get('speaker_consistency', 0) < 0.6:
                    suggestions.append("说话人一致性较低，建议使用更稳定的参考音频")

                if features.get('parameter_complexity', 0) > 0.7:
                    suggestions.append("参数设置较复杂，建议简化配置")

                if features.get('content_difficulty', 0) > 0.7:
                    suggestions.append("内容复杂度较高，建议调整语速和停顿")

                if features.get('embedding_stability', 0) < 0.6:
                    suggestions.append("嵌入不够稳定，建议使用更多参考样本")

            elif predicted_score > 0.8:
                suggestions.append("预测质量良好，当前设置适合")

            return suggestions

        except Exception as e:
            print(f"[QualityPredictor] 生成建议失败: {e}")
            return []

    def _identify_risk_factors(self, features: Dict[str, float]) -> List[str]:
        """识别风险因素"""
        risk_factors = []

        try:
            if features.get('historical_quality', 0) < 0.4:
                risk_factors.append("历史质量较低")

            if features.get('speaker_consistency', 0) < 0.5:
                risk_factors.append("说话人一致性不足")

            if features.get('embedding_stability', 0) < 0.4:
                risk_factors.append("嵌入稳定性差")

            if features.get('content_difficulty', 0) > 0.8:
                risk_factors.append("内容复杂度过高")

            return risk_factors

        except Exception as e:
            print(f"[QualityPredictor] 识别风险因素失败: {e}")
            return []

    def update_prediction_accuracy(self, predicted_score: float, actual_score: float):
        """更新预测准确性"""
        try:
            # 计算预测误差
            error = abs(predicted_score - actual_score)
            accuracy = 1.0 - min(1.0, error)

            self.prediction_accuracy.append(accuracy)

            # 更新统计
            if accuracy > 0.8:  # 认为是准确预测
                self.prediction_stats['accurate_predictions'] += 1

            # 更新最近的预测记录
            if self.prediction_history:
                self.prediction_history[-1]['actual_quality'] = actual_score

        except Exception as e:
            print(f"[QualityPredictor] 更新预测准确性失败: {e}")

    def get_prediction_stats(self) -> Dict[str, Any]:
        """获取预测统计信息"""
        try:
            if not self.prediction_accuracy:
                return {
                    'total_predictions': self.prediction_stats['total_predictions'],
                    'average_accuracy': 0.0,
                    'accuracy_trend': 'insufficient_data'
                }

            avg_accuracy = float(np.mean(self.prediction_accuracy))
            recent_accuracy = float(np.mean(list(self.prediction_accuracy)[-20:])) if len(self.prediction_accuracy) >= 20 else avg_accuracy

            # 计算准确性趋势
            if len(self.prediction_accuracy) >= 10:
                trend_slope = np.polyfit(range(len(self.prediction_accuracy)), list(self.prediction_accuracy), 1)[0]
                if trend_slope > 0.01:
                    accuracy_trend = 'improving'
                elif trend_slope < -0.01:
                    accuracy_trend = 'declining'
                else:
                    accuracy_trend = 'stable'
            else:
                accuracy_trend = 'insufficient_data'

            return {
                'total_predictions': self.prediction_stats['total_predictions'],
                'accurate_predictions': self.prediction_stats['accurate_predictions'],
                'accuracy_rate': self.prediction_stats['accurate_predictions'] / max(1, self.prediction_stats['total_predictions']),
                'average_accuracy': avg_accuracy,
                'recent_accuracy': recent_accuracy,
                'accuracy_trend': accuracy_trend,
                'quality_improvements_suggested': self.prediction_stats['quality_improvements_suggested'],
                'preventive_adjustments': self.prediction_stats['preventive_adjustments']
            }

        except Exception as e:
            print(f"[QualityPredictor] 获取预测统计失败: {e}")
            return {}


class AdaptiveCacheStrategy:
    """自适应缓存策略系统"""

    def __init__(self, parameter_learner: IntelligentParameterLearner):
        self.parameter_learner = parameter_learner

        # 使用模式分析
        self.usage_patterns = {
            'speaker_frequency': defaultdict(int),      # 说话人使用频率
            'time_patterns': defaultdict(list),         # 时间使用模式
            'session_lengths': deque(maxlen=1000),      # 会话长度历史
            'cache_hit_rates': deque(maxlen=100),       # 缓存命中率历史
            'performance_metrics': deque(maxlen=500)    # 性能指标历史
        }

        # 缓存策略配置
        self.cache_strategies = {
            'lru': {'weight': 0.4, 'effectiveness': deque(maxlen=50)},
            'lfu': {'weight': 0.3, 'effectiveness': deque(maxlen=50)},
            'time_based': {'weight': 0.2, 'effectiveness': deque(maxlen=50)},
            'predictive': {'weight': 0.1, 'effectiveness': deque(maxlen=50)}
        }

        # 动态配置
        self.dynamic_config = {
            'cache_size_multiplier': 1.0,      # 缓存大小倍数
            'eviction_threshold': 0.8,         # 驱逐阈值
            'preload_threshold': 0.7,          # 预加载阈值
            'adaptation_rate': 0.05            # 适应率
        }

        # 性能预测模型
        self.performance_model = {
            'cache_size_impact': 0.3,          # 缓存大小对性能的影响
            'hit_rate_impact': 0.5,            # 命中率对性能的影响
            'strategy_impact': 0.2             # 策略对性能的影响
        }

        # 统计信息
        self.adaptation_stats = {
            'strategy_switches': 0,
            'cache_resizes': 0,
            'preload_actions': 0,
            'performance_improvements': 0
        }

        print("[AdaptiveCacheStrategy] 自适应缓存策略系统初始化完成")

    def analyze_usage_patterns(self, speaker_id: str, session_duration: float,
                              cache_performance: Dict[str, float]):
        """分析使用模式"""
        try:
            current_time = datetime.now()

            # 记录说话人使用频率
            self.usage_patterns['speaker_frequency'][speaker_id] += 1

            # 记录时间模式
            hour = current_time.hour
            self.usage_patterns['time_patterns'][hour].append({
                'speaker_id': speaker_id,
                'timestamp': current_time,
                'duration': session_duration
            })

            # 记录会话长度
            self.usage_patterns['session_lengths'].append(session_duration)

            # 记录缓存性能
            if 'hit_rate' in cache_performance:
                self.usage_patterns['cache_hit_rates'].append(cache_performance['hit_rate'])

            if 'response_time' in cache_performance:
                self.usage_patterns['performance_metrics'].append({
                    'timestamp': current_time,
                    'response_time': cache_performance['response_time'],
                    'hit_rate': cache_performance.get('hit_rate', 0.0),
                    'cache_size': cache_performance.get('cache_size', 0)
                })

            # 定期分析和调整
            if len(self.usage_patterns['performance_metrics']) % 20 == 0:
                self._analyze_and_adapt()

        except Exception as e:
            print(f"[AdaptiveCacheStrategy] 分析使用模式失败: {e}")

    def _analyze_and_adapt(self):
        """分析模式并自适应调整"""
        try:
            # 分析缓存命中率趋势
            if len(self.usage_patterns['cache_hit_rates']) >= 10:
                recent_hit_rates = list(self.usage_patterns['cache_hit_rates'])[-10:]
                avg_hit_rate = np.mean(recent_hit_rates)
                hit_rate_trend = np.polyfit(range(len(recent_hit_rates)), recent_hit_rates, 1)[0]

                # 如果命中率下降，考虑调整策略
                if avg_hit_rate < 0.6 or hit_rate_trend < -0.01:
                    self._adjust_cache_strategy()

            # 分析性能趋势
            if len(self.usage_patterns['performance_metrics']) >= 20:
                recent_metrics = list(self.usage_patterns['performance_metrics'])[-20:]
                avg_response_time = np.mean([m['response_time'] for m in recent_metrics])

                # 如果响应时间过长，考虑增加缓存大小
                if avg_response_time > 1.0:  # 1秒阈值
                    self._adjust_cache_size(increase=True)
                elif avg_response_time < 0.3:  # 0.3秒阈值
                    self._adjust_cache_size(increase=False)

            # 分析说话人使用模式
            self._analyze_speaker_patterns()

            # 分析时间使用模式
            self._analyze_time_patterns()

        except Exception as e:
            print(f"[AdaptiveCacheStrategy] 自适应调整失败: {e}")

    def _adjust_cache_strategy(self):
        """调整缓存策略"""
        try:
            # 评估各策略的效果
            strategy_scores = {}

            for strategy, config in self.cache_strategies.items():
                if config['effectiveness']:
                    avg_effectiveness = np.mean(config['effectiveness'])
                    strategy_scores[strategy] = avg_effectiveness
                else:
                    strategy_scores[strategy] = 0.5  # 默认分数

            # 重新分配权重
            total_score = sum(strategy_scores.values())
            if total_score > 0:
                for strategy in self.cache_strategies:
                    old_weight = self.cache_strategies[strategy]['weight']
                    new_weight = strategy_scores[strategy] / total_score

                    # 平滑调整
                    self.cache_strategies[strategy]['weight'] = (
                        old_weight * 0.8 + new_weight * 0.2
                    )

            self.adaptation_stats['strategy_switches'] += 1
            print("[AdaptiveCacheStrategy] 缓存策略权重已调整")

        except Exception as e:
            print(f"[AdaptiveCacheStrategy] 调整缓存策略失败: {e}")

    def _adjust_cache_size(self, increase: bool):
        """调整缓存大小"""
        try:
            current_multiplier = self.dynamic_config['cache_size_multiplier']

            if increase and current_multiplier < 2.0:
                self.dynamic_config['cache_size_multiplier'] = min(2.0, current_multiplier * 1.2)
                self.adaptation_stats['cache_resizes'] += 1
                print(f"[AdaptiveCacheStrategy] 缓存大小增加到 {self.dynamic_config['cache_size_multiplier']:.2f}x")
            elif not increase and current_multiplier > 0.5:
                self.dynamic_config['cache_size_multiplier'] = max(0.5, current_multiplier * 0.9)
                self.adaptation_stats['cache_resizes'] += 1
                print(f"[AdaptiveCacheStrategy] 缓存大小减少到 {self.dynamic_config['cache_size_multiplier']:.2f}x")

        except Exception as e:
            print(f"[AdaptiveCacheStrategy] 调整缓存大小失败: {e}")

    def _analyze_speaker_patterns(self):
        """分析说话人使用模式"""
        try:
            # 找出高频使用的说话人
            sorted_speakers = sorted(
                self.usage_patterns['speaker_frequency'].items(),
                key=lambda x: x[1], reverse=True
            )

            # 为高频说话人提供预加载建议
            high_frequency_speakers = [
                speaker_id for speaker_id, count in sorted_speakers[:5]
                if count >= 5
            ]

            if high_frequency_speakers:
                print(f"[AdaptiveCacheStrategy] 识别到高频说话人: {len(high_frequency_speakers)} 个")
                # 这里可以触发预加载逻辑

        except Exception as e:
            print(f"[AdaptiveCacheStrategy] 分析说话人模式失败: {e}")

    def _analyze_time_patterns(self):
        """分析时间使用模式"""
        try:
            current_hour = datetime.now().hour

            # 分析当前时间段的使用模式
            if current_hour in self.usage_patterns['time_patterns']:
                recent_usage = self.usage_patterns['time_patterns'][current_hour]
                if len(recent_usage) >= 3:
                    # 预测可能的高使用时段
                    avg_duration = np.mean([usage['duration'] for usage in recent_usage[-10:]])
                    if avg_duration > 30:  # 30秒以上认为是高使用
                        print(f"[AdaptiveCacheStrategy] 检测到高使用时段: {current_hour}:00")

        except Exception as e:
            print(f"[AdaptiveCacheStrategy] 分析时间模式失败: {e}")

    def get_optimal_cache_config(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """获取最优缓存配置"""
        try:
            optimal_config = current_config.copy()

            # 应用动态缓存大小
            if 'cache_size' in optimal_config:
                base_size = optimal_config['cache_size']
                optimal_config['cache_size'] = int(base_size * self.dynamic_config['cache_size_multiplier'])

            # 应用策略权重
            optimal_config['strategy_weights'] = {
                strategy: config['weight']
                for strategy, config in self.cache_strategies.items()
            }

            # 应用其他动态配置
            optimal_config.update({
                'eviction_threshold': self.dynamic_config['eviction_threshold'],
                'preload_threshold': self.dynamic_config['preload_threshold']
            })

            return optimal_config

        except Exception as e:
            print(f"[AdaptiveCacheStrategy] 获取最优配置失败: {e}")
            return current_config

    def predict_cache_performance(self, config: Dict[str, Any]) -> Dict[str, float]:
        """预测缓存性能"""
        try:
            # 基于历史数据预测性能
            if not self.usage_patterns['performance_metrics']:
                return {'predicted_hit_rate': 0.7, 'predicted_response_time': 0.5}

            recent_metrics = list(self.usage_patterns['performance_metrics'])[-50:]

            # 计算基准性能
            baseline_hit_rate = np.mean([m['hit_rate'] for m in recent_metrics])
            baseline_response_time = np.mean([m['response_time'] for m in recent_metrics])

            # 预测缓存大小影响
            cache_size_factor = config.get('cache_size', 200) / 200  # 基准大小200
            hit_rate_improvement = (cache_size_factor - 1) * self.performance_model['cache_size_impact']
            response_time_improvement = (1 - cache_size_factor) * self.performance_model['cache_size_impact']

            # 预测策略影响
            strategy_weights = config.get('strategy_weights', {})
            strategy_effectiveness = 0
            for strategy, weight in strategy_weights.items():
                if strategy in self.cache_strategies and self.cache_strategies[strategy]['effectiveness']:
                    effectiveness = np.mean(self.cache_strategies[strategy]['effectiveness'])
                    strategy_effectiveness += weight * effectiveness

            strategy_improvement = strategy_effectiveness * self.performance_model['strategy_impact']

            # 综合预测
            predicted_hit_rate = min(1.0, baseline_hit_rate + hit_rate_improvement + strategy_improvement)
            predicted_response_time = max(0.1, baseline_response_time + response_time_improvement)

            return {
                'predicted_hit_rate': predicted_hit_rate,
                'predicted_response_time': predicted_response_time,
                'confidence': min(1.0, len(recent_metrics) / 50)
            }

        except Exception as e:
            print(f"[AdaptiveCacheStrategy] 预测缓存性能失败: {e}")
            return {'predicted_hit_rate': 0.7, 'predicted_response_time': 0.5, 'confidence': 0.5}

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """获取自适应统计信息"""
        try:
            # 计算使用模式统计
            total_sessions = sum(self.usage_patterns['speaker_frequency'].values())
            unique_speakers = len(self.usage_patterns['speaker_frequency'])

            avg_session_length = (
                np.mean(self.usage_patterns['session_lengths'])
                if self.usage_patterns['session_lengths'] else 0
            )

            avg_hit_rate = (
                np.mean(self.usage_patterns['cache_hit_rates'])
                if self.usage_patterns['cache_hit_rates'] else 0
            )

            # 计算策略分布
            strategy_distribution = {
                strategy: config['weight']
                for strategy, config in self.cache_strategies.items()
            }

            return {
                'adaptation_stats': self.adaptation_stats.copy(),
                'usage_summary': {
                    'total_sessions': total_sessions,
                    'unique_speakers': unique_speakers,
                    'avg_session_length': avg_session_length,
                    'avg_hit_rate': avg_hit_rate
                },
                'current_config': self.dynamic_config.copy(),
                'strategy_distribution': strategy_distribution,
                'cache_size_multiplier': self.dynamic_config['cache_size_multiplier']
            }

        except Exception as e:
            print(f"[AdaptiveCacheStrategy] 获取自适应统计失败: {e}")
            return {}
