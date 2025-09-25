#!/usr/bin/env python3
"""
IndexTTS2 高级音频系统
Advanced Audio Systems for IndexTTS2 - Phase 2 Improvements
"""

import torch
import torch.nn.functional as F
import numpy as np
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, OrderedDict
import threading
import json
import os

class SpeakerEmbeddingCache:
    """智能说话人嵌入缓存系统"""
    
    def __init__(self, cache_size: int = 200, similarity_threshold: float = 0.95,
                 enable_multi_sample_fusion: bool = True, adaptive_cache_strategy=None):
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.enable_multi_sample_fusion = enable_multi_sample_fusion
        self.adaptive_cache_strategy = adaptive_cache_strategy
        
        # 缓存存储
        self.cache = OrderedDict()  # LRU缓存
        self.access_count = defaultdict(int)
        self.creation_time = {}
        self.sample_groups = defaultdict(list)  # 相似样本分组
        
        # 性能统计
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'fusion_operations': 0,
            'similarity_matches': 0
        }
        
        # 线程安全
        self._lock = threading.RLock()
        
    def _compute_audio_hash(self, audio: torch.Tensor, metadata: Dict = None) -> str:
        """计算音频哈希值"""
        # 使用音频的统计特征和元数据计算哈希
        features = []
        
        # 音频统计特征
        features.extend([
            audio.mean().item(),
            audio.std().item(), 
            audio.max().item(),
            audio.min().item(),
            audio.shape[-1]  # 长度
        ])
        
        # 频域特征
        if audio.shape[-1] > 1024:
            fft = torch.fft.fft(audio.flatten()[:1024])
            magnitude = torch.abs(fft)
            features.extend([
                magnitude.mean().item(),
                magnitude.std().item(),
                torch.argmax(magnitude).item()  # 主频位置
            ])
        
        # 元数据
        if metadata:
            features.append(str(metadata.get('sample_rate', 22050)))
            features.append(str(metadata.get('speaker_id', 'unknown')))
        
        # 生成哈希
        hash_input = json.dumps(features, sort_keys=True).encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def _compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """计算嵌入相似度"""
        try:
            # 确保两个嵌入具有相同的形状
            emb1_flat = emb1.flatten()
            emb2_flat = emb2.flatten()

            # 检查维度是否匹配
            if emb1_flat.shape != emb2_flat.shape:
                # 如果维度不匹配，使用较小的维度进行比较
                min_size = min(emb1_flat.shape[0], emb2_flat.shape[0])
                emb1_flat = emb1_flat[:min_size]
                emb2_flat = emb2_flat[:min_size]

                # 如果维度差异太大，认为不相似
                size_ratio = max(emb1.numel(), emb2.numel()) / min(emb1.numel(), emb2.numel())
                if size_ratio > 2.0:  # 如果大小差异超过2倍，认为不相似
                    return 0.0

            # 余弦相似度
            cos_sim = F.cosine_similarity(emb1_flat, emb2_flat, dim=0)

            # L2距离相似度
            l2_dist = torch.norm(emb1_flat - emb2_flat)
            l2_sim = 1.0 / (1.0 + l2_dist.item())

            # 加权组合
            similarity = 0.7 * cos_sim.item() + 0.3 * l2_sim
            return similarity

        except Exception as e:
            # 如果计算失败，返回0相似度
            print(f"[SpeakerEmbeddingCache] 相似度计算失败: {e}")
            return 0.0
    
    def _find_similar_embeddings(self, target_embedding: torch.Tensor,
                                audio_hash: str) -> List[Tuple[str, torch.Tensor, float]]:
        """查找相似的嵌入"""
        similar_embeddings = []
        target_shape = target_embedding.shape
        target_size = target_embedding.numel()

        for cached_hash, cached_data in self.cache.items():
            if cached_hash == audio_hash:
                continue

            cached_embedding = cached_data['embedding']
            cached_shape = cached_data.get('embedding_shape', cached_embedding.shape)
            cached_size = cached_data.get('embedding_size', cached_embedding.numel())

            # 只比较形状和大小相似的嵌入
            if cached_shape == target_shape or cached_size == target_size:
                similarity = self._compute_similarity(target_embedding, cached_embedding)

                if similarity > self.similarity_threshold:
                    similar_embeddings.append((cached_hash, cached_embedding, similarity))
            else:
                # 跳过维度不匹配的嵌入
                continue

        # 按相似度排序
        similar_embeddings.sort(key=lambda x: x[2], reverse=True)
        return similar_embeddings[:5]  # 最多返回5个相似嵌入
    
    def _fuse_embeddings(self, embeddings: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """多样本嵌入融合"""
        if len(embeddings) == 1:
            return embeddings[0][0]

        try:
            # 检查所有嵌入的形状是否一致
            reference_shape = embeddings[0][0].shape
            compatible_embeddings = []

            for embedding, weight in embeddings:
                if embedding.shape == reference_shape:
                    compatible_embeddings.append((embedding, weight))
                else:
                    # 如果形状不匹配，尝试调整或跳过
                    if embedding.numel() == embeddings[0][0].numel():
                        # 如果元素数量相同，重塑形状
                        reshaped_embedding = embedding.reshape(reference_shape)
                        compatible_embeddings.append((reshaped_embedding, weight))
                    else:
                        # 形状和大小都不匹配，跳过这个嵌入
                        print(f"[SpeakerEmbeddingCache] 跳过不兼容的嵌入: {embedding.shape} vs {reference_shape}")
                        continue

            # 如果没有兼容的嵌入，返回第一个
            if len(compatible_embeddings) <= 1:
                return embeddings[0][0]

            # 加权平均融合
            total_weight = sum(weight for _, weight in compatible_embeddings)
            fused_embedding = torch.zeros_like(compatible_embeddings[0][0])

            for embedding, weight in compatible_embeddings:
                fused_embedding += embedding * (weight / total_weight)

            return fused_embedding

        except Exception as e:
            print(f"[SpeakerEmbeddingCache] 嵌入融合失败: {e}")
            # 返回第一个嵌入作为备用
            return embeddings[0][0]
    
    def get_or_compute_embedding(self, audio: torch.Tensor, 
                               extractor_func,
                               metadata: Dict = None,
                               force_recompute: bool = False) -> torch.Tensor:
        """获取或计算说话人嵌入"""
        with self._lock:
            start_time = time.time()
            self.stats['total_requests'] += 1

            # 计算音频哈希
            audio_hash = self._compute_audio_hash(audio, metadata)

            # 检查缓存
            if not force_recompute and audio_hash in self.cache:
                # 缓存命中
                self.stats['cache_hits'] += 1
                self.access_count[audio_hash] += 1

                # 移动到末尾（LRU）
                cached_data = self.cache.pop(audio_hash)
                self.cache[audio_hash] = cached_data

                # 记录性能数据
                response_time = time.time() - start_time
                if self.adaptive_cache_strategy:
                    cache_performance = {
                        'hit_rate': self.stats['cache_hits'] / self.stats['total_requests'],
                        'response_time': response_time,
                        'cache_size': len(self.cache)
                    }
                    self.adaptive_cache_strategy.analyze_usage_patterns(
                        speaker_id=audio_hash[:8],  # 简化的说话人ID
                        session_duration=response_time,
                        cache_performance=cache_performance
                    )

                return cached_data['embedding']
            
            # 缓存未命中，计算新嵌入
            self.stats['cache_misses'] += 1
            embedding = extractor_func(audio)
            
            # 多样本融合（如果启用）
            if self.enable_multi_sample_fusion:
                similar_embeddings = self._find_similar_embeddings(embedding, audio_hash)
                
                if similar_embeddings:
                    self.stats['similarity_matches'] += len(similar_embeddings)
                    
                    # 准备融合数据
                    fusion_data = [(embedding, 1.0)]  # 当前嵌入权重为1.0
                    
                    for _, similar_emb, similarity in similar_embeddings:
                        # 相似度越高，权重越大
                        weight = similarity ** 2  # 平方增强差异
                        fusion_data.append((similar_emb, weight))
                    
                    # 执行融合
                    embedding = self._fuse_embeddings(fusion_data)
                    self.stats['fusion_operations'] += 1
            
            # 缓存管理
            if len(self.cache) >= self.cache_size:
                # 移除最少使用的嵌入
                lru_hash = next(iter(self.cache))
                del self.cache[lru_hash]
                del self.access_count[lru_hash]
                del self.creation_time[lru_hash]
            
            # 添加到缓存
            self.cache[audio_hash] = {
                'embedding': embedding,
                'embedding_shape': embedding.shape,
                'embedding_size': embedding.numel(),
                'metadata': metadata or {},
                'audio_shape': audio.shape,
                'creation_time': time.time()
            }
            self.access_count[audio_hash] = 1
            self.creation_time[audio_hash] = time.time()
            
            return embedding
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        with self._lock:
            hit_rate = (self.stats['cache_hits'] / max(self.stats['total_requests'], 1)) * 100
            
            return {
                'cache_size': len(self.cache),
                'max_cache_size': self.cache_size,
                'hit_rate': hit_rate,
                'total_requests': self.stats['total_requests'],
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'fusion_operations': self.stats['fusion_operations'],
                'similarity_matches': self.stats['similarity_matches'],
                'average_access_count': np.mean(list(self.access_count.values())) if self.access_count else 0
            }
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.access_count.clear()
            self.creation_time.clear()
            self.sample_groups.clear()
    
    def save_cache_to_disk(self, filepath: str):
        """保存缓存到磁盘"""
        with self._lock:
            cache_data = {
                'cache': {k: {
                    'embedding': v['embedding'].cpu().numpy().tolist(),
                    'metadata': v['metadata'],
                    'audio_shape': v['audio_shape'],
                    'creation_time': v['creation_time']
                } for k, v in self.cache.items()},
                'access_count': dict(self.access_count),
                'stats': self.stats
            }
            
            with open(filepath, 'w') as f:
                json.dump(cache_data, f, indent=2)
    
    def load_cache_from_disk(self, filepath: str):
        """从磁盘加载缓存"""
        if not os.path.exists(filepath):
            return
            
        with self._lock:
            try:
                with open(filepath, 'r') as f:
                    cache_data = json.load(f)
                
                # 恢复缓存
                self.cache.clear()
                for k, v in cache_data.get('cache', {}).items():
                    self.cache[k] = {
                        'embedding': torch.tensor(v['embedding']),
                        'metadata': v['metadata'],
                        'audio_shape': v['audio_shape'],
                        'creation_time': v['creation_time']
                    }
                
                # 恢复统计信息
                self.access_count.update(cache_data.get('access_count', {}))
                self.stats.update(cache_data.get('stats', {}))
                
            except Exception as e:
                print(f"[SpeakerEmbeddingCache] 加载缓存失败: {e}")

class VoiceConsistencyController:
    """声音一致性控制器"""
    
    def __init__(self, consistency_threshold: float = 0.8, adaptation_rate: float = 0.1):
        self.consistency_threshold = consistency_threshold
        self.adaptation_rate = adaptation_rate
        
        # 说话人档案
        self.speaker_profiles = {}
        self.consistency_history = defaultdict(list)
        
        # 全局一致性统计
        self.global_stats = {
            'total_checks': 0,
            'consistency_violations': 0,
            'corrections_applied': 0,
            'average_consistency': 0.0
        }
        
        # 线程安全
        self._lock = threading.RLock()
    
    def register_speaker(self, speaker_id: str, reference_embedding: torch.Tensor, 
                        metadata: Dict = None):
        """注册说话人参考嵌入"""
        with self._lock:
            self.speaker_profiles[speaker_id] = {
                'reference_embedding': reference_embedding.clone(),
                'embedding_history': [reference_embedding.clone()],
                'consistency_scores': [],
                'metadata': metadata or {},
                'registration_time': time.time(),
                'update_count': 0
            }
    
    def compute_consistency_score(self, current_embedding: torch.Tensor, 
                                speaker_id: str) -> float:
        """计算一致性分数"""
        with self._lock:
            if speaker_id not in self.speaker_profiles:
                return 1.0
            
            profile = self.speaker_profiles[speaker_id]
            reference = profile['reference_embedding']
            
            # 多维度一致性评估
            cos_sim = F.cosine_similarity(
                current_embedding.flatten(), 
                reference.flatten(), 
                dim=0
            ).item()
            
            # L2距离一致性
            l2_dist = torch.norm(current_embedding.flatten() - reference.flatten()).item()
            l2_consistency = 1.0 / (1.0 + l2_dist)
            
            # 历史一致性（与最近几个嵌入的平均相似度）
            if len(profile['embedding_history']) > 1:
                recent_embeddings = profile['embedding_history'][-3:]  # 最近3个
                hist_similarities = []
                
                for hist_emb in recent_embeddings:
                    hist_sim = F.cosine_similarity(
                        current_embedding.flatten(),
                        hist_emb.flatten(),
                        dim=0
                    ).item()
                    hist_similarities.append(hist_sim)
                
                hist_consistency = np.mean(hist_similarities)
            else:
                hist_consistency = cos_sim
            
            # 加权组合
            consistency_score = (
                0.5 * cos_sim + 
                0.3 * l2_consistency + 
                0.2 * hist_consistency
            )
            
            return consistency_score
    
    def apply_consistency_constraint(self, current_embedding: torch.Tensor,
                                   speaker_id: str,
                                   constraint_strength: float = None) -> torch.Tensor:
        """应用一致性约束"""
        with self._lock:
            self.global_stats['total_checks'] += 1
            
            if speaker_id not in self.speaker_profiles:
                return current_embedding
            
            consistency_score = self.compute_consistency_score(current_embedding, speaker_id)
            profile = self.speaker_profiles[speaker_id]
            
            # 记录一致性历史
            profile['consistency_scores'].append(consistency_score)
            if len(profile['consistency_scores']) > 20:
                profile['consistency_scores'].pop(0)
            
            # 更新全局统计
            self.global_stats['average_consistency'] = (
                (self.global_stats['average_consistency'] * (self.global_stats['total_checks'] - 1) + 
                 consistency_score) / self.global_stats['total_checks']
            )
            
            if consistency_score < self.consistency_threshold:
                # 一致性不足，应用约束
                self.global_stats['consistency_violations'] += 1
                self.global_stats['corrections_applied'] += 1
                
                # 自适应约束强度
                if constraint_strength is None:
                    # 根据一致性分数动态调整约束强度
                    constraint_strength = (self.consistency_threshold - consistency_score) * 0.5
                    constraint_strength = min(constraint_strength, 0.5)  # 最大约束强度
                
                reference = profile['reference_embedding']
                
                # 加权平均约束
                constrained_embedding = (
                    current_embedding * (1 - constraint_strength) + 
                    reference * constraint_strength
                )
                
                return constrained_embedding
            
            return current_embedding
    
    def update_speaker_profile(self, speaker_id: str, 
                             new_embedding: torch.Tensor,
                             consistency_score: float):
        """更新说话人档案"""
        with self._lock:
            if speaker_id not in self.speaker_profiles:
                return
            
            profile = self.speaker_profiles[speaker_id]
            
            # 更新嵌入历史
            profile['embedding_history'].append(new_embedding.clone())
            if len(profile['embedding_history']) > 10:
                profile['embedding_history'].pop(0)
            
            # 自适应更新参考嵌入
            if consistency_score > 0.9:  # 高质量嵌入
                old_ref = profile['reference_embedding']
                profile['reference_embedding'] = (
                    old_ref * (1 - self.adaptation_rate) + 
                    new_embedding * self.adaptation_rate
                )
                profile['update_count'] += 1
    
    def get_consistency_stats(self) -> Dict:
        """获取一致性统计信息"""
        with self._lock:
            speaker_stats = {}
            
            for speaker_id, profile in self.speaker_profiles.items():
                if profile['consistency_scores']:
                    speaker_stats[speaker_id] = {
                        'average_consistency': np.mean(profile['consistency_scores']),
                        'min_consistency': np.min(profile['consistency_scores']),
                        'max_consistency': np.max(profile['consistency_scores']),
                        'consistency_trend': np.mean(profile['consistency_scores'][-5:]) if len(profile['consistency_scores']) >= 5 else 0,
                        'update_count': profile['update_count'],
                        'history_length': len(profile['embedding_history'])
                    }
            
            violation_rate = (self.global_stats['consistency_violations'] / 
                            max(self.global_stats['total_checks'], 1)) * 100
            
            return {
                'global_stats': self.global_stats,
                'violation_rate': violation_rate,
                'speaker_count': len(self.speaker_profiles),
                'speaker_stats': speaker_stats
            }

class AdaptiveQualityMonitor:
    """自适应质量监控器"""

    def __init__(self, quality_thresholds: Dict = None, enable_auto_improvement: bool = False):
        self.quality_thresholds = quality_thresholds or {
            'snr': 20.0,  # dB
            'thd': 0.05,  # 5%
            'spectral_flatness': 0.5,
            'dynamic_range': 40.0,  # dB
            'peak_level': -3.0  # dB
        }

        # 自动改进功能
        self.enable_auto_improvement = enable_auto_improvement
        
        # 质量历史
        self.quality_history = []
        self.quality_trends = defaultdict(list)
        
        # 自适应阈值
        self.adaptive_thresholds = self.quality_thresholds.copy()
        
        # 统计信息
        self.stats = {
            'total_assessments': 0,
            'quality_violations': 0,
            'auto_corrections': 0,
            'threshold_adaptations': 0,
            'improvements_applied': 0
        }
        
        # 线程安全
        self._lock = threading.RLock()
    
    def compute_snr(self, audio: torch.Tensor) -> float:
        """计算信噪比"""
        # 信号功率（使用RMS）
        signal_power = torch.mean(audio ** 2)
        
        # 噪声估计（使用最低10%的能量）
        sorted_power = torch.sort(audio ** 2)[0]
        noise_power = torch.mean(sorted_power[:len(sorted_power)//10])
        
        if noise_power > 0:
            snr = 10 * torch.log10(signal_power / noise_power)
            return snr.item()
        
        return float('inf')
    
    def compute_thd(self, audio: torch.Tensor, sample_rate: int = 22050) -> float:
        """计算总谐波失真"""
        if audio.shape[-1] < 2048:
            return 0.0
        
        # FFT分析
        fft = torch.fft.fft(audio.flatten())
        magnitude = torch.abs(fft[:len(fft)//2])
        
        # 找到基频
        freqs = torch.fft.fftfreq(len(audio.flatten()), 1/sample_rate)[:len(fft)//2]
        fundamental_idx = torch.argmax(magnitude[1:]) + 1  # 跳过DC分量
        
        # 计算谐波失真
        fundamental_power = magnitude[fundamental_idx] ** 2
        harmonic_power = 0
        
        for harmonic in range(2, 6):  # 2-5次谐波
            harmonic_freq = freqs[fundamental_idx] * harmonic
            harmonic_idx = torch.argmin(torch.abs(freqs - harmonic_freq))
            
            if harmonic_idx < len(magnitude):
                harmonic_power += magnitude[harmonic_idx] ** 2
        
        if fundamental_power > 0:
            thd = torch.sqrt(harmonic_power / fundamental_power)
            return thd.item()
        
        return 0.0
    
    def compute_spectral_flatness(self, audio: torch.Tensor) -> float:
        """计算频谱平坦度"""
        if audio.shape[-1] < 1024:
            return 0.5
        
        fft = torch.fft.fft(audio.flatten())
        magnitude = torch.abs(fft[:len(fft)//2])
        magnitude = magnitude[1:]  # 去除DC分量
        
        # 几何平均 / 算术平均
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude + 1e-8)))
        arithmetic_mean = torch.mean(magnitude)
        
        if arithmetic_mean > 0:
            return (geometric_mean / arithmetic_mean).item()
        
        return 0.0
    
    def compute_dynamic_range(self, audio: torch.Tensor) -> float:
        """计算动态范围"""
        max_level = torch.max(torch.abs(audio))
        
        # 噪声底限（最低1%的能量）
        sorted_abs = torch.sort(torch.abs(audio.flatten()))[0]
        noise_floor = torch.mean(sorted_abs[:len(sorted_abs)//100])
        
        if noise_floor > 0 and max_level > 0:
            dynamic_range = 20 * torch.log10(max_level / noise_floor)
            return dynamic_range.item()
        
        return 0.0
    
    def compute_peak_level(self, audio: torch.Tensor) -> float:
        """计算峰值电平"""
        peak = torch.max(torch.abs(audio))
        if peak > 0:
            peak_db = 20 * torch.log10(peak)
            return peak_db.item()
        
        return -float('inf')
    
    def assess_quality(self, audio: torch.Tensor, sample_rate: int = 22050) -> Dict[str, float]:
        """综合质量评估"""
        with self._lock:
            self.stats['total_assessments'] += 1
            
            # 计算各项指标
            metrics = {
                'snr': self.compute_snr(audio),
                'thd': self.compute_thd(audio, sample_rate),
                'spectral_flatness': self.compute_spectral_flatness(audio),
                'dynamic_range': self.compute_dynamic_range(audio),
                'peak_level': self.compute_peak_level(audio)
            }
            
            # 质量评分
            quality_scores = {}
            violations = 0
            
            for metric, value in metrics.items():
                threshold = self.adaptive_thresholds[metric]
                
                if metric in ['snr', 'dynamic_range']:
                    # 越高越好
                    score = min(value / threshold, 1.0) if threshold > 0 else 1.0
                    if value < threshold:
                        violations += 1
                elif metric in ['thd']:
                    # 越低越好
                    score = max(1.0 - value / threshold, 0.0) if threshold > 0 else 1.0
                    if value > threshold:
                        violations += 1
                elif metric == 'peak_level':
                    # 接近阈值最好
                    score = max(1.0 - abs(value - threshold) / 10.0, 0.0)
                    if value > -1.0:  # 过载检查
                        violations += 1
                else:
                    # spectral_flatness等，接近阈值最好
                    score = max(1.0 - abs(value - threshold) / threshold, 0.0) if threshold > 0 else 1.0
                
                quality_scores[metric] = score
            
            # 综合质量分数
            overall_quality = np.mean(list(quality_scores.values()))
            
            # 记录历史
            quality_record = {
                'timestamp': time.time(),
                'metrics': metrics,
                'scores': quality_scores,
                'overall_quality': overall_quality,
                'violations': violations
            }
            
            self.quality_history.append(quality_record)
            if len(self.quality_history) > 100:
                self.quality_history.pop(0)
            
            # 更新趋势
            for metric, value in metrics.items():
                self.quality_trends[metric].append(value)
                if len(self.quality_trends[metric]) > 50:
                    self.quality_trends[metric].pop(0)
            
            # 统计违规
            if violations > 0:
                self.stats['quality_violations'] += 1
            
            # 自适应阈值调整
            self._adapt_thresholds()

            result = {
                'metrics': metrics,
                'scores': quality_scores,
                'overall_quality': overall_quality,
                'violations': violations,
                'adaptive_thresholds': self.adaptive_thresholds.copy(),
                'improved_audio': None,
                'improvement_applied': False
            }

            # 自动质量改进 - 已禁用
            # if self.enable_auto_improvement and violations > 0:
            #     try:
            #         improved_audio = self._auto_improve_audio(audio, metrics, sample_rate)
            #         if improved_audio is not None:
            #             result['improved_audio'] = improved_audio
            #             result['improvement_applied'] = True
            #             self.stats['auto_corrections'] += 1
            #     except Exception as e:
            #         print(f"[AdaptiveQualityMonitor] 自动改进失败: {e}")
            print("[AdaptiveQualityMonitor] ℹ️ 自动改进功能已禁用，使用原始音频")

            return result

    def _auto_improve_audio(self, audio: torch.Tensor, metrics: Dict, sample_rate: int) -> torch.Tensor:
        """自动音频质量改进"""
        improved_audio = audio.clone()
        improvements_applied = []

        # 1. 处理过高的峰值电平 (削波/过载)
        if metrics['peak_level'] > -1.0:  # 峰值过高
            # 计算当前峰值
            current_peak = torch.max(torch.abs(improved_audio)).item()

            # 只有在真正过载时才进行限幅
            if current_peak > 0.95:  # 接近削波
                # 温和的软限幅，保持更多动态
                target_peak = 0.9
                reduction_factor = target_peak / current_peak
                improved_audio = improved_audio * reduction_factor
                improvements_applied.append(f"峰值限幅({reduction_factor:.2f})")
            elif current_peak > 0.8:  # 轻微过载
                # 非常温和的处理
                improved_audio = torch.tanh(improved_audio * 0.95)
                improvements_applied.append("软限幅")

        # 2. 处理高THD (总谐波失真)
        if metrics['thd'] > 0.1:  # THD > 10%
            # 应用低通滤波减少高频失真
            nyquist = sample_rate // 2
            cutoff_freq = min(8000, nyquist * 0.8)  # 8kHz或80%奈奎斯特频率

            # 简单的低通滤波 (移动平均)
            kernel_size = max(3, int(sample_rate / cutoff_freq))
            if kernel_size % 2 == 0:
                kernel_size += 1

            # 创建低通滤波核
            kernel = torch.ones(kernel_size) / kernel_size
            kernel = kernel.unsqueeze(0).unsqueeze(0)

            # 应用卷积滤波
            if improved_audio.dim() == 2:
                improved_audio = improved_audio.unsqueeze(0)
                filtered = torch.nn.functional.conv1d(
                    improved_audio, kernel, padding=kernel_size//2
                )
                improved_audio = filtered.squeeze(0)

            improvements_applied.append("低通滤波")

        # 3. 处理动态范围问题
        if metrics['dynamic_range'] > 100.0:  # 动态范围过大
            # 应用温和的压缩
            threshold = 0.7
            ratio = 3.0

            # 简单的压缩算法
            abs_audio = torch.abs(improved_audio)
            mask = abs_audio > threshold

            if mask.any():
                # 对超过阈值的部分应用压缩
                compressed_part = threshold + (abs_audio[mask] - threshold) / ratio
                improved_audio[mask] = torch.sign(improved_audio[mask]) * compressed_part

            improvements_applied.append("动态压缩")

        # 4. 标准化音量
        if len(improvements_applied) > 0:
            # 标准化到合适的音量 - 修复音量过小问题
            current_rms = torch.sqrt(torch.mean(improved_audio ** 2))

            # 动态选择目标RMS，确保音频有足够的音量
            if current_rms > 0.5:  # 如果当前音量很大
                target_rms = 0.3  # 适度降低
            elif current_rms > 0.1:  # 如果当前音量适中
                target_rms = max(0.2, current_rms * 0.8)  # 轻微调整
            else:  # 如果当前音量很小
                target_rms = 0.2  # 提升到合理水平

            if current_rms > 1e-6:  # 避免除零
                normalization_factor = target_rms / current_rms
                # 限制标准化范围，避免过度放大或缩小
                normalization_factor = max(0.1, min(5.0, normalization_factor))
                improved_audio = improved_audio * normalization_factor
                improvements_applied.append(f"音量标准化(x{normalization_factor:.2f})")

        if improvements_applied:
            print(f"[AdaptiveQualityMonitor] 🔧 自动改进应用: {', '.join(improvements_applied)}")
            return improved_audio

        return None  # 没有改进

    def _adapt_thresholds(self):
        """自适应阈值调整"""
        if len(self.quality_history) < 10:
            return
        
        # 分析最近的质量趋势
        recent_records = self.quality_history[-10:]
        
        for metric in self.quality_thresholds.keys():
            recent_values = [r['metrics'][metric] for r in recent_records]
            
            # 计算趋势
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)
            
            # 自适应调整
            current_threshold = self.adaptive_thresholds[metric]
            
            if metric in ['snr', 'dynamic_range']:
                # 如果平均值持续高于阈值，可以适当提高阈值
                if mean_value > current_threshold * 1.2:
                    new_threshold = current_threshold * 1.05
                    self.adaptive_thresholds[metric] = min(new_threshold, current_threshold * 1.5)
                    self.stats['threshold_adaptations'] += 1
            elif metric in ['thd']:
                # 如果平均值持续低于阈值，可以适当降低阈值
                if mean_value < current_threshold * 0.8:
                    new_threshold = current_threshold * 0.95
                    self.adaptive_thresholds[metric] = max(new_threshold, current_threshold * 0.5)
                    self.stats['threshold_adaptations'] += 1
    
    def get_quality_stats(self) -> Dict:
        """获取质量统计信息"""
        with self._lock:
            if not self.quality_history:
                return {'message': 'No quality assessments yet'}
            
            # 计算趋势
            trends = {}
            for metric, values in self.quality_trends.items():
                if len(values) >= 5:
                    recent_trend = np.mean(values[-5:]) - np.mean(values[-10:-5]) if len(values) >= 10 else 0
                    trends[metric] = recent_trend
            
            # 最近质量统计
            recent_qualities = [r['overall_quality'] for r in self.quality_history[-10:]]
            
            violation_rate = (self.stats['quality_violations'] / 
                            max(self.stats['total_assessments'], 1)) * 100
            
            return {
                'stats': self.stats,
                'violation_rate': violation_rate,
                'average_quality': np.mean(recent_qualities),
                'quality_trend': np.mean(recent_qualities[-5:]) - np.mean(recent_qualities[-10:-5]) if len(recent_qualities) >= 10 else 0,
                'current_thresholds': self.adaptive_thresholds,
                'original_thresholds': self.quality_thresholds,
                'metric_trends': trends,
                'assessment_count': len(self.quality_history)
            }
