import numpy as np
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """推荐引擎 - 整合向量检索和缓存，提供推荐服务"""

    def __init__(self, vector_retriever, user_embeddings=None,
                 user_mapping=None, recall_cache=None, top_k=20):
        """
        初始化推荐引擎

        :param vector_retriever: 向量检索器
        :param user_embeddings: 用户向量矩阵
        :param user_mapping: 用户ID映射 {原始ID: 索引}
        :param recall_cache: 召回缓存（可选）
        :param top_k: 默认返回数量
        """
        self.vector_retriever = vector_retriever
        self.user_embeddings = user_embeddings
        self.user_mapping = user_mapping or {}
        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()} if user_mapping else {}
        self.recall_cache = recall_cache
        self.top_k = top_k

        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_latency_ms': 0
        }

    def load_user_embeddings(self, embedding_path='./output/embeddings/user_embedding.npy'):
        """
        加载用户向量

        :param embedding_path: 向量文件路径
        """
        if not os.path.exists(embedding_path):
            logger.error(f"用户向量文件不存在: {embedding_path}")
            return False

        self.user_embeddings = np.load(embedding_path)
        logger.info(f"用户向量已加载: {self.user_embeddings.shape}")
        return True

    def load_user_mapping(self, mapping_path='./output/graph/user_mapping.json'):
        """
        加载用户ID映射

        :param mapping_path: 映射文件路径
        """
        if not os.path.exists(mapping_path):
            logger.warning(f"用户映射文件不存在: {mapping_path}")
            return False

        with open(mapping_path, 'r') as f:
            raw_mapping = json.load(f)
            self.user_mapping = {int(k): v for k, v in raw_mapping.items()}
            self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}

        logger.info(f"用户ID映射已加载: {len(self.user_mapping)} 个用户")
        return True

    def get_user_embedding(self, user_id):
        """
        获取用户向量

        :param user_id: 用户原始ID
        :return: 用户向量或 None
        """
        if self.user_embeddings is None:
            logger.error("用户向量未加载")
            return None

        user_idx = self.user_mapping.get(user_id)
        if user_idx is None:
            # 如果用户不在映射中，尝试直接使用ID作为索引
            if user_id < len(self.user_embeddings):
                user_idx = user_id
            else:
                logger.warning(f"用户ID {user_id} 不在映射中，使用随机向量")
                return np.random.randn(self.user_embeddings.shape[1]).astype(np.float32)

        return self.user_embeddings[user_idx]

    def recommend(self, user_id, top_k=None, use_cache=True):
        """
        为用户推荐商品

        :param user_id: 用户ID
        :param top_k: 返回数量
        :param use_cache: 是否使用缓存
        :return: 推荐结果列表
        """
        import time
        start_time = time.time()
        self._stats['total_requests'] += 1

        top_k = top_k or self.top_k

        if use_cache and self.recall_cache:
            cached = self.recall_cache.get(user_id, top_k)
            if cached is not None:
                self._stats['cache_hits'] += 1
                logger.debug(f"缓存命中: user_id={user_id}")
                return cached

        user_embedding = self.get_user_embedding(user_id)
        if user_embedding is None:
            logger.error(f"无法获取用户向量: user_id={user_id}")
            return []

        results = self.vector_retriever.retrieve(user_embedding, top_k=top_k, use_cache=False)

        if use_cache and self.recall_cache:
            self.recall_cache.set(user_id, results, top_k)

        elapsed = (time.time() - start_time) * 1000
        self._stats['avg_latency_ms'] = (
            self._stats['avg_latency_ms'] * (self._stats['total_requests'] - 1) + elapsed
        ) / self._stats['total_requests']

        return results

    def recommend_batch(self, user_ids, top_k=None, use_cache=True):
        """
        批量推荐

        :param user_ids: 用户ID列表
        :param top_k: 返回数量
        :param use_cache: 是否使用缓存
        :return: 推荐结果字典 {user_id: results}
        """
        results_dict = {}
        for user_id in user_ids:
            results_dict[user_id] = self.recommend(user_id, top_k=top_k, use_cache=use_cache)
        return results_dict

    def get_item_ids(self, results):
        """
        从推荐结果中提取商品ID列表

        :param results: 推荐结果
        :return: 商品ID列表
        """
        return [r['original_item_id'] for r in results]

    def get_stats(self):
        """获取推荐引擎统计信息"""
        total = self._stats['total_requests']
        cache_hit_rate = (self._stats['cache_hits'] / total * 100) if total > 0 else 0

        return {
            'total_requests': total,
            'cache_hits': self._stats['cache_hits'],
            'cache_hit_rate': round(cache_hit_rate, 2),
            'avg_latency_ms': round(self._stats['avg_latency_ms'], 2),
            'top_k': self.top_k,
            'has_cache': self.recall_cache is not None
        }