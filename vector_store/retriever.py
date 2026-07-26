import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量检索器 - 根据用户向量检索相似商品"""

    def __init__(self, qdrant_client, top_k=20):
        """
        初始化向量检索器

        :param qdrant_client: Qdrant 客户端
        :param top_k: 返回的 Top-K 数量
        """
        self.qdrant_client = qdrant_client
        self.top_k = top_k
        self._cache = {}  # 简单内存缓存
        self._stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'total_time_ms': 0
        }

    def retrieve(self, user_embedding, top_k=None, use_cache=True):
        """
        检索与用户向量最相似的商品

        :param user_embedding: 用户向量
        :param top_k: 返回数量，默认使用初始化值
        :param use_cache: 是否使用缓存
        :return: 检索结果列表 [{item_idx, original_item_id, score}]
        """
        top_k = top_k or self.top_k
        self._stats['total_queries'] += 1

        start_time = time.time()

        if use_cache:
            cache_key = str(user_embedding.tobytes()) if hasattr(user_embedding, 'tobytes') else str(user_embedding)
            if cache_key in self._cache:
                self._stats['cache_hits'] += 1
                return self._cache[cache_key][:top_k]

        results = self.qdrant_client.search(
            query_vector=user_embedding,
            limit=top_k
        )

        formatted_results = []
        for result in results:
            payload = result.get('payload', {})
            formatted_results.append({
                'item_idx': result['id'],
                'original_item_id': payload.get('original_item_id', result['id']),
                'score': result['score']
            })

        elapsed = time.time() - start_time
        self._stats['total_time_ms'] += elapsed * 1000

        if use_cache:
            self._cache[cache_key] = formatted_results
            if len(self._cache) > 10000:
                self._cache = {}  # 简单的缓存清理

        return formatted_results

    def batch_retrieve(self, user_embeddings, top_k=None, use_cache=True):
        """
        批量检索

        :param user_embeddings: 用户向量列表
        :param top_k: 返回数量
        :param use_cache: 是否使用缓存
        :return: 检索结果列表的列表
        """
        all_results = []
        for user_emb in user_embeddings:
            results = self.retrieve(user_emb, top_k=top_k, use_cache=use_cache)
            all_results.append(results)
        return all_results

    def get_item_ids(self, results):
        """
        从检索结果中提取商品ID列表

        :param results: 检索结果
        :return: 商品ID列表
        """
        return [r['original_item_id'] for r in results]

    def get_item_indices(self, results):
        """
        从检索结果中提取商品索引列表

        :param results: 检索结果
        :return: 商品索引列表
        """
        return [r['item_idx'] for r in results]

    def get_stats(self):
        """获取检索统计信息"""
        total_queries = self._stats['total_queries']
        return {
            'total_queries': total_queries,
            'cache_hits': self._stats['cache_hits'],
            'cache_hit_rate': float(self._stats['cache_hits'] / total_queries * 100) if total_queries > 0 else 0,
            'avg_latency_ms': float(self._stats['total_time_ms'] / total_queries) if total_queries > 0 else 0,
            'total_time_ms': float(self._stats['total_time_ms'])
        }

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("缓存已清空")