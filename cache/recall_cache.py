import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecallCache:
    """召回结果缓存 - 缓存用户的向量检索结果"""

    def __init__(self, redis_client, ttl=3600, prefix='recall'):
        """
        初始化召回缓存

        :param redis_client: Redis 客户端
        :param ttl: 缓存过期时间（秒），默认1小时
        :param prefix: 缓存键前缀
        """
        self.redis_client = redis_client
        self.ttl = ttl
        self.prefix = prefix
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }

    def _build_key(self, user_id, top_k=20):
        """构建缓存键"""
        return f"{self.prefix}:user:{user_id}:top{top_k}"

    def get(self, user_id, top_k=20):
        """
        获取缓存的召回结果

        :param user_id: 用户ID
        :param top_k: Top-K 数量
        :return: 召回结果列表或 None
        """
        key = self._build_key(user_id, top_k)
        result = self.redis_client.get(key)

        if result is not None:
            self._stats['hits'] += 1
            logger.debug(f"缓存命中: user_id={user_id}")
        else:
            self._stats['misses'] += 1
            logger.debug(f"缓存未命中: user_id={user_id}")

        return result

    def set(self, user_id, results, top_k=20):
        """
        设置缓存的召回结果

        :param user_id: 用户ID
        :param results: 召回结果列表
        :param top_k: Top-K 数量
        :return: 是否成功
        """
        key = self._build_key(user_id, top_k)
        success = self.redis_client.set(key, results, ttl=self.ttl)

        if success:
            self._stats['sets'] += 1

        return success

    def invalidate(self, user_id, top_k=20):
        """
        使缓存失效

        :param user_id: 用户ID
        :param top_k: Top-K 数量
        :return: 是否成功
        """
        key = self._build_key(user_id, top_k)
        return self.redis_client.delete(key)

    def invalidate_all(self):
        """使所有召回缓存失效"""
        pattern = f"{self.prefix}:*"
        keys = self.redis_client.keys(pattern)
        for key in keys:
            self.redis_client.delete(key)
        logger.info(f"已清除 {len(keys)} 个召回缓存")
        return len(keys)

    def get_with_fallback(self, user_id, top_k, retrieve_func):
        """
        获取缓存，未命中则执行检索并缓存

        :param user_id: 用户ID
        :param top_k: Top-K 数量
        :param retrieve_func: 检索函数
        :return: 召回结果列表
        """
        cached = self.get(user_id, top_k)
        if cached is not None:
            return cached

        results = retrieve_func()
        self.set(user_id, results, top_k)
        return results

    def get_stats(self):
        """获取缓存统计信息"""
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total * 100) if total > 0 else 0

        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'sets': self._stats['sets'],
            'hit_rate': round(hit_rate, 2),
            'ttl': self.ttl,
            'prefix': self.prefix,
            'local_mode': self.redis_client.is_local_mode()
        }

    def reset_stats(self):
        """重置统计信息"""
        self._stats = {'hits': 0, 'misses': 0, 'sets': 0}