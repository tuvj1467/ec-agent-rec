import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisClientWrapper:
    """Redis 客户端封装"""

    def __init__(self, host='localhost', port=6379, db=0, password=None):
        """
        初始化 Redis 客户端

        :param host: Redis 地址
        :param port: 端口
        :param db: 数据库索引
        :param password: 密码
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.client = None
        self._local_mode = False
        self._local_store = {}
        self._local_ttl = {}

        self._init_client()

    def _init_client(self):
        """初始化 Redis 连接"""
        try:
            import redis

            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            self.client.ping()
            logger.info(f"Redis 连接成功: {self.host}:{self.port}")
        except ImportError:
            logger.warning("redis 包未安装，使用本地内存模式")
            self._local_mode = True
        except Exception as e:
            logger.warning(f"Redis 连接失败，使用本地内存模式: {e}")
            self._local_mode = True

    def get(self, key):
        """
        获取键值

        :param key: 键
        :return: 值（自动反序列化 JSON）
        """
        if self._local_mode:
            return self._local_store.get(key)

        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis GET 失败: {e}")
            return None

    def set(self, key, value, ttl=None):
        """
        设置键值

        :param key: 键
        :param value: 值（自动序列化为 JSON）
        :param ttl: 过期时间（秒）
        :return: 是否成功
        """
        serialized = json.dumps(value, ensure_ascii=False)

        if self._local_mode:
            self._local_store[key] = value
            if ttl:
                self._local_ttl[key] = ttl
            return True

        try:
            if ttl:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis SET 失败: {e}")
            return False

    def delete(self, key):
        """
        删除键

        :param key: 键
        :return: 是否成功
        """
        if self._local_mode:
            self._local_store.pop(key, None)
            self._local_ttl.pop(key, None)
            return True

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE 失败: {e}")
            return False

    def exists(self, key):
        """
        检查键是否存在

        :param key: 键
        :return: 是否存在
        """
        if self._local_mode:
            return key in self._local_store

        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS 失败: {e}")
            return False

    def keys(self, pattern='*'):
        """
        获取匹配的键列表

        :param pattern: 匹配模式
        :return: 键列表
        """
        if self._local_mode:
            import fnmatch
            return [k for k in self._local_store.keys() if fnmatch.fnmatch(k, pattern)]

        try:
            return self.client.keys(pattern)
        except Exception as e:
            logger.error(f"Redis KEYS 失败: {e}")
            return []

    def flush_db(self):
        """清空当前数据库"""
        if self._local_mode:
            self._local_store.clear()
            self._local_ttl.clear()
            logger.info("[本地模式] 数据库已清空")
            return True

        try:
            self.client.flushdb()
            logger.info("Redis 数据库已清空")
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHDB 失败: {e}")
            return False

    def is_local_mode(self):
        """是否为本地内存模式"""
        return self._local_mode

    def get_stats(self):
        """获取统计信息"""
        if self._local_mode:
            return {
                'local_mode': True,
                'total_keys': len(self._local_store)
            }

        try:
            return {
                'local_mode': False,
                'total_keys': self.client.dbsize(),
                'info': self.client.info()
            }
        except Exception as e:
            return {'error': str(e)}