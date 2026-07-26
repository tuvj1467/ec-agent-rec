import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """Qdrant 客户端封装"""

    def __init__(self, host='localhost', port=6333, collection_name='ecommerce_items',
                 embedding_dim=64, distance='Cosine'):
        """
        初始化 Qdrant 客户端

        :param host: Qdrant 服务地址
        :param port: Qdrant 服务端口
        :param collection_name: 集合名称
        :param embedding_dim: 向量维度
        :param distance: 距离度量 ('Cosine', 'Euclid', 'Dot')
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance = distance
        self.client = None
        self._local_mode = False
        self._local_vectors = {}
        self._local_payloads = {}

        self._init_client()

    def _init_client(self):
        """初始化 Qdrant 客户端"""
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Qdrant 客户端已连接: {self.host}:{self.port}")
        except ImportError:
            logger.warning("qdrant_client 未安装，使用本地内存模式")
            self._local_mode = True
            self.client = None
        except Exception as e:
            logger.warning(f"Qdrant 连接失败，使用本地内存模式: {e}")
            self._local_mode = True
            self.client = None

    def create_collection(self, collection_name=None, embedding_dim=None, distance=None):
        """
        创建集合

        :param collection_name: 集合名称
        :param embedding_dim: 向量维度
        :param distance: 距离度量
        :return: 是否成功
        """
        collection_name = collection_name or self.collection_name
        embedding_dim = embedding_dim or self.embedding_dim
        distance = distance or self.distance

        if self._local_mode:
            self._local_vectors[collection_name] = []
            self._local_payloads[collection_name] = []
            logger.info(f"[本地模式] 集合已创建: {collection_name}")
            return True

        try:
            from qdrant_client.http.models import Distance, VectorParams

            distance_map = {
                'Cosine': Distance.COSINE,
                'Euclid': Distance.EUCLID,
                'Dot': Distance.DOT
            }

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=distance_map.get(distance, Distance.COSINE)
                )
            )
            logger.info(f"集合已创建: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def delete_collection(self, collection_name=None):
        """
        删除集合

        :param collection_name: 集合名称
        :return: 是否成功
        """
        collection_name = collection_name or self.collection_name

        if self._local_mode:
            if collection_name in self._local_vectors:
                del self._local_vectors[collection_name]
                del self._local_payloads[collection_name]
            logger.info(f"[本地模式] 集合已删除: {collection_name}")
            return True

        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"集合已删除: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    def collection_exists(self, collection_name=None):
        """
        检查集合是否存在

        :param collection_name: 集合名称
        :return: 是否存在
        """
        collection_name = collection_name or self.collection_name

        if self._local_mode:
            return collection_name in self._local_vectors

        try:
            collections = self.client.get_collections()
            return collection_name in [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"检查集合失败: {e}")
            return False

    def upsert_points(self, vectors, payloads=None, ids=None, collection_name=None):
        """
        批量插入/更新向量点

        :param vectors: 向量列表 [(embedding_dim,)
        :param payloads: 负载信息列表 [{...}
        :param ids: 点ID列表
        :param collection_name: 集合名称
        :return: 是否成功
        """
        collection_name = collection_name or self.collection_name

        if ids is None:
            ids = list(range(len(vectors)))

        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]

        if self._local_mode:
            if collection_name not in self._local_vectors:
                self._local_vectors[collection_name] = []
                self._local_payloads[collection_name] = []

            for i, vec, payload in zip(ids, vectors, payloads):
                if i < len(self._local_vectors[collection_name]):
                    self._local_vectors[collection_name][i] = vec
                    self._local_payloads[collection_name][i] = payload
                else:
                    self._local_vectors[collection_name].append(vec)
                    self._local_payloads[collection_name].append(payload)

            logger.info(f"[本地模式] 已插入 {len(vectors)} 个向量点")
            return True

        try:
            from qdrant_client.http.models import PointStruct

            points = [
                PointStruct(id=id_, vector=vec.tolist(), payload=payload)
                for id_, vec, payload in zip(ids, vectors, payloads)
            ]

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"已插入 {len(vectors)} 个向量点")
            return True

        except Exception as e:
            logger.error(f"插入向量失败: {e}")
            return False

    def search(self, query_vector, limit=20, collection_name=None):
        """
        向量相似度检索

        :param query_vector: 查询向量
        :param limit: 返回数量
        :param collection_name: 集合名称
        :return: 检索结果列表 [{id, score, payload}]
        """
        collection_name = collection_name or self.collection_name

        if self._local_mode:
            return self._local_search(query_vector, limit, collection_name)

        try:
            from qdrant_client.http.models import Filter

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector,
                limit=limit
            )

            return [
                {'id': hit.id, 'score': hit.score, 'payload': hit.payload}
                for hit in results
            ]

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def _local_search(self, query_vector, limit, collection_name):
        """本地内存模式下的相似度检索"""
        import numpy as np

        if collection_name not in self._local_vectors:
            logger.warning(f"[本地模式] 集合不存在: {collection_name}")
            return []

        vectors = np.array(self._local_vectors[collection_name])
        if len(vectors) == 0:
            return []

        if hasattr(query_vector, 'tolist'):
            query_vector = np.array(query_vector)
        else:
            query_vector = np.array(query_vector)

        # 余弦相似度
        if self.distance == 'Cosine':
            norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
            scores = np.dot(vectors, query_vector) / (norms + 1e-8)
        elif self.distance == 'Dot':
            scores = np.dot(vectors, query_vector)
        else:  # Euclid
            scores = -np.linalg.norm(vectors - query_vector, axis=1)

        top_indices = np.argsort(scores)[::-1][:limit]

        results = []
        for idx in top_indices:
            results.append({
                'id': int(idx),
                'score': float(scores[idx]),
                'payload': self._local_payloads[collection_name][idx] if idx < len(self._local_payloads[collection_name]) else {}
            })

        return results

    def get_collection_info(self, collection_name=None):
        """
        获取集合信息

        :param collection_name: 集合名称
        :return: 集合信息
        """
        collection_name = collection_name or self.collection_name

        if self._local_mode:
            if collection_name in self._local_vectors:
                return {
                    'name': collection_name,
                    'points_count': len(self._local_vectors[collection_name]),
                    'local_mode': True
                }
            return {}

        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                'name': collection_name,
                'points_count': info.points_count,
                'vectors_count': info.vectors_count
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}

    def is_local_mode(self):
        """是否为本地内存模式"""
        return self._local_mode