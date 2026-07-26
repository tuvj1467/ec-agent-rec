import numpy as np
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorIndexer:
    """向量索引器 - 批量将商品向量入库 Qdrant"""

    def __init__(self, qdrant_client, item_mapping_path=None, output_dir='./output/embeddings'):
        """
        初始化向量索引器

        :param qdrant_client: Qdrant 客户端
        :param item_mapping_path: 商品ID映射文件路径
        :param output_dir: 向量文件目录
        """
        self.qdrant_client = qdrant_client
        self.item_mapping_path = item_mapping_path
        self.output_dir = output_dir

        self.item_mapping = {}
        self.reverse_item_mapping = {}

        if item_mapping_path and os.path.exists(item_mapping_path):
            self._load_item_mapping()

    def _load_item_mapping(self):
        """加载商品ID映射"""
        with open(self.item_mapping_path, 'r') as f:
            raw_mapping = json.load(f)
            self.item_mapping = {int(k): v for k, v in raw_mapping.items()}
            self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}

        logger.info(f"商品ID映射已加载: {len(self.item_mapping)} 个商品")

    def index_item_embeddings(self, item_embeddings=None, item_ids=None, batch_size=1000):
        """
        批量索引商品向量

        :param item_embeddings: 商品向量矩阵 (num_items, embedding_dim)
        :param item_ids: 商品原始ID列表
        :param batch_size: 批次大小
        :return: 是否成功
        """
        if item_embeddings is None:
            item_embeddings = self._load_item_embeddings()

        if item_embeddings is None:
            logger.error("商品向量为空")
            return False

        num_items = len(item_embeddings)
        logger.info(f"开始索引商品向量: {num_items} 个商品")

        if not self.qdrant_client.collection_exists():
            self.qdrant_client.create_collection()

        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            batch_vectors = item_embeddings[start:end]

            batch_payloads = []
            batch_ids = []
            for i in range(start, end):
                item_idx = i
                original_item_id = self.reverse_item_mapping.get(item_idx, item_idx) if self.reverse_item_mapping else item_idx
                batch_ids.append(item_idx)
                batch_payloads.append({
                    'item_idx': int(item_idx),
                    'original_item_id': int(original_item_id)
                })

            self.qdrant_client.upsert_points(
                vectors=batch_vectors,
                payloads=batch_payloads,
                ids=batch_ids
            )

            if (end // batch_size) % 10 == 0:
                logger.info(f"索引进度: {end}/{num_items} ({end / num_items * 100:.1f}%)")

        logger.info("商品向量索引完成")
        return True

    def _load_item_embeddings(self):
        """加载商品向量"""
        embedding_path = os.path.join(self.output_dir, 'item_embedding.npy')
        if not os.path.exists(embedding_path):
            logger.error(f"向量文件不存在: {embedding_path}")
            return None

        embeddings = np.load(embedding_path)
        logger.info(f"商品向量已加载: {embeddings.shape}")
        return embeddings

    def get_index_stats(self):
        """获取索引统计信息"""
        info = self.qdrant_client.get_collection_info()
        return {
            'collection': info,
            'item_mapping_size': len(self.item_mapping),
            'local_mode': self.qdrant_client.is_local_mode()
        }