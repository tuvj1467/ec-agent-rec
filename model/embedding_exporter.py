import numpy as np
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingExporter:
    """向量导出器 - 导出用户和商品的嵌入向量"""

    def __init__(self, model, output_dir='./output/embeddings'):
        """
        初始化向量导出器

        :param model: LightGCN 模型
        :param output_dir: 输出目录
        """
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_embeddings(self):
        """
        导出用户和商品的嵌入向量

        :return: user_embeddings, item_embeddings (numpy arrays)
        """
        logger.info("开始导出嵌入向量")

        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            user_embeddings = self.model.get_user_embeddings()
            item_embeddings = self.model.get_item_embeddings()

            user_embeddings = user_embeddings.cpu().numpy()
            item_embeddings = item_embeddings.cpu().numpy()

        logger.info(f"用户向量: {user_embeddings.shape}")
        logger.info(f"商品向量: {item_embeddings.shape}")

        return user_embeddings, item_embeddings

    def save_embeddings(self, user_embeddings=None, item_embeddings=None):
        """
        保存嵌入向量到文件

        :param user_embeddings: 用户向量 (numpy array)
        :param item_embeddings: 商品向量 (numpy array)
        :return: 是否保存成功
        """
        if user_embeddings is None or item_embeddings is None:
            user_embeddings, item_embeddings = self.export_embeddings()

        try:
            user_path = os.path.join(self.output_dir, 'user_embedding.npy')
            item_path = os.path.join(self.output_dir, 'item_embedding.npy')

            np.save(user_path, user_embeddings)
            np.save(item_path, item_embeddings)

            logger.info(f"用户向量已保存: {user_path}")
            logger.info(f"商品向量已保存: {item_path}")

            return True

        except Exception as e:
            logger.error(f"保存向量失败: {e}")
            return False

    def load_embeddings(self):
        """
        从文件加载嵌入向量

        :return: user_embeddings, item_embeddings
        """
        user_path = os.path.join(self.output_dir, 'user_embedding.npy')
        item_path = os.path.join(self.output_dir, 'item_embedding.npy')

        if not os.path.exists(user_path) or not os.path.exists(item_path):
            logger.error("向量文件不存在")
            return None, None

        user_embeddings = np.load(user_path)
        item_embeddings = np.load(item_path)

        logger.info(f"向量已加载: user={user_embeddings.shape}, item={item_embeddings.shape}")

        return user_embeddings, item_embeddings

    def get_embedding_stats(self, user_embeddings=None, item_embeddings=None):
        """
        获取向量统计信息

        :param user_embeddings: 用户向量
        :param item_embeddings: 商品向量
        :return: 统计信息字典
        """
        if user_embeddings is None or item_embeddings is None:
            user_embeddings, item_embeddings = self.export_embeddings()

        stats = {
            'user_embedding': {
                'shape': user_embeddings.shape,
                'mean': float(user_embeddings.mean()),
                'std': float(user_embeddings.std()),
                'min': float(user_embeddings.min()),
                'max': float(user_embeddings.max()),
                'norm_mean': float(np.linalg.norm(user_embeddings, axis=1).mean())
            },
            'item_embedding': {
                'shape': item_embeddings.shape,
                'mean': float(item_embeddings.mean()),
                'std': float(item_embeddings.std()),
                'min': float(item_embeddings.min()),
                'max': float(item_embeddings.max()),
                'norm_mean': float(np.linalg.norm(item_embeddings, axis=1).mean())
            }
        }

        return stats