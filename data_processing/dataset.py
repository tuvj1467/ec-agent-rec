import numpy as np
import pandas as pd
import scipy.sparse as sp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationDataset:
    """推荐系统数据集 - 用于 LightGCN 训练"""

    def __init__(self, adj_matrix, user_items_dict, num_negatives=1, mode='train'):
        """
        初始化数据集

        :param adj_matrix: 用户-商品邻接矩阵 (sparse)
        :param user_items_dict: 用户交互商品字典 {user_idx: set(item_idx)}
        :param num_negatives: 负采样数量
        :param mode: 'train' / 'val' / 'test'
        """
        import torch
        from torch.utils.data import Dataset as TorchDataset
        self._torch = torch
        self._Dataset = TorchDataset

        self.adj_matrix = adj_matrix
        self.user_items_dict = user_items_dict
        self.num_negatives = num_negatives
        self.mode = mode

        self.num_users = adj_matrix.shape[0]
        self.num_items = adj_matrix.shape[1]

        self.user_list = list(user_items_dict.keys())

        logger.info(f"数据集初始化: {mode}模式, {self.num_users} 用户, {self.num_items} 商品")

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        """
        获取一个样本

        :return: user_idx, pos_item_idx, neg_item_idx (neg_items数量为num_negatives)
        """
        user_idx = self.user_list[index]
        pos_items = self.user_items_dict[user_idx]

        if len(pos_items) == 0:
            pos_item = np.random.randint(0, self.num_items)
        else:
            pos_item = np.random.choice(list(pos_items))

        neg_items = []
        for _ in range(self.num_negatives):
            while True:
                neg_item = np.random.randint(0, self.num_items)
                if neg_item not in pos_items:
                    neg_items.append(neg_item)
                    break

        user_tensor = self._torch.LongTensor([user_idx])
        pos_item_tensor = self._torch.LongTensor([pos_item])
        neg_item_tensor = self._torch.LongTensor(neg_items)

        return user_tensor, pos_item_tensor, neg_item_tensor


def split_data(adj_matrix, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    划分数据集

    按用户划分: 每个用户的交互按比例分到训练/验证/测试集

    :param adj_matrix: 原始邻接矩阵
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :param seed: 随机种子
    :return: train_adj, val_adj, test_adj, train_user_items, val_user_items, test_user_items
    """
    np.random.seed(seed)
    logger.info("开始划分数据集")

    num_users = adj_matrix.shape[0]
    num_items = adj_matrix.shape[1]

    train_rows, train_cols = [], []
    val_rows, val_cols = [], []
    test_rows, test_cols = [], []

    train_user_items = {}
    val_user_items = {}
    test_user_items = {}

    for user_idx in range(num_users):
        pos_items = list(adj_matrix[user_idx].indices)
        np.random.shuffle(pos_items)

        n = len(pos_items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val

        if n <= 2:
            n_train = n - 1
            n_val = 1
            n_test = 0

        train_items = pos_items[:n_train]
        val_items = pos_items[n_train:n_train + n_val]
        test_items = pos_items[n_train + n_val:]

        train_rows.extend([user_idx] * len(train_items))
        train_cols.extend(train_items)
        val_rows.extend([user_idx] * len(val_items))
        val_cols.extend(val_items)
        test_rows.extend([user_idx] * len(test_items))
        test_cols.extend(test_items)

        train_user_items[user_idx] = set(train_items)
        val_user_items[user_idx] = set(val_items)
        test_user_items[user_idx] = set(test_items)

    train_adj = sp.csr_matrix(
        (np.ones(len(train_rows)), (train_rows, train_cols)),
        shape=(num_users, num_items)
    )
    val_adj = sp.csr_matrix(
        (np.ones(len(val_rows)), (val_rows, val_cols)),
        shape=(num_users, num_items)
    )
    test_adj = sp.csr_matrix(
        (np.ones(len(test_rows)), (test_rows, test_cols)),
        shape=(num_users, num_items)
    )

    logger.info(f"数据集划分完成:")
    logger.info(f"  训练集: {train_adj.nnz} 条交互")
    logger.info(f"  验证集: {val_adj.nnz} 条交互")
    logger.info(f"  测试集: {test_adj.nnz} 条交互")

    return train_adj, val_adj, test_adj, train_user_items, val_user_items, test_user_items


def build_dataloaders(adj_matrix, batch_size=1024, num_negatives=1, num_workers=0):
    """
    构建数据加载器

    :param adj_matrix: 训练集邻接矩阵
    :param batch_size: 批次大小
    :param num_negatives: 负采样数量
    :param num_workers: 工作线程数
    :return: train_loader
    """
    user_items = {}
    for user_idx in range(adj_matrix.shape[0]):
        items = set(adj_matrix[user_idx].indices)
        if len(items) > 0:
            user_items[user_idx] = items

    train_dataset = RecommendationDataset(
        adj_matrix=adj_matrix,
        user_items_dict=user_items,
        num_negatives=num_negatives,
        mode='train'
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    logger.info(f"数据加载器构建完成: batch_size={batch_size}, "
                f"num_batches={len(train_loader)}")

    return train_loader