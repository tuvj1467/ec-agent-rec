import torch
import torch.optim as optim
import os
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器 - 负责 LightGCN 模型的训练"""

    def __init__(self, model, train_loader, evaluator=None, config=None):
        """
        初始化训练器

        :param model: LightGCN 模型
        :param train_loader: 训练数据加载器
        :param evaluator: 评估器
        :param config: 训练配置
        """
        self.model = model
        self.train_loader = train_loader
        self.evaluator = evaluator

        self.config = config or {}
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.weight_decay = self.config.get('weight_decay', 0.0001)
        self.epochs = self.config.get('epochs', 100)
        self.eval_every = self.config.get('eval_every', 10)
        self.save_dir = self.config.get('save_dir', './output/checkpoints')
        self.patience = self.config.get('patience', 10)

        os.makedirs(self.save_dir, exist_ok=True)

        self.device = self._get_device()
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,
            gamma=0.5
        )

        self.best_recall = 0
        self.best_epoch = 0
        self.patience_counter = 0

        logger.info(f"训练器初始化: device={self.device}, lr={self.learning_rate}, "
                    f"weight_decay={self.weight_decay}, epochs={self.epochs}")

    def _get_device(self):
        """获取可用设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("使用 CPU")
        return device

    def train(self):
        """执行训练"""
        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)

        train_history = {
            'loss': [],
            'recall': [],
            'ndcg': []
        }

        for epoch in range(1, self.epochs + 1):
            epoch_loss = self._train_one_epoch(epoch)
            train_history['loss'].append(epoch_loss)

            self.scheduler.step()

            if epoch % self.eval_every == 0 and self.evaluator is not None:
                metrics = self.evaluator.evaluate(self.model)
                train_history['recall'].append(metrics['recall@20'])
                train_history['ndcg'].append(metrics['ndcg@20'])

                recall = metrics['recall@20']
                if recall > self.best_recall:
                    self.best_recall = recall
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"最优模型已保存 (Recall@20: {recall:.4f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(f"早停触发，{self.patience} 轮无提升")
                        break

                logger.info(f"Epoch {epoch:3d}/{self.epochs} | "
                            f"Loss: {epoch_loss:.4f} | "
                            f"Recall@20: {metrics['recall@20']:.4f} | "
                            f"NDCG@20: {metrics['ndcg@20']:.4f}")
            else:
                logger.info(f"Epoch {epoch:3d}/{self.epochs} | Loss: {epoch_loss:.4f}")

        self._save_checkpoint(epoch, is_best=False)

        logger.info("=" * 60)
        logger.info(f"训练完成: 最优 Epoch {self.best_epoch}, "
                    f"最佳 Recall@20: {self.best_recall:.4f}")
        logger.info("=" * 60)

        return train_history

    def _train_one_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch_data in self.train_loader:
            user_indices, pos_item_indices, neg_item_indices = batch_data

            user_indices = user_indices.to(self.device)
            pos_item_indices = pos_item_indices.to(self.device)
            neg_item_indices = neg_item_indices.to(self.device)

            self.optimizer.zero_grad()

            loss = self.model.bpr_loss(user_indices, pos_item_indices, neg_item_indices)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time

        return avg_loss

    def _save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_recall': self.best_recall,
            'config': {
                'num_users': self.model.num_users,
                'num_items': self.model.num_items,
                'embedding_dim': self.model.embedding_dim,
                'num_layers': self.model.num_layers
            }
        }

        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pt')
        else:
            path = os.path.join(self.save_dir, f'model_epoch_{epoch}.pt')

        torch.save(checkpoint, path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        if not os.path.exists(checkpoint_path):
            logger.error(f"检查点文件不存在: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_recall = checkpoint.get('best_recall', 0)

        logger.info(f"检查点已加载: {checkpoint_path} (epoch {checkpoint['epoch']})")
        return True