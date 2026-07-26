import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EcommerceAgent:
    """电商推荐 Agent - 主类"""

    def __init__(self, recommendation_engine, response_generator=None):
        """
        初始化电商 Agent

        :param recommendation_engine: 推荐引擎
        :param response_generator: 回复生成器（可选，默认用Mock）
        """
        self.recommendation_engine = recommendation_engine

        if response_generator is None:
            from .response_generator import ResponseGenerator
            self.response_generator = ResponseGenerator(use_mock=True)
        else:
            self.response_generator = response_generator

        self.conversation_history = {}
        self.current_user_id = None

        logger.info("电商推荐 Agent 初始化完成")

    def chat(self, user_id, user_message, top_k=20):
        """
        与用户对话

        :param user_id: 用户ID
        :param user_message: 用户消息
        :param top_k: 推荐商品数量
        :return: 回复文本 + 商品ID列表
        """
        self.current_user_id = user_id

        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        self.conversation_history[user_id].append({
            'role': 'user',
            'content': user_message
        })

        template_type = self._detect_intent(user_message)

        candidates = self.recommendation_engine.recommend(user_id, top_k=top_k)

        response_text, item_ids = self.response_generator.generate(
            user_query=user_message,
            user_id=user_id,
            candidates=candidates,
            template_type=template_type
        )

        self.conversation_history[user_id].append({
            'role': 'assistant',
            'content': response_text,
            'item_ids': item_ids
        })

        return response_text, item_ids

    def _detect_intent(self, user_message):
        """
        检测用户意图

        :param user_message: 用户消息
        :return: 意图类型
        """
        message = user_message.lower()

        if re.search(r'(你好|您好|hi|hello|在吗|有人吗|欢迎|第一次)', message):
            return 'welcome'

        if re.search(r'(介绍|详情|是什么|怎么样|看看|解释|说明)', message):
            return 'explain'

        if re.search(r'(分类|品类|类型|哪类|哪种|数码|服装|食品|家居|美妆)', message):
            return 'category'

        return 'recommend'

    def get_recommendations(self, user_id, top_k=20):
        """
        获取推荐商品（直接返回结构化结果）

        :param user_id: 用户ID
        :param top_k: 返回数量
        :return: 推荐结果列表
        """
        return self.recommendation_engine.recommend(user_id, top_k=top_k)

    def get_conversation_history(self, user_id):
        """获取对话历史"""
        return self.conversation_history.get(user_id, [])

    def clear_conversation(self, user_id):
        """清除对话历史"""
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
            logger.info(f"已清除用户 {user_id} 的对话历史")

    def get_stats(self):
        """获取统计信息"""
        return {
            'total_users': len(self.conversation_history),
            'recommendation_stats': self.recommendation_engine.get_stats()
        }