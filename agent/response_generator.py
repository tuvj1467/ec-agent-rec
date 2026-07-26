import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """回复生成器 - 生成电商导购回复"""

    def __init__(self, llm_client=None, use_mock=True):
        """
        初始化回复生成器

        :param llm_client: LLM 客户端
        :param use_mock: 是否使用 Mock 模式（无 LLM 时）
        """
        self.llm_client = llm_client
        self.use_mock = use_mock or (llm_client is None)

        if self.use_mock:
            logger.info("使用 Mock LLM 模式生成回复")

    def generate(self, user_query, user_id, candidates, template_type='recommend'):
        """
        生成导购回复

        :param user_query: 用户问题
        :param user_id: 用户ID
        :param candidates: 候选商品列表
        :param template_type: 模板类型
        :return: 回复文本 + 商品ID列表
        """
        if self.use_mock:
            return self._generate_mock(user_query, user_id, candidates, template_type)
        else:
            return self._generate_with_llm(user_query, user_id, candidates, template_type)

    def _generate_mock(self, user_query, user_id, candidates, template_type):
        """
        Mock 模式生成回复（模板化）

        :param user_query: 用户问题
        :param user_id: 用户ID
        :param candidates: 候选商品列表
        :param template_type: 模板类型
        :return: 回复文本 + 商品ID列表
        """
        item_ids = [str(c['original_item_id']) for c in candidates[:10]]
        scores = [f"{c['score']:.3f}" for c in candidates[:10]]

        if template_type == 'welcome':
            response = self._mock_welcome(user_query, user_id, candidates, item_ids, scores)
        elif template_type == 'explain':
            response = self._mock_explain(user_query, user_id, candidates, item_ids, scores)
        elif template_type == 'category':
            response = self._mock_category(user_query, user_id, candidates, item_ids, scores)
        else:
            response = self._mock_recommend(user_query, user_id, candidates, item_ids, scores)

        return response, item_ids

    def _mock_recommend(self, user_query, user_id, candidates, item_ids, scores):
        """Mock 推荐回复"""
        response = f"""
亲，您好！👋 根据您的喜好，我为您精心挑选了 {len(item_ids)} 款好物推荐：

🎉 **为您推荐**：
"""

        for i, (item_id, score) in enumerate(zip(item_ids, scores), 1):
            response += f"{i}. 商品 ID: {item_id} (匹配度: {float(score)*100:.1f}%)\n"

        response += f"""
✨ 这些都是根据您的浏览和购买偏好智能推荐的哦~
💡 您可以告诉我更具体的需求，比如：
   • 想要什么类型的商品？
   • 预算范围是多少？
   • 有没有特别关注的功能？

需要我为您详细介绍哪款商品吗？😊
"""
        return response.strip()

    def _mock_welcome(self, user_query, user_id, candidates, item_ids, scores):
        """Mock 欢迎回复"""
        response = f"""
欢迎来到智能导购！我是您的专属导购"小荐" 🎉

根据热门商品和您的潜在喜好，我先为您推荐几款热门好物：

🔥 **热门推荐**：
"""

        for i, (item_id, score) in enumerate(zip(item_ids[:5], scores[:5]), 1):
            response += f"{i}. 商品 ID: {item_id} (热度: {float(score)*100:.1f}%)\n"

        response += f"""
💝 您可以这样和我互动：
   • "帮我推荐一些数码产品"
   • "我想买个礼物"
   • "介绍一下商品 12345"
   • "有什么性价比高的？"

告诉我您的需求，让我为您找到最合适的商品吧！🛍️
"""
        return response.strip()

    def _mock_explain(self, user_query, user_id, candidates, item_ids, scores):
        """Mock 商品详情回复"""
        response = f"""
好的，我来为您详细介绍一下这款商品~ 🔍

📦 **商品详情**：
   商品 ID: {candidates[0]['original_item_id'] if candidates else 'N/A'}
   匹配度: {float(scores[0])*100:.1f}%

✨ **商品特点**：
   • 品质保证，正品行货
   • 超高性价比
   • 用户评价优秀
   • 售后无忧

👉 **相似推荐**：
"""

        for i, (item_id, score) in enumerate(zip(item_ids[1:6], scores[1:6]), 1):
            response += f"   {i}. 商品 ID: {item_id} (相似度: {float(score)*100:.1f}%)\n"

        response += """
需要我为您对比这些商品吗？或者您还有其他想了解的吗？😊
"""
        return response.strip()

    def _mock_category(self, user_query, user_id, candidates, item_ids, scores):
        """Mock 品类推荐回复"""
        response = f"""
好的！这是我为您精选的好物推荐 🎯

🛒 **为您挑选**：
"""

        for i, (item_id, score) in enumerate(zip(item_ids, scores), 1):
            response += f"{i}. 商品 ID: {item_id} (推荐度: {float(score)*100:.1f}%)\n"

        response += """
💡 这些都是根据您的兴趣偏好精心筛选的哦~
还有其他需要了解的吗？随时告诉我！😉
"""
        return response.strip()

    def _generate_with_llm(self, user_query, user_id, candidates, template_type):
        """
        使用 LLM 生成回复

        :param user_query: 用户问题
        :param user_id: 用户ID
        :param candidates: 候选商品列表
        :param template_type: 模板类型
        :return: 回复文本 + 商品ID列表
        """
        from .prompt_templates import PROMPT_TEMPLATES

        template = PROMPT_TEMPLATES.get(template_type, PROMPT_TEMPLATES['recommend'])
        system_prompt = PROMPT_TEMPLATES['system']

        candidate_text = "\n".join([
            f"  - 商品ID: {c['original_item_id']}, 匹配度: {c['score']:.4f}"
            for c in candidates[:20]
        ])

        user_prompt = template.format(
            user_query=user_query,
            user_id=user_id,
            candidate_items=candidate_text,
            target_item=candidates[0]['original_item_id'] if candidates else 'N/A',
            category='相关品类'
        )

        try:
            response_text = self.llm_client.chat(system_prompt, user_prompt)
            response, item_ids = self._parse_llm_response(response_text)
            return response, item_ids
        except Exception as e:
            logger.error(f"LLM 生成失败，使用 Mock 模式: {e}")
            return self._generate_mock(user_query, user_id, candidates, template_type)

    def _parse_llm_response(self, response_text):
        """
        解析 LLM 回复，提取回复文本和商品ID

        :param response_text: LLM 原始回复
        :return: 回复文本, 商品ID列表
        """
        response_match = re.search(r'<response>(.*?)</response>', response_text, re.DOTALL)
        items_match = re.search(r'<items>(.*?)</items>', response_text, re.DOTALL)

        response = response_match.group(1).strip() if response_match else response_text

        item_ids = []
        if items_match:
            items_text = items_match.group(1).strip()
            item_ids = [item.strip() for item in items_text.split(',') if item.strip()]

        return response, item_ids