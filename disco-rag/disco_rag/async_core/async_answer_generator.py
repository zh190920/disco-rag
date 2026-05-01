"""
异步答案生成器 - 高性能版

答案生成是单次LLM调用，优化点：
1. 异步非阻塞等待
2. 并行准备提纲文本和证据文本
"""

import asyncio
import logging
from typing import List

from disco_rag.rst_parser import RSTTree
from disco_rag.relation_net import RelationNet
from disco_rag.outline_generator import Outline
from disco_rag.async_core.async_llm_client import AsyncLLMClient

# 复用同步版的Prompt模板
from disco_rag.answer_generator import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class AsyncAnswerGenerator:
    """异步答案生成器"""

    def __init__(self, llm_client: AsyncLLMClient):
        self.llm = llm_client

    async def generate(
        self,
        question: str,
        outline: Outline,
        rst_trees: List[RSTTree],
        relation_net: RelationNet,
    ) -> str:
        """异步生成最终答案"""
        logger.info("生成最终答案...")

        # 并行准备两类输入
        outline_text, evidence_text = await asyncio.gather(
            asyncio.to_thread(self._format_outline, outline),
            asyncio.to_thread(self._format_evidence, rst_trees, relation_net),
        )

        user_prompt = ANSWER_USER_TEMPLATE.format(
            question=question,
            outline_text=outline_text,
            evidence_text=evidence_text,
        )

        answer = await self.llm.chat(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
        )

        logger.info(f"答案生成完成: {len(answer)} 字符")
        return answer

    @staticmethod
    def _format_outline(outline: Outline) -> str:
        """格式化提纲文本"""
        lines = []
        lines.append(f"整体策略: {outline.overall_strategy}")
        if outline.conflict_resolution:
            lines.append(f"矛盾协调: {outline.conflict_resolution}")
        if outline.conclusion_strategy:
            lines.append(f"结论策略: {outline.conclusion_strategy}")
        lines.append("\n章节规划:")
        for section in outline.sections:
            lines.append(f"\n{section.section_id}. {section.title}")
            lines.append(f"   内容: {section.content_plan}")
            if section.strategy:
                lines.append(f"   策略: {section.strategy}")
            if section.key_points:
                for kp in section.key_points:
                    lines.append(f"   • {kp}")
        return "\n".join(lines)

    @staticmethod
    def _format_evidence(rst_trees: List[RSTTree], relation_net: RelationNet) -> str:
        """格式化可引用的证据文本"""
        lines = []
        for tree in rst_trees:
            lines.append(f"\n[段落 {tree.paragraph_id}]")
            lines.append(f"原文: {tree.paragraph_text}")
            lines.append(f"核心论点: {tree.get_core_claim()}")
            conditions = tree.get_conditions()
            if conditions:
                lines.append(f"⚠️ 限定条件: {'；'.join(conditions)}")
            relations = relation_net.get_relations_of(tree.paragraph_id)
            for rel in relations:
                if rel.relation_type.value != "neutral":
                    direction = "→" if rel.source_id == tree.paragraph_id else "←"
                    other_id = rel.target_id if rel.source_id == tree.paragraph_id else rel.source_id
                    lines.append(f"🔗 关系 {direction} 段落{other_id}: {rel.relation_type.value} - {rel.explanation}")
        return "\n".join(lines)
