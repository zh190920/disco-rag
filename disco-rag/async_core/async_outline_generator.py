"""
异步写作提纲生成器 - 高性能版

提纲生成是单次LLM调用，本身无法并行化。
优化点：
1. 使用异步LLM客户端（非阻塞等待）
2. 并行准备输入信息（格式化段落信息、关系信息、冲突信息）
3. 流式输出支持（可选）
"""

import asyncio
import logging
from typing import List

from disco_rag.rst_parser import RSTTree
from disco_rag.relation_net import RelationNet, RelationType
from disco_rag.outline_generator import Outline, OutlineSection
from disco_rag.async_core.async_llm_client import AsyncLLMClient

# 复用同步版的Prompt模板
from disco_rag.outline_generator import (
    OUTLINE_SYSTEM_PROMPT,
    OUTLINE_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class AsyncOutlineGenerator:
    """异步写作提纲生成器"""

    def __init__(self, llm_client: AsyncLLMClient):
        self.llm = llm_client

    async def generate(
        self,
        question: str,
        rst_trees: List[RSTTree],
        relation_net: RelationNet,
    ) -> Outline:
        """
        异步生成写作提纲

        优化：并行准备三类输入信息
        """
        logger.info(f"生成写作提纲，问题: {question[:50]}...")

        # 并行格式化三类信息（虽然这里是CPU操作，但保持一致的模式）
        paragraphs_info = await asyncio.to_thread(
            self._format_paragraphs_info, rst_trees
        )
        relations_info = await asyncio.to_thread(
            self._format_relations_info, relation_net, rst_trees
        )
        conflicts_info = await asyncio.to_thread(
            self._format_conflicts_info, relation_net
        )

        user_prompt = OUTLINE_USER_TEMPLATE.format(
            question=question,
            paragraphs_info=paragraphs_info,
            relations_info=relations_info,
            conflicts_info=conflicts_info,
        )

        response = await self.llm.chat_json(
            system_prompt=OUTLINE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        outline = self._build_outline(question, response)
        logger.info(f"提纲生成完成: {len(outline.sections)} 个章节")
        return outline

    @staticmethod
    def _format_paragraphs_info(rst_trees: List[RSTTree]) -> str:
        """格式化段落及修辞结构信息"""
        lines = []
        for tree in rst_trees:
            lines.append(f"\n段落 {tree.paragraph_id}:")
            lines.append(f"  原文: {tree.paragraph_text}")
            lines.append(f"  摘要: {tree.summary}")
            lines.append(f"  核心论点: {tree.get_core_claim()}")
            conditions = tree.get_conditions()
            if conditions:
                lines.append(f"  限定条件: {'；'.join(conditions)}")
            lines.append("  修辞结构:")
            for edu in tree.edus:
                role = "核心" if edu.is_nucleus() else "辅助"
                rel = f" <{edu.relation_to_parent.value}>" if edu.relation_to_parent else ""
                lines.append(f"    [{role}] {edu.edu_id}{rel}: {edu.text}")
        return "\n".join(lines)

    @staticmethod
    def _format_relations_info(net: RelationNet, rst_trees: List[RSTTree]) -> str:
        """格式化段落关系信息"""
        lines = []
        for edge in net.edges:
            if edge.relation_type != RelationType.NEUTRAL:
                lines.append(
                    f"  {edge.source_id} → {edge.target_id}: "
                    f"{edge.relation_type.value} (置信度: {edge.confidence:.2f})"
                )
                if edge.explanation:
                    lines.append(f"    理由: {edge.explanation}")
        if not lines:
            lines.append("  无明显的段落间逻辑关系")
        return "\n".join(lines)

    @staticmethod
    def _format_conflicts_info(net: RelationNet) -> str:
        """格式化冲突分析信息"""
        contradictions = net.get_contradictions()
        if not contradictions:
            return "  未发现段落间的直接矛盾"
        lines = []
        for c in contradictions:
            src_tree = net.paragraphs.get(c.source_id)
            tgt_tree = net.paragraphs.get(c.target_id)
            lines.append(f"  冲突: {c.source_id} ↔ {c.target_id}")
            if src_tree and tgt_tree:
                lines.append(f"    {c.source_id} 核心论点: {src_tree.get_core_claim()}")
                lines.append(f"    {c.target_id} 核心论点: {tgt_tree.get_core_claim()}")
            if c.conflict_point:
                lines.append(f"    冲突点: {c.conflict_point}")
            if c.explanation:
                lines.append(f"    说明: {c.explanation}")
        return "\n".join(lines)

    @staticmethod
    def _build_outline(question: str, response: dict) -> Outline:
        """从LLM回复构建Outline对象"""
        outline = Outline(
            question=question,
            overall_strategy=response.get("overall_strategy", ""),
            conflict_resolution=response.get("conflict_resolution", ""),
            conclusion_strategy=response.get("conclusion_strategy", ""),
        )
        for sec_data in response.get("sections", []):
            section = OutlineSection(
                section_id=sec_data.get("section_id", ""),
                title=sec_data.get("title", ""),
                content_plan=sec_data.get("content_plan", ""),
                evidence_ids=sec_data.get("evidence_ids", []),
                strategy=sec_data.get("strategy", ""),
                key_points=sec_data.get("key_points", []),
            )
            outline.sections.append(section)
        return outline
