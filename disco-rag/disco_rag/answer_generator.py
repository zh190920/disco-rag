"""
答案生成模块

核心功能：
1. 基于写作提纲，引导模型产出最终回答
2. 确保回答有层次、有条件、有依据
3. 正确处理矛盾信息，给出有条件的综合判断
"""

import logging
from typing import List, Optional

from disco_rag.llm_client import LLMClient
from disco_rag.rst_parser import RSTTree
from disco_rag.relation_net import RelationNet
from disco_rag.outline_generator import Outline

logger = logging.getLogger(__name__)


# ============ LLM Prompt模板 ============

ANSWER_SYSTEM_PROMPT = """你是一位严谨的学术分析助手。你必须按照给定的写作提纲来组织你的回答。

核心原则：
1. 严格遵循提纲的章节结构和写作策略
2. 引用证据时要准确，不能歪曲原文意思
3. 限定条件必须明确标注，不能把有条件的结论当作普遍事实
4. 对于矛盾的文献，要客观呈现，分析原因，给出有条件的判断
5. 回答要有层次、有条件、有依据，避免简单粗暴的是/否结论
6. 使用专业但易懂的语言"""

ANSWER_USER_TEMPLATE = """请根据以下写作提纲，撰写最终回答。

【用户问题】
{question}

【写作提纲】
{outline_text}

【可引用的原文段落】
{evidence_text}

【写作要求】
1. 严格按照提纲的章节顺序组织回答
2. 每个章节都要落实提纲中的内容规划和关键要点
3. 引用证据时，注明来源段落编号
4. 限定条件要明确标注（如"在...条件下"、"仅限于..."等）
5. 对于矛盾信息，要客观呈现不同观点，分析可能的原因
6. 结论部分要给出有条件的综合判断
7. 回答要完整、专业、有逻辑"""


class AnswerGenerator:
    """
    答案生成器
    
    基于写作提纲，引导LLM产出高质量的结构化回答。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(
        self,
        question: str,
        outline: Outline,
        rst_trees: List[RSTTree],
        relation_net: RelationNet,
    ) -> str:
        """
        生成最终答案
        
        Args:
            question: 用户问题
            outline: 写作提纲
            rst_trees: RST解析树列表
            relation_net: 段落关系网络
        
        Returns:
            str: 最终生成的答案
        """
        logger.info("开始生成最终答案...")

        # 构建提纲文本
        outline_text = self._format_outline(outline)

        # 构建证据文本
        evidence_text = self._format_evidence(rst_trees, relation_net)

        user_prompt = ANSWER_USER_TEMPLATE.format(
            question=question,
            outline_text=outline_text,
            evidence_text=evidence_text,
        )

        answer = self.llm.chat(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,  # 稍高温度让回答更自然
        )

        logger.info(f"答案生成完成，长度: {len(answer)} 字符")
        return answer

    def _format_outline(self, outline: Outline) -> str:
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

    def _format_evidence(self, rst_trees: List[RSTTree], relation_net: RelationNet) -> str:
        """格式化可引用的证据文本"""
        lines = []
        for tree in rst_trees:
            lines.append(f"\n[段落 {tree.paragraph_id}]")
            lines.append(f"原文: {tree.paragraph_text}")
            lines.append(f"核心论点: {tree.get_core_claim()}")
            conditions = tree.get_conditions()
            if conditions:
                lines.append(f"⚠️ 限定条件: {'；'.join(conditions)}")

            # 标注该段落与其他段落的关系
            relations = relation_net.get_relations_of(tree.paragraph_id)
            for rel in relations:
                if rel.relation_type.value != "neutral":
                    direction = "→" if rel.source_id == tree.paragraph_id else "←"
                    other_id = rel.target_id if rel.source_id == tree.paragraph_id else rel.source_id
                    lines.append(f"🔗 关系 {direction} 段落{other_id}: {rel.relation_type.value} - {rel.explanation}")

        return "\n".join(lines)
