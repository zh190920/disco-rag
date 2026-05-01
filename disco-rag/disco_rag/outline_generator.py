"""
写作提纲生成模块

核心功能：
1. 综合用户提问、原始段落、论证树和关系网
2. 自动生成一份"写作提纲"
3. 提纲标明要引用的关键证据、叙述的先后顺序、以及如何协调矛盾信息
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from disco_rag.llm_client import LLMClient
from disco_rag.rst_parser import RSTTree
from disco_rag.relation_net import RelationNet, RelationType

logger = logging.getLogger(__name__)


@dataclass
class OutlineSection:
    """提纲的一个章节"""
    section_id: str                     # 章节编号，如 "1", "2.1"
    title: str                          # 章节标题
    content_plan: str                   # 内容规划（要写什么）
    evidence_ids: List[str] = field(default_factory=list)  # 要引用的段落/EDU ID
    strategy: str = ""                  # 写作策略说明
    key_points: List[str] = field(default_factory=list)    # 要涵盖的关键点

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content_plan": self.content_plan,
            "evidence_ids": self.evidence_ids,
            "strategy": self.strategy,
            "key_points": self.key_points,
        }


@dataclass
class Outline:
    """写作提纲"""
    question: str                           # 用户问题
    sections: List[OutlineSection] = field(default_factory=list)
    overall_strategy: str = ""              # 整体写作策略
    conflict_resolution: str = ""           # 矛盾协调方案
    conclusion_strategy: str = ""           # 结论写作策略

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "overall_strategy": self.overall_strategy,
            "conflict_resolution": self.conflict_resolution,
            "conclusion_strategy": self.conclusion_strategy,
            "sections": [s.to_dict() for s in self.sections],
        }


# ============ LLM Prompt模板 ============

OUTLINE_SYSTEM_PROMPT = """你是一位学术写作规划专家。你的任务是根据用户的提问、检索到的文献段落及其修辞结构分析结果，制定一份详细的写作提纲。

写作提纲的核心原则：
1. 先呈现各方证据，再进行综合分析
2. 对于存在矛盾的文献，要明确指出矛盾所在，分析可能的原因（如研究对象、方法、条件差异等）
3. 区分普适性结论和有条件的结论，不能把限定条件下的结论当作普遍事实
4. 给出有层次、有条件、有依据的综合判断
5. 提纲要具体到每个段落应该写什么内容、引用哪些证据

请严格按照JSON格式输出。"""

OUTLINE_USER_TEMPLATE = """请根据以下信息，制定一份详细的写作提纲：

【用户问题】
{question}

【检索到的文献段落及修辞结构】
{paragraphs_info}

【段落间关系网络】
{relations_info}

【冲突分析】
{conflicts_info}

请输出JSON格式：
{{
  "overall_strategy": "整体写作策略（2-3句话描述如何组织答案）",
  "conflict_resolution": "如何协调矛盾信息的方案（如有矛盾）",
  "conclusion_strategy": "结论部分的写作策略",
  "sections": [
    {{
      "section_id": "1",
      "title": "章节标题",
      "content_plan": "这个章节要写什么内容（详细描述）",
      "evidence_ids": ["段落编号如P1", "段落编号如P2"],
      "strategy": "这个章节的写作策略说明",
      "key_points": ["要点1", "要点2"]
    }},
    ...
  ]
}}

注意事项：
1. 提纲要覆盖所有重要的证据和观点
2. 对于矛盾的段落，要安排专门的章节进行分析和协调
3. 限定条件要明确标注，不能忽略
4. 结论要基于证据给出有条件的判断，而非简单的是/否
5. 仅输出JSON，不要输出其他内容"""


class OutlineGenerator:
    """
    写作提纲生成器
    
    综合用户问题、段落修辞结构、段落关系网络，
    生成一份指导最终回答的写作提纲。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(
        self,
        question: str,
        rst_trees: List[RSTTree],
        relation_net: RelationNet,
    ) -> Outline:
        """
        生成写作提纲
        
        Args:
            question: 用户问题
            rst_trees: RST解析树列表
            relation_net: 段落关系网络
        
        Returns:
            Outline: 写作提纲
        """
        logger.info(f"开始生成写作提纲，问题: {question[:50]}...")

        # 构建段落信息文本
        paragraphs_info = self._format_paragraphs_info(rst_trees)

        # 构建关系网络文本
        relations_info = self._format_relations_info(relation_net, rst_trees)

        # 构建冲突分析文本
        conflicts_info = self._format_conflicts_info(relation_net)

        user_prompt = OUTLINE_USER_TEMPLATE.format(
            question=question,
            paragraphs_info=paragraphs_info,
            relations_info=relations_info,
            conflicts_info=conflicts_info,
        )

        response = self.llm.chat_json(
            system_prompt=OUTLINE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        outline = self._build_outline_from_response(question, response)
        logger.info(f"写作提纲生成完成: {len(outline.sections)} 个章节")
        return outline

    def _format_paragraphs_info(self, rst_trees: List[RSTTree]) -> str:
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

    def _format_relations_info(self, net: RelationNet, rst_trees: List[RSTTree]) -> str:
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

    def _format_conflicts_info(self, net: RelationNet) -> str:
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
    def _build_outline_from_response(question: str, response: dict) -> Outline:
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

    @staticmethod
    def format_outline_visual(outline: Outline) -> str:
        """以可视化方式展示提纲"""
        lines = []
        lines.append("=" * 60)
        lines.append("📝 写作提纲")
        lines.append("=" * 60)
        lines.append(f"\n问题: {outline.question}")
        lines.append(f"\n整体策略: {outline.overall_strategy}")

        if outline.conflict_resolution:
            lines.append(f"\n矛盾协调方案: {outline.conflict_resolution}")

        if outline.conclusion_strategy:
            lines.append(f"\n结论策略: {outline.conclusion_strategy}")

        lines.append("\n" + "-" * 40)
        for section in outline.sections:
            lines.append(f"\n{section.section_id}. {section.title}")
            lines.append(f"   内容规划: {section.content_plan}")
            if section.evidence_ids:
                lines.append(f"   引用证据: {', '.join(section.evidence_ids)}")
            if section.strategy:
                lines.append(f"   写作策略: {section.strategy}")
            if section.key_points:
                lines.append(f"   关键要点:")
                for kp in section.key_points:
                    lines.append(f"     • {kp}")

        return "\n".join(lines)
