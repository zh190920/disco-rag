"""
RST解析器 - 修辞结构理论（Rhetorical Structure Theory）解析模块

核心功能：
1. 将段落拆解为最小语义单元（EDU - Elementary Discourse Unit）
2. 标记每个单元是"核心内容"（Nucleus）还是"辅助说明"（Satellite）
3. 识别单元之间的修辞关系类型（因果、对比、展开、条件等）
4. 构建论证树结构
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from disco_rag.llm_client import LLMClient

logger = logging.getLogger(__name__)


class EDURole(Enum):
    """EDU的角色：核心 or 辅助"""
    NUCLEUS = "nucleus"       # 核心单元 - 段落的主要论点/结论
    SATELLITE = "satellite"   # 辅助单元 - 前提条件、限定、证据等


class RSTRelation(Enum):
    """RST修辞关系类型"""
    # 核心关系
    CAUSE = "cause"                    # 因果关系：A导致B
    RESULT = "result"                  # 结果关系：A的结果是B
    REASON = "reason"                  # 原因关系：A的原因是B
    EVIDENCE = "evidence"              # 证据关系：A为B提供证据
    EXPLANATION = "explanation"        # 解释关系：A解释B
    
    # 对比关系
    CONTRAST = "contrast"              # 对比关系：A与B形成对比
    CONCESSION = "concession"          # 让步关系：虽然A，但B
    ANTITHESIS = "antithesis"          # 对立关系：A与B对立
    
    # 条件关系
    CONDITION = "condition"            # 条件关系：如果A，则B
    OTHERWISE = "otherwise"            # 否则关系：如果不A，则B
    
    # 展开关系
    ELABORATION = "elaboration"        # 展开关系：A展开说明B
    ADDITIONAL = "additional"          # 补充关系：A补充B
    EXAMPLE = "example"                # 举例关系：A是B的例子
    
    # 时序关系
    SEQUENCE = "sequence"              # 时序关系：A先于B
    BACKGROUND = "background"          # 背景关系：A为B提供背景
    
    # 总结关系
    SUMMARY = "summary"                # 总结关系：A总结B
    RESTATEMENT = "restatement"        # 重述关系：A重述B
    
    # 其他
    ENABLEMENT = "enablement"          # 使能关系
    EVALUATION = "evaluation"          # 评价关系
    INTERPRETATION = "interpretation"  # 解释关系
    SOLUTIONHOOD = "solutionhood"      # 问题-解决方案关系
    NO_RELATION = "no_relation"        # 无明确关系


@dataclass
class EDU:
    """最小语义单元（Elementary Discourse Unit）"""
    edu_id: str                           # EDU编号，如 "E1"
    text: str                             # 原始文本
    role: EDURole                         # 核心or辅助
    relation_to_parent: Optional[RSTRelation] = None  # 与父节点的关系
    parent_id: Optional[str] = None       # 父节点ID
    children_ids: List[str] = field(default_factory=list)  # 子节点ID列表
    position: int = 0                     # 在段落中的位置

    def is_nucleus(self) -> bool:
        return self.role == EDURole.NUCLEUS

    def to_dict(self) -> dict:
        return {
            "edu_id": self.edu_id,
            "text": self.text,
            "role": self.role.value,
            "relation_to_parent": self.relation_to_parent.value if self.relation_to_parent else None,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "position": self.position,
        }


@dataclass
class RSTTree:
    """修辞结构树 - 一个段落的完整论证结构"""
    paragraph_id: str                     # 段落编号
    paragraph_text: str                   # 段落原始文本
    edus: List[EDU] = field(default_factory=list)   # 所有EDU列表
    root_id: Optional[str] = None         # 根节点ID
    summary: str = ""                     # 段落摘要

    def get_nucleus_edus(self) -> List[EDU]:
        """获取所有核心单元"""
        return [edu for edu in self.edus if edu.is_nucleus()]

    def get_satellite_edus(self) -> List[EDU]:
        """获取所有辅助单元"""
        return [edu for edu in self.edus if not edu.is_nucleus()]

    def get_edu_by_id(self, edu_id: str) -> Optional[EDU]:
        """根据ID获取EDU"""
        for edu in self.edus:
            if edu.edu_id == edu_id:
                return edu
        return None

    def get_core_claim(self) -> str:
        """获取段落的核心论点（所有核心单元的拼接）"""
        nucleus_edus = self.get_nucleus_edus()
        nucleus_edus.sort(key=lambda e: e.position)
        return " ".join([edu.text for edu in nucleus_edus])

    def get_conditions(self) -> List[str]:
        """获取段落的限定条件（辅助单元中的条件和背景）"""
        conditions = []
        for edu in self.get_satellite_edus():
            if edu.relation_to_parent in [
                RSTRelation.CONDITION,
                RSTRelation.BACKGROUND,
                RSTRelation.CONCESSION,
            ]:
                conditions.append(edu.text)
        return conditions

    def to_dict(self) -> dict:
        return {
            "paragraph_id": self.paragraph_id,
            "paragraph_text": self.paragraph_text,
            "edus": [edu.to_dict() for edu in self.edus],
            "root_id": self.root_id,
            "summary": self.summary,
            "core_claim": self.get_core_claim(),
            "conditions": self.get_conditions(),
        }


# ============ LLM Prompt模板 ============

RST_PARSE_SYSTEM_PROMPT = """你是一位修辞结构理论（Rhetorical Structure Theory, RST）分析专家。
你的任务是将输入的文本段落拆解为最小语义单元（EDU），并为每个单元标注：
1. 角色：核心（nucleus）还是辅助（satellite）
2. 与父节点的关系类型

核心单元（Nucleus）：段落的主要论点、结论或核心信息
辅助单元（Satellite）：前提条件、限定范围、证据、背景等支撑性信息

关系类型包括：
- cause: 因果关系（A导致B）
- result: 结果关系（A的结果是B）
- evidence: 证据关系（A为B提供证据）
- explanation: 解释关系（A解释B）
- contrast: 对比关系（A与B形成对比）
- concession: 让步关系（虽然A，但B）
- condition: 条件关系（如果A，则B）
- elaboration: 展开关系（A展开说明B）
- additional: 补充关系（A补充B）
- example: 举例关系（A是B的例子）
- background: 背景关系（A为B提供背景）
- summary: 总结关系
- sequence: 时序关系
- no_relation: 无明确关系

请严格按照JSON格式输出。"""

RST_PARSE_USER_TEMPLATE = """请分析以下段落的修辞结构：

段落编号：{paragraph_id}
段落内容：{paragraph_text}

请输出JSON格式，包含以下字段：
{{
  "summary": "段落的简要摘要（一句话）",
  "edus": [
    {{
      "edu_id": "E1",
      "text": "该语义单元的原文",
      "role": "nucleus或satellite",
      "relation_to_parent": "与父节点的关系类型（根节点为null）",
      "parent_id": "父节点EDU的id（根节点为null）"
    }},
    ...
  ],
  "root_id": "根节点EDU的id"
}}

注意事项：
1. 每个EDU应该是一个最小的、不可再分的语义单元
2. 核心论点/结论标记为nucleus，限定条件和支撑信息标记为satellite
3. 关系类型要准确反映单元之间的逻辑关系
4. 确保所有EDU构成一棵完整的树结构
5. 仅输出JSON，不要输出其他内容"""


class RSTParser:
    """
    RST修辞结构解析器
    
    利用LLM将段落拆解为EDU，构建论证树，
    区分核心论点和辅助条件。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def parse_paragraph(self, paragraph_id: str, paragraph_text: str) -> RSTTree:
        """
        解析单个段落的修辞结构，生成论证树
        
        Args:
            paragraph_id: 段落编号
            paragraph_text: 段落文本
        
        Returns:
            RSTTree: 段落的修辞结构树
        """
        logger.info(f"解析段落 {paragraph_id}: {paragraph_text[:50]}...")

        user_prompt = RST_PARSE_USER_TEMPLATE.format(
            paragraph_id=paragraph_id,
            paragraph_text=paragraph_text,
        )

        response = self.llm.chat_json(
            system_prompt=RST_PARSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.05,  # 低温度确保结构化输出稳定
        )

        tree = self._build_tree_from_response(paragraph_id, paragraph_text, response)
        logger.info(
            f"段落 {paragraph_id} 解析完成: {len(tree.edus)} 个EDU, "
            f"{len(tree.get_nucleus_edus())} 个核心, {len(tree.get_satellite_edus())} 个辅助"
        )
        return tree

    def parse_paragraphs(self, paragraphs: List[tuple]) -> List[RSTTree]:
        """
        批量解析多个段落
        
        Args:
            paragraphs: [(paragraph_id, paragraph_text), ...] 列表
        
        Returns:
            RSTTree列表
        """
        trees = []
        for pid, ptext in paragraphs:
            tree = self.parse_paragraph(pid, ptext)
            trees.append(tree)
        return trees

    def _build_tree_from_response(
        self, paragraph_id: str, paragraph_text: str, response: dict
    ) -> RSTTree:
        """从LLM的JSON回复构建RSTTree对象"""
        tree = RSTTree(
            paragraph_id=paragraph_id,
            paragraph_text=paragraph_text,
        )

        # 提取摘要
        tree.summary = response.get("summary", "")

        # 解析EDU列表
        edus_data = response.get("edus", [])
        for i, edu_data in enumerate(edus_data):
            role = EDURole.NUCLEUS
            if edu_data.get("role", "").lower() == "satellite":
                role = EDURole.SATELLITE

            # 解析关系类型
            relation = None
            rel_str = edu_data.get("relation_to_parent")
            if rel_str and rel_str.lower() != "null":
                relation = self._parse_relation(rel_str)

            edu = EDU(
                edu_id=edu_data.get("edu_id", f"E{i+1}"),
                text=edu_data.get("text", ""),
                role=role,
                relation_to_parent=relation,
                parent_id=edu_data.get("parent_id") if edu_data.get("parent_id") and edu_data.get("parent_id").lower() != "null" else None,
                position=i,
            )
            tree.edus.append(edu)

        # 设置根节点
        root_id = response.get("root_id")
        if root_id and root_id.lower() != "null":
            tree.root_id = root_id
        elif tree.edus:
            # 尝试找到没有父节点的EDU作为根
            for edu in tree.edus:
                if edu.parent_id is None:
                    tree.root_id = edu.edu_id
                    break
            if tree.root_id is None and tree.edus:
                tree.root_id = tree.edus[0].edu_id

        # 建立children关系
        for edu in tree.edus:
            if edu.parent_id:
                parent = tree.get_edu_by_id(edu.parent_id)
                if parent and edu.edu_id not in parent.children_ids:
                    parent.children_ids.append(edu.edu_id)

        return tree

    @staticmethod
    def _parse_relation(rel_str: str) -> RSTRelation:
        """将字符串解析为RSTRelation枚举"""
        rel_str = rel_str.lower().strip()
        relation_map = {r.value: r for r in RSTRelation}
        return relation_map.get(rel_str, RSTRelation.NO_RELATION)

    @staticmethod
    def format_tree_visual(tree: RSTTree, indent: int = 0) -> str:
        """以可视化方式打印论证树"""
        lines = []
        prefix = "  " * indent

        def _print_node(edu_id: str, level: int):
            edu = tree.get_edu_by_id(edu_id)
            if not edu:
                return
            p = "  " * level
            role_marker = "[核心]" if edu.is_nucleus() else "[辅助]"
            relation_info = f" <{edu.relation_to_parent.value}>" if edu.relation_to_parent else ""
            lines.append(f"{p}{role_marker} {edu.edu_id}{relation_info}: {edu.text}")
            for child_id in edu.children_ids:
                _print_node(child_id, level + 1)

        # 从根节点开始打印
        if tree.root_id:
            _print_node(tree.root_id, 0)
        else:
            # 没有根节点，按位置顺序打印
            for edu in tree.edus:
                role_marker = "[核心]" if edu.is_nucleus() else "[辅助]"
                relation_info = f" <{edu.relation_to_parent.value}>" if edu.relation_to_parent else ""
                lines.append(f"{prefix}{role_marker} {edu.edu_id}{relation_info}: {edu.text}")

        return "\n".join(lines)
