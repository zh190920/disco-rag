"""
段落关系网络构建模块

核心功能：
1. 对检索到的所有段落进行两两配对分析
2. 预测段落间的关系类型：支持、反驳、补充、无关
3. 构建有向关系图
4. 识别段落间的逻辑脉络和冲突点
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple

from disco_rag.llm_client import LLMClient
from disco_rag.rst_parser import RSTTree

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """段落间关系类型"""
    SUPPORT = "support"           # 支持：B段落支持A段落的论点
    CONTRADICT = "contradict"     # 反驳：B段落与A段落立场冲突
    SUPPLEMENT = "supplement"     # 补充：B段落补充A段落的信息
    ELABORATE = "elaborate"       # 详述：B段落详细展开A段落的内容
    CONDITION = "condition"       # 限定：B段落为A段落提供限定条件
    NEUTRAL = "neutral"           # 中立：B段落与A段落无直接逻辑关联


@dataclass
class ParagraphRelation:
    """段落间关系边"""
    source_id: str                    # 源段落ID
    target_id: str                    # 目标段落ID
    relation_type: RelationType       # 关系类型
    confidence: float                 # 置信度 (0-1)
    explanation: str = ""             # 关系说明
    conflict_point: str = ""          # 冲突点描述（仅CONTRADICT时有意义）

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "conflict_point": self.conflict_point,
        }


@dataclass
class RelationNet:
    """段落关系网络 - 有向图"""
    paragraphs: Dict[str, RSTTree] = field(default_factory=dict)
    edges: List[ParagraphRelation] = field(default_factory=list)

    def get_relations_of(self, paragraph_id: str) -> List[ParagraphRelation]:
        """获取与某个段落相关的所有关系"""
        return [
            e for e in self.edges
            if e.source_id == paragraph_id or e.target_id == paragraph_id
        ]

    def get_contradictions(self) -> List[ParagraphRelation]:
        """获取所有反驳/矛盾关系"""
        return [e for e in self.edges if e.relation_type == RelationType.CONTRADICT]

    def get_supports(self) -> List[ParagraphRelation]:
        """获取所有支持关系"""
        return [e for e in self.edges if e.relation_type == RelationType.SUPPORT]

    def get_supplements(self) -> List[ParagraphRelation]:
        """获取所有补充关系"""
        return [e for e in self.edges if e.relation_type == RelationType.SUPPLEMENT]

    def get_neighbors(self, paragraph_id: str) -> Dict[str, List[ParagraphRelation]]:
        """获取某个段落的邻接关系，按类型分组"""
        result: Dict[str, List[ParagraphRelation]] = {
            "outgoing": [],
            "incoming": [],
        }
        for e in self.edges:
            if e.source_id == paragraph_id:
                result["outgoing"].append(e)
            if e.target_id == paragraph_id:
                result["incoming"].append(e)
        return result

    def has_conflict_between(self, pid1: str, pid2: str) -> bool:
        """判断两个段落之间是否存在冲突"""
        for e in self.edges:
            if e.relation_type == RelationType.CONTRADICT:
                if (e.source_id == pid1 and e.target_id == pid2) or \
                   (e.source_id == pid2 and e.target_id == pid1):
                    return True
        return False

    def to_dict(self) -> dict:
        return {
            "paragraphs": {pid: tree.to_dict() for pid, tree in self.paragraphs.items()},
            "edges": [e.to_dict() for e in self.edges],
            "summary": {
                "total_paragraphs": len(self.paragraphs),
                "total_edges": len(self.edges),
                "contradictions": len(self.get_contradictions()),
                "supports": len(self.get_supports()),
                "supplements": len(self.get_supplements()),
            },
        }


# ============ LLM Prompt模板 ============

RELATION_ANALYSIS_SYSTEM_PROMPT = """你是一位学术文献分析专家，擅长识别不同文本段落之间的逻辑关系。
你的任务是分析两段文本之间的关系，判断它们是互相支持、互相反驳、互相补充还是无关。

关系类型定义：
- support（支持）：B段落的证据或论点支持A段落的结论
- contradict（反驳）：B段落的证据或论点与A段落形成矛盾或对立
- supplement（补充）：B段落提供额外信息，扩展A段落的内容，但不直接支持或反驳
- elaborate（详述）：B段落是对A段落内容的详细展开或具体说明
- condition（限定）：B段落为A段落的结论提供了限定条件或适用范围
- neutral（中立）：两段文字没有直接的逻辑关联

请严格按照JSON格式输出。"""

RELATION_ANALYSIS_USER_TEMPLATE = """请分析以下两段文本之间的逻辑关系：

段落A（编号：{source_id}）：
{source_text}

段落A核心论点：{source_claim}
段落A限定条件：{source_conditions}

段落B（编号：{target_id}）：
{target_text}

段落B核心论点：{target_claim}
段落B限定条件：{target_conditions}

请输出JSON格式：
{{
  "relation_type": "support/contradict/supplement/elaborate/condition/neutral",
  "confidence": 0.0到1.0之间的置信度,
  "explanation": "关系判断的详细理由（2-3句话）",
  "conflict_point": "如果是contradict，描述具体的矛盾点；否则为空字符串"
}}

注意：
1. 重点关注两段文本的核心论点是否一致或冲突
2. 注意限定条件的差异，同样结论在不同条件下可能互不矛盾
3. 置信度要反映你对关系判断的确定程度
4. 仅输出JSON，不要输出其他内容"""


class RelationNetBuilder:
    """
    段落关系网络构建器
    
    对所有检索到的段落进行两两配对分析，
    预测它们之间的逻辑关系，构建有向关系图。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def build(self, rst_trees: List[RSTTree]) -> RelationNet:
        """
        构建段落关系网络
        
        Args:
            rst_trees: 所有段落的RST解析树
        
        Returns:
            RelationNet: 段落关系网络
        """
        net = RelationNet()

        # 注册所有段落
        for tree in rst_trees:
            net.paragraphs[tree.paragraph_id] = tree

        # 两两配对分析
        paragraph_ids = [tree.paragraph_id for tree in rst_trees]
        total_pairs = len(paragraph_ids) * (len(paragraph_ids) - 1)
        analyzed = 0

        logger.info(f"开始分析 {total_pairs} 对段落关系...")

        for i, src_tree in enumerate(rst_trees):
            for j, tgt_tree in enumerate(rst_trees):
                if i == j:
                    continue

                relation = self._analyze_pair(src_tree, tgt_tree)
                if relation and relation.relation_type != RelationType.NEUTRAL:
                    net.edges.append(relation)

                analyzed += 1
                if analyzed % 5 == 0:
                    logger.info(f"已分析 {analyzed}/{total_pairs} 对段落关系")

        # 即使是neutral也记录（方便完整展示），但低置信度的neutral可以跳过
        # 重新遍历把高置信度的neutral也加入
        for i, src_tree in enumerate(rst_trees):
            for j, tgt_tree in enumerate(rst_trees):
                if i == j:
                    continue
                # 检查是否已经添加
                exists = any(
                    e.source_id == src_tree.paragraph_id and e.target_id == tgt_tree.paragraph_id
                    for e in net.edges
                )
                if not exists:
                    # 添加neutral关系
                    net.edges.append(ParagraphRelation(
                        source_id=src_tree.paragraph_id,
                        target_id=tgt_tree.paragraph_id,
                        relation_type=RelationType.NEUTRAL,
                        confidence=0.5,
                        explanation="两段落无直接逻辑关联",
                    ))

        logger.info(
            f"关系网络构建完成: {len(net.edges)} 条边, "
            f"{len(net.get_contradictions())} 个冲突, "
            f"{len(net.get_supports())} 个支持, "
            f"{len(net.get_supplements())} 个补充"
        )
        return net

    def _analyze_pair(self, source: RSTTree, target: RSTTree) -> Optional[ParagraphRelation]:
        """分析一对段落之间的关系"""
        user_prompt = RELATION_ANALYSIS_USER_TEMPLATE.format(
            source_id=source.paragraph_id,
            source_text=source.paragraph_text,
            source_claim=source.get_core_claim(),
            source_conditions="；".join(source.get_conditions()) if source.get_conditions() else "无明确限定条件",
            target_id=target.paragraph_id,
            target_text=target.paragraph_text,
            target_claim=target.get_core_claim(),
            target_conditions="；".join(target.get_conditions()) if target.get_conditions() else "无明确限定条件",
        )

        try:
            response = self.llm.chat_json(
                system_prompt=RELATION_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.05,
            )
            return self._parse_relation_response(
                source.paragraph_id, target.paragraph_id, response
            )
        except Exception as e:
            logger.warning(f"分析段落对 ({source.paragraph_id}, {target.paragraph_id}) 失败: {e}")
            return None

    @staticmethod
    def _parse_relation_response(
        source_id: str, target_id: str, response: dict
    ) -> ParagraphRelation:
        """从LLM回复解析段落关系"""
        rel_type_str = response.get("relation_type", "neutral").lower()
        relation_map = {r.value: r for r in RelationType}
        relation_type = relation_map.get(rel_type_str, RelationType.NEUTRAL)

        confidence = min(1.0, max(0.0, float(response.get("confidence", 0.5))))

        return ParagraphRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            explanation=response.get("explanation", ""),
            conflict_point=response.get("conflict_point", ""),
        )

    @staticmethod
    def format_net_visual(net: RelationNet) -> str:
        """以可视化方式展示关系网络"""
        lines = []
        lines.append("=" * 60)
        lines.append("段落关系网络")
        lines.append("=" * 60)

        # 统计信息
        summary = net.to_dict()["summary"]
        lines.append(f"\n📊 统计: {summary['total_paragraphs']} 个段落, {summary['total_edges']} 条关系边")
        lines.append(f"   冲突: {summary['contradictions']} | 支持: {summary['supports']} | 补充: {summary['supplements']}")

        # 按类型分组展示
        type_labels = {
            RelationType.CONTRADICT: "⚔️ 冲突/反驳",
            RelationType.SUPPORT: "✅ 支持",
            RelationType.SUPPLEMENT: "➕ 补充",
            RelationType.ELABORATE: "📝 详述",
            RelationType.CONDITION: "⚙️ 限定",
            RelationType.NEUTRAL: "➖ 中立",
        }

        for rel_type, label in type_labels.items():
            edges = [e for e in net.edges if e.relation_type == rel_type]
            if not edges:
                continue
            lines.append(f"\n{label}:")
            for e in edges:
                lines.append(f"  {e.source_id} → {e.target_id} (置信度: {e.confidence:.2f})")
                if e.explanation:
                    lines.append(f"    理由: {e.explanation}")
                if e.conflict_point:
                    lines.append(f"    ⚠️ 冲突点: {e.conflict_point}")

        return "\n".join(lines)
