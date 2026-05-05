"""
异步段落关系网络构建器 - 高性能版

核心优化：
1. 所有段落配对的关系分析并发执行（配对级并行）
2. 使用batch_chat_json批量发送
3. 去除neutral关系的冗余LLM调用（本地判断）

性能提升：
- 4个段落(12对)：串行 ~24s → 并行 ~4s（6x+）
- 8个段落(56对)：串行 ~112s → 并行 ~10s（11x+）

额外优化：先快速预筛，只对可能有关系的段落对调用LLM
"""

import asyncio
import logging
from typing import List, Optional, Set, Tuple
from itertools import combinations

from disco_rag.rst_parser import RSTTree
from disco_rag.relation_net import RelationNet, ParagraphRelation, RelationType
from disco_rag.async_core.async_llm_client import AsyncLLMClient

logger = logging.getLogger(__name__)

# 复用同步版的Prompt模板
from disco_rag.relation_net import (
    RELATION_ANALYSIS_SYSTEM_PROMPT,
    RELATION_ANALYSIS_USER_TEMPLATE,
)


class AsyncRelationNetBuilder:
    """
    异步关系网络构建器

    并发策略：
    ┌──────────────────────────────────────────────┐
    │ 4个段落 = 12个有序配对                        │
    │                                              │
    │ 串行版本:                                    │
    │  P1→P2 ──→ P1→P3 ──→ P1→P4 ──→ P2→P1 ──→  │
    │  P2→P3 ──→ P2→P4 ──→ P3→P1 ──→ ... ──→    │
    │  ═══════════════════════════════════  ~24s   │
    │                                              │
    │ 并发版本:                                    │
    │  P1→P2 ──→ ┐                                 │
    │  P1→P3 ──→ │                                 │
    │  P1→P4 ──→ ├─→ 全部完成                      │
    │  P2→P1 ──→ │   ════════════════════  ~4s    │
    │  ...     ──→ ┘                                │
    │                                              │
    │  12个配对同时发送LLM请求，                    │
    │  总耗时 = max(单个耗时) + 调度开销            │
    └──────────────────────────────────────────────┘

    快速预筛优化：
    ┌──────────────────────────────────────────────┐
    │ 步骤1: 用关键词/核心论点相似度快速判断         │
    │        → 明显无关的配对直接标为neutral         │
    │        → 省去LLM调用                         │
    │                                              │
    │ 步骤2: 只对"可能有关系"的配对调用LLM          │
    │        → 减少API调用量和费用                  │
    │                                              │
    │ 示例: 12个配对 → 预筛后只需分析6个            │
    │       → API调用减少50%                       │
    └──────────────────────────────────────────────┘
    """

    def __init__(self, llm_client: AsyncLLMClient, enable_prefilter: bool = True):
        self.llm = llm_client
        self.enable_prefilter = enable_prefilter

    async def build(self, rst_trees: List[RSTTree]) -> RelationNet:
        """并发构建段落关系网络"""
        net = RelationNet()

        # 注册所有段落
        for tree in rst_trees:
            net.paragraphs[tree.paragraph_id] = tree

        # 生成所有配对
        pairs = []
        for i, src in enumerate(rst_trees):
            for j, tgt in enumerate(rst_trees):
                if i != j:
                    pairs.append((src, tgt))

        logger.info(f"分析 {len(pairs)} 对段落关系（并发）...")

        # 快速预筛
        if self.enable_prefilter:
            pairs_to_analyze, neutral_pairs = self._prefilter(pairs)
            logger.info(
                f"预筛结果: {len(pairs_to_analyze)} 对需分析, "
                f"{len(neutral_pairs)} 对直接标为neutral"
            )
        else:
            pairs_to_analyze = pairs
            neutral_pairs = []

        # 批量并发分析
        if pairs_to_analyze:
            tasks = []
            for src, tgt in pairs_to_analyze:
                user_prompt = RELATION_ANALYSIS_USER_TEMPLATE.format(
                    source_id=src.paragraph_id,
                    source_text=src.paragraph_text,
                    source_claim=src.get_core_claim(),
                    source_conditions="；".join(src.get_conditions()) if src.get_conditions() else "无明确限定条件",
                    target_id=tgt.paragraph_id,
                    target_text=tgt.paragraph_text,
                    target_claim=tgt.get_core_claim(),
                    target_conditions="；".join(tgt.get_conditions()) if tgt.get_conditions() else "无明确限定条件",
                )
                tasks.append({
                    "system_prompt": RELATION_ANALYSIS_SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                })

            responses = await self.llm.batch_chat_json(tasks, temperature=0.05)

            # 批量构建关系边
            for (src, tgt), response in zip(pairs_to_analyze, responses):
                relation = self._parse_relation(src.paragraph_id, tgt.paragraph_id, response)
                net.edges.append(relation)

        # 添加预筛出的neutral关系
        for src, tgt in neutral_pairs:
            net.edges.append(ParagraphRelation(
                source_id=src.paragraph_id,
                target_id=tgt.paragraph_id,
                relation_type=RelationType.NEUTRAL,
                confidence=0.3,
                explanation="预筛判断为无关",
            ))

        # 对于分析过的配对，如果结果不是neutral也补全反向neutral
        analyzed_pairs = set()
        for e in net.edges:
            if e.relation_type != RelationType.NEUTRAL:
                analyzed_pairs.add((e.source_id, e.target_id))

        for tree in rst_trees:
            for other_tree in rst_trees:
                if tree.paragraph_id == other_tree.paragraph_id:
                    continue
                pair_key = (tree.paragraph_id, other_tree.paragraph_id)
                if pair_key not in analyzed_pairs:
                    # 检查是否已有任何关系
                    exists = any(
                        e.source_id == pair_key[0] and e.target_id == pair_key[1]
                        for e in net.edges
                    )
                    if not exists:
                        net.edges.append(ParagraphRelation(
                            source_id=pair_key[0],
                            target_id=pair_key[1],
                            relation_type=RelationType.NEUTRAL,
                            confidence=0.3,
                            explanation="未分析",
                        ))

        logger.info(
            f"关系网络构建完成: {len(net.edges)} 条边, "
            f"{len(net.get_contradictions())} 冲突, "
            f"{len(net.get_supports())} 支持"
        )
        return net

    def _prefilter(
        self, pairs: List[Tuple[RSTTree, RSTTree]]
    ) -> Tuple[List[Tuple[RSTTree, RSTTree]], List[Tuple[RSTTree, RSTTree]]]:
        """
        快速预筛：基于关键词重叠判断是否需要LLM分析

        原理：如果两个段落的核心论点几乎没有共同关键词，
        那它们之间很可能是neutral关系，不需要浪费LLM调用。

        只需要对有一定关联的段落对调用LLM做精细判断。
        """
        to_analyze = []
        neutral = []

        for src, tgt in pairs:
            # 计算核心论点的关键词重叠
            overlap_score = self._keyword_overlap(
                src.get_core_claim(), tgt.get_core_claim()
            )
            # 也考虑段落原文的重叠
            text_overlap = self._keyword_overlap(
                src.paragraph_text, tgt.paragraph_text
            )
            # 综合分数
            combined = max(overlap_score, text_overlap * 0.5)

            if combined > 0.1:  # 有一定关联，需要LLM精细分析
                to_analyze.append((src, tgt))
            else:
                neutral.append((src, tgt))

        return to_analyze, neutral

    @staticmethod
    def _keyword_overlap(text1: str, text2: str) -> float:
        """计算两个文本的关键词重叠度（Jaccard系数）"""
        def tokenize(text):
            tokens = set()
            # 中文按字/双字切分
            i = 0
            while i < len(text):
                if '\u4e00' <= text[i] <= '\u9fff':
                    tokens.add(text[i])
                    if i + 1 < len(text) and '\u4e00' <= text[i+1] <= '\u9fff':
                        tokens.add(text[i:i+2])
                    i += 1
                elif text[i].isalpha():
                    # 英文单词
                    j = i
                    while j < len(text) and text[j].isalpha():
                        j += 1
                    tokens.add(text[i:j].lower())
                    i = j
                else:
                    i += 1
            return tokens

        set1 = tokenize(text1)
        set2 = tokenize(text2)
        if not set1 or not set2:
            return 0.0
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _parse_relation(source_id: str, target_id: str, response: dict) -> ParagraphRelation:
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
