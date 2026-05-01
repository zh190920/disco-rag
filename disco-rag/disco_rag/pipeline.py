"""
Disco-RAG 主流程编排

完整的三步流程：
1. 给每个段落画一棵"论证树"（RST解析）
2. 给所有段落织一张"关系网"（段落间关系图）
3. 先列提纲，再写答案（提纲引导生成）

支持：
- 完整流程运行
- 逐步调试查看中间结果
- 与传统RAG对比
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from disco_rag.llm_client import LLMClient, LLMConfig
from disco_rag.rst_parser import RSTParser, RSTTree
from disco_rag.relation_net import RelationNetBuilder, RelationNet
from disco_rag.outline_generator import OutlineGenerator, Outline
from disco_rag.answer_generator import AnswerGenerator
from disco_rag.retriever import SimpleRetriever, Document

logger = logging.getLogger(__name__)


@dataclass
class DiscoRAGResult:
    """Disco-RAG完整运行结果"""
    question: str
    rst_trees: List[RSTTree] = field(default_factory=list)
    relation_net: Optional[RelationNet] = None
    outline: Optional[Outline] = None
    answer: str = ""
    traditional_answer: str = ""  # 传统RAG的答案（用于对比）
    timing: Dict[str, float] = field(default_factory=dict)

    def save_to_file(self, filepath: str):
        """将结果保存到JSON文件"""
        data = {
            "question": self.question,
            "answer": self.answer,
            "traditional_answer": self.traditional_answer,
            "timing": self.timing,
            "rst_trees": [tree.to_dict() for tree in self.rst_trees],
            "relation_net": self.relation_net.to_dict() if self.relation_net else None,
            "outline": self.outline.to_dict() if self.outline else None,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def format_report(self) -> str:
        """格式化完整报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("  Disco-RAG 运行报告")
        lines.append("=" * 70)

        # 问题
        lines.append(f"\n📌 用户问题: {self.question}")

        # 耗时
        lines.append("\n⏱️ 各阶段耗时:")
        for stage, t in self.timing.items():
            lines.append(f"   {stage}: {t:.2f}s")

        # Step 1: 论证树
        lines.append("\n" + "=" * 70)
        lines.append("  Step 1: 论证树（RST解析）")
        lines.append("=" * 70)
        for tree in self.rst_trees:
            lines.append(f"\n段落 {tree.paragraph_id}: {tree.paragraph_text[:60]}...")
            lines.append(f"  核心论点: {tree.get_core_claim()}")
            conditions = tree.get_conditions()
            if conditions:
                lines.append(f"  限定条件: {'；'.join(conditions)}")
            lines.append(RSTParser.format_tree_visual(tree, indent=2))

        # Step 2: 关系网
        if self.relation_net:
            lines.append("\n" + "=" * 70)
            lines.append("  Step 2: 段落关系网络")
            lines.append("=" * 70)
            lines.append(RelationNetBuilder.format_net_visual(self.relation_net))

        # Step 3: 写作提纲
        if self.outline:
            lines.append("\n" + "=" * 70)
            lines.append("  Step 3: 写作提纲")
            lines.append("=" * 70)
            lines.append(OutlineGenerator.format_outline_visual(self.outline))

        # 最终答案
        lines.append("\n" + "=" * 70)
        lines.append("  Disco-RAG 最终答案")
        lines.append("=" * 70)
        lines.append(f"\n{self.answer}")

        # 对比传统RAG
        if self.traditional_answer:
            lines.append("\n" + "=" * 70)
            lines.append("  传统RAG答案（对比）")
            lines.append("=" * 70)
            lines.append(f"\n{self.traditional_answer}")

        return "\n".join(lines)


class DiscoRAGPipeline:
    """
    Disco-RAG 完整流程管线
    
    三步走：
    1. RST解析 → 论证树
    2. 关系分析 → 关系网络
    3. 提纲+生成 → 最终答案
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[SimpleRetriever] = None,
    ):
        # 初始化LLM客户端
        if llm_client:
            self.llm = llm_client
        elif llm_config:
            self.llm = LLMClient(llm_config)
        else:
            self.llm = LLMClient(LLMConfig())

        # 初始化各模块
        self.rst_parser = RSTParser(self.llm)
        self.relation_builder = RelationNetBuilder(self.llm)
        self.outline_generator = OutlineGenerator(self.llm)
        self.answer_generator = AnswerGenerator(self.llm)

        # 检索器
        self.retriever = retriever or SimpleRetriever()

    def run(
        self,
        question: str,
        paragraphs: Optional[List[str]] = None,
        top_k: int = 5,
        compare_traditional: bool = True,
        verbose: bool = True,
    ) -> DiscoRAGResult:
        """
        运行完整的Disco-RAG流程
        
        Args:
            question: 用户问题
            paragraphs: 预设的段落列表（如果为None，则从检索器搜索）
            top_k: 检索返回的段落数量
            compare_traditional: 是否同时生成传统RAG答案用于对比
            verbose: 是否打印中间结果
        
        Returns:
            DiscoRAGResult: 完整运行结果
        """
        result = DiscoRAGResult(question=question)
        timing = {}

        # ======== 获取段落 ========
        if paragraphs is None:
            logger.info("从检索器搜索相关段落...")
            search_results = self.retriever.search(question, top_k=top_k)
            paragraphs = [doc.content for doc, score in search_results]
            if not paragraphs:
                result.answer = "抱歉，未找到与您问题相关的文档。"
                return result

        paragraph_items = [(f"P{i+1}", p) for i, p in enumerate(paragraphs)]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Disco-RAG 处理流程")
            print(f"{'='*60}")
            print(f"\n📌 问题: {question}")
            print(f"📄 检索到 {len(paragraphs)} 个段落")
            for pid, ptext in paragraph_items:
                print(f"  [{pid}]: {ptext[:80]}...")

        # ======== Step 1: RST解析 - 构建论证树 ========
        if verbose:
            print(f"\n{'─'*60}")
            print("  Step 1/3: 构建论证树（RST解析）...")
            print(f"{'─'*60}")

        t0 = time.time()
        rst_trees = self.rst_parser.parse_paragraphs(paragraph_items)
        timing["Step1_RST解析"] = time.time() - t0
        result.rst_trees = rst_trees

        if verbose:
            for tree in rst_trees:
                print(f"\n  📖 段落 {tree.paragraph_id}:")
                print(f"     核心论点: {tree.get_core_claim()}")
                conditions = tree.get_conditions()
                if conditions:
                    print(f"     限定条件: {'；'.join(conditions)}")
                print(RSTParser.format_tree_visual(tree, indent=4))

        # ======== Step 2: 构建关系网络 ========
        if verbose:
            print(f"\n{'─'*60}")
            print("  Step 2/3: 构建段落关系网络...")
            print(f"{'─'*60}")

        t0 = time.time()
        relation_net = self.relation_builder.build(rst_trees)
        timing["Step2_关系网络"] = time.time() - t0
        result.relation_net = relation_net

        if verbose:
            print(RelationNetBuilder.format_net_visual(relation_net))

        # ======== Step 3: 生成提纲 + 写答案 ========
        if verbose:
            print(f"\n{'─'*60}")
            print("  Step 3/3: 生成写作提纲 + 撰写答案...")
            print(f"{'─'*60}")

        # 3a. 生成提纲
        t0 = time.time()
        outline = self.outline_generator.generate(question, rst_trees, relation_net)
        timing["Step3a_提纲生成"] = time.time() - t0
        result.outline = outline

        if verbose:
            print(OutlineGenerator.format_outline_visual(outline))

        # 3b. 生成答案
        t0 = time.time()
        answer = self.answer_generator.generate(question, outline, rst_trees, relation_net)
        timing["Step3b_答案生成"] = time.time() - t0
        result.answer = answer

        if verbose:
            print(f"\n{'='*60}")
            print("  ✅ Disco-RAG 最终答案")
            print(f"{'='*60}")
            print(f"\n{answer}")

        # ======== 对比传统RAG ========
        if compare_traditional:
            if verbose:
                print(f"\n{'─'*60}")
                print("  对比: 传统RAG答案...")
                print(f"{'─'*60}")

            t0 = time.time()
            traditional_answer = self._generate_traditional_rag(question, paragraphs)
            timing["传统RAG"] = time.time() - t0
            result.traditional_answer = traditional_answer

            if verbose:
                print(f"\n  📋 传统RAG答案:\n")
                print(traditional_answer)

        # 总结耗时
        timing["总耗时"] = sum(timing.values())
        result.timing = timing

        if verbose:
            print(f"\n{'='*60}")
            print("  ⏱️ 耗时统计")
            print(f"{'='*60}")
            for stage, t in timing.items():
                print(f"   {stage}: {t:.2f}s")

        return result

    def _generate_traditional_rag(self, question: str, paragraphs: List[str]) -> str:
        """生成传统RAG的答案（直接拼接段落，不加分析）"""
        context = "\n\n".join([f"段落{i+1}: {p}" for i, p in enumerate(paragraphs)])

        system_prompt = """你是一位知识问答助手。请根据提供的参考信息回答用户问题。
直接基于参考信息作答，如果参考信息不足以回答问题，请如实说明。"""

        user_prompt = f"""参考信息：
{context}

用户问题：{question}

请根据以上参考信息回答问题。"""

        return self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
        )

    # ======== 便捷方法 ========

    def add_documents(self, documents: List[Document]):
        """添加文档到检索库"""
        self.retriever.add_documents(documents)

    def ask(self, question: str, top_k: int = 5, **kwargs) -> str:
        """
        便捷方法：提问并获取答案
        
        Args:
            question: 用户问题
            top_k: 检索返回的段落数量
        
        Returns:
            str: 最终答案
        """
        result = self.run(question=question, top_k=top_k, **kwargs)
        return result.answer
