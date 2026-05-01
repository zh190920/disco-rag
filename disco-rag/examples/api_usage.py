"""
Disco-RAG API 编程接口使用示例

展示如何通过Python代码调用Disco-RAG的各个模块。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disco_rag.llm_client import LLMClient, LLMConfig
from disco_rag.rst_parser import RSTParser, RSTTree
from disco_rag.relation_net import RelationNetBuilder, RelationNet, RelationType
from disco_rag.outline_generator import OutlineGenerator, Outline
from disco_rag.answer_generator import AnswerGenerator
from disco_rag.pipeline import DiscoRAGPipeline


def example_step_by_step():
    """逐步调用各模块的示例"""
    print("=" * 60)
    print("  Disco-RAG 逐步调用示例")
    print("=" * 60)

    # 1. 配置LLM
    config = LLMConfig(
        api_base=os.environ.get("DISCO_RAG_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=os.environ.get("DISCO_RAG_API_KEY", ""),
        model_name=os.environ.get("DISCO_RAG_MODEL", "glm-4-flash"),
    )
    llm = LLMClient(config)

    # 2. 准备段落
    paragraphs = [
        ("P1", "在冬季维生素D水平偏低的成年人群中，额外补充维生素D后流感发病率下降了12%。"),
        ("P2", "大规模随机对照试验未发现维生素D补充与流感风险之间存在统计学上的显著关联。"),
    ]

    # 3. Step 1: RST解析
    print("\n📖 Step 1: RST解析...")
    rst_parser = RSTParser(llm)
    rst_trees = rst_parser.parse_paragraphs(paragraphs)

    for tree in rst_trees:
        print(f"\n  段落 {tree.paragraph_id}:")
        print(f"    核心论点: {tree.get_core_claim()}")
        conditions = tree.get_conditions()
        if conditions:
            print(f"    限定条件: {'；'.join(conditions)}")

    # 4. Step 2: 构建关系网络
    print("\n🔗 Step 2: 构建关系网络...")
    relation_builder = RelationNetBuilder(llm)
    relation_net = relation_builder.build(rst_trees)

    contradictions = relation_net.get_contradictions()
    if contradictions:
        print(f"  发现 {len(contradictions)} 个冲突:")
        for c in contradictions:
            print(f"    {c.source_id} ↔ {c.target_id}: {c.conflict_point}")

    # 5. Step 3: 生成提纲
    print("\n📝 Step 3: 生成写作提纲...")
    outline_gen = OutlineGenerator(llm)
    outline = outline_gen.generate(
        question="补充维生素D能预防流感吗？",
        rst_trees=rst_trees,
        relation_net=relation_net,
    )
    for section in outline.sections:
        print(f"  {section.section_id}. {section.title}")

    # 6. Step 4: 生成答案
    print("\n✍️ Step 4: 生成最终答案...")
    answer_gen = AnswerGenerator(llm)
    answer = answer_gen.generate(
        question="补充维生素D能预防流感吗？",
        outline=outline,
        rst_trees=rst_trees,
        relation_net=relation_net,
    )
    print(f"\n  答案:\n{answer}")


def example_pipeline():
    """使用Pipeline一键运行的示例"""
    print("=" * 60)
    print("  Disco-RAG Pipeline 一键运行示例")
    print("=" * 60)

    config = LLMConfig(
        api_base=os.environ.get("DISCO_RAG_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=os.environ.get("DISCO_RAG_API_KEY", ""),
        model_name=os.environ.get("DISCO_RAG_MODEL", "glm-4-flash"),
    )

    pipeline = DiscoRAGPipeline(llm_config=config)

    # 一行代码运行
    answer = pipeline.ask(
        "补充维生素D能预防流感吗？",
        paragraphs=[
            "在冬季维生素D水平偏低的成年人群中，额外补充维生素D后流感发病率下降了12%。",
            "大规模随机对照试验未发现维生素D补充与流感风险之间存在统计学上的显著关联。",
        ],
    )

    print(f"\n答案: {answer}")


if __name__ == "__main__":
    # 选择一个示例运行
    print("请确保已设置 DISCO_RAG_API_KEY 环境变量\n")

    # 逐步调用
    example_step_by_step()

    # 或者一键运行
    # example_pipeline()
