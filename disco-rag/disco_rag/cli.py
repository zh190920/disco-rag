"""
Disco-RAG 命令行接口

用法:
    python -m disco_rag.cli --question "维生素D能预防流感吗？"
    python -m disco_rag.cli --question "问题" --paragraphs p1.txt p2.txt
    python -m disco_rag.cli --question "问题" --corpus docs.json
    python -m disco_rag.cli --demo
"""

import argparse
import json
import logging
import sys
import os

from disco_rag.llm_client import LLMConfig
from disco_rag.pipeline import DiscoRAGPipeline
from disco_rag.retriever import SimpleRetriever, Document


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_paragraphs_from_files(files: list) -> list:
    """从文件列表加载段落"""
    paragraphs = []
    for fpath in files:
        if os.path.isfile(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if text:
                paragraphs.append(text)
    return paragraphs


def load_corpus(filepath: str) -> SimpleRetriever:
    """从JSON文件加载文档库"""
    retriever = SimpleRetriever()
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for item in data:
        doc = Document(
            doc_id=item.get("doc_id", ""),
            content=item.get("content", ""),
            title=item.get("title", ""),
            source=item.get("source", ""),
        )
        documents.append(doc)

    retriever.add_documents(documents)
    return retriever


def run_demo():
    """运行维生素D的演示案例"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║              Disco-RAG 演示：维生素D与流感               ║
║                                                          ║
║   展示Disco-RAG如何"读懂"文档，而非仅仅"搜到"文档       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    question = "补充维生素D能预防流感吗？"

    paragraphs = [
        "在冬季维生素D水平偏低的成年人群中，额外补充维生素D后流感发病率下降了12%。这一发现来自一项针对该特定人群的观察性研究，研究者在冬季对血清维生素D浓度低于30ng/mL的成年人进行了为期6个月的追踪，发现补充组相比对照组流感发病率具有统计学意义上的降低。",
        "大规模随机对照试验未发现维生素D补充与流感风险之间存在统计学上的显著关联。该试验纳入超过15000名受试者，年龄覆盖18-75岁，随访时间长达3年。亚组分析显示，无论在哪个年龄组或季节，补充组与安慰剂组的流感发生率均无显著差异。",
        "维生素D在免疫调节中发挥重要作用，其活性形式1,25-二羟维生素D3能够促进抗菌肽的表达，增强先天免疫防御。多项体外实验和动物模型均证实维生素D缺乏会导致免疫功能下降。",
        "流感疫苗仍是预防流感最有效的手段，每年可减少数百万重症病例。维生素D作为一种潜在的辅助预防策略，其证据等级尚不足以支持常规推荐。临床指南建议优先接种疫苗，同时维持正常的维生素D水平。",
    ]

    # 需要配置LLM
    config = _get_llm_config()
    pipeline = DiscoRAGPipeline(llm_config=config)

    result = pipeline.run(
        question=question,
        paragraphs=paragraphs,
        compare_traditional=True,
        verbose=True,
    )

    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # 保存JSON结果
    result_path = os.path.join(output_dir, "demo_result.json")
    result.save_to_file(result_path)
    print(f"\n📁 详细结果已保存至: {result_path}")

    # 保存可读报告
    report_path = os.path.join(output_dir, "demo_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(result.format_report())
    print(f"📁 可读报告已保存至: {report_path}")


def _get_llm_config() -> LLMConfig:
    """获取LLM配置"""
    config = LLMConfig()

    # 从环境变量读取配置
    api_base = os.environ.get("DISCO_RAG_API_BASE")
    api_key = os.environ.get("DISCO_RAG_API_KEY")
    model_name = os.environ.get("DISCO_RAG_MODEL")

    if api_base:
        config.api_base = api_base
    if api_key:
        config.api_key = api_key
    if model_name:
        config.model_name = model_name

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Disco-RAG: 基于修辞结构理论的检索增强生成框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行维生素D演示案例
  python -m disco_rag.cli --demo

  # 直接提问（使用预设段落）
  python -m disco_rag.cli --question "维生素D能预防流感吗？" --paragraphs p1.txt p2.txt

  # 使用文档库
  python -m disco_rag.cli --question "问题" --corpus docs.json

  # 配置LLM
  export DISCO_RAG_API_BASE="https://open.bigmodel.cn/api/paas/v4"
  export DISCO_RAG_API_KEY="your-api-key"
  export DISCO_RAG_MODEL="glm-4-flash"
        """,
    )

    parser.add_argument("--demo", action="store_true", help="运行维生素D演示案例")
    parser.add_argument("--question", "-q", type=str, help="用户问题")
    parser.add_argument("--paragraphs", "-p", nargs="+", help="段落文件路径列表")
    parser.add_argument("--corpus", "-c", type=str, help="文档库JSON文件路径")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="检索返回段落数量")
    parser.add_argument("--output", "-o", type=str, help="输出文件路径")
    parser.add_argument("--no-compare", action="store_true", help="不生成传统RAG对比答案")
    parser.add_argument("--save-json", action="store_true", help="保存详细JSON结果")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志输出")

    args = parser.parse_args()

    setup_logging(args.verbose)

    # 演示模式
    if args.demo:
        run_demo()
        return

    # 必须提供问题
    if not args.question:
        parser.error("请提供 --question 参数，或使用 --demo 运行演示案例")

    # 获取LLM配置
    config = _get_llm_config()

    # 加载段落
    paragraphs = None
    retriever = None

    if args.paragraphs:
        paragraphs = load_paragraphs_from_files(args.paragraphs)
        if not paragraphs:
            print("错误: 未从文件中加载到任何段落内容", file=sys.stderr)
            sys.exit(1)

    if args.corpus:
        retriever = load_corpus(args.corpus)

    # 创建管线
    pipeline = DiscoRAGPipeline(
        llm_config=config,
        retriever=retriever,
    )

    # 运行
    result = pipeline.run(
        question=args.question,
        paragraphs=paragraphs,
        top_k=args.top_k,
        compare_traditional=not args.no_compare,
        verbose=True,
    )

    # 保存结果
    if args.output or args.save_json:
        output_path = args.output or "disco_rag_result.json"
        result.save_to_file(output_path)
        print(f"\n📁 结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
