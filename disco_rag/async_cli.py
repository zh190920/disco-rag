"""
Disco-RAG 异步高性能CLI

用法:
    python -m disco_rag.async_cli --demo
    python -m disco_rag.async_cli -q "问题" --paragraphs p1.txt p2.txt
    python -m disco_rag.async_cli -q "问题" --corpus docs.json
    python -m disco_rag.async_cli --benchmark        # 性能基准测试
    python -m disco_rag.async_cli -q "问题" --concurrency 20  # 自定义并发数
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disco_rag.async_core.async_llm_client import AsyncLLMClient, AsyncLLMConfig
from disco_rag.async_core.async_pipeline import AsyncDiscoRAGPipeline, AsyncPipelineConfig
from disco_rag.async_core.async_retriever import AsyncSimpleRetriever
from disco_rag.retriever import Document


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_llm_config(concurrency: int = 10) -> AsyncLLMConfig:
    """获取LLM配置"""
    return AsyncLLMConfig(
        api_base=os.environ.get("DISCO_RAG_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=os.environ.get("DISCO_RAG_API_KEY", ""),
        model_name=os.environ.get("DISCO_RAG_MODEL", "glm-4-flash"),
        max_concurrency=concurrency,
        requests_per_minute=int(os.environ.get("DISCO_RAG_RPM", "60")),
        max_retries=3,
        enable_cache=True,
    )


async def run_demo(concurrency: int = 10):
    """运行维生素D演示案例"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║        Disco-RAG 异步高性能版 演示                       ║
║        维生素D与流感                                     ║
║                                                          ║
║   并发RST解析 + 并发关系分析 + 预筛选优化                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    question = "补充维生素D能预防流感吗？"

    paragraphs = [
        "在冬季维生素D水平偏低的成年人群中，额外补充维生素D后流感发病率下降了12%。"
        "这一发现来自一项针对该特定人群的观察性研究，研究者在冬季对血清维生素D浓度"
        "低于30ng/mL的成年人进行了为期6个月的追踪，发现补充组相比对照组流感发病率"
        "具有统计学意义上的降低。",

        "大规模随机对照试验未发现维生素D补充与流感风险之间存在统计学上的显著关联。"
        "该试验纳入超过15000名受试者，年龄覆盖18-75岁，随访时间长达3年。亚组分析"
        "显示，无论在哪个年龄组或季节，补充组与安慰剂组的流感发生率均无显著差异。",

        "维生素D在免疫调节中发挥重要作用，其活性形式1,25-二羟维生素D3能够促进抗菌肽"
        "的表达，增强先天免疫防御。多项体外实验和动物模型均证实维生素D缺乏会导致"
        "免疫功能下降。",

        "流感疫苗仍是预防流感最有效的手段，每年可减少数百万重症病例。维生素D作为"
        "一种潜在的辅助预防策略，其证据等级尚不足以支持常规推荐。临床指南建议优先"
        "接种疫苗，同时维持正常的维生素D水平。",
    ]

    config = AsyncPipelineConfig(
        llm=get_llm_config(concurrency),
        enable_prefilter=True,
        parallel_traditional=True,
    )

    async with AsyncDiscoRAGPipeline(config=config) as pipeline:
        result = await pipeline.run(
            question=question,
            paragraphs=paragraphs,
            compare_traditional=True,
            verbose=True,
        )

        # 保存结果
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
        os.makedirs(output_dir, exist_ok=True)

        result.save_to_file(os.path.join(output_dir, "async_demo_result.json"))
        with open(os.path.join(output_dir, "async_demo_report.txt"), 'w', encoding='utf-8') as f:
            f.write(result.format_report())

        # 打印统计
        stats = pipeline.get_stats()
        print(f"\n{'='*60}")
        print("  📊 LLM客户端统计")
        print(f"{'='*60}")
        print(f"   总请求数: {stats['total_requests']}")
        print(f"   缓存命中: {stats['cache_hits']}")
        print(f"   缓存命中率: {stats['cache_hit_rate']:.1%}")
        print(f"   重试次数: {stats['retries']}")
        print(f"   平均耗时: {stats['avg_time']:.2f}s/请求")


async def run_benchmark(concurrency: int = 10):
    """性能基准测试：对比同步 vs 异步"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║        Disco-RAG 性能基准测试                            ║
║        同步串行 vs 异步并发                              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    question = "补充维生素D能预防流感吗？"

    # 不同段落规模测试
    base_paragraphs = [
        "在冬季维生素D水平偏低的成年人群中，额外补充维生素D后流感发病率下降了12%。",
        "大规模随机对照试验未发现维生素D补充与流感风险之间存在统计学上的显著关联。",
        "维生素D在免疫调节中发挥重要作用，其活性形式能够促进抗菌肽的表达。",
        "流感疫苗仍是预防流感最有效的手段，每年可减少数百万重症病例。",
    ]

    test_configs = [
        ("2段落", base_paragraphs[:2]),
        ("4段落", base_paragraphs[:4]),
    ]

    for name, paragraphs in test_configs:
        print(f"\n{'─'*40}")
        print(f"  测试: {name}")
        print(f"{'─'*40}")

        # 异步版本
        config = AsyncPipelineConfig(
            llm=get_llm_config(concurrency),
            enable_prefilter=True,
            parallel_traditional=False,  # 不生成对比答案
        )

        t0 = time.monotonic()
        async with AsyncDiscoRAGPipeline(config=config) as pipeline:
            result = await pipeline.run(
                question=question,
                paragraphs=paragraphs,
                compare_traditional=False,
                verbose=False,
            )
        async_time = time.monotonic() - t0

        print(f"\n  ⚡ 异步版本耗时: {async_time:.2f}s")
        for stage, t in result.timing.items():
            print(f"     {stage}: {t:.2f}s")

    print(f"\n{'='*60}")
    print("  基准测试完成")
    print(f"{'='*60}")


async def run_question(
    question: str,
    paragraph_files: list = None,
    corpus_file: str = None,
    top_k: int = 5,
    concurrency: int = 10,
    output: str = None,
    save_json: bool = False,
    no_compare: bool = False,
):
    """运行自定义问题"""
    config = AsyncPipelineConfig(
        llm=get_llm_config(concurrency),
        enable_prefilter=True,
        parallel_traditional=True,
    )

    paragraphs = None
    retriever = None

    if paragraph_files:
        paragraphs = []
        for fpath in paragraph_files:
            if os.path.isfile(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if text:
                    paragraphs.append(text)

    if corpus_file:
        retriever = await AsyncSimpleRetriever.from_file(corpus_file)

    async with AsyncDiscoRAGPipeline(config=config, retriever=retriever) as pipeline:
        result = await pipeline.run(
            question=question,
            paragraphs=paragraphs,
            top_k=top_k,
            compare_traditional=not no_compare,
            verbose=True,
        )

        if output or save_json:
            output_path = output or "async_disco_rag_result.json"
            result.save_to_file(output_path)
            print(f"\n📁 结果已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Disco-RAG 异步高性能版 - 基于修辞结构理论的检索增强生成框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行演示
  python -m disco_rag.async_cli --demo

  # 自定义并发数
  python -m disco_rag.async_cli --demo --concurrency 20

  # 性能基准测试
  python -m disco_rag.async_cli --benchmark

  # 自定义提问
  python -m disco_rag.async_cli -q "你的问题" -p p1.txt p2.txt

  # 配置LLM
  export DISCO_RAG_API_BASE="https://open.bigmodel.cn/api/paas/v4"
  export DISCO_RAG_API_KEY="your-api-key"
  export DISCO_RAG_MODEL="glm-4-flash"
        """,
    )

    parser.add_argument("--demo", action="store_true", help="运行维生素D演示案例")
    parser.add_argument("--benchmark", action="store_true", help="运行性能基准测试")
    parser.add_argument("--question", "-q", type=str, help="用户问题")
    parser.add_argument("--paragraphs", "-p", nargs="+", help="段落文件路径列表")
    parser.add_argument("--corpus", "-c", type=str, help="文档库JSON文件路径")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="检索返回段落数量")
    parser.add_argument("--concurrency", "-n", type=int, default=10, help="最大并发请求数")
    parser.add_argument("--output", "-o", type=str, help="输出文件路径")
    parser.add_argument("--no-compare", action="store_true", help="不生成传统RAG对比")
    parser.add_argument("--save-json", action="store_true", help="保存详细JSON结果")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.demo:
        asyncio.run(run_demo(args.concurrency))
    elif args.benchmark:
        asyncio.run(run_benchmark(args.concurrency))
    elif args.question:
        asyncio.run(run_question(
            question=args.question,
            paragraph_files=args.paragraphs,
            corpus_file=args.corpus,
            top_k=args.top_k,
            concurrency=args.concurrency,
            output=args.output,
            save_json=args.save_json,
            no_compare=args.no_compare,
        ))
    else:
        parser.error("请提供 --demo, --benchmark 或 --question 参数")


if __name__ == "__main__":
    main()
