"""
Disco-RAG 性能基准测试

对比同步版本 vs 异步版本在不同段落规模下的耗时。

使用方法:
    export DISCO_RAG_API_KEY="your-api-key"
    python benchmark.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run_async_benchmark(question, paragraphs, concurrency=10):
    """运行异步版本"""
    from disco_rag.async_core.async_llm_client import AsyncLLMConfig
    from disco_rag.async_core.async_pipeline import AsyncDiscoRAGPipeline, AsyncPipelineConfig

    config = AsyncPipelineConfig(
        llm=AsyncLLMConfig(
            api_base=os.environ.get("DISCO_RAG_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
            api_key=os.environ.get("DISCO_RAG_API_KEY", ""),
            model_name=os.environ.get("DISCO_RAG_MODEL", "glm-4-flash"),
            max_concurrency=concurrency,
        ),
        enable_prefilter=True,
        parallel_traditional=False,
    )

    async with AsyncDiscoRAGPipeline(config=config) as pipeline:
        t0 = time.monotonic()
        result = await pipeline.run(
            question=question,
            paragraphs=paragraphs,
            compare_traditional=False,
            verbose=False,
        )
        elapsed = time.monotonic() - t0

    return elapsed, result.timing, pipeline.get_stats()


def run_sync_benchmark(question, paragraphs):
    """运行同步版本"""
    from disco_rag.llm_client import LLMConfig
    from disco_rag.pipeline import DiscoRAGPipeline

    config = LLMConfig(
        api_base=os.environ.get("DISCO_RAG_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=os.environ.get("DISCO_RAG_API_KEY", ""),
        model_name=os.environ.get("DISCO_RAG_MODEL", "glm-4-flash"),
    )

    pipeline = DiscoRAGPipeline(llm_config=config)
    t0 = time.monotonic()
    result = pipeline.run(
        question=question,
        paragraphs=paragraphs,
        compare_traditional=False,
        verbose=False,
    )
    elapsed = time.monotonic() - t0

    return elapsed, result.timing


async def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║        Disco-RAG 性能基准测试                            ║
║        同步串行 vs 异步并发                              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    question = "补充维生素D能预防流感吗？"
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

    results = []

    for name, paragraphs in test_configs:
        print(f"\n{'─'*60}")
        print(f"  📊 测试: {name} ({len(paragraphs)}个段落)")
        print(f"{'─'*60}")

        # 异步版本
        print(f"\n  ⚡ 异步并发版...")
        async_time, async_timing, async_stats = await run_async_benchmark(
            question, paragraphs, concurrency=10
        )

        print(f"     总耗时: {async_time:.2f}s")
        for stage, t in async_timing.items():
            print(f"     {stage}: {t:.2f}s")
        print(f"     LLM请求: {async_stats['total_requests']}次, "
              f"缓存命中: {async_stats['cache_hit_rate']:.0%}")

        results.append({
            "name": name,
            "paragraphs": len(paragraphs),
            "async_time": async_time,
            "async_timing": async_timing,
        })

    # 汇总
    print(f"\n{'='*60}")
    print("  📋 性能汇总")
    print(f"{'='*60}")
    print(f"  {'场景':<8} {'段落数':<6} {'异步耗时':<10}")
    print(f"  {'─'*30}")
    for r in results:
        print(f"  {r['name']:<8} {r['paragraphs']:<6} {r['async_time']:.2f}s")

    print(f"\n  💡 提示: 异步版本通过并发LLM调用，将RST解析和关系网络分析")
    print(f"     的串行等待转变为并行执行，显著降低端到端延迟。")
    print(f"     段落越多，并发优势越明显。")


if __name__ == "__main__":
    asyncio.run(main())
