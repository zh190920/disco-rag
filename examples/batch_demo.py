"""
Disco-RAG 批量并发处理示例

展示如何使用 batch_ask 同时处理多个用户问题，
实现高并发场景下的高效推理。

使用方法:
    export DISCO_RAG_API_KEY="your-api-key"
    python batch_demo.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disco_rag.async_core.async_llm_client import AsyncLLMConfig
from disco_rag.async_core.async_pipeline import AsyncDiscoRAGPipeline, AsyncPipelineConfig


async def main():
    config = AsyncPipelineConfig(
        llm=AsyncLLMConfig(
            api_base=os.environ.get("DISCO_RAG_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
            api_key=os.environ.get("DISCO_RAG_API_KEY", ""),
            model_name=os.environ.get("DISCO_RAG_MODEL", "glm-4-flash"),
            max_concurrency=20,          # 更高的并发上限
            requests_per_minute=120,     # 更高的RPM限制
        ),
    )

    # 多个用户的问题
    questions = [
        "补充维生素D能预防流感吗？",
        "咖啡因对运动表现有帮助吗？",
        "间歇性禁食真的能减肥吗？",
    ]

    paragraphs_list = [
        # 维生素D相关
        [
            "在冬季维生素D水平偏低的成年人群中，额外补充维生素D后流感发病率下降了12%。",
            "大规模随机对照试验未发现维生素D补充与流感风险之间存在统计学上的显著关联。",
        ],
        # 咖啡因相关
        [
            "咖啡因可以提高运动表现。研究表明摄入3-6mg/kg体重的咖啡因可以使耐力提升2-5%。",
            "长期大量摄入咖啡因可能导致耐受性增加，建议赛前1-2周减少摄入以恢复敏感性。",
        ],
        # 间歇性禁食相关
        [
            "间歇性禁食在短期内可以减少热量摄入，从而促进体重下降。",
            "与持续性热量限制相比，间歇性禁食的长期减重效果并无显著差异。",
        ],
    ]

    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║        Disco-RAG 批量并发处理演示                        ║
║        同时处理 3 个用户问题                              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    async with AsyncDiscoRAGPipeline(config=config) as pipeline:
        print("  🚀 开始批量处理...\n")
        t0 = time.time()

        results = await pipeline.batch_ask(questions, paragraphs_list)

        elapsed = time.time() - t0

        for i, (q, r) in enumerate(zip(questions, results)):
            print(f"\n{'─'*50}")
            print(f"  问题 {i+1}: {q}")
            print(f"  答案: {r.answer[:200]}...")
            print(f"  耗时: {r.timing.get('总耗时', 0):.2f}s")

        print(f"\n{'='*50}")
        print(f"  ⚡ 批量处理总耗时: {elapsed:.2f}s")
        print(f"  平均每个问题: {elapsed/len(questions):.2f}s")
        print(f"  (并发处理，总耗时远小于各问题串行之和)")

        stats = pipeline.get_stats()
        print(f"\n  📊 LLM统计: {stats['total_requests']}次请求, "
              f"缓存命中{stats['cache_hit_rate']:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
