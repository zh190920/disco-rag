"""
Disco-RAG 异步高性能版 - 维生素D演示

使用方法:
    export DISCO_RAG_API_KEY="your-api-key"
    python async_demo.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disco_rag.async_core.async_llm_client import AsyncLLMConfig
from disco_rag.async_core.async_pipeline import AsyncDiscoRAGPipeline, AsyncPipelineConfig


async def main():
    # ======== 配置 ========
    config = AsyncPipelineConfig(
        llm=AsyncLLMConfig(
            api_base="https://api.siliconflow.cn/v1",
            api_key="",
            model_name="",
            max_concurrency=20,          # 最多10个并发LLM请求
            requests_per_minute=60,      # 每分钟60个请求
            enable_cache=True,           # 启用缓存
        ),
        enable_prefilter=True,           # 启用关系网络预筛选
        parallel_traditional=True,       # 并行生成传统RAG对比
    )

    # ======== 数据 ========
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

    # ======== 运行 ========
    async with AsyncDiscoRAGPipeline(config=config) as pipeline:
        result = await pipeline.run(
            question=question,
            paragraphs=paragraphs,
            compare_traditional=True,
            verbose=True,
        )

        # 打印客户端统计
        stats = pipeline.get_stats()
        print(f"\n📊 LLM统计: {stats['total_requests']}次请求, "
              f"缓存命中{stats['cache_hit_rate']:.0%}, "
              f"平均{stats['avg_time']:.2f}s/请求")

    # 保存
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    result.save_to_file(os.path.join(output_dir, "async_vitamin_d_result.json"))
    print(f"\n📁 结果已保存至: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
