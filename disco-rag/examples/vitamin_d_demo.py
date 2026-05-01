"""
Disco-RAG 维生素D演示案例

这个示例展示了Disco-RAG如何"读懂"文档中的逻辑关系，
而非传统RAG那样简单拼接段落。

使用方法:
    # 1. 配置LLM API（以智谱AI为例）
    export DISCO_RAG_API_BASE="https://open.bigmodel.cn/api/paas/v4"
    export DISCO_RAG_API_KEY="your-api-key"
    export DISCO_RAG_MODEL="glm-4-flash"

    # 2. 运行演示
    python vitamin_d_demo.py
"""

import os
import sys

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disco_rag.llm_client import LLMConfig
from disco_rag.pipeline import DiscoRAGPipeline


def main():
    # ======== 配置LLM ========
    # 方式1: 通过环境变量配置
    # export DISCO_RAG_API_BASE="https://open.bigmodel.cn/api/paas/v4"
    # export DISCO_RAG_API_KEY="your-api-key"
    # export DISCO_RAG_MODEL="glm-4-flash"

    # 方式2: 在代码中直接配置
    config = LLMConfig(
        api_base=os.environ.get("DISCO_RAG_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=os.environ.get("DISCO_RAG_API_KEY", ""),
        model_name=os.environ.get("DISCO_RAG_MODEL", "glm-4-flash"),
        temperature=0.1,
        max_tokens=4096,
    )

    # ======== 准备数据 ========
    question = "补充维生素D能预防流感吗？"

    # 模拟检索返回的段落（真实场景中这些由检索器返回）
    paragraphs = [
        # 段落A：限定条件下的正面结论
        "在冬季维生素D水平偏低的成年人群中，额外补充维生素D后流感发病率下降了12%。"
        "这一发现来自一项针对该特定人群的观察性研究，研究者在冬季对血清维生素D浓度"
        "低于30ng/mL的成年人进行了为期6个月的追踪，发现补充组相比对照组流感发病率"
        "具有统计学意义上的降低。",

        # 段落B：大规模试验的否定结论
        "大规模随机对照试验未发现维生素D补充与流感风险之间存在统计学上的显著关联。"
        "该试验纳入超过15000名受试者，年龄覆盖18-75岁，随访时间长达3年。亚组分析"
        "显示，无论在哪个年龄组或季节，补充组与安慰剂组的流感发生率均无显著差异。",

        # 段落C：维生素D的免疫学基础
        "维生素D在免疫调节中发挥重要作用，其活性形式1,25-二羟维生素D3能够促进抗菌肽"
        "的表达，增强先天免疫防御。多项体外实验和动物模型均证实维生素D缺乏会导致"
        "免疫功能下降。",

        # 段落D：临床实践建议
        "流感疫苗仍是预防流感最有效的手段，每年可减少数百万重症病例。维生素D作为"
        "一种潜在的辅助预防策略，其证据等级尚不足以支持常规推荐。临床指南建议优先"
        "接种疫苗，同时维持正常的维生素D水平。",
    ]

    # ======== 运行Disco-RAG ========
    pipeline = DiscoRAGPipeline(llm_config=config)

    result = pipeline.run(
        question=question,
        paragraphs=paragraphs,
        compare_traditional=True,  # 同时生成传统RAG答案用于对比
        verbose=True,
    )

    # ======== 保存结果 ========
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    result.save_to_file(os.path.join(output_dir, "vitamin_d_result.json"))
    with open(os.path.join(output_dir, "vitamin_d_report.txt"), 'w', encoding='utf-8') as f:
        f.write(result.format_report())

    print(f"\n📁 结果已保存至: {output_dir}/")


if __name__ == "__main__":
    main()
