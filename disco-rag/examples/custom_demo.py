"""
Disco-RAG 自定义文档库示例

演示如何：
1. 创建自定义文档库
2. 使用检索器搜索相关文档
3. 运行Disco-RAG完整流程

使用方法:
    export DISCO_RAG_API_KEY="your-api-key"
    python custom_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disco_rag.llm_client import LLMConfig
from disco_rag.pipeline import DiscoRAGPipeline
from disco_rag.retriever import SimpleRetriever, Document


def main():
    # ======== 配置LLM ========
    config = LLMConfig(
        api_base=os.environ.get("DISCO_RAG_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=os.environ.get("DISCO_RAG_API_KEY", ""),
        model_name=os.environ.get("DISCO_RAG_MODEL", "glm-4-flash"),
    )

    # ======== 创建文档库 ========
    retriever = SimpleRetriever(use_semantic=False)

    # 添加文档（实际使用中可以从数据库、文件等加载）
    documents = [
        Document(
            doc_id="doc_001",
            content="咖啡因可以提高运动表现。研究表明，在运动前60分钟摄入3-6mg/kg体重的咖啡因，"
                    "可以使耐力运动表现提升2-5%。这一效果在跑步、骑行等有氧运动中尤为显著。",
            title="咖啡因与运动表现",
            source="运动营养学杂志",
        ),
        Document(
            doc_id="doc_002",
            content="长期大量摄入咖啡因可能导致耐受性增加，从而削弱其对运动表现的提升效果。"
                    "一项针对习惯性咖啡饮用者的研究发现，停用咖啡因7天后恢复摄入，运动表现提升效果"
                    "明显优于持续使用者。建议在重要比赛前1-2周减少咖啡因摄入以恢复敏感性。",
            title="咖啡因耐受性与运动",
            source="应用生理学杂志",
        ),
        Document(
            doc_id="doc_003",
            content="咖啡因可能增加心率和不规律心律的风险。对有心血管疾病史的人群，"
                    "运动前摄入高剂量咖啡因需要格外谨慎。欧洲食品安全局建议单次摄入不超过200mg，"
                    "每日总摄入量不超过400mg。",
            title="咖啡因的安全剂量",
            source="食品安全评估报告",
        ),
        Document(
            doc_id="doc_004",
            content="低剂量咖啡因（1-3mg/kg）同样能带来运动表现提升，且副作用更少。"
                    "对于不常饮用咖啡的人群，即使是100mg的低剂量（约一杯咖啡）也可能产生明显效果。"
                    "运动生理学家建议从低剂量开始，根据个人反应逐步调整。",
            title="咖啡因的剂量效应",
            source="运动医学综述",
        ),
    ]

    retriever.add_documents(documents)

    # ======== 运行Disco-RAG ========
    pipeline = DiscoRAGPipeline(
        llm_config=config,
        retriever=retriever,
    )

    # 提问
    question = "运动前应该摄入多少咖啡因？"

    result = pipeline.run(
        question=question,
        top_k=4,  # 检索4个最相关文档
        compare_traditional=True,
        verbose=True,
    )

    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    result.save_to_file(os.path.join(output_dir, "caffeine_result.json"))
    with open(os.path.join(output_dir, "caffeine_report.txt"), 'w', encoding='utf-8') as f:
        f.write(result.format_report())

    print(f"\n📁 结果已保存至: {output_dir}/")


if __name__ == "__main__":
    main()
