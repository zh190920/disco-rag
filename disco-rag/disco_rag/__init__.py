"""
Disco-RAG: 基于修辞结构理论的检索增强生成框架

在"搜"和"答"之间加入"读懂"环节，让模型真正理解检索文档的内在逻辑。
三步核心流程：
1. 给每个段落画一棵"论证树"（RST解析）
2. 给所有段落织一张"关系网"（段落间关系图）
3. 先列提纲，再写答案（提纲引导生成）

提供两个版本：
- 同步版本：简单易用，适合低频调用
- 异步高性能版本：并发执行，适合高频/批量调用
"""

__version__ = "2.0.0"
__author__ = "Disco-RAG Implementation"

# 同步版本
from disco_rag.pipeline import DiscoRAGPipeline
from disco_rag.rst_parser import RSTParser, EDU, RSTTree
from disco_rag.relation_net import RelationNetBuilder, ParagraphRelation, RelationType
from disco_rag.outline_generator import OutlineGenerator, Outline
from disco_rag.answer_generator import AnswerGenerator
from disco_rag.retriever import SimpleRetriever, Document

# 异步高性能版本
from disco_rag.async_core.async_llm_client import AsyncLLMClient, AsyncLLMConfig
from disco_rag.async_core.async_rst_parser import AsyncRSTParser
from disco_rag.async_core.async_relation_net import AsyncRelationNetBuilder
from disco_rag.async_core.async_outline_generator import AsyncOutlineGenerator
from disco_rag.async_core.async_answer_generator import AsyncAnswerGenerator
from disco_rag.async_core.async_pipeline import AsyncDiscoRAGPipeline
from disco_rag.async_core.async_retriever import AsyncSimpleRetriever

__all__ = [
    # 同步版本
    "DiscoRAGPipeline",
    "RSTParser",
    "EDU",
    "RSTTree",
    "RelationNetBuilder",
    "ParagraphRelation",
    "RelationType",
    "OutlineGenerator",
    "Outline",
    "AnswerGenerator",
    "SimpleRetriever",
    "Document",
    # 异步高性能版本
    "AsyncLLMClient",
    "AsyncLLMConfig",
    "AsyncRSTParser",
    "AsyncRelationNetBuilder",
    "AsyncOutlineGenerator",
    "AsyncAnswerGenerator",
    "AsyncDiscoRAGPipeline",
    "AsyncSimpleRetriever",
]
