"""
Disco-RAG 异步高性能核心模块

提供全异步并发实现，相比同步版本可获得 3-10x 性能提升：
- 异步LLM客户端：连接池复用 + 并发请求
- 段落级并发：多个段落RST解析同时进行
- 配对级并发：段落间关系分析同时进行
- 信号量控制：防止API限流
- LRU缓存：相同输入不重复调用LLM
"""

from disco_rag.async_core.async_llm_client import AsyncLLMClient, AsyncLLMConfig
from disco_rag.async_core.async_rst_parser import AsyncRSTParser
from disco_rag.async_core.async_relation_net import AsyncRelationNetBuilder
from disco_rag.async_core.async_outline_generator import AsyncOutlineGenerator
from disco_rag.async_core.async_answer_generator import AsyncAnswerGenerator
from disco_rag.async_core.async_pipeline import AsyncDiscoRAGPipeline
from disco_rag.async_core.async_retriever import AsyncSimpleRetriever

__all__ = [
    "AsyncLLMClient",
    "AsyncLLMConfig",
    "AsyncRSTParser",
    "AsyncRelationNetBuilder",
    "AsyncOutlineGenerator",
    "AsyncAnswerGenerator",
    "AsyncDiscoRAGPipeline",
    "AsyncSimpleRetriever",
]
