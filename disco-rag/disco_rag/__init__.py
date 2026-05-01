"""
Disco-RAG: 基于修辞结构理论的检索增强生成框架

在"搜"和"答"之间加入"读懂"环节，让模型真正理解检索文档的内在逻辑。
三步核心流程：
1. 给每个段落画一棵"论证树"（RST解析）
2. 给所有段落织一张"关系网"（段落间关系图）
3. 先列提纲，再写答案（提纲引导生成）
"""

__version__ = "1.0.0"
__author__ = "Disco-RAG Implementation"

from disco_rag.pipeline import DiscoRAGPipeline
from disco_rag.rst_parser import RSTParser, EDU, RSTTree
from disco_rag.relation_net import RelationNetBuilder, ParagraphRelation, RelationType
from disco_rag.outline_generator import OutlineGenerator, Outline
from disco_rag.answer_generator import AnswerGenerator
from disco_rag.retriever import SimpleRetriever, Document

__all__ = [
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
]
