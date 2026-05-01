"""
异步简易检索器 - 高性能版

优化点：
1. 异步搜索接口（为未来对接异步向量数据库预留）
2. 批量嵌入向量计算（可选语义模式）
3. 文档库并发索引构建
"""

import asyncio
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from disco_rag.retriever import Document

logger = logging.getLogger(__name__)


class AsyncSimpleRetriever:
    """
    异步简易检索器

    与同步版本功能一致，但提供async接口，
    便于在异步Pipeline中统一使用。
    """

    def __init__(self, use_semantic: bool = False):
        self.documents: List[Document] = []
        self.use_semantic = use_semantic
        self._tfidf_index = None
        self._embedding_model = None
        self._embeddings = None

        if use_semantic:
            self._init_embedding_model()

    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("语义嵌入模型加载成功")
        except ImportError:
            logger.warning("sentence-transformers未安装，仅使用关键词检索")
            self.use_semantic = False

    def add_documents(self, documents: List[Document]):
        """添加文档到检索库"""
        self.documents.extend(documents)
        self._tfidf_index = None
        self._embeddings = None
        logger.info(f"添加 {len(documents)} 个文档，总计 {len(self.documents)} 个")

    def add_document(self, document: Document):
        """添加单个文档"""
        self.documents.append(document)
        self._tfidf_index = None
        self._embeddings = None

    async def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        异步检索

        当前实现：在线程池中执行同步搜索（避免阻塞事件循环）
        未来可对接异步向量数据库（Milvus、Qdrant等）
        """
        if not self.documents:
            return []

        if self.use_semantic and self._embedding_model:
            return await asyncio.to_thread(self._semantic_search, query, top_k)
        else:
            return await asyncio.to_thread(self._keyword_search, query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """基于TF-IDF的关键词检索"""
        self._build_tfidf_index()
        query_tokens = self._tokenize(query)
        scores = []
        for i, doc in enumerate(self.documents):
            score = self._compute_tfidf_score(query_tokens, i)
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """基于语义相似度的检索"""
        if self._embeddings is None:
            self._build_embeddings()
        import numpy as np
        query_embedding = self._embedding_model.encode([query])
        scores = []
        for i, doc in enumerate(self.documents):
            similarity = float(np.dot(query_embedding[0], self._embeddings[i]) /
                              (np.linalg.norm(query_embedding[0]) * np.linalg.norm(self._embeddings[i]) + 1e-8))
            scores.append((doc, similarity))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _build_tfidf_index(self):
        """构建TF-IDF索引"""
        if self._tfidf_index is not None:
            return
        doc_freq = defaultdict(int)
        doc_tokens_list = []
        for doc in self.documents:
            tokens = set(self._tokenize(doc.content))
            doc_tokens_list.append(tokens)
            for token in tokens:
                doc_freq[token] += 1
        n_docs = len(self.documents)
        self._tfidf_index = {
            "doc_freq": doc_freq,
            "doc_tokens_list": doc_tokens_list,
            "n_docs": n_docs,
        }

    def _compute_tfidf_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算TF-IDF得分"""
        if self._tfidf_index is None:
            return 0.0
        doc_freq = self._tfidf_index["doc_freq"]
        doc_tokens = self._tfidf_index["doc_tokens_list"][doc_idx]
        n_docs = self._tfidf_index["n_docs"]
        score = 0.0
        for token in query_tokens:
            if token in doc_tokens:
                tf = query_tokens.count(token)
                df = doc_freq.get(token, 0)
                idf = math.log((n_docs + 1) / (df + 1)) + 1
                score += tf * idf
        return score

    def _build_embeddings(self):
        """构建文档嵌入向量"""
        if not self._embedding_model:
            return
        texts = [doc.content for doc in self.documents]
        self._embeddings = self._embedding_model.encode(texts)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简易分词"""
        tokens = []
        for word in text.lower().split():
            if word.isascii():
                tokens.append(word)
            else:
                for char in word:
                    if '\u4e00' <= char <= '\u9fff':
                        tokens.append(char)
                    elif char.isalnum():
                        tokens.append(char)
        return tokens

    @classmethod
    async def from_file(cls, filepath: str, use_semantic: bool = False) -> "AsyncSimpleRetriever":
        """异步加载文档库"""
        retriever = cls(use_semantic=use_semantic)
        content = await asyncio.to_thread(cls._read_file, filepath)
        data = json.loads(content)
        documents = [
            Document(
                doc_id=item.get("doc_id", ""),
                content=item.get("content", ""),
                title=item.get("title", ""),
                source=item.get("source", ""),
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]
        retriever.add_documents(documents)
        return retriever

    @staticmethod
    def _read_file(filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
