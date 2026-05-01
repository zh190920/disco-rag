"""
简易检索模块

提供基础的文档检索功能：
1. 关键词匹配检索
2. 语义相似度检索（基于嵌入向量）
3. 文档管理

注：本模块为简易实现，生产环境建议替换为专业向量数据库（如FAISS、Milvus等）
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """文档对象"""
    doc_id: str                      # 文档ID
    content: str                     # 文档内容
    title: str = ""                  # 文档标题
    source: str = ""                 # 来源
    metadata: Dict = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "title": self.title,
            "source": self.source,
            "metadata": self.metadata,
        }


class SimpleRetriever:
    """
    简易检索器
    
    支持：
    1. TF-IDF关键词检索
    2. 可选的语义检索（需要sentence-transformers）
    
    对于Disco-RAG的演示，关键词检索已足够。
    生产环境建议使用专业向量检索引擎。
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
            logger.warning("sentence-transformers未安装，将仅使用关键词检索")
            self.use_semantic = False

    def add_documents(self, documents: List[Document]):
        """添加文档到检索库"""
        self.documents.extend(documents)
        self._tfidf_index = None  # 重置索引
        self._embeddings = None
        logger.info(f"添加 {len(documents)} 个文档，总计 {len(self.documents)} 个")

    def add_document(self, document: Document):
        """添加单个文档"""
        self.documents.append(document)
        self._tfidf_index = None
        self._embeddings = None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        检索与查询最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
        
        Returns:
            [(Document, score), ...] 列表，按相关度降序排列
        """
        if not self.documents:
            logger.warning("检索库为空")
            return []

        if self.use_semantic and self._embedding_model:
            return self._semantic_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)

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

        query_embedding = self._embedding_model.encode([query])
        scores = []

        for i, doc in enumerate(self.documents):
            # 余弦相似度
            similarity = self._cosine_similarity(
                query_embedding[0], self._embeddings[i]
            )
            scores.append((doc, float(similarity)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _build_tfidf_index(self):
        """构建TF-IDF索引"""
        if self._tfidf_index is not None:
            return

        # 文档频率
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
        """计算查询与文档的TF-IDF得分"""
        if self._tfidf_index is None:
            return 0.0

        doc_freq = self._tfidf_index["doc_freq"]
        doc_tokens = self._tfidf_index["doc_tokens_list"][doc_idx]
        n_docs = self._tfidf_index["n_docs"]

        score = 0.0
        for token in query_tokens:
            if token in doc_tokens:
                # TF: 词在查询中出现的次数
                tf = query_tokens.count(token)
                # IDF: 逆文档频率
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
        logger.info(f"已构建 {len(self.documents)} 个文档的嵌入向量")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        简易分词
        
        对中文按字/词切分，对英文按空格切分
        """
        tokens = []
        # 英文按空格切分
        for word in text.lower().split():
            # 保留英文单词
            if word.isascii():
                tokens.append(word)
            else:
                # 中文按字符切分（简易版）
                for char in word:
                    if '\u4e00' <= char <= '\u9fff':
                        tokens.append(char)
                    elif char.isalnum():
                        tokens.append(char)
        return tokens

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """计算余弦相似度"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    @classmethod
    def from_file(cls, filepath: str, use_semantic: bool = False) -> "SimpleRetriever":
        """从JSON文件加载文档库"""
        retriever = cls(use_semantic=use_semantic)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for item in data:
            doc = Document(
                doc_id=item.get("doc_id", ""),
                content=item.get("content", ""),
                title=item.get("title", ""),
                source=item.get("source", ""),
                metadata=item.get("metadata", {}),
            )
            documents.append(doc)

        retriever.add_documents(documents)
        logger.info(f"从 {filepath} 加载了 {len(documents)} 个文档")
        return retriever
