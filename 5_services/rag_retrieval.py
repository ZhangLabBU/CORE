# services/rag_retrieval.py
import sys
import os

# 将 Pre_processing 目录添加到 sys.path 中
PRE_PROCESSING_DIR = "/home/yzt/AgentDNS/src/Pre_processing"
if PRE_PROCESSING_DIR not in sys.path:
    sys.path.append(PRE_PROCESSING_DIR)

# 现在可以导入 langchain_es_bm25_demo 模块了
from langchain_es_bm25_demo import CapabilityHybridRetriever
from elasticsearch import Elasticsearch

async def rag_retrieval(query: str, top_k: int = 5) -> list:
    # 进行检索并返回相关文档
    retriever = CapabilityHybridRetriever(
        es=Elasticsearch(),
        index="capability_registry",
        k=top_k
    )
    docs = retriever.invoke(query) or []
    # 对 docs 进行进一步处理并返回
    return docs
