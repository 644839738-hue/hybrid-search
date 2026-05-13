"""
向量存储管理模块

负责 ChromaDB 向量存储操作，包括文档索引和检索。
支持本地持久化存储和远程 Docker/HTTP 服务器两种模式。
支持 OpenAI、HuggingFace 和 DeepSeek 三种 Embedding 提供商。
"""

import os
import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    管理 ChromaDB 向量存储操作。

    处理与 ChromaDB 向量存储的所有交互，包括：
    - 向量存储初始化和持久化
    - 文档 Embedding 和索引
    - 相似度搜索和检索
    - 集合管理（清空、删除）

    支持两种连接模式：
    - 本地模式: 使用持久化目录存储（默认）
    - Docker/HTTP 模式: 通过 HTTP 连接 ChromaDB 服务器

    支持三种 Embedding 提供商：
    - openai: OpenAI Embeddings（需 OPENAI_API_KEY）
    - huggingface: 本地 HuggingFace sentence-transformers（无需 API Key）
    - deepseek: DeepSeek Embeddings（需 DEEPSEEK_API_KEY，兼容 OpenAI API）

    属性:
        embedding_model: Embedding 模型实例
        embedding_provider: 提供商名称 ("openai" / "huggingface" / "deepseek")
        collection_name: ChromaDB 集合名称
        persist_directory: 持久化存储目录（本地模式）
        chroma_host: ChromaDB 服务器主机（Docker 模式）
        chroma_port: ChromaDB 服务器端口（Docker 模式）
        vector_store: ChromaDB 向量存储实例

    示例:
        >>> # OpenAI Embeddings (本地模式)
        >>> manager = VectorStoreManager(
        ...     embedding_model_name="text-embedding-3-large",
        ...     embedding_provider="openai",
        ...     collection_name="my_docs"
        ... )
        >>> # DeepSeek Embeddings (本地模式)
        >>> manager = VectorStoreManager(
        ...     embedding_model_name="deepseek-embedding",
        ...     embedding_provider="deepseek",
        ...     collection_name="my_docs"
        ... )
        >>> ids = manager.add_documents(chunks)
        >>> retriever = manager.get_retriever(search_k=3)
    """

    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-large",
        embedding_provider: str = "openai",
        embedding_base_url: str = "",
        collection_name: str = "semantic_search_docs_streamlit",
        persist_directory: str = "./chroma/db",
        use_docker: bool = False,
        chroma_host: str = "localhost",
        chroma_port: int = 8000
    ):
        """
        初始化向量存储管理器。

        Args:
            embedding_model_name: Embedding 模型名称
            embedding_provider: Embedding 提供商 ("openai" / "huggingface" / "deepseek")
            embedding_base_url: Embedding API 基础地址（用于 deepseek）
            collection_name: ChromaDB 集合名称
            persist_directory: 持久化目录路径（本地模式）
            use_docker: 为 True 时连接 ChromaDB Docker 服务器
            chroma_host: ChromaDB 服务器主机名（Docker 模式）
            chroma_port: ChromaDB 服务器端口（Docker 模式）
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_docker = use_docker
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self._chroma_client = None

        # 根据提供商初始化 Embedding 模型
        if embedding_provider == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                logger.info(f"已初始化 HuggingFace Embeddings: {embedding_model_name} (device=cpu)")
            except ImportError:
                raise ImportError(
                    "使用 HuggingFace Embeddings 需要安装 langchain-huggingface。"
                    "请运行: pip install langchain-huggingface"
                )
        elif embedding_provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "未找到 DEEPSEEK_API_KEY 环境变量。"
                    "请在 .env 文件中设置或执行 export DEEPSEEK_API_KEY=your_key"
                )
            self.embedding_model = OpenAIEmbeddings(
                model=embedding_model_name,
                base_url=embedding_base_url or "https://api.deepseek.com",
                api_key=api_key
            )
            logger.info(f"已初始化 DeepSeek Embeddings: {embedding_model_name}")
        else:
            self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
            logger.info(f"已初始化 OpenAI Embeddings: {embedding_model_name}")

        # 初始化向量存储
        self.vector_store = self._initialize_vector_store()
        mode = "Docker" if use_docker else "本地"
        logger.info(f"向量存储已初始化 ({mode} 模式): collection={collection_name}")

    def _initialize_vector_store(self) -> Chroma:
        """
        初始化 ChromaDB 向量存储。

        支持两种模式：
        - 本地: 使用持久化目录和 SQLite 后端
        - Docker: 通过 HTTP 客户端连接 ChromaDB 服务器

        Returns:
            Chroma 向量存储实例

        Raises:
            ConnectionError: Docker 模式已启用但服务器不可达时抛出
        """
        if self.use_docker:
            # Docker/HTTP 客户端模式 - 连接 ChromaDB 服务器
            logger.info(f"正在连接 ChromaDB 服务器 {self.chroma_host}:{self.chroma_port}")
            try:
                self._chroma_client = chromadb.HttpClient(
                    host=self.chroma_host,
                    port=self.chroma_port,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                # 测试连接
                self._chroma_client.heartbeat()
                logger.info("成功连接到 ChromaDB 服务器")

                return Chroma(
                    client=self._chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model
                )
            except Exception as e:
                logger.error(f"连接 ChromaDB 服务器失败: {e}")
                raise ConnectionError(
                    f"无法连接到 ChromaDB {self.chroma_host}:{self.chroma_port}。"
                    "请确保 Docker 容器正在运行: docker run -p 8000:8000 chromadb/chroma"
                ) from e
        else:
            # 本地持久化模式
            logger.info(f"使用本地 ChromaDB，路径: {self.persist_directory}")
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        将文档及其 Embedding 添加到向量存储。

        Args:
            documents: 要索引的 LangChain Document 对象列表

        Returns:
            文档 ID 列表

        Raises:
            ValueError: documents 为空或 Embedding 生成失败时抛出
            Exception: 索引失败时抛出

        示例:
            >>> ids = manager.add_documents(chunks)
            >>> print(f"已索引 {len(ids)} 个文档")
        """
        if not documents:
            raise ValueError("无法索引空文档列表：documents 为空，请检查 PDF 是否成功解析出文本。")

        logger.info(f"正在添加 {len(documents)} 个文档到向量存储...")

        # 预检查：验证 Embedding 模型能正常生成向量
        try:
            test_embedding = self.embedding_model.embed_query("test")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError(
                    f"Embedding 生成失败：embed_query 返回空结果。"
                    f"请检查 embedding provider '{self.embedding_provider}' 的配置和依赖。"
                )
            logger.debug(f"Embedding 预检查通过，向量维度: {len(test_embedding)}")
        except Exception as e:
            if "Embedding 生成失败" in str(e):
                raise
            raise ValueError(
                f"Embedding 生成失败：{str(e)}。"
                f"请检查 embedding provider '{self.embedding_provider}' 的配置和依赖。"
            ) from e

        ids = self.vector_store.add_documents(documents=documents)
        logger.info(f"成功索引 {len(ids)} 个文档")
        return ids

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_k: int = 3,
        filter: Optional[dict] = None
    ) -> VectorStoreRetriever:
        """
        获取用于相似度搜索的检索器。

        Args:
            search_type: 搜索类型 ("similarity" 或 "mmr")
            search_k: 返回的文档数量
            filter: 可选的 ChromaDB 过滤字典，用于限定搜索范围

        Returns:
            配置好的检索器实例

        示例:
            >>> # 基础检索器
            >>> retriever = manager.get_retriever(search_k=5)

            >>> # 集合范围检索器
            >>> retriever = manager.get_retriever(
            ...     search_k=5,
            ...     filter={"collection_id": "abc123"}
            ... )
        """
        search_kwargs = {"k": search_k}

        if filter:
            search_kwargs["filter"] = filter
            logger.info(f"创建检索器: type={search_type}, k={search_k}, filter={filter}")
        else:
            logger.info(f"创建检索器: type={search_type}, k={search_k}")

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def get_collection_count(self) -> int:
        """
        获取集合中的文档数量。

        Returns:
            向量存储中的文档数量

        示例:
            >>> count = manager.get_collection_count()
            >>> print(f"数据库包含 {count} 个文本块")
        """
        try:
            count = self.vector_store._collection.count()
            logger.debug(f"集合数量: {count}")
            return count
        except Exception as e:
            logger.error(f"获取集合数量失败: {e}")
            return 0

    def clear_collection(self) -> None:
        """
        清空向量存储中的所有文档。

        删除集合并重新创建空集合。

        Raises:
            Exception: 清空失败时抛出

        示例:
            >>> manager.clear_collection()
            >>> # 集合现在为空
        """
        try:
            logger.info("正在清空向量存储集合...")
            self.vector_store.delete_collection()

            # 重新创建空集合
            self.vector_store = self._initialize_vector_store()
            logger.info("向量存储已清空并重新创建")

        except Exception as e:
            logger.error(f"清空向量存储失败: {e}", exc_info=True)
            raise

    def get_indexed_documents(self) -> List[str]:
        """
        获取向量存储中唯一文档来源的列表。

        Returns:
            唯一来源文件名列表

        示例:
            >>> docs = manager.get_indexed_documents()
            >>> print(f"已索引的文档: {docs}")
        """
        try:
            collection = self.vector_store._collection
            result = collection.get(include=["metadatas"])

            sources = set()
            for metadata in result.get("metadatas", []):
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])

            logger.debug(f"找到 {len(sources)} 个已索引文档")
            return sorted(list(sources))

        except Exception as e:
            logger.error(f"获取已索引文档失败: {e}")
            return []

    def document_exists(self, filename: str) -> bool:
        """
        检查指定文件名的文档是否存在于向量存储中。

        Args:
            filename: 要检查的文件名

        Returns:
            文档存在返回 True，否则返回 False

        示例:
            >>> exists = manager.document_exists("report.pdf")
            >>> if exists:
            ...     print("文档已索引！")
        """
        indexed_docs = self.get_indexed_documents()
        return filename in indexed_docs

    def search_similar(
        self,
        query: str,
        k: int = 3,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        搜索相似文档。

        Args:
            query: 搜索查询文本
            k: 返回结果数量
            filter: 可选的 ChromaDB 过滤条件

        Returns:
            相似 Document 对象列表

        示例:
            >>> docs = manager.search_similar("机器学习", k=5)
        """
        query_preview = query[:50] if len(query) > 50 else query
        logger.info(f"正在搜索相似文档: query='{query_preview}...', k={k}, filter={filter}")

        if filter:
            results = self.vector_store.similarity_search(query, k=k, filter=filter)
        else:
            results = self.vector_store.similarity_search(query, k=k)

        logger.info(f"找到 {len(results)} 个相似文档")
        return results

    def search_by_collection(
        self,
        query: str,
        collection_id: str,
        k: int = 3
    ) -> List[Document]:
        """
        在指定集合内搜索。

        Args:
            query: 搜索查询文本
            collection_id: 要搜索的集合 ID
            k: 返回结果数量

        Returns:
            来自该集合的相似 Document 对象列表

        示例:
            >>> docs = manager.search_by_collection(
            ...     "机器学习",
            ...     collection_id="abc123",
            ...     k=5
            ... )
        """
        return self.search_similar(
            query=query,
            k=k,
            filter={"collection_id": {"$eq": collection_id}}
        )

    def search_by_documents(
        self,
        query: str,
        document_ids: List[str],
        k: int = 3
    ) -> List[Document]:
        """
        在指定文档内搜索。

        Args:
            query: 搜索查询文本
            document_ids: 要搜索的文档 ID 列表
            k: 返回结果数量

        Returns:
            来自指定文档的相似 Document 对象列表

        示例:
            >>> docs = manager.search_by_documents(
            ...     "机器学习",
            ...     document_ids=["doc1", "doc2"],
            ...     k=5
            ... )
        """
        if len(document_ids) == 1:
            filter_dict = {"document_id": {"$eq": document_ids[0]}}
        else:
            filter_dict = {"document_id": {"$in": document_ids}}

        return self.search_similar(query=query, k=k, filter=filter_dict)

    def delete_by_document_id(self, document_id: str) -> int:
        """
        从向量存储中删除指定文档的所有文本块。

        Args:
            document_id: 要删除的文档 ID

        Returns:
            删除的文本块数量

        示例:
            >>> deleted = manager.delete_by_document_id("xyz789")
            >>> print(f"已删除 {deleted} 个文本块")
        """
        try:
            collection = self.vector_store._collection

            results = collection.get(
                where={"document_id": {"$eq": document_id}},
                include=[]
            )

            chunk_ids = results.get("ids", [])
            if not chunk_ids:
                logger.info(f"未找到文档 {document_id} 的文本块")
                return 0

            collection.delete(ids=chunk_ids)
            logger.info(f"已删除文档 {document_id} 的 {len(chunk_ids)} 个文本块")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"删除文档 {document_id} 的文本块失败: {e}")
            return 0

    def delete_by_collection_id(self, collection_id: str) -> int:
        """
        从向量存储中删除指定集合的所有文本块。

        Args:
            collection_id: 要删除的集合 ID

        Returns:
            删除的文本块数量

        示例:
            >>> deleted = manager.delete_by_collection_id("abc123")
            >>> print(f"已删除 {deleted} 个文本块")
        """
        try:
            collection = self.vector_store._collection

            results = collection.get(
                where={"collection_id": {"$eq": collection_id}},
                include=[]
            )

            chunk_ids = results.get("ids", [])
            if not chunk_ids:
                logger.info(f"未找到集合 {collection_id} 的文本块")
                return 0

            collection.delete(ids=chunk_ids)
            logger.info(f"已删除集合 {collection_id} 的 {len(chunk_ids)} 个文本块")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"删除集合 {collection_id} 的文本块失败: {e}")
            return 0

    def delete_by_source(self, source: str) -> int:
        """
        删除向量存储中特定来源（文件名）的所有文本块。

        Args:
            source: 要删除的源文件名

        Returns:
            删除的文本块数量

        示例:
            >>> deleted = manager.delete_by_source("document.pdf")
            >>> print(f"已删除 {deleted} 个文本块")
        """
        try:
            collection = self.vector_store._collection

            results = collection.get(
                where={"source": {"$eq": source}},
                include=[]
            )

            chunk_ids = results.get("ids", [])
            if not chunk_ids:
                logger.info(f"未找到来源 {source} 的文本块")
                return 0

            collection.delete(ids=chunk_ids)
            logger.info(f"已删除来源 {source} 的 {len(chunk_ids)} 个文本块")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"删除来源 {source} 的文本块失败: {e}")
            return 0

    def get_chunks_by_document(self, document_id: str) -> List[Document]:
        """
        获取指定文档的所有文本块。

        Args:
            document_id: 要获取文本块的文档 ID

        Returns:
            Document 文本块列表

        示例:
            >>> chunks = manager.get_chunks_by_document("xyz789")
            >>> print(f"找到 {len(chunks)} 个文本块")
        """
        try:
            collection = self.vector_store._collection
            results = collection.get(
                where={"document_id": {"$eq": document_id}},
                include=["documents", "metadatas"]
            )

            chunks = []
            for i, content in enumerate(results.get("documents", [])):
                metadata = results.get("metadatas", [])[i] if results.get("metadatas") else {}
                chunks.append(Document(page_content=content, metadata=metadata))

            logger.debug(f"找到文档 {document_id} 的 {len(chunks)} 个文本块")
            return chunks

        except Exception as e:
            logger.error(f"获取文档 {document_id} 的文本块失败: {e}")
            return []

    def clear_non_collection_documents(self) -> int:
        """
        清空所有不属于任何集合的文档。

        这些是通过首页文件上传器上传的、元数据中没有 collection_id 的旧文档。

        Returns:
            删除的文本块数量

        示例:
            >>> deleted = manager.clear_non_collection_documents()
            >>> print(f"已删除 {deleted} 个旧文本块")
        """
        try:
            collection = self.vector_store._collection

            all_results = collection.get(include=["metadatas"])
            all_ids = all_results.get("ids", [])
            all_metadatas = all_results.get("metadatas", [])

            ids_to_delete = []
            for i, metadata in enumerate(all_metadatas):
                if not metadata or "collection_id" not in metadata:
                    ids_to_delete.append(all_ids[i])

            if not ids_to_delete:
                logger.info("未找到非集合文档")
                return 0

            # 批量删除（ChromaDB 有数量限制）
            batch_size = 5000
            total_deleted = 0
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i + batch_size]
                collection.delete(ids=batch)
                total_deleted += len(batch)

            logger.info(f"已删除 {total_deleted} 个非集合文本块")
            return total_deleted

        except Exception as e:
            logger.error(f"清空非集合文档失败: {e}", exc_info=True)
            return 0

    def clear_all_collection_documents(self) -> int:
        """
        清空所有属于集合的文档。

        这些是元数据中有 collection_id 的文档，通过集合页面添加。

        Returns:
            删除的文本块数量

        示例:
            >>> deleted = manager.clear_all_collection_documents()
            >>> print(f"已删除 {deleted} 个集合文本块")
        """
        try:
            collection = self.vector_store._collection

            all_results = collection.get(include=["metadatas"])
            all_ids = all_results.get("ids", [])
            all_metadatas = all_results.get("metadatas", [])

            ids_to_delete = []
            for i, metadata in enumerate(all_metadatas):
                if metadata and "collection_id" in metadata:
                    ids_to_delete.append(all_ids[i])

            if not ids_to_delete:
                logger.info("未找到集合文档")
                return 0

            batch_size = 5000
            total_deleted = 0
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i + batch_size]
                collection.delete(ids=batch)
                total_deleted += len(batch)

            logger.info(f"已删除 {total_deleted} 个集合文本块")
            return total_deleted

        except Exception as e:
            logger.error(f"清空集合文档失败: {e}", exc_info=True)
            return 0

    def get_non_collection_count(self) -> int:
        """
        获取不属于任何集合的文本块数量。

        Returns:
            非集合文本块数量
        """
        try:
            collection = self.vector_store._collection
            all_results = collection.get(include=["metadatas"])
            all_metadatas = all_results.get("metadatas", [])

            count = sum(1 for m in all_metadatas if not m or "collection_id" not in m)
            return count
        except Exception as e:
            logger.error(f"获取非集合数量失败: {e}")
            return 0

    def get_collection_documents_count(self) -> int:
        """
        获取属于集合的文本块数量。

        Returns:
            集合文本块数量
        """
        try:
            collection = self.vector_store._collection
            all_results = collection.get(include=["metadatas"])
            all_metadatas = all_results.get("metadatas", [])

            count = sum(1 for m in all_metadatas if m and "collection_id" in m)
            return count
        except Exception as e:
            logger.error(f"获取集合文档数量失败: {e}")
            return 0

    def get_all_documents(self, collection_id: Optional[str] = None) -> List[Document]:
        """
        获取向量存储中的所有文档。

        Args:
            collection_id: 可选的集合 ID 过滤条件。
                          为 None 时返回非集合文档。

        Returns:
            Document 对象列表
        """
        try:
            collection = self.vector_store._collection
            all_results = collection.get(include=["documents", "metadatas"])

            documents = []
            contents = all_results.get("documents", [])
            metadatas = all_results.get("metadatas", [])

            for content, metadata in zip(contents, metadatas):
                if collection_id:
                    if metadata and metadata.get("collection_id") == collection_id:
                        documents.append(Document(page_content=content, metadata=metadata or {}))
                else:
                    if not metadata or "collection_id" not in metadata:
                        documents.append(Document(page_content=content, metadata=metadata or {}))

            logger.info(f"从向量存储中获取了 {len(documents)} 个文档")
            return documents

        except Exception as e:
            logger.error(f"获取所有文档失败: {e}", exc_info=True)
            return []
