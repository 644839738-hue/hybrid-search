"""
配置加载模块

负责加载和解析语义搜索应用的 YAML 配置文件。
提供类型安全的配置值访问和验证。
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    语义搜索应用配置加载器。

    加载 YAML 配置文件，提供类型安全的配置值访问。
    验证配置结构并提供清晰的错误信息。

    属性:
        config_path: YAML 配置文件的路径
        config: 加载的配置字典

    示例:
        >>> config = ConfigLoader("config.yaml")
        >>> chunk_size = config.get_chunk_size()
        >>> embedding_model = config.get_embedding_model()
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置加载器。

        Args:
            config_path: YAML 配置文件路径 (默认: config.yaml)

        Raises:
            FileNotFoundError: 配置文件不存在时抛出
            yaml.YAMLError: YAML 文件格式错误时抛出
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """
        从 YAML 文件加载配置。

        Raises:
            FileNotFoundError: 配置文件不存在时抛出
            yaml.YAMLError: YAML 解析失败时抛出
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"配置文件未找到: {self.config_path}\n"
                f"请确保 config.yaml 存在于项目目录中。"
            )

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"配置已从 {self.config_path} 加载")
        except yaml.YAMLError as e:
            logger.error(f"YAML 配置解析失败: {e}")
            raise

    def get_embedding_model(self) -> str:
        """获取 Embedding 模型名称。"""
        return self.config.get("models", {}).get("embedding", {}).get("name", "text-embedding-3-large")

    def get_embedding_provider(self) -> str:
        """获取 Embedding 提供商: 'openai' / 'huggingface' / 'deepseek'。"""
        return self.config.get("models", {}).get("embedding", {}).get("provider", "openai")

    def get_embedding_base_url(self) -> str:
        """获取 Embedding API 基础地址（用于 deepseek 等兼容 OpenAI API 的服务）。"""
        return self.config.get("models", {}).get("embedding", {}).get("base_url", "")

    def get_chat_model(self) -> str:
        """获取对话模型名称。"""
        return self.config.get("models", {}).get("chat", {}).get("name", "gpt-4o-mini")

    def get_chat_provider(self) -> str:
        """获取对话模型提供商: 'openai' / 'deepseek'。"""
        return self.config.get("models", {}).get("chat", {}).get("provider", "openai")

    def get_chat_temperature(self) -> float:
        """获取对话模型温度参数。"""
        return self.config.get("models", {}).get("chat", {}).get("temperature", 0.0)

    def get_chat_base_url(self) -> str:
        """获取对话模型 API 基础地址（用于 DeepSeek 等兼容 OpenAI API 的服务）。"""
        return self.config.get("models", {}).get("chat", {}).get("base_url", "")

    def get_chunk_size(self) -> int:
        """获取文档分块大小。"""
        return self.config.get("document_processing", {}).get("chunk_size", 1000)

    def get_chunk_overlap(self) -> int:
        """获取文档分块重叠长度。"""
        return self.config.get("document_processing", {}).get("chunk_overlap", 200)

    def get_add_start_index(self) -> bool:
        """获取是否为分块添加起始索引。"""
        return self.config.get("document_processing", {}).get("add_start_index", True)

    def get_chunking_strategy(self) -> str:
        """获取切分策略: 'fixed_size' / 'nlp_dynamic'。"""
        return self.config.get("document_processing", {}).get("chunking_strategy", "fixed_size")

    def get_min_chunk_size(self) -> int:
        """获取 NLP 动态切分的最小块大小。"""
        return self.config.get("document_processing", {}).get("min_chunk_size", 300)

    def get_max_chunk_size(self) -> int:
        """获取 NLP 动态切分的最大块大小。"""
        return self.config.get("document_processing", {}).get("max_chunk_size", 1200)

    def get_nlp_chunking_config(self) -> Dict[str, Any]:
        """获取完整的 NLP 动态切分配置。"""
        return self.config.get("document_processing", {}).get("nlp", {
            "sentence_splitter": "regex",
            "enable_semantic_similarity": True,
            "semantic_similarity_threshold": 0.55,
            "semantic_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "overlap_by_sentence": True,
            "preserve_headings": True
        })

    def is_ocr_enabled(self) -> bool:
        """检查是否启用 OCR（扫描版 PDF 自动识别）。"""
        return self.config.get("document_processing", {}).get("enable_ocr", True)

    def get_ocr_engine(self) -> str:
        """获取 OCR 引擎。"""
        return self.config.get("document_processing", {}).get("ocr_engine", "tesseract")

    def get_ocr_languages(self) -> str:
        """获取 OCR 语言包配置。"""
        return self.config.get("document_processing", {}).get("ocr_languages", "chi_sim+eng")

    def get_ocr_dpi(self) -> int:
        """获取 OCR 渲染 DPI。"""
        return self.config.get("document_processing", {}).get("ocr_dpi", 200)

    def get_tesseract_cmd(self) -> str:
        """获取 Tesseract 可执行文件路径。"""
        return self.config.get("document_processing", {}).get("tesseract_cmd", "")

    def get_ocr_config(self) -> Dict[str, Any]:
        """获取完整的 OCR 配置。"""
        return self.config.get("document_processing", {}).get("ocr", {
            "enabled": True,
            "engine": "tesseract",
            "languages": "chi_sim+eng",
            "dpi": 200,
            "tesseract_cmd": ""
        })

    def get_collection_name(self) -> str:
        """获取 ChromaDB 集合名称。"""
        return self.config.get("vector_store", {}).get("collection_name", "semantic_search_docs_streamlit")

    def get_persist_directory(self) -> str:
        """获取 ChromaDB 持久化目录。"""
        return self.config.get("vector_store", {}).get("persist_directory", "./chroma/db")

    def use_chroma_docker(self) -> bool:
        """检查是否启用 ChromaDB Docker 模式。"""
        return self.config.get("vector_store", {}).get("use_docker", False)

    def get_chroma_host(self) -> str:
        """获取 ChromaDB 服务器主机名（Docker 模式）。"""
        return self.config.get("vector_store", {}).get("chroma_host", "localhost")

    def get_chroma_port(self) -> int:
        """获取 ChromaDB 服务器端口（Docker 模式）。"""
        return self.config.get("vector_store", {}).get("chroma_port", 8000)

    def get_search_type(self) -> str:
        """获取向量搜索类型。"""
        return self.config.get("vector_store", {}).get("search_type", "similarity")

    def get_search_k(self) -> int:
        """获取检索返回的文本块数量。"""
        return self.config.get("vector_store", {}).get("search_k", 3)

    def get_retry_config(self) -> Dict[str, int]:
        """获取重试配置。"""
        return self.config.get("retry", {
            "max_attempts": 3,
            "min_wait": 2,
            "max_wait": 10,
            "multiplier": 1
        })

    def get_logging_config(self) -> Dict[str, str]:
        """获取日志配置。"""
        return self.config.get("logging", {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "semantic_search.log"
        })

    def get_qa_system_prompt(self) -> str:
        """获取问答系统提示词模板。"""
        return self.config.get("prompts", {}).get("qa_system",
            "你是一个乐于助人的助手。请仅使用以下信息 {document} 来回答下面的问题 {question}。如果你无法回答问题，请直接说你无法回答该问题。")

    def get_follow_up_system_prompt(self) -> str:
        """获取追问系统提示词模板。"""
        return self.config.get("prompts", {}).get("follow_up_system",
            """你是一个正在参与文档对话的乐于助人的助手。
            请利用之前的对话上下文来理解引用内容并保持连贯性。
            之前的对话: {conversation_context}
            文档上下文: {document}
            请回答以下问题: {question}
            如果你无法回答问题，请直接说你无法回答该问题。""")

    # 混合检索配置
    def is_hybrid_retrieval_enabled(self) -> bool:
        """检查是否启用混合检索。"""
        return self.config.get("hybrid_retrieval", {}).get("enabled", True)

    def get_default_retrieval_method(self) -> str:
        """获取默认检索方式。"""
        return self.config.get("hybrid_retrieval", {}).get("default_method", "hybrid")

    def get_hybrid_alpha(self) -> float:
        """获取混合模式下语义搜索的权重。"""
        return self.config.get("hybrid_retrieval", {}).get("alpha", 0.5)

    def get_rrf_k(self) -> int:
        """获取 RRF 排序融合常数。"""
        return self.config.get("hybrid_retrieval", {}).get("rrf_k", 60)

    def get_bm25_k1(self) -> float:
        """获取 BM25 k1 参数（词频饱和度）。"""
        return self.config.get("hybrid_retrieval", {}).get("bm25", {}).get("k1", 1.5)

    def get_bm25_b(self) -> float:
        """获取 BM25 b 参数（长度归一化）。"""
        return self.config.get("hybrid_retrieval", {}).get("bm25", {}).get("b", 0.75)

    def is_reranking_enabled(self) -> bool:
        """检查是否启用重排序。"""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get("enabled", True)

    def get_reranker_provider(self) -> str:
        """获取重排序提供商。"""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get("provider", "auto")

    def get_cohere_rerank_model(self) -> str:
        """获取 Cohere 重排序模型名称。"""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get(
            "cohere_model", "rerank-english-v3.0")

    def get_jina_rerank_model(self) -> str:
        """获取 Jina 重排序模型名称。"""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get(
            "jina_model", "jinaai/jina-reranker-v1-tiny-en")

    def get_fetch_k_multiplier(self) -> int:
        """获取重排序前候选数量的倍率。"""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get(
            "fetch_k_multiplier", 3)

    # 对话配置
    def is_conversation_enabled(self) -> bool:
        """检查是否启用对话历史。"""
        return self.config.get("conversation", {}).get("enabled", True)

    def get_conversation_storage_dir(self) -> str:
        """获取对话存储目录。"""
        return self.config.get("conversation", {}).get("storage_dir", "./conversation_history")

    def get_max_conversation_history(self) -> int:
        """获取每个会话最多保留的查询数。"""
        return self.config.get("conversation", {}).get("max_history", 50)

    def is_follow_up_optimization_enabled(self) -> bool:
        """检查是否启用追问优化。"""
        return self.config.get("conversation", {}).get("follow_up_optimization", True)

    def get_conversation_context_window(self) -> int:
        """获取追问上下文中使用的最近问答对数。"""
        return self.config.get("conversation", {}).get("context_window", 3)

    # A/B 测试配置
    def is_ab_testing_enabled(self) -> bool:
        """检查是否启用 A/B 测试。"""
        return self.config.get("ab_testing", {}).get("enabled", True)

    def get_ab_testing_storage_dir(self) -> str:
        """获取 A/B 测试存储目录。"""
        return self.config.get("ab_testing", {}).get("storage_dir", "./ab_testing_results")

    def get_default_ab_variants(self) -> list:
        """获取默认 A/B 测试对比方案。"""
        return self.config.get("ab_testing", {}).get("default_variants",
            ["semantic", "bm25", "hybrid", "hybrid_rerank"])

    def get_hybrid_retrieval_config(self) -> Dict[str, Any]:
        """获取完整的混合检索配置。"""
        return self.config.get("hybrid_retrieval", {
            "enabled": True,
            "default_method": "hybrid",
            "alpha": 0.5,
            "rrf_k": 60,
            "bm25": {"k1": 1.5, "b": 0.75},
            "reranking": {
                "enabled": True,
                "provider": "auto",
                "cohere_model": "rerank-english-v3.0",
                "jina_model": "jinaai/jina-reranker-v1-tiny-en",
                "fetch_k_multiplier": 3
            }
        })

    def get_conversation_config(self) -> Dict[str, Any]:
        """获取完整的对话配置。"""
        return self.config.get("conversation", {
            "enabled": True,
            "storage_dir": "./conversation_history",
            "max_history": 50,
            "follow_up_optimization": True,
            "context_window": 3
        })

    def get_ab_testing_config(self) -> Dict[str, Any]:
        """获取完整的 A/B 测试配置。"""
        return self.config.get("ab_testing", {
            "enabled": True,
            "storage_dir": "./ab_testing_results",
            "default_variants": ["semantic", "bm25", "hybrid", "hybrid_rerank"]
        })

    # 检索预设配置
    def get_retrieval_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有检索预设。

        Returns:
            以预设名称为键的预设配置字典。

        示例:
            >>> presets = config.get_retrieval_presets()
            >>> print(presets["high_precision"]["display_name"])
            '高精度'
        """
        presets_config = self.config.get("retrieval_presets", {})

        # 未配置时的默认预设
        default_presets = {
            "high_precision": {
                "display_name": "高精度",
                "description": "更少的结果，更高的相关性。适合精确问题。",
                "icon": "🎯",
                "k": 3,
                "alpha": 0.7,
                "rerank": True,
                "method": "hybrid"
            },
            "balanced": {
                "display_name": "平衡",
                "description": "精度与覆盖率的良好平衡。推荐默认使用。",
                "icon": "⚖️",
                "k": 5,
                "alpha": 0.5,
                "rerank": True,
                "method": "hybrid"
            },
            "high_recall": {
                "display_name": "高召回",
                "description": "更多结果，更广覆盖。适合探索性搜索。",
                "icon": "🔍",
                "k": 10,
                "alpha": 0.3,
                "rerank": False,
                "method": "hybrid"
            }
        }

        # 过滤掉非预设键如 'default_preset'
        presets = {}
        for key, value in presets_config.items():
            if isinstance(value, dict) and "display_name" in value:
                presets[key] = value

        return presets if presets else default_presets

    def get_preset_by_name(self, preset_name: str) -> Dict[str, Any]:
        """
        按名称获取特定预设。

        Args:
            preset_name: 预设名称 (如 "high_precision", "balanced")

        Returns:
            预设配置字典

        Raises:
            KeyError: 预设名称不存在时抛出

        示例:
            >>> preset = config.get_preset_by_name("high_precision")
            >>> print(preset["k"])
            3
        """
        presets = self.get_retrieval_presets()
        if preset_name not in presets:
            raise KeyError(f"未知预设: {preset_name}。可用预设: {list(presets.keys())}")
        return presets[preset_name]

    def get_default_preset(self) -> str:
        """
        获取默认预设名称。

        Returns:
            默认预设名称
        """
        return self.config.get("retrieval_presets", {}).get("default_preset", "balanced")

    def get_preset_names(self) -> list:
        """
        获取所有可用预设名称列表。

        Returns:
            预设名称列表
        """
        return list(self.get_retrieval_presets().keys())


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    从 YAML 文件加载配置。

    便捷函数，创建并返回 ConfigLoader 实例。

    Args:
        config_path: 配置文件路径 (默认: config.yaml)

    Returns:
        配置好的 ConfigLoader 实例

    示例:
        >>> config = load_config()
        >>> chunk_size = config.get_chunk_size()
    """
    return ConfigLoader(config_path)
