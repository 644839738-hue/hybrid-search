"""
文档处理模块

负责 PDF 文档加载和文本分块，支持扫描版 PDF 的 OCR 识别。
支持按集合范围索引以实现过滤检索。

提供两种切分策略:
  - fixed_size: 基于字符数固定长度切分（RecursiveCharacterTextSplitter）
  - nlp_dynamic: 基于 NLP 语义动态切分 + 句子级重叠窗口
"""

import os
import re
import uuid
import tempfile
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class NLPDynamicChunker:
    """
    基于 NLP 的语义动态文本切分器。

    工作流程:
    1. 用正则表达式将文本分割为句子（支持中英文混合）
    2. 可选：用 sentence-transformers 计算句子级语义嵌入
    3. 按语义相似度 + chunk_size 贪婪合并句子
    4. 重叠窗口按完整句子保留，不在字符中间截断
    5. 识别章节标题作为自然切分边界

    属性:
        target_chunk_size: 目标块大小（字符数）
        min_chunk_size: 最小块大小
        max_chunk_size: 最大块大小
        chunk_overlap: 块间重叠字符数
        overlap_by_sentence: 是否按完整句子重叠
        enable_semantic: 是否启用语义相似度计算
        similarity_threshold: 语义相似度阈值
        embedding_model_name: 语义嵌入模型名称
        preserve_headings: 是否保留标题边界
        add_start_index: 是否添加起始索引 metadata
    """

    # 中英文混合句子分割正则
    SENTENCE_PATTERN = re.compile(
        r'(?:[^。！？\n!?.;：；，,]+(?:\.\.\.|…|[。！？.!?]))'
        r'|(?:[^。！？\n.!?;：；，,]+(?:\\n|\n|$))'
        r'|(?:[^。！？\n.!?]+)'
    )

    # 标题/章节识别模式
    HEADING_PATTERN = re.compile(
        r'^\s*(?:第[一二三四五六七八九十\d]+[章节]|'
        r'(?:chapter|section|part)\s+\d+|'
        r'(?:\d+[.、．])\s*\S|'
        r'^(?:[A-Z][^.。！？\n]{0,60}$)|'
        r'^(?:[一二三四五六七八九十]、|（[一二三四五六七八九十]）))',
        re.IGNORECASE | re.MULTILINE
    )

    def __init__(
        self,
        target_chunk_size: int = 800,
        min_chunk_size: int = 300,
        max_chunk_size: int = 1200,
        chunk_overlap: int = 150,
        overlap_by_sentence: bool = True,
        enable_semantic: bool = True,
        similarity_threshold: float = 0.55,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        preserve_headings: bool = True,
        add_start_index: bool = True
    ):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.overlap_by_sentence = overlap_by_sentence
        self.enable_semantic = enable_semantic
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model_name
        self.preserve_headings = preserve_headings
        self.add_start_index = add_start_index
        self._embedding_model = None

        logger.info(
            f"NLPDynamicChunker 初始化: target={target_chunk_size}, "
            f"min={min_chunk_size}, max={max_chunk_size}, "
            f"overlap={chunk_overlap}, semantic={enable_semantic}"
        )

    def _get_embedding_model(self):
        """懒加载语义嵌入模型。"""
        if self._embedding_model is None and self.enable_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device="cpu"
                )
                logger.info(f"语义嵌入模型已加载: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"语义嵌入模型加载失败，回退到纯长度切分: {e}")
                self.enable_semantic = False
        return self._embedding_model

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表按 NLP 动态策略切分为文本块。

        Args:
            documents: LangChain Document 列表（通常每页一个）

        Returns:
            切分后的 Document 列表，每个包含 page_content 和 metadata
        """
        if not documents:
            return []

        all_chunks = []

        for doc in documents:
            text = doc.page_content
            page = doc.metadata.get("page", 1)
            source = doc.metadata.get("source", "")

            if not text.strip():
                continue

            sentences, heading_flags = self._split_sentences(text)

            if self.enable_semantic:
                self._get_embedding_model()
                embeddings = self._compute_embeddings(sentences)
            else:
                embeddings = None

            chunk_texts = self._merge_sentences_into_chunks(
                sentences, embeddings, heading_flags
            )

            for i, chunk_text in enumerate(chunk_texts):
                chunk_meta = {
                    "page": page,
                    "source": source,
                    "chunk_index": i
                }
                if self.add_start_index:
                    chunk_meta["start_index"] = i * self.target_chunk_size

                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_meta
                ))

        # 跨页重新编号 chunk_index
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_index"] = i

        logger.info(
            f"NLP 动态切分完成: {len(documents)} 页 → {len(all_chunks)} 块"
        )
        return all_chunks

    def _split_sentences(self, text: str) -> Tuple[List[str], List[bool]]:
        """
        将文本分割为句子列表。

        支持中英文混合句子边界: . ! ? 。！？换行符。
        同时识别章节标题。

        Args:
            text: 输入文本

        Returns:
            (句子列表, 是否为标题的布尔列表)
        """
        raw_sentences = []
        heading_flags = []

        # 按换行预分割段落
        paragraphs = text.split('\n')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            is_heading = bool(self.HEADING_PATTERN.match(para)) if self.preserve_headings else False

            # 对段落内进行句子分割
            para_sentences = self._regex_split_sentences(para)

            for sent in para_sentences:
                sent = sent.strip()
                if sent:
                    raw_sentences.append(sent)
                    heading_flags.append(is_heading)
                    is_heading = False  # 仅段落首句标记为标题

        return raw_sentences, heading_flags

    def _regex_split_sentences(self, text: str) -> List[str]:
        """
        用正则表达式分割句子，保留标点符号。

        处理:
        - 中文标点: 。！？作为句子边界
        - 英文标点: . ! ? 后跟空格和大写字母作为边界
        - 避免: 数字中的小数点、缩写中的点号
        """
        sentences = []
        buf = ""

        for i, ch in enumerate(text):
            buf += ch

            if ch in '。！？':
                sentences.append(buf)
                buf = ""
            elif ch in '.!?':
                # 英文句子边界: 需要看后续是否为空格+大写或结尾
                rest = text[i + 1:]
                if not rest or (rest[0] in ' \n' and (
                    len(rest) < 2 or rest.lstrip()[0].isupper() or rest.lstrip()[0] in '"\'')):
                    sentences.append(buf)
                    buf = ""

        if buf.strip():
            sentences.append(buf)

        return sentences

    def _compute_embeddings(self, sentences: List[str]):
        """计算句子级语义嵌入向量。"""
        if not self._embedding_model:
            return None
        try:
            embeddings = self._embedding_model.encode(
                sentences,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.warning(f"嵌入计算失败，回退到纯长度切分: {e}")
            self.enable_semantic = False
            return None

    def _merge_sentences_into_chunks(
        self,
        sentences: List[str],
        embeddings,
        heading_flags: List[bool]
    ) -> List[str]:
        """
        核心合并算法: 按语义相似度 + 大小约束贪婪合并句子为文本块。

        Args:
            sentences: 句子列表
            embeddings: 句子嵌入数组（可为 None）
            heading_flags: 每个句子是否为标题

        Returns:
            合并后的文本块列表
        """
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_embeddings = []
        current_len = 0

        for i, (sent, is_heading) in enumerate(zip(sentences, heading_flags)):
            sent_len = len(sent)
            sent_emb = embeddings[i] if embeddings is not None else None

            # 规则1: 遇到标题强制开启新块
            if is_heading and current_sentences and current_len >= self.min_chunk_size:
                chunks.append("".join(current_sentences))
                current_sentences, current_embeddings, current_len = self._start_new_chunk(
                    sentences, embeddings, current_sentences, i, sent, sent_emb
                )
                continue

            # 首个句子直接加入
            if not current_sentences:
                current_sentences.append(sent)
                if sent_emb is not None:
                    current_embeddings.append(sent_emb)
                current_len = sent_len
                continue

            # 规则2: 超过最大块大小，强制开启新块（带重叠）
            if current_len + sent_len > self.max_chunk_size:
                chunks.append("".join(current_sentences))
                current_sentences, current_embeddings, current_len = self._start_new_chunk(
                    sentences, embeddings, current_sentences, i, sent, sent_emb
                )
                continue

            # 规则3: 语义相似度判断
            should_merge = True
            if self.enable_semantic and sent_emb is not None and current_embeddings:
                chunk_centroid = np.mean(current_embeddings, axis=0)
                similarity = float(np.dot(sent_emb, chunk_centroid))
                # 余弦相似度（已归一化）= 内积

                if similarity < self.similarity_threshold:
                    should_merge = False

            if should_merge:
                # 语义相近，合并
                current_sentences.append(sent)
                if sent_emb is not None:
                    current_embeddings.append(sent_emb)
                current_len += sent_len
            elif current_len >= self.min_chunk_size:
                # 语义不同且已达最小块，开启新块
                chunks.append("".join(current_sentences))
                current_sentences, current_embeddings, current_len = self._start_new_chunk(
                    sentences, embeddings, current_sentences, i, sent, sent_emb
                )
            else:
                # 未达最小块，即使语义不同也强制合并
                current_sentences.append(sent)
                if sent_emb is not None:
                    current_embeddings.append(sent_emb)
                current_len += sent_len

        # 最后一个块
        if current_sentences:
            chunks.append("".join(current_sentences))

        return chunks

    def _start_new_chunk(
        self,
        all_sentences: List[str],
        all_embeddings,
        prev_sentences: List[str],
        current_idx: int,
        new_sent: str,
        new_emb
    ) -> Tuple[List[str], List, int]:
        """
        开启新文本块，包含来自前一块的句子级重叠。

        Args:
            all_sentences: 所有句子列表（仅用于非重叠模式）
            all_embeddings: 所有嵌入列表
            prev_sentences: 前一个块的句子列表
            current_idx: 当前句子索引
            new_sent: 新块的第一个句子
            new_emb: 新块第一个句子的嵌入

        Returns:
            (新块句子列表, 新块嵌入列表, 新块总长度)
        """
        if not self.overlap_by_sentence:
            return [new_sent], ([new_emb] if new_emb is not None else []), len(new_sent)

        overlap_sentences = self._get_overlap_sentences(prev_sentences)
        new_sentences = list(overlap_sentences)
        new_sentences.append(new_sent)

        new_len = sum(len(s) for s in new_sentences)

        new_embeddings = []
        if all_embeddings is not None and new_emb is not None:
            new_embeddings = []  # 重叠句子的嵌入暂不复制，避免复杂性
            if new_emb is not None:
                new_embeddings.append(new_emb)

        return new_sentences, new_embeddings, new_len

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        从前一个块末尾取完整句子作为下一个块的重叠前缀。

        根据 chunk_overlap 字符数计算需要保留的句子数，
        确保不从句子中间截断。

        Args:
            sentences: 前一个块的句子列表

        Returns:
            重叠句子列表（从前一块末尾取）
        """
        if not sentences or self.chunk_overlap <= 0:
            return []

        target_overlap = self.chunk_overlap
        accumulated = 0
        overlap_count = 0

        # 从末尾向前数，直到达到目标重叠字符数
        for sent in reversed(sentences):
            accumulated += len(sent)
            overlap_count += 1
            if accumulated >= target_overlap:
                break

        return sentences[-overlap_count:] if overlap_count > 0 else []


class DocumentProcessor:
    """
    Handles PDF document loading and text chunking.

    This class manages the entire document processing pipeline including:
    - Temporary file handling for uploaded PDFs
    - PDF parsing and text extraction
    - Text chunking with configurable parameters

    Attributes:
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between consecutive chunks
        add_start_index: Whether to add start index metadata to chunks

    Example:
        >>> processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        >>> chunks = processor.process_uploaded_file(uploaded_file)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True,
        enable_ocr: bool = True,
        ocr_languages: str = "chi_sim+eng",
        ocr_dpi: int = 200,
        tesseract_cmd: str = "",
        chunking_strategy: str = "fixed_size",
        min_chunk_size: int = 300,
        max_chunk_size: int = 1200,
        nlp_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化文档处理器。

        Args:
            chunk_size: 文本块大小（字符数）
            chunk_overlap: 文本块之间的重叠长度
            add_start_index: 是否为文本块添加起始索引
            enable_ocr: 是否启用 OCR 识别扫描版 PDF
            ocr_languages: OCR 语言包（如 "chi_sim+eng"）
            ocr_dpi: OCR 渲染 DPI
            tesseract_cmd: Tesseract 可执行文件路径
            chunking_strategy: 切分策略 "fixed_size" / "nlp_dynamic"
            min_chunk_size: NLP 动态切分最小块大小
            max_chunk_size: NLP 动态切分最大块大小
            nlp_config: NLP 动态切分详细配置字典
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index
        self.enable_ocr = enable_ocr
        self.ocr_languages = ocr_languages
        self.ocr_dpi = ocr_dpi
        self.tesseract_cmd = tesseract_cmd
        self.chunking_strategy = chunking_strategy
        self.nlp_config = nlp_config or {}

        # 初始化对应的切分器
        if chunking_strategy == "nlp_dynamic":
            self.chunker = NLPDynamicChunker(
                target_chunk_size=chunk_size,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                overlap_by_sentence=self.nlp_config.get("overlap_by_sentence", True),
                enable_semantic=self.nlp_config.get("enable_semantic_similarity", True),
                similarity_threshold=self.nlp_config.get("semantic_similarity_threshold", 0.55),
                embedding_model_name=self.nlp_config.get(
                    "semantic_embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                preserve_headings=self.nlp_config.get("preserve_headings", True),
                add_start_index=add_start_index
            )
            self.text_splitter = None
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=add_start_index
            )
            self.chunker = None

        logger.info(
            f"DocumentProcessor 初始化: chunk_size={chunk_size}, overlap={chunk_overlap}, "
            f"strategy={chunking_strategy}, ocr={'enabled' if enable_ocr else 'disabled'}"
        )

    def process_uploaded_file(
        self,
        uploaded_file,
        collection_id: Optional[str] = None,
        document_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process an uploaded PDF file into text chunks.

        Args:
            uploaded_file: Streamlit uploaded file object
            collection_id: Optional collection ID for scoped retrieval
            document_id: Optional document ID for scoped retrieval
            extra_metadata: Additional metadata to add to all chunks

        Returns:
            List of LangChain Document objects (chunks)

        Raises:
            ValueError: If file is not a PDF
            Exception: If PDF processing fails

        Example:
            >>> chunks = processor.process_uploaded_file(
            ...     uploaded_file,
            ...     collection_id="abc123",
            ...     document_id="xyz789"
            ... )
            >>> print(f"Created {len(chunks)} chunks")
        """
        if not uploaded_file.name.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are supported")

        original_filename = uploaded_file.name
        logger.info(f"Processing file: {original_filename}, size: {uploaded_file.size} bytes")

        # Create temporary file
        temp_file_path = self._create_temp_file(uploaded_file)

        try:
            # Load PDF
            docs = self._load_pdf(temp_file_path)
            logger.info(f"Loaded {len(docs)} pages from PDF")

            # Split into chunks using the configured strategy
            if self.chunking_strategy == "nlp_dynamic" and self.chunker:
                chunks = self.chunker.split_documents(docs)
            else:
                chunks = self.text_splitter.split_documents(docs)
            logger.info(f"Document split into {len(chunks)} chunks")

            # Update metadata for each chunk
            for chunk in chunks:
                chunk.metadata["source"] = original_filename

                # Add collection/document scoping metadata
                if collection_id:
                    chunk.metadata["collection_id"] = collection_id
                if document_id:
                    chunk.metadata["document_id"] = document_id

                # Add any extra metadata
                if extra_metadata:
                    chunk.metadata.update(extra_metadata)

            # Log chunk statistics
            self._log_chunk_stats(chunks)

            return chunks

        finally:
            # Clean up temporary file
            self._cleanup_temp_file(temp_file_path)

    def _create_temp_file(self, uploaded_file) -> str:
        """
        Create a temporary file from uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Path to temporary file
        """
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_extension}")

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Created temporary file: {temp_file_path}")
        return temp_file_path

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        加载 PDF 文档（PyMuPDF 为主，Tesseract OCR 为备）。

        优先使用 PyMuPDF 提取文本；若文本为空且启用 OCR，
        则将每页渲染为图像后调用 Tesseract 识别。

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 对象列表（每页一个）

        Raises:
            Exception: 若 PDF 加载或 OCR 识别全部失败
        """
        logger.info(f"Loading PDF from: {file_path}")

        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) 未安装。请运行: pip install pymupdf"
            )

        pdf_doc = fitz.open(file_path)
        documents = []
        ocr_pages = 0

        for page_num, page in enumerate(pdf_doc):
            text = page.get_text("text").strip()

            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"page": page_num + 1, "source": file_path}
                ))
            elif self.enable_ocr:
                logger.info(f"第 {page_num + 1} 页无文本，尝试 OCR 识别...")
                ocr_text = self._ocr_page(page, page_num)
                if ocr_text:
                    documents.append(Document(
                        page_content=ocr_text,
                        metadata={"page": page_num + 1, "source": file_path, "ocr": True}
                    ))
                    ocr_pages += 1

        pdf_doc.close()

        if not documents:
            raise ValueError(
                "PDF 未提取到任何文本。\n"
                "可能原因:\n"
                "  1. PDF 为扫描版图片，且 OCR 未启用（请在 config.yaml 中设置 enable_ocr: true）\n"
                "  2. Tesseract 未安装或路径不正确（请检查 config.yaml 中的 tesseract_cmd）\n"
                "  3. PDF 文件已损坏或为空白文档"
            )

        if ocr_pages > 0:
            logger.info(f"OCR 识别了 {ocr_pages}/{page_num + 1} 页")

        logger.info(f"Loaded {len(documents)} pages from PDF")
        return documents

    def _ocr_page(self, page, page_num: int) -> str:
        """
        对单页 PDF 执行 OCR 识别。

        Args:
            page: PyMuPDF Page 对象
            page_num: 页码（0-based）

        Returns:
            识别出的文本，失败返回空字符串
        """
        try:
            from PIL import Image
            import pytesseract
        except ImportError as e:
            logger.error(f"OCR 依赖未安装: {e}")
            return ""

        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        try:
            mat = page.get_pixmap(dpi=self.ocr_dpi)
            img = Image.frombytes("RGB", [mat.width, mat.height], mat.samples)
            text = pytesseract.image_to_string(img, lang=self.ocr_languages).strip()
            return text
        except Exception as e:
            logger.warning(f"第 {page_num + 1} 页 OCR 失败: {e}")
            return ""

    def _log_chunk_stats(self, chunks: List[Document]) -> None:
        """
        Log statistics about document chunks.

        Args:
            chunks: List of document chunks
        """
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i}: {len(chunk.page_content)} characters")

        if chunks:
            avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
            logger.info(f"Average chunk size: {avg_size:.0f} characters")

    def _cleanup_temp_file(self, file_path: str) -> None:
        """
        Remove temporary file.

        Args:
            file_path: Path to temporary file
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")

    def get_chunk_info(self, chunks: List[Document]) -> List[dict]:
        """
        Get information about each chunk.

        Args:
            chunks: List of document chunks

        Returns:
            List of dictionaries with chunk metadata

        Example:
            >>> info = processor.get_chunk_info(chunks)
            >>> for item in info:
            ...     print(f"Chunk {item['index']}: {item['size']} chars")
        """
        return [
            {
                "index": i + 1,
                "size": len(chunk.page_content),
                "metadata": chunk.metadata
            }
            for i, chunk in enumerate(chunks)
        ]
