"""
语义搜索引擎 - Streamlit 应用

基于模块化 RAG（检索增强生成）架构，使用 ChromaDB 向量存储和可配置的 Embedding/LLM，
实现对 PDF 文档的语义搜索。

功能:
    - PDF 文档上传与处理
    - 可配置参数的文本分块
    - 可配置提供商的向量 Embedding（OpenAI / HuggingFace / DeepSeek）
    - ChromaDB 向量存储（持久化）
    - 混合检索（BM25 + 语义搜索）
    - Cohere/Jina 重排序
    - 对话历史与追问优化
    - 检索方式 A/B 测试框架
    - 可配置 LLM 的问答（OpenAI / DeepSeek）
    - 上下文展示（透明可查）

所需环境变量:
    DEEPSEEK_API_KEY / OPENAI_API_KEY: 所选提供商的 API Key
    COHERE_API_KEY: (可选) Cohere 重排序 API Key

使用方式:
    streamlit run app.py

作者: Harsh
仓库: https://github.com/shrimpy8/semantic-serach
"""

import time
import streamlit as st
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config_loader import load_config
from core import (
    DocumentProcessor,
    VectorStoreManager,
    QAChain,
    HybridRetriever,
    RetrievalMethod,
    create_hybrid_retriever,
    ConversationManager,
    ABTestingManager,
    TestVariant
)
from utils import add_documents_with_retry, stream_llm_with_retry
from ui import (
    apply_page_styles,
    render_sidebar_header,
    render_retrieval_settings,
    render_configuration_display,
)

# Load configuration
config = load_config()

# Configure structured logging
logging_config = config.get_logging_config()
logging.basicConfig(
    level=getattr(logging, logging_config["level"]),
    format=logging_config["format"],
    handlers=[
        logging.FileHandler(logging_config["file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize all session state variables."""
    if 'vector_store_manager' not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager(
            embedding_model_name=config.get_embedding_model(),
            embedding_provider=config.get_embedding_provider(),
            embedding_base_url=config.get_embedding_base_url(),
            collection_name=config.get_collection_name(),
            persist_directory=config.get_persist_directory(),
            use_docker=config.use_chroma_docker(),
            chroma_host=config.get_chroma_host(),
            chroma_port=config.get_chroma_port()
        )

    if 'qa_chain' not in st.session_state:
        retriever = st.session_state.vector_store_manager.get_retriever(
            search_type=config.get_search_type(),
            search_k=config.get_search_k()
        )
        st.session_state.qa_chain = QAChain(
            model_name=config.get_chat_model(),
            temperature=config.get_chat_temperature(),
            retriever=retriever,
            system_prompt=config.get_qa_system_prompt(),
            provider=config.get_chat_provider(),
            base_url=config.get_chat_base_url()
        )

    if 'hybrid_retriever' not in st.session_state:
        st.session_state.hybrid_retriever = None

    if 'documents' not in st.session_state:
        st.session_state.documents = []

    # Flag for triggering sidebar refresh after upload
    if 'needs_sidebar_refresh' not in st.session_state:
        st.session_state.needs_sidebar_refresh = False

    # Check for existing documents and restore state on restart
    if 'processed_file' not in st.session_state:
        existing_doc_count = st.session_state.vector_store_manager.get_non_collection_count()
        if existing_doc_count > 0:
            logger.info(f"Found {existing_doc_count} existing documents in vector store, restoring state")
            st.session_state.processed_file = True
            # Initialize hybrid retriever for existing documents
            if st.session_state.hybrid_retriever is None:
                semantic_retriever = st.session_state.vector_store_manager.get_retriever(
                    search_k=config.get_search_k() * config.get_fetch_k_multiplier()
                )
                # Get existing documents for BM25
                existing_docs = st.session_state.vector_store_manager.get_all_documents()
                st.session_state.hybrid_retriever = create_hybrid_retriever(
                    semantic_retriever=semantic_retriever,
                    documents=existing_docs,
                    enable_reranker=config.is_reranking_enabled(),
                    reranker_provider=config.get_reranker_provider(),
                    alpha=config.get_hybrid_alpha(),
                    rrf_k=config.get_rrf_k(),
                    bm25_k1=config.get_bm25_k1(),
                    bm25_b=config.get_bm25_b()
                )
        else:
            st.session_state.processed_file = False

    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager(
            storage_dir=config.get_conversation_storage_dir(),
            max_history=config.get_max_conversation_history()
        )

    if 'ab_testing_manager' not in st.session_state:
        st.session_state.ab_testing_manager = ABTestingManager(
            storage_dir=config.get_ab_testing_storage_dir()
        )

    if 'current_retrieval_method' not in st.session_state:
        st.session_state.current_retrieval_method = config.get_default_retrieval_method()


def render_sidebar():
    """Render sidebar with configuration and management options."""
    # Branding and navigation (shared component)
    render_sidebar_header()

    # Initialize preset in session state
    if 'current_preset' not in st.session_state:
        st.session_state.current_preset = config.get_default_preset()

    # Retrieval settings (shared component)
    render_retrieval_settings(config)

    # Configuration display (shared component)
    render_configuration_display(config)

@st.dialog("清空所有文档")
def confirm_clear_documents():
    """清空非集合文档的确认对话框。

    仅清空直接上传到首页的文档。
    集合文档需在集合页面单独管理。
    """
    non_collection_count = st.session_state.vector_store_manager.get_non_collection_count()

    if non_collection_count == 0:
        st.info("没有可清空的文档。数据库中暂无独立文档。")
        if st.button("关闭", use_container_width=True):
            st.rerun()
        return

    st.warning(
        f"⚠️ 这将永久删除 **{non_collection_count}** 个直接上传到本页面的文档文本块。"
    )
    st.caption("注意：集合中的文档不受影响。请在集合页面管理它们。")
    st.markdown("确定要继续吗？")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("确认清空", type="primary", use_container_width=True):
            try:
                deleted = st.session_state.vector_store_manager.clear_non_collection_documents()
                st.session_state.processed_file = False
                st.session_state.documents = []
                st.session_state.hybrid_retriever = None
                logger.info(f"用户清空了 {deleted} 个非集合文档")
                st.rerun()
            except Exception as e:
                st.error(f"清空数据库失败: {str(e)}")
                logger.error(f"清空向量存储失败: {e}", exc_info=True)
    with col2:
        if st.button("取消", use_container_width=True):
            st.rerun()


def render_database_management():
    """Render database management section at bottom of sidebar.

    Shows count of non-collection documents only. Collection documents
    are managed separately through the Collections page.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 数据库管理")

    try:
        non_collection_count = st.session_state.vector_store_manager.get_non_collection_count()
        if non_collection_count > 0:
            st.sidebar.success(f"**{non_collection_count}** 个文档文本块已索引")
        else:
            st.sidebar.info("暂无已索引的文档。请上传文档以开始使用。")
    except Exception as e:
        st.sidebar.warning(f"无法检查数据库状态: {str(e)}")
        logger.error(f"检查数据库状态失败: {e}")

    if st.sidebar.button("清空所有文档"):
        confirm_clear_documents()


@st.dialog("清空对话历史")
def confirm_clear_history():
    """清空对话历史的确认对话框。"""
    st.warning("⚠️ 这将删除当前的对话历史。")
    st.markdown("确定要继续吗？")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("确认清空", type="primary", use_container_width=True):
            if st.session_state.conversation_manager.current_session:
                st.session_state.conversation_manager.delete_session(
                    st.session_state.conversation_manager.current_session.session_id
                )
                st.session_state.conversation_manager.start_session()
                logger.info("用户清空了对话历史")
            st.rerun()
    with col2:
        if st.button("取消", use_container_width=True):
            st.rerun()


def render_conversation_history():
    """Render conversation history panel."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 对话历史")

    if st.session_state.conversation_manager.current_session:
        history = st.session_state.conversation_manager.get_query_history(n=5)
        if history:
            with st.sidebar.expander(f"最近查询 ({len(history)})", expanded=False):
                for i, item in enumerate(reversed(history)):
                    st.markdown(f"**Q{len(history)-i}:** {item['query'][:50]}...")
                    st.caption(f"方式: {item['retrieval_method']}")
                    st.markdown("---")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("新建会话"):
            st.session_state.conversation_manager.start_session(
                document_name=getattr(st.session_state, 'current_doc_name', None)
            )
            st.rerun()
    with col2:
        if st.button("清空历史"):
            confirm_clear_history()


def render_documents_panel():
    """Render documents panel showing stats and document list."""
    chunk_count = st.session_state.vector_store_manager.get_non_collection_count()

    if chunk_count == 0:
        return

    with st.expander(f"📄 文档 ({chunk_count} 个文本块)", expanded=False):
        stat_cols = st.columns(4)
        with stat_cols[0]:
            docs = st.session_state.vector_store_manager.get_all_documents()
            unique_sources = set(doc.metadata.get("source", "未知") for doc in docs)
            st.metric("文档数", len(unique_sources))
        with stat_cols[1]:
            st.metric("文本块", chunk_count)
        with stat_cols[2]:
            st.metric("分块大小", config.get_chunk_size())
        with stat_cols[3]:
            st.metric("重叠长度", config.get_chunk_overlap())

        st.divider()

        if docs:
            doc_stats = {}
            for doc in docs:
                source = doc.metadata.get("source", "未知")
                if source not in doc_stats:
                    doc_stats[source] = {"chunks": 0, "pages": set()}
                doc_stats[source]["chunks"] += 1
                if doc.metadata.get("page"):
                    doc_stats[source]["pages"].add(doc.metadata.get("page"))

            st.markdown("**文档列表:**")
            for source, stats in doc_stats.items():
                filename = source.split("/")[-1] if "/" in source else source
                page_info = f", {len(stats['pages'])} 页" if stats['pages'] else ""
                st.markdown(f"📄 **{filename}** — {stats['chunks']} 个文本块{page_info}")
        else:
            st.info("暂未上传文档。")


def render_ab_testing_panel():
    """Render A/B testing panel."""
    if not config.is_ab_testing_enabled():
        return

    with st.expander("A/B 测试", expanded=False):
        st.markdown("### 对比检索方式")

        if st.session_state.processed_file and st.session_state.hybrid_retriever:
            test_query = st.text_input(
                "测试查询",
                placeholder="输入查询以对比不同检索方式...",
                key="ab_test_query"
            )

            if st.button("运行对比") and test_query:
                with st.spinner("正在运行 A/B 对比..."):
                    run_ab_comparison(test_query)

            if st.session_state.ab_testing_manager.current_experiment:
                summary = st.session_state.ab_testing_manager.get_comparison_summary()
                if summary.get("total_tests", 0) > 0:
                    st.markdown("### 结果")

                    cols = st.columns(len(summary.get("variants", {})))
                    for i, (variant, stats) in enumerate(summary.get("variants", {}).items()):
                        if stats:
                            with cols[i]:
                                st.metric(
                                    label=variant.upper(),
                                    value=f"{stats.get('avg_score', {}).get('mean', 0):.3f}",
                                    delta=f"{stats.get('latency', {}).get('mean', 0):.0f}ms"
                                )

                    if summary.get("recommendation", {}).get("best_variant"):
                        st.success(
                            f"推荐: **{summary['recommendation']['best_variant']}** "
                            f"(平均分: {summary['recommendation']['best_avg_score']:.3f})"
                        )

                    if st.button("导出结果 (CSV)"):
                        csv_data = st.session_state.ab_testing_manager.export_results("csv")
                        if csv_data:
                            st.download_button(
                                label="下载 CSV",
                                data=csv_data,
                                file_name="ab_test_results.csv",
                                mime="text/csv"
                            )
        else:
            st.info("请上传文档以启用 A/B 测试。")


def run_ab_comparison(query: str):
    """Run A/B comparison test for a query."""
    if not st.session_state.ab_testing_manager.current_experiment:
        st.session_state.ab_testing_manager.create_experiment(
            name=f"Comparison - {query[:30]}",
            description="Automated A/B comparison"
        )

    def retriever_func(q, method, k):
        method_map = {
            "semantic": RetrievalMethod.SEMANTIC,
            "bm25": RetrievalMethod.BM25,
            "hybrid": RetrievalMethod.HYBRID,
            "hybrid_rerank": RetrievalMethod.HYBRID
        }
        use_rerank = method == "hybrid_rerank"
        return st.session_state.hybrid_retriever.retrieve(
            q, k=k,
            method=method_map.get(method, RetrievalMethod.HYBRID),
            use_reranker=use_rerank
        )

    variants = [TestVariant.CONTROL, TestVariant.VARIANT_A, TestVariant.VARIANT_B]
    if st.session_state.hybrid_retriever.reranker:
        variants.append(TestVariant.VARIANT_C)

    st.session_state.ab_testing_manager.run_comparison(
        query=query,
        retriever_func=retriever_func,
        variants=variants,
        k=st.session_state.search_k
    )


def process_uploaded_file(uploaded_file, force_reindex: bool = False):
    """Process an uploaded PDF file.

    Args:
        uploaded_file: Streamlit uploaded file object
        force_reindex: If True, delete existing chunks before re-indexing
    """
    if not uploaded_file.name.lower().endswith('.pdf'):
        st.error("目前仅支持 PDF 文件。")
        logger.warning(f"上传了无效的文件类型: {uploaded_file.name}")
        return

    if force_reindex:
        with st.spinner("正在移除旧文档数据..."):
            deleted = st.session_state.vector_store_manager.delete_by_source(uploaded_file.name)
            if deleted > 0:
                logger.info(f"已删除 {uploaded_file.name} 的 {deleted} 个旧文本块")

    with st.spinner("正在处理文档..."):
        try:
            doc_processor = DocumentProcessor(
                chunk_size=config.get_chunk_size(),
                chunk_overlap=config.get_chunk_overlap(),
                add_start_index=config.get_add_start_index(),
                enable_ocr=config.is_ocr_enabled(),
                ocr_languages=config.get_ocr_languages(),
                ocr_dpi=config.get_ocr_dpi(),
                tesseract_cmd=config.get_tesseract_cmd(),
                chunking_strategy=config.get_chunking_strategy(),
                min_chunk_size=config.get_min_chunk_size(),
                max_chunk_size=config.get_max_chunk_size(),
                nlp_config=config.get_nlp_chunking_config()
            )

            st.info(f"正在处理: **{uploaded_file.name}**")
            chunks = doc_processor.process_uploaded_file(uploaded_file)

            if not chunks:
                st.error(
                    "PDF 未解析出任何文本。\n\n"
                    "可能原因:\n"
                    "  1. PDF 为扫描版图片，且 OCR 未启用（请在 config.yaml 中设置 document_processing.enable_ocr: true）\n"
                    "  2. Tesseract OCR 未安装或路径不正确（请检查 config.yaml 中的 document_processing.tesseract_cmd）\n"
                    "  3. PDF 文件已损坏或为空白文档\n\n"
                    "Windows Tesseract 安装: https://github.com/UB-Mannheim/tesseract/wiki"
                )
                logger.warning(f"PDF 解析为空: {uploaded_file.name}（OCR 也未识别出文本）")
                return

            st.session_state.documents = chunks
            st.session_state.current_doc_name = uploaded_file.name

            with st.spinner("正在创建 Embedding 并索引..."):
                chroma_ids = add_documents_with_retry(
                    st.session_state.vector_store_manager.vector_store,
                    chunks
                )

            st.success(f"✅ 文档处理完成: 创建了 **{len(chunks)}** 个文本块, **{len(chroma_ids)}** 个 Embedding 已索引")

            chunk_info = doc_processor.get_chunk_info(chunks)
            with st.expander("查看文本块详情", expanded=False):
                for info in chunk_info[:5]:
                    st.write(f"文本块 {info['index']}: {info['size']} 字符")
                if len(chunk_info) > 5:
                    st.write(f"... 还有 {len(chunk_info) - 5} 个")

            # Initialize hybrid retriever
            semantic_retriever = st.session_state.vector_store_manager.get_retriever(
                search_k=st.session_state.search_k * config.get_fetch_k_multiplier()
            )

            st.session_state.hybrid_retriever = create_hybrid_retriever(
                semantic_retriever=semantic_retriever,
                documents=chunks,
                enable_reranker=config.is_reranking_enabled(),
                reranker_provider=config.get_reranker_provider(),
                alpha=config.get_hybrid_alpha(),
                rrf_k=config.get_rrf_k(),
                bm25_k1=config.get_bm25_k1(),
                bm25_b=config.get_bm25_b()
            )

            # Start conversation session
            st.session_state.conversation_manager.start_session(
                document_name=uploaded_file.name
            )

            # Update session state
            st.session_state.processed_file = True
            st.session_state.needs_sidebar_refresh = True
            logger.info(f"File processing complete: {uploaded_file.name}")

        except ValueError as e:
            st.error(f"验证错误: {str(e)}")
            logger.error(f"验证错误: {e}")

        except Exception as e:
            st.error(f"处理文件失败: {str(e)}")
            logger.error(f"处理文件 {uploaded_file.name} 失败: {e}", exc_info=True)


def handle_question(prompt: str):
    """处理用户问题并生成回答。"""
    if not st.session_state.processed_file:
        st.error("请先上传文档再提问。")
        return

    st.markdown(f"**问题:** {prompt}")

    optimized_query = prompt
    if config.is_follow_up_optimization_enabled():
        optimized_query = st.session_state.conversation_manager.optimize_follow_up_query(
            prompt,
            include_context=True
        )

    with st.spinner("正在搜索答案..."):
        try:
            start_time = time.perf_counter()

            if st.session_state.hybrid_retriever:
                method_map = {
                    "semantic": RetrievalMethod.SEMANTIC,
                    "bm25": RetrievalMethod.BM25,
                    "hybrid": RetrievalMethod.HYBRID
                }
                method = method_map.get(
                    st.session_state.current_retrieval_method,
                    RetrievalMethod.HYBRID
                )

                alpha = getattr(st.session_state, 'hybrid_alpha', config.get_hybrid_alpha())

                results = st.session_state.hybrid_retriever.retrieve(
                    optimized_query,
                    k=st.session_state.search_k,
                    method=method,
                    alpha=alpha,
                    use_reranker=st.session_state.use_reranking
                )

                retrieval_time = (time.perf_counter() - start_time) * 1000

                if not results:
                    logger.warning("未找到相关信息")
                    st.warning("未找到相关信息。请尝试换个方式提问。")
                    return

                docs_retrieved = [r.document for r in results]
                scores = [r.final_score for r in results]

                with st.expander("回答所使用的上下文", expanded=False):
                    preset_info = f"预设: **{st.session_state.current_preset}**" if st.session_state.current_preset != "custom" else "预设: **自定义**"
                    st.caption(f"{preset_info} | 方式: **{method.value}** | 耗时: {retrieval_time:.0f}ms")

                    for i, result in enumerate(results):
                        score_parts = []
                        if result.semantic_score is not None:
                            score_parts.append(f"语义: {result.semantic_score:.3f}")
                        if result.bm25_score is not None:
                            score_parts.append(f"BM25: {result.bm25_score:.3f}")
                        if result.rerank_score is not None:
                            score_parts.append(f"重排序: {result.rerank_score:.3f}")

                        st.markdown(f"**文本块 {i+1}** (最终得分: {result.final_score:.4f})")

                        if score_parts:
                            st.caption(" | ".join(score_parts))

                        st.code(result.document.page_content, language=None)

            else:
                docs_retrieved = st.session_state.qa_chain.retrieve_context(prompt)
                scores = [1.0 / (i + 1) for i in range(len(docs_retrieved))]
                retrieval_time = 0

                if not docs_retrieved:
                    logger.warning("未找到相关信息")
                    st.warning("未找到相关信息。请尝试换个方式提问。")
                    return

                with st.expander("回答所使用的上下文", expanded=False):
                    for i, doc in enumerate(docs_retrieved):
                        st.markdown(f"**文本块 {i+1}:**\n```\n{doc.page_content}\n```")

            context = st.session_state.qa_chain.format_context(docs_retrieved)

            st.subheader("回答:")
            answer_placeholder = st.empty()

            full_answer = ""
            for chunk in stream_llm_with_retry(
                st.session_state.qa_chain.llm_model,
                st.session_state.qa_chain.prompt_template.invoke({
                    "question": prompt,
                    "document": context
                })
            ):
                full_answer += chunk.content
                answer_placeholder.write(full_answer)

            st.session_state.conversation_manager.add_query(
                query=prompt,
                answer=full_answer,
                retrieved_docs=docs_retrieved,
                scores=scores,
                retrieval_method=st.session_state.current_retrieval_method
            )

            logger.info(f"回答已生成: {len(full_answer)} 字符")

        except Exception as e:
            st.error(f"生成回答失败: {str(e)}")
            logger.error(f"生成回答失败: {e}", exc_info=True)


def render_help_section():
    """Render help section with 2-column layout and learn more link."""
    with st.expander("应用使用说明", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### RAG 流程概览

            1. **上传 PDF**: 应用处理 PDF 并将其分割成较小的文本块。
            2. **文档索引**: 每个文本块转换为向量 Embedding 并存储到 ChromaDB。
            3. **混合搜索**: 当你提问时，应用通过以下方式查找相关文本块:
               - **语义搜索**: 查找概念上相似的内容
               - **BM25 (关键词)**: 查找精确匹配的词汇
               - **重排序**: 可选地对结果重新评分，提高准确性
            4. **生成回答**: LLM 仅根据检索到的文本块生成回答。

            ---

            ### 检索方式

            - **仅语义**: 使用 Embedding 相似度（适合概念性问题）
            - **仅 BM25**: 使用关键词匹配（适合精确词汇）
            - **混合**: 结合两种方式（推荐日常使用）
            """)

        with col2:
            st.markdown("""
            ### 获取更好结果的技巧

            - 提出文档中可能回答的具体问题
            - 如果收到"无法回答"的回复，请尝试换个方式提问
            - 使用 alpha 滑块调节语义搜索和关键词搜索的平衡
            - 查看上下文展开区域了解使用了哪些信息
            - 追问会自动利用对话上下文

            ---

            ### 技术栈

            - **Embeddings**: 可配置 (OpenAI / HuggingFace / DeepSeek)
            - **向量存储**: ChromaDB（持久化）
            - **LLM**: 可配置 (OpenAI / DeepSeek)
            - **重排序**: Cohere / Jina（可选）
            - **框架**: LangChain + Streamlit
            """)

        st.markdown("---")
        st.page_link(
            "pages/1_How_It_Works.py",
            label="📚 了解更多关于优化语义搜索的方法",
            icon="🔗"
        )


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Page configuration
    st.set_page_config(
        page_title="语义搜索引擎",
        page_icon="magnifying_glass_tilted_left:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check if sidebar needs refresh (after document upload)
    if st.session_state.get('needs_sidebar_refresh', False):
        st.session_state.needs_sidebar_refresh = False
        st.rerun()

    # Apply shared page styles (hide nav + base styles)
    apply_page_styles()

    # Render sidebar
    render_sidebar()
    render_conversation_history()
    render_database_management()

    # Main content
    st.title("文档搜索")
    st.markdown("上传 PDF 文档，使用混合语义搜索进行提问。")

    render_help_section()

    st.page_link(
        "pages/1_How_It_Works.py",
        label="📚 了解更多关于优化语义搜索的方法 →",
        icon=None
    )

    st.page_link(
        "pages/2_Collections.py",
        label="📁 高级搜索: 将文档组织到可搜索的集合中 →",
        icon=None
    )

    # File upload
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "选择 PDF 文件",
        type=['pdf'],
        help="上传 PDF 文档以启用语义搜索"
    )

    if uploaded_file is not None:
        if st.session_state.vector_store_manager.document_exists(uploaded_file.name):
            st.info(
                f"ℹ️ **'{uploaded_file.name}'** 已索引，可以直接搜索！"
                "在下方输入问题即可搜索该文档。"
            )
            with st.expander("重新索引此文档？"):
                st.caption("仅在文件内容发生变化时需要。")
                if st.button("重新索引文档", type="secondary"):
                    process_uploaded_file(uploaded_file, force_reindex=True)
        else:
            process_uploaded_file(uploaded_file)

    render_documents_panel()
    render_ab_testing_panel()

    st.markdown("---")
    st.subheader("对文档提问")
    st.caption("搜索范围包括所有已上传的文档和集合。")

    col1, col2 = st.columns([5, 1])
    with col1:
        question_input = st.text_input(
            "你的问题",
            placeholder="在此输入你的问题...",
            label_visibility="collapsed",
            key="question_input"
        )
    with col2:
        ask_button = st.button("提问", type="primary", use_container_width=True)

    if ask_button and question_input:
        handle_question(question_input)
    elif ask_button and not question_input:
        st.warning("请输入问题。")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "基于 LangChain、ChromaDB、DeepSeek 构建 | "
        f"<a href='https://github.com/shrimpy8/semantic-serach'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
