"""
侧边栏 UI 组件。

侧边栏检索设置和配置显示组件。
"""

import streamlit as st
from typing import Any


def render_retrieval_settings(config: Any):
    """渲染侧边栏中的检索设置区域。

    Args:
        config: 包含检索设置的应用配置对象。
    """
    st.sidebar.markdown("### 检索设置")

    presets = config.get_retrieval_presets()
    preset_names = list(presets.keys()) + ["custom"]

    def format_preset(preset_name: str) -> str:
        if preset_name == "custom":
            return "⚙️ 自定义"
        preset = presets.get(preset_name, {})
        icon = preset.get("icon", "")
        display = preset.get("display_name", preset_name)
        return f"{icon} {display}"

    selected_preset = st.sidebar.selectbox(
        "检索预设",
        options=preset_names,
        index=preset_names.index(st.session_state.current_preset)
        if st.session_state.current_preset in preset_names else 0,
        format_func=format_preset,
        help="针对不同场景预配置的检索设置"
    )
    st.session_state.current_preset = selected_preset

    if selected_preset != "custom":
        preset = presets[selected_preset]

        st.sidebar.info(f"{preset.get('icon', '')} {preset.get('description', '')}")

        st.session_state.search_k = preset.get("k", 5)
        st.session_state.hybrid_alpha = preset.get("alpha", 0.5)
        st.session_state.use_reranking = preset.get("rerank", True)
        st.session_state.current_retrieval_method = preset.get("method", "hybrid")

        with st.sidebar.expander("预设参数", expanded=False):
            st.markdown(f"**结果数**: {st.session_state.search_k}")
            st.markdown(f"**Alpha**: {st.session_state.hybrid_alpha}")
            st.markdown(f"**重排序**: {'开启' if st.session_state.use_reranking else '关闭'}")
            st.markdown(f"**方式**: {st.session_state.current_retrieval_method}")

    else:
        _render_custom_retrieval_controls(config)

    st.sidebar.markdown("---")


def _render_custom_retrieval_controls(config: Any):
    """渲染自定义检索方式控件。

    Args:
        config: 应用配置对象。
    """
    retrieval_methods = {
        "仅语义搜索": "semantic",
        "仅 BM25": "bm25",
        "混合 (BM25 + 语义)": "hybrid"
    }

    selected_method = st.sidebar.selectbox(
        "检索方式",
        options=list(retrieval_methods.keys()),
        index=list(retrieval_methods.values()).index(
            st.session_state.current_retrieval_method
        ) if st.session_state.current_retrieval_method in retrieval_methods.values() else 2,
        help="选择检索策略"
    )
    st.session_state.current_retrieval_method = retrieval_methods[selected_method]

    if st.session_state.current_retrieval_method == "hybrid":
        st.session_state.hybrid_alpha = st.sidebar.slider(
            "语义搜索权重 (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=getattr(st.session_state, 'hybrid_alpha', config.get_hybrid_alpha()),
            step=0.1,
            help="0 = 仅 BM25, 1 = 仅语义搜索"
        )

    st.session_state.use_reranking = st.sidebar.checkbox(
        "启用重排序",
        value=getattr(st.session_state, 'use_reranking', config.is_reranking_enabled()),
        help="使用交叉编码器重排序以提高准确性"
    )

    st.session_state.search_k = st.sidebar.slider(
        "返回结果数",
        min_value=1,
        max_value=10,
        value=getattr(st.session_state, 'search_k', config.get_search_k()),
        help="要检索的文档文本块数量"
    )


def render_configuration_display(config: Any):
    """渲染侧边栏中的配置展示区域。

    Args:
        config: 应用配置对象。
    """
    with st.sidebar.expander("当前配置", expanded=False):
        st.markdown(f"**Embedding**: {config.get_embedding_model()}")
        st.markdown(f"**对话模型**: {config.get_chat_model()}")
        st.markdown(f"**分块大小**: {config.get_chunk_size()}")

        reranker_provider = config.get_reranker_provider()
        if reranker_provider == "auto":
            st.markdown("**重排序**: auto")
            st.caption("优先级: jina (本地) → cohere (云端)")
        else:
            st.markdown(f"**重排序**: {reranker_provider}")
