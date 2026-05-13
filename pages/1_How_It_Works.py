"""
How It Works - Interactive Documentation Page

This page provides interactive documentation explaining how the semantic search
engine works, including retrieval methods, configurations, and A/B testing.
"""

import streamlit as st

st.set_page_config(
    page_title="How It Works - Semantic Search",
    page_icon="📚",
    layout="wide"
)

from ui import apply_page_styles, render_sidebar_header

# Apply shared page styles (hide nav + base styles)
apply_page_styles()

# Sidebar branding and navigation (shared component)
render_sidebar_header()

st.title("📚 How It Works")
st.markdown("*Interactive guide to understanding and optimizing your semantic search*")

# Brief Overview Section
st.subheader("🔄 How Search Works")
st.markdown("""
When you ask a question, here's what happens behind the scenes:
""")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <div style="font-size: 2rem;">📄</div>
        <div style="font-weight: bold;">1. Chunk</div>
        <div style="font-size: 0.8rem; color: gray;">Document split into searchable pieces</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <div style="font-size: 2rem;">🧮</div>
        <div style="font-weight: bold;">2. Embed</div>
        <div style="font-size: 0.8rem; color: gray;">Embedding model creates vectors</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <div style="font-size: 2rem;">🔀</div>
        <div style="font-weight: bold;">3. Hybrid Search</div>
        <div style="font-size: 0.8rem; color: gray;">BM25 + Semantic combined</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <div style="font-size: 2rem;">🏆</div>
        <div style="font-weight: bold;">4. Rerank</div>
        <div style="font-size: 0.8rem; color: gray;">Jina cross-encoder refines results</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <div style="font-size: 2rem;">💬</div>
        <div style="font-weight: bold;">5. Answer</div>
        <div style="font-size: 0.8rem; color: gray;">GPT-4o-mini generates response</div>
    </div>
    """, unsafe_allow_html=True)

st.caption("This is a RAG (Retrieval-Augmented Generation) pipeline — the LLM only answers based on your documents, not general knowledge.")

st.divider()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Retrieval Methods",
    "📁 Collections",
    "📊 Precision vs Recall",
    "⚙️ Configuration Guide",
    "🧪 A/B Testing",
    "💬 Conversation History",
    "🎯 Quick Reference"
])

# =============================================================================
# TAB 1: Retrieval Methods
# =============================================================================
with tab1:
    st.header("Understanding Retrieval Methods")

    st.markdown("""
    The search engine supports multiple retrieval strategies. Each has strengths
    and weaknesses depending on your query type and document content.
    """)

    # Method comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🎯 Semantic Search")
        st.markdown("""
        **How it works:**
        - Converts query to a vector (embedding)
        - Finds similar vectors in the database
        - Understands meaning, not just keywords

        **Best for:**
        - Conceptual questions
        - Paraphrased queries
        - "What is..." questions

        **Example:**
        > Query: "How do neural networks learn?"
        >
        > Finds: "backpropagation", "gradient descent", "training process"
        """)
        st.info("💡 Use when asking about **concepts** and **ideas**")

    with col2:
        st.subheader("📝 BM25 Search")
        st.markdown("""
        **How it works:**
        - Scores based on term frequency
        - Exact and partial keyword matching
        - Classic information retrieval algorithm

        **Best for:**
        - Exact term searches
        - Technical terminology
        - Known phrases

        **Example:**
        > Query: "API authentication endpoint"
        >
        > Finds: Exact matches for those terms
        """)
        st.info("💡 Use when searching for **specific terms** or **technical keywords**")

    with col3:
        st.subheader("🔀 Hybrid Search")
        st.markdown("""
        **How it works:**
        - Runs both methods in parallel
        - Combines using Reciprocal Rank Fusion
        - Alpha controls the balance

        **Best for:**
        - General-purpose queries
        - When unsure which method is better
        - Balancing precision and recall

        **Example:**
        > Query: "machine learning classification"
        >
        > Gets best of both: exact matches + related concepts
        """)
        st.success("✅ **Recommended** as the default method")

    st.divider()

    # Interactive Alpha Explainer
    st.subheader("🎚️ Understanding the Alpha Parameter")

    alpha_demo = st.slider(
        "Move the slider to see how Alpha affects retrieval:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Alpha controls the balance between semantic and BM25 search"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Visual representation
        semantic_pct = int(alpha_demo * 100)
        bm25_pct = 100 - semantic_pct

        st.markdown(f"""
        <div style="display: flex; height: 40px; border-radius: 5px; overflow: hidden; margin: 20px 0;">
            <div style="width: {semantic_pct}%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                Semantic {semantic_pct}%
            </div>
            <div style="width: {bm25_pct}%; background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                BM25 {bm25_pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        if alpha_demo == 0:
            st.warning("⚠️ Pure BM25 - Only keyword matching, no semantic understanding")
        elif alpha_demo < 0.3:
            st.info("📝 BM25-heavy - Good for technical documentation with specific terms")
        elif alpha_demo < 0.7:
            st.success("✅ Balanced - Recommended for most use cases")
        elif alpha_demo < 1.0:
            st.info("🧠 Semantic-heavy - Good for conceptual questions")
        else:
            st.warning("⚠️ Pure Semantic - May miss exact keyword matches")

    st.divider()

    # Re-ranking explanation
    st.subheader("🏆 Re-ranking: The Quality Boost")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **What is Re-ranking?**

        Re-ranking is a second-stage process that improves result quality:

        1. **Initial Retrieval**: Fast but approximate (BM25/Semantic)
        2. **Fetch More Candidates**: Get 3x more results than needed
        3. **Re-rank with Cross-Encoder**: Score each query-document pair
        4. **Return Top Results**: Best quality results

        **Why it helps:**
        - Cross-encoders see query AND document together
        - Better understanding of relevance
        - Typically improves top-result quality by 15-30%
        """)

    with col2:
        st.markdown("""
        **Available Re-rankers:**

        | Provider | Type | Speed | Quality | Priority |
        |----------|------|-------|---------|----------|
        | **Jina** | Local | ⚡ No network | ⭐⭐ Good | 1st |
        | **Cohere** | Cloud API | Fast | ⭐⭐⭐ Excellent | 2nd |

        **When to use:**
        - ✅ Enable for important/production queries
        - ✅ When quality matters more than speed
        - ❌ Disable for rapid prototyping
        - ❌ Disable if latency is critical
        """)

        st.info("💡 Auto mode uses Jina (local) first, falls back to Cohere (cloud)")

# =============================================================================
# TAB 2: Collections
# =============================================================================
with tab2:
    st.header("📁 Document Collections")

    st.markdown("""
    Collections help you **organize documents** into logical groups for better
    search results and easier management. Think of them as folders for your
    semantic search database.
    """)

    # Why Collections
    st.subheader("Why Use Collections?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h4 style="color: white; margin: 0;">🎯 Focused Search</h4>
            <p style="margin: 10px 0;">Search within specific document sets</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Get more relevant results by limiting scope")

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h4 style="color: white; margin: 0;">📂 Organization</h4>
            <p style="margin: 10px 0;">Group related documents together</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Keep projects, topics, or domains separate")

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h4 style="color: white; margin: 0;">⚡ Performance</h4>
            <p style="margin: 10px 0;">Faster searches on smaller sets</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Reduce noise and improve response time")

    st.divider()

    # How Collections Work
    st.subheader("How Collections Work")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Semantic Search Engine                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  📁 Collection: "Research Papers"                               │
    │  ├── 📄 paper_neural_networks.pdf (45 chunks)                   │
    │  ├── 📄 paper_transformers.pdf (62 chunks)                      │
    │  └── 📄 paper_attention.pdf (38 chunks)                         │
    │                                                                 │
    │  📁 Collection: "Product Documentation"                         │
    │  ├── 📄 api_reference.pdf (120 chunks)                          │
    │  ├── 📄 user_guide.pdf (85 chunks)                              │
    │  └── 📄 faq.pdf (30 chunks)                                     │
    │                                                                 │
    │  📁 Collection: "Legal Contracts"                               │
    │  ├── 📄 terms_of_service.pdf (40 chunks)                        │
    │  └── 📄 privacy_policy.pdf (25 chunks)                          │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.info("💡 Each collection has its own settings for chunk size and overlap, allowing you to optimize for different document types.")

    st.divider()

    # Creating Collections
    st.subheader("Creating & Managing Collections")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Creating a Collection:**

        1. Navigate to **📁 Collections** page
        2. Enter a descriptive name
        3. (Optional) Add a description
        4. Configure chunk settings:
           - **Chunk Size**: Characters per chunk (default: 1000)
           - **Chunk Overlap**: Overlap between chunks (default: 200)
        5. Click **Create Collection**

        **Best Practices:**
        - Use descriptive names ("Q4 Reports", not "docs1")
        - Group by topic, project, or document type
        - Consider search patterns when organizing
        """)

    with col2:
        st.markdown("""
        **Managing Documents:**

        1. Select a collection from the list
        2. Use the **Upload** tab to add PDFs
        3. View documents in the **Documents** tab
        4. Delete individual documents as needed

        **Document Features:**
        - Automatic duplicate detection (by file hash)
        - Status tracking (Processing → Ready)
        - Chunk count visibility
        - File size display

        **Soft Limits:**
        - 3 collections recommended
        - 5 documents per collection recommended
        - (Limits are soft - you can exceed them)
        """)

    st.divider()

    # Scoped Search
    st.subheader("🔍 Searching Within Collections")

    st.markdown("""
    Each collection has its own **Search** tab, allowing you to:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Scoped Search Benefits:**

        - Only searches documents in that collection
        - Same retrieval settings (presets, alpha, reranking)
        - Faster results on smaller document sets
        - More relevant answers for specific topics

        **Example:**
        > Searching "authentication" in "API Documentation"
        > collection only returns API auth info, not general
        > auth concepts from other documents.
        """)

    with col2:
        st.markdown("""
        **Search Settings:**

        All sidebar retrieval settings apply:
        - Retrieval Profile (presets)
        - Retrieval Method (semantic/BM25/hybrid)
        - Alpha (for hybrid mode)
        - Re-ranking (on/off)
        - Number of results

        The same answer generation (GPT-4o-mini) is used
        for consistent response quality.
        """)

    st.divider()

    # When to Use Collections
    st.subheader("📋 When to Use Collections")

    configs = {
        "Scenario": [
            "Multiple projects with different docs",
            "Different document types (legal, technical, etc.)",
            "Separating by time period (Q1, Q2, etc.)",
            "Different audiences (internal, customer)",
            "A/B testing document organization",
            "Single topic, few documents"
        ],
        "Use Collections?": [
            "✅ Yes",
            "✅ Yes",
            "✅ Yes",
            "✅ Yes",
            "✅ Yes",
            "❌ No - use Home page"
        ],
        "Reason": [
            "Keep project context separate",
            "Optimize chunk settings per type",
            "Historical search without cross-contamination",
            "Different access patterns and terminology",
            "Compare search quality between organizations",
            "Collections add overhead for simple use cases"
        ]
    }

    st.dataframe(configs, use_container_width=True, hide_index=True)

    st.divider()

    # Database Management
    st.subheader("🗄️ Database Management")

    st.markdown("""
    The system separates document storage:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Home Page Documents:**
        - Uploaded directly on the home page
        - Not part of any collection
        - Cleared via "Clear All Documents" on home
        - For quick, single-document searches
        """)

    with col2:
        st.markdown("""
        **Collection Documents:**
        - Organized within collections
        - Cleared via "Clear All Collection Documents"
        - Persist with collection metadata
        - For organized, multi-document search
        """)

    st.info("💡 Clearing home page documents does NOT affect collection documents, and vice versa.")

# =============================================================================
# TAB 3: Precision vs Recall
# =============================================================================
with tab3:
    st.header("Understanding Precision vs Recall")

    st.markdown("""
    Search systems face a fundamental tradeoff between **precision** and **recall**.
    Understanding this helps you choose the right settings for your use case.
    """)

    # Visual explanation
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Precision")
        st.markdown("""
        **Definition:** *What percentage of returned results are actually relevant?*

        ```
        Precision = Relevant Results Retrieved
                    ────────────────────────────
                    Total Results Retrieved
        ```

        **High Precision means:**
        - Most results are relevant
        - Fewer "noise" documents
        - More confident in top results

        **Example:**
        > Query: "Python async programming"
        >
        > Results: 5 documents, 4 are about async Python
        >
        > Precision = 4/5 = **80%**
        """)

    with col2:
        st.subheader("🔍 Recall")
        st.markdown("""
        **Definition:** *What percentage of all relevant documents were retrieved?*

        ```
        Recall = Relevant Results Retrieved
                 ───────────────────────────
                 Total Relevant Documents
        ```

        **High Recall means:**
        - Most relevant docs are found
        - Less likely to miss important info
        - Broader coverage of the topic

        **Example:**
        > Query: "Python async programming"
        >
        > Your database has 10 relevant docs
        > Search returns 6 of them
        >
        > Recall = 6/10 = **60%**
        """)

    st.divider()

    # The Tradeoff
    st.subheader("⚖️ The Precision-Recall Tradeoff")

    st.markdown("""
    You typically **cannot maximize both** at the same time. Here's why:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h4 style="color: white; margin: 0;">Return Few Results</h4>
            <p style="font-size: 2rem; margin: 10px 0;">3 docs</p>
            <p style="margin: 0;">High Precision ✓</p>
            <p style="margin: 0;">Low Recall ✗</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Confident but may miss relevant docs")

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h4 style="color: white; margin: 0;">Balanced</h4>
            <p style="font-size: 2rem; margin: 10px 0;">5 docs</p>
            <p style="margin: 0;">Good Precision ✓</p>
            <p style="margin: 0;">Good Recall ✓</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Best for most use cases")

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h4 style="color: white; margin: 0;">Return Many Results</h4>
            <p style="font-size: 2rem; margin: 10px 0;">10 docs</p>
            <p style="margin: 0;">Low Precision ✗</p>
            <p style="margin: 0;">High Recall ✓</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Broad coverage but more noise")

    st.divider()

    # Retrieval Presets
    st.subheader("🎛️ Retrieval Presets")

    st.markdown("""
    To make this tradeoff easy, we provide **three preset profiles** that optimize
    for different use cases. Select them from the sidebar dropdown.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🎯 High Precision

        **Settings:**
        - Results: 3
        - Alpha: 0.7 (semantic-heavy)
        - Reranking: Enabled

        **Best for:**
        - Specific, focused questions
        - When you need the most relevant answer
        - "What is the exact syntax for...?"
        - Executive summaries

        **Trade-off:**
        May miss some relevant documents in favor of
        returning only the most confident matches.
        """)
        st.info("💡 Use when **quality > quantity**")

    with col2:
        st.markdown("""
        ### ⚖️ Balanced (Default)

        **Settings:**
        - Results: 5
        - Alpha: 0.5 (equal weight)
        - Reranking: Enabled

        **Best for:**
        - General questions
        - When you're not sure what you need
        - Research and exploration
        - Most everyday use cases

        **Trade-off:**
        Good balance between finding relevant docs
        and not overwhelming with results.
        """)
        st.success("✅ **Recommended** starting point")

    with col3:
        st.markdown("""
        ### 🔍 High Recall

        **Settings:**
        - Results: 10
        - Alpha: 0.3 (BM25-heavy)
        - Reranking: Disabled

        **Best for:**
        - Comprehensive research
        - "Find everything about..."
        - Legal/compliance (can't miss anything)
        - Topic exploration

        **Trade-off:**
        Returns more results but some may be
        less relevant. Faster (no reranking).
        """)
        st.info("💡 Use when **completeness > precision**")

    st.divider()

    # When to use each
    st.subheader("📋 Decision Guide")

    decision_data = {
        "Scenario": [
            "Quick specific question",
            "General research",
            "Must not miss anything important",
            "Exploring a new topic",
            "Generating a summary",
            "Looking for exact phrase/term"
        ],
        "Recommended Preset": [
            "🎯 High Precision",
            "⚖️ Balanced",
            "🔍 High Recall",
            "🔍 High Recall",
            "⚖️ Balanced",
            "🎯 High Precision"
        ],
        "Why": [
            "Need the most relevant answer fast",
            "Good balance for varied queries",
            "Can't afford to miss relevant docs",
            "Want broad coverage of information",
            "Quality context matters more than quantity",
            "Precise matching reduces noise"
        ]
    }

    st.dataframe(decision_data, use_container_width=True, hide_index=True)

    st.divider()

    # Score visibility explanation
    st.subheader("📊 Understanding Scores")

    st.markdown("""
    When you search, each result shows **relevance scores** to help you understand
    why it was returned and how confident the system is.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Score Types:**

        | Score | Meaning | Range |
        |-------|---------|-------|
        | **Semantic** | Vector similarity | 0.0 - 1.0 |
        | **BM25** | Keyword match score | Varies |
        | **Rerank** | Cross-encoder confidence | 0.0 - 1.0 |

        **Interpreting Scores:**
        - **> 0.8**: Very high confidence
        - **0.6 - 0.8**: Good match
        - **0.4 - 0.6**: Moderate relevance
        - **< 0.4**: May be less relevant
        """)

    with col2:
        st.markdown("""
        **Why scores matter:**

        1. **Compare results**: Higher scores = more relevant
        2. **Tune settings**: If top scores are low, adjust alpha
        3. **Validate answers**: Low-scoring context may be unreliable
        4. **A/B testing**: Compare methods by score quality

        **Tips:**
        - Semantic score sensitive to concept matching
        - BM25 score rewards exact keyword presence
        - Rerank score is usually most accurate
        """)

    st.info("💡 Scores are shown in the expandable 'Source' sections of each search result")

# =============================================================================
# TAB 4: Configuration Guide
# =============================================================================
with tab4:
    st.header("Configuration Guide")

    st.markdown("""
    All settings are controlled via `config.yaml`. Here's how each setting affects your search.
    """)

    # Document Processing
    st.subheader("📄 Document Processing")

    with st.expander("Chunk Size & Overlap", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Chunk Size** (`chunk_size`)

            How many characters per text chunk.

            | Size | Use Case |
            |------|----------|
            | 500-800 | Code, short precise answers |
            | 1000 | General documents (default) |
            | 1500-2000 | Long-form content, legal docs |

            **Rule of thumb:** Chunk should contain one complete idea.
            """)

        with col2:
            st.markdown("""
            **Chunk Overlap** (`chunk_overlap`)

            How much text overlaps between chunks.

            - **Purpose:** Prevents information loss at boundaries
            - **Typical:** 20% of chunk size
            - **Example:** 200 overlap for 1000 chunk size

            **Higher overlap:**
            - ✅ Better coverage
            - ❌ More storage, slower indexing
            """)

    # Retrieval Settings
    st.subheader("🔍 Retrieval Settings")

    with st.expander("Hybrid Retrieval Configuration", expanded=True):
        st.code("""
# config.yaml
hybrid_retrieval:
  enabled: true           # Enable hybrid search
  default_method: "hybrid"  # "semantic", "bm25", or "hybrid"
  alpha: 0.5              # Balance (0=BM25, 1=Semantic)
  rrf_k: 60               # RRF constant (usually 60)

  bm25:
    k1: 1.5   # Term frequency saturation
    b: 0.75  # Length normalization

  reranking:
    enabled: true
    # "auto" = jina first, then cohere
    # "jina" = force local, "cohere" = force cloud
    provider: "auto"
    fetch_k_multiplier: 3  # Fetch 3x candidates for reranking
        """, language="yaml")

        st.markdown("""
        **BM25 Parameters Explained:**

        | Parameter | Effect of Higher Value |
        |-----------|----------------------|
        | `k1` | More weight on term frequency (repeated terms score higher) |
        | `b` | More penalty for long documents |

        **Default values (k1=1.5, b=0.75) work well for most cases.**
        """)

    # Vector Store
    st.subheader("🗄️ Vector Store (ChromaDB)")

    with st.expander("ChromaDB Configuration"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Docker Mode (Recommended)**
            ```yaml
            vector_store:
              use_docker: true
              chroma_host: "localhost"
              chroma_port: 8000
            ```

            **Benefits:**
            - Stable and isolated
            - No SQLite mutex issues
            - Easy to reset/manage
            """)

        with col2:
            st.markdown("""
            **Local Mode**
            ```yaml
            vector_store:
              use_docker: false
              persist_directory: "./chroma/db"
            ```

            **Use when:**
            - Docker not available
            - Simple local development
            - Single-user scenarios
            """)

# =============================================================================
# TAB 5: A/B Testing
# =============================================================================
with tab5:
    st.header("🧪 A/B Testing Framework")

    st.markdown("""
    The A/B testing framework lets you empirically compare retrieval methods
    to find the best configuration for your specific documents and query patterns.
    """)

    # Why A/B Test
    st.subheader("Why A/B Test?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **The Problem:**
        - No single retrieval method is best for all cases
        - Document type affects optimal settings
        - Query patterns vary by use case

        **The Solution:**
        - Test multiple methods on real queries
        - Measure objective metrics
        - Make data-driven decisions
        """)

    with col2:
        st.markdown("""
        **Metrics Tracked:**

        | Metric | What It Measures |
        |--------|-----------------|
        | **Latency** | Speed (ms) |
        | **Top Score** | Best match quality |
        | **Avg Score** | Overall quality |
        | **Variance** | Consistency |
        """)

    st.divider()

    # How to Use
    st.subheader("How to Use A/B Testing")

    st.markdown("""
    ### Step 1: Upload a Document

    A/B testing is enabled after uploading a document to the main Search page.
    Once uploaded, expand the "A/B Testing" panel.

    ### Step 2: All Methods Compared Automatically

    When you run a comparison, all 4 retrieval methods are tested automatically:
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.checkbox("Semantic", value=True, disabled=True)
        st.caption("Pure vector search")
    with col2:
        st.checkbox("BM25", value=True, disabled=True)
        st.caption("Pure keyword search")
    with col3:
        st.checkbox("Hybrid", value=True, disabled=True)
        st.caption("Combined search")
    with col4:
        st.checkbox("Hybrid+Rerank", value=True, disabled=True)
        st.caption("Combined + reranking")

    st.markdown("""
    ### Step 3: Run Comparisons

    Enter representative queries and click **"Run Comparison"**:

    ```
    Example queries to test:
    - "What is [specific concept]?"          → Tests conceptual understanding
    - "[exact technical term]"               → Tests keyword matching
    - "How does [X] compare to [Y]?"         → Tests complex queries
    - "List all [items]"                     → Tests enumeration
    ```

    ### Step 4: View Results & Export

    After running queries:
    - View average score and latency per method
    - System recommends the best variant automatically
    - Click "Export Results (CSV)" to download for deeper analysis

    ### Step 5: Make a Decision

    Based on results:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Semantic Wins?**

        → Set `alpha: 0.7-1.0`

        Your queries are conceptual
        """)

    with col2:
        st.info("""
        **BM25 Wins?**

        → Set `alpha: 0.0-0.3`

        Your queries are keyword-heavy
        """)

    with col3:
        st.success("""
        **Mixed Results?**

        → Keep `alpha: 0.5`

        Hybrid is your best bet
        """)

    st.divider()

    # Example A/B Test Interpretation
    st.subheader("Example: Interpreting A/B Results")

    st.markdown("**Scenario:** Testing a technical API documentation")

    # Sample results table
    sample_data = {
        "Query": ["authentication flow", "OAuth 2.0 implementation", "What is rate limiting?", "error handling best practices"],
        "Semantic": [0.82, 0.75, 0.91, 0.88],
        "BM25": [0.89, 0.92, 0.71, 0.79],
        "Hybrid": [0.88, 0.87, 0.85, 0.86],
        "Hybrid+Rerank": [0.91, 0.93, 0.89, 0.90]
    }

    st.dataframe(sample_data, use_container_width=True)

    st.markdown("""
    **Analysis:**
    - Technical terms ("OAuth 2.0") → BM25 wins
    - Conceptual ("What is...") → Semantic wins
    - **Hybrid+Rerank** is consistently good across all query types

    **Recommendation:** Use Hybrid with reranking enabled (`alpha: 0.5`)
    """)

# =============================================================================
# TAB 6: Conversation History
# =============================================================================
with tab6:
    st.header("💬 Conversation History")

    st.markdown("""
    The conversation history feature maintains context across queries,
    enabling natural follow-up questions.
    """)

    # How it works
    st.subheader("How It Works")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ Session: "machine_learning.pdf"                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │ Q1: "What is supervised learning?"                              │
    │ A1: "Supervised learning is a type of ML where..."              │
    │                                                                 │
    │ Q2: "How does it differ from unsupervised?"                     │
    │     ↓ System detects this is a follow-up                        │
    │     ↓ Expands query using context from Q1                       │
    │ A2: "Unlike supervised learning, unsupervised..."               │
    │                                                                 │
    │ Q3: "Give me examples"                                          │
    │     ↓ Uses context from Q1 AND Q2                               │
    │ A3: "Examples of supervised: regression, classification..."     │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)

    # Follow-up Detection
    st.subheader("Follow-up Query Detection")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Triggers follow-up optimization:**
        - Short queries (< 5 words)
        - Pronouns: "it", "they", "this", "that"
        - Continuity words: "also", "another", "more"
        - References: "the same", "similar"

        **Example:**
        > Previous: "What is gradient descent?"
        >
        > Follow-up: "How does it work?"
        >
        > Expanded: "How does gradient descent work?"
        """)

    with col2:
        st.markdown("""
        **Does NOT trigger optimization:**
        - Long, specific queries
        - New topics
        - Queries with full context

        **Example:**
        > "Explain the backpropagation algorithm
        > in neural networks step by step"
        >
        > → Query is complete, no expansion needed
        """)

    # Configuration
    st.subheader("Configuration")

    st.code("""
# config.yaml
conversation:
  enabled: true
  storage_dir: "./conversation_history"  # Where sessions are saved
  max_history: 50                        # Max queries per session
  follow_up_optimization: true           # Enable query expansion
  context_window: 3                      # Recent Q&As for context
    """, language="yaml")

    st.markdown("""
    **context_window Effect:**

    | Value | Behavior |
    |-------|----------|
    | 1 | Only last Q&A for context |
    | 3 | Last 3 Q&As (recommended) |
    | 5+ | More context, but may confuse model |
    """)

    # Tips
    st.subheader("Tips for Better Conversations")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **Do:**
        - Start with broad questions
        - Ask follow-ups naturally
        - Reference previous answers
        - Use "Start New Session" for new topics
        """)

    with col2:
        st.error("""
        **Don't:**
        - Mix unrelated topics in one session
        - Expect perfect pronoun resolution
        - Use very long context windows
        - Forget to save important sessions
        """)

# =============================================================================
# TAB 7: Quick Reference
# =============================================================================
with tab7:
    st.header("🎯 Quick Reference")

    # Recommended Configs by Use Case
    st.subheader("Recommended Configurations by Use Case")

    configs = {
        "Use Case": [
            "Technical Documentation",
            "General Knowledge Base",
            "Academic Papers",
            "Legal Documents",
            "Code Documentation",
            "Customer Support FAQ"
        ],
        "Alpha": ["0.3-0.4", "0.5", "0.6-0.7", "0.4", "0.3", "0.5"],
        "Chunk Size": ["800-1000", "1000", "1200-1500", "1500-2000", "500-800", "800-1000"],
        "Reranking": ["Optional", "Recommended", "Recommended", "Required", "Optional", "Recommended"],
        "Reason": [
            "Technical terms need keyword matching",
            "Balanced for varied queries",
            "Conceptual content benefits from semantic",
            "Precise language, longer context needed",
            "Short, precise code blocks",
            "Mix of exact and conceptual queries"
        ]
    }

    st.dataframe(configs, use_container_width=True, hide_index=True)

    st.divider()

    # Troubleshooting
    st.subheader("Troubleshooting Guide")

    with st.expander("🔴 Poor Search Results"):
        st.markdown("""
        **Symptoms:** Irrelevant results, missing obvious matches

        **Solutions:**
        1. **Adjust Alpha:**
           - Getting irrelevant semantic matches? Lower alpha (more BM25)
           - Missing conceptual matches? Raise alpha (more semantic)

        2. **Check Chunk Size:**
           - Results lack context? Increase chunk size
           - Results too broad? Decrease chunk size

        3. **Enable Reranking:**
           - Often improves top results by 15-30%

        4. **Run A/B Test:**
           - Empirically find the best config for your docs
        """)

    with st.expander("🔴 Slow Performance"):
        st.markdown("""
        **Symptoms:** Long response times, timeouts

        **Solutions:**
        1. **Disable Reranking:** Biggest latency impact
        2. **Reduce search_k:** Fewer results = faster
        3. **Lower fetch_k_multiplier:** Less reranking work
        4. **Use ChromaDB Docker:** More stable than local
        5. **Check network:** If using Cohere, API latency adds up
        """)

    with st.expander("🔴 Follow-up Questions Not Working"):
        st.markdown("""
        **Symptoms:** Context lost, pronouns not resolved

        **Solutions:**
        1. **Check config:** `follow_up_optimization: true`
        2. **Increase context_window:** Try 3-5
        3. **Be explicit:** "How does [X] work?" vs "How does it work?"
        4. **Start new session:** Old context may be confusing
        """)

    with st.expander("🔴 ChromaDB Issues"):
        st.markdown("""
        **Symptoms:** Mutex errors, connection failures

        **Solutions:**
        1. **Use Docker mode:**
           ```yaml
           vector_store:
             use_docker: true
             chroma_host: "localhost"
             chroma_port: 8000
           ```

        2. **Start container:**
           ```bash
           docker run -d -p 8000:8000 chromadb/chroma
           ```

        3. **Clear local DB:** Delete `./chroma/` directory
        """)

    st.divider()

    # Key Formulas
    st.subheader("Key Formulas")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Reciprocal Rank Fusion (RRF):**
        ```
        score = α × 1/(k + sem_rank) + (1-α) × 1/(k + bm25_rank)
        ```

        Where:
        - α = alpha (semantic weight)
        - k = rrf_k constant (60)
        - rank = position in results (1, 2, 3...)
        """)

    with col2:
        st.markdown("""
        **BM25 Scoring:**
        ```
        score = IDF × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × len/avg_len))
        ```

        Where:
        - IDF = inverse document frequency
        - tf = term frequency
        - k1, b = tuning parameters
        """)

    st.divider()

    # Environment Variables
    st.subheader("Environment Variables")

    st.code("""
# .env file
# Chat provider (set in config.yaml models.chat.provider)
DEEPSEEK_API_KEY=sk-...    # Required if using DeepSeek for chat
OPENAI_API_KEY=sk-...      # Required if using OpenAI for chat or embeddings
COHERE_API_KEY=...         # Optional: For Cohere reranker (cloud fallback)
    """)

    st.info("💡 Jina (local) is preferred and requires no API key - just `pip install sentence-transformers`")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    📚 Semantic Search Engine |
    <a href="https://github.com/shrimpy8/semantic-serach" target="_blank">GitHub</a> |
    Built with Streamlit, LangChain, and ChromaDB
</div>
""", unsafe_allow_html=True)
