# How the Semantic Search Engine Works

This document explains the core concepts, configurations, and how different settings affect search quality and performance.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Retrieval Methods Explained](#retrieval-methods-explained)
3. [Precision vs Recall](#precision-vs-recall)
4. [Configuration Deep Dive](#configuration-deep-dive)
5. [A/B Testing Framework](#ab-testing-framework)
6. [Conversation History](#conversation-history)
7. [Practical Examples](#practical-examples)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Processing                             │
│  • Follow-up optimization (if conversation context exists)      │
│  • Query expansion for short queries                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Retrieval Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Semantic   │  │    BM25      │  │       Hybrid         │   │
│  │   Search     │  │   Search     │  │  (RRF Fusion)        │   │
│  │  (Vectors)   │  │  (Keywords)  │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Re-ranking Layer (Optional)                   │
│  • Jina (Local) or Cohere (Cloud) - auto-selects local first   │
│  • Cross-encoder scoring for better relevance                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Answer Generation                             │
│  • GPT-4o-mini with retrieved context                           │
│  • Conversation-aware prompting                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Retrieval Methods Explained

### 1. Semantic Search (Vector-based)

**How it works:**
- Converts your query into a dense vector (embedding) using OpenAI's text-embedding-3-large
- Finds document chunks with similar vectors using cosine similarity
- Understands meaning and context, not just keywords

**Best for:**
- Questions about concepts ("What is machine learning?")
- Paraphrased queries ("How do neural networks learn?" finds "training process of deep learning models")
- Semantic similarity ("happy" matches "joyful", "content")

**Limitations:**
- May miss exact keyword matches
- Can return semantically similar but irrelevant content
- Requires quality embeddings

**Configuration:**
```yaml
# No special config - this is the base retrieval
vector_store:
  search_k: 3  # Number of results to return
```

---

### 2. BM25 Search (Keyword-based)

**How it works:**
- Uses BM25 (Best Matching 25) algorithm - a proven information retrieval formula
- Scores documents based on term frequency and document length
- Exact and partial keyword matching

**The BM25 Formula:**
```
Score = IDF(term) × (term_freq × (k1 + 1)) / (term_freq + k1 × (1 - b + b × doc_len/avg_doc_len))
```

**Best for:**
- Exact term searches ("API endpoint", "function name")
- Technical queries with specific terminology
- When you know the exact words in the document

**Limitations:**
- No understanding of synonyms ("car" won't match "automobile")
- Sensitive to exact wording
- Doesn't understand context

**Configuration:**
```yaml
hybrid_retrieval:
  bm25:
    k1: 1.5   # Term frequency saturation (higher = more weight on frequency)
    b: 0.75  # Length normalization (0 = no normalization, 1 = full normalization)
```

**Parameter Effects:**

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| `k1` | Diminishing returns on term frequency quickly | Term frequency continues to boost score |
| `b` | Short and long docs treated equally | Short docs boosted, long docs penalized |

---

### 3. Hybrid Search (BM25 + Semantic)

**How it works:**
1. Runs both BM25 and Semantic search in parallel
2. Combines results using Reciprocal Rank Fusion (RRF)
3. Alpha parameter controls the balance

**The RRF Formula:**
```
RRF_score = α × (1/(k + semantic_rank)) + (1-α) × (1/(k + bm25_rank))
```

Where:
- `α` (alpha) = weight for semantic search (0 to 1)
- `k` (rrf_k) = constant to prevent division issues (typically 60)

**Best for:**
- General-purpose queries
- When you're unsure which method would work better
- Combining precision (BM25) with recall (semantic)

**Configuration:**
```yaml
hybrid_retrieval:
  enabled: true
  default_method: "hybrid"
  alpha: 0.5    # 0 = BM25 only, 1 = Semantic only, 0.5 = equal weight
  rrf_k: 60     # RRF constant
```

**Alpha Effects:**

| Alpha | Behavior | Best For |
|-------|----------|----------|
| 0.0 | Pure BM25 | Exact keyword searches |
| 0.3 | BM25-heavy hybrid | Technical documentation |
| 0.5 | Balanced | General purpose |
| 0.7 | Semantic-heavy hybrid | Conceptual questions |
| 1.0 | Pure Semantic | Abstract queries |

---

### 4. Re-ranking

**How it works:**
- Takes initial retrieval results (more candidates than needed)
- Uses a cross-encoder model to score query-document pairs
- Re-orders results based on fine-grained relevance

**Why it helps:**
- Initial retrieval is fast but approximate
- Re-ranking is slower but more accurate
- Cross-encoders see query and document together (better context)

**Providers:**

| Provider | Type | Speed | Quality | Cost | Priority |
|----------|------|-------|---------|------|----------|
| Jina | Local | No network latency | Good | Free | 1st (preferred) |
| Cohere | Cloud API | Fast | Excellent | Per-request | 2nd (fallback) |

**Configuration:**
```yaml
hybrid_retrieval:
  reranking:
    enabled: true
    # Provider options:
    #   - "auto": tries jina (local) first, then cohere (cloud)
    #   - "jina": force local Jina model
    #   - "cohere": force cloud Cohere API
    provider: "auto"
    fetch_k_multiplier: 3  # Fetch 3x more candidates for reranking
```

💡 **Auto mode** tries Jina (local) first, then falls back to Cohere (cloud) if unavailable.

⚙️ **Override default**: Set `provider: "cohere"` to force Cohere even when Jina is available.

**fetch_k_multiplier Effect:**
- `fetch_k_multiplier: 1` → No extra candidates (defeats purpose)
- `fetch_k_multiplier: 3` → Fetch 9 docs, rerank to top 3 (recommended)
- `fetch_k_multiplier: 5` → Fetch 15 docs, rerank to top 3 (slower but thorough)

---

## Precision vs Recall

### Understanding the Tradeoff

Search systems face a fundamental tradeoff between **precision** and **recall**:

**Precision**: What percentage of returned results are actually relevant?
```
Precision = Relevant Results Retrieved / Total Results Retrieved
```

**Recall**: What percentage of all relevant documents were retrieved?
```
Recall = Relevant Results Retrieved / Total Relevant Documents
```

### The Tradeoff in Practice

| Strategy | Results | Precision | Recall |
|----------|---------|-----------|--------|
| Return few results (k=3) | More focused | High ✓ | Low ✗ |
| Balanced (k=5) | Good mix | Good ✓ | Good ✓ |
| Return many results (k=10) | Broad coverage | Low ✗ | High ✓ |

You typically **cannot maximize both** at the same time:
- **Fewer results** → Higher precision (most results relevant) but lower recall (may miss some)
- **More results** → Higher recall (find most relevant) but lower precision (more noise)

---

### Retrieval Presets

To simplify this tradeoff, we provide **three preset profiles** accessible from the sidebar:

#### 🎯 High Precision

| Setting | Value |
|---------|-------|
| Results (k) | 3 |
| Alpha | 0.7 (semantic-heavy) |
| Reranking | Enabled |

**Best for:**
- Specific, focused questions
- When you need the most relevant answer
- "What is the exact syntax for...?"
- Executive summaries

**Trade-off:** May miss some relevant documents in favor of returning only the most confident matches.

---

#### ⚖️ Balanced (Default)

| Setting | Value |
|---------|-------|
| Results (k) | 5 |
| Alpha | 0.5 (equal weight) |
| Reranking | Enabled |

**Best for:**
- General questions
- When you're not sure what you need
- Research and exploration
- Most everyday use cases

**Trade-off:** Good balance between finding relevant docs and not overwhelming with results.

---

#### 🔍 High Recall

| Setting | Value |
|---------|-------|
| Results (k) | 10 |
| Alpha | 0.3 (BM25-heavy) |
| Reranking | Disabled |

**Best for:**
- Comprehensive research
- "Find everything about..."
- Legal/compliance (can't miss anything)
- Topic exploration

**Trade-off:** Returns more results but some may be less relevant. Faster (no reranking).

---

### Decision Guide

| Scenario | Recommended Preset | Why |
|----------|-------------------|-----|
| Quick specific question | 🎯 High Precision | Need the most relevant answer fast |
| General research | ⚖️ Balanced | Good balance for varied queries |
| Must not miss anything | 🔍 High Recall | Can't afford to miss relevant docs |
| Exploring a new topic | 🔍 High Recall | Want broad coverage of information |
| Generating a summary | ⚖️ Balanced | Quality context matters more than quantity |
| Looking for exact phrase | 🎯 High Precision | Precise matching reduces noise |

---

### Understanding Scores

When you search, each result shows **relevance scores** to help you understand why it was returned:

| Score | Meaning | Range | Interpretation |
|-------|---------|-------|----------------|
| **Semantic** | Vector similarity | 0.0 - 1.0 | Higher = more conceptually similar |
| **BM25** | Keyword match score | Varies | Higher = more keyword matches |
| **Rerank** | Cross-encoder confidence | 0.0 - 1.0 | Most accurate relevance estimate |

**Score Guidelines:**
- **> 0.8**: Very high confidence
- **0.6 - 0.8**: Good match
- **0.4 - 0.6**: Moderate relevance
- **< 0.4**: May be less relevant

---

## Configuration Deep Dive

### Document Processing

```yaml
document_processing:
  chunk_size: 1000      # Characters per chunk
  chunk_overlap: 200    # Overlap between chunks
  add_start_index: true # Track position in original document
```

**Chunk Size Effects:**

| Size | Pros | Cons |
|------|------|------|
| Small (500) | More precise retrieval | May lose context |
| Medium (1000) | Good balance | Standard choice |
| Large (2000) | More context per chunk | May include irrelevant content |

**Chunk Overlap:**
- Prevents information loss at chunk boundaries
- 20% overlap (200 for 1000 chunk) is typical
- Higher overlap = more redundancy but better coverage

---

### Vector Store

```yaml
vector_store:
  use_docker: true        # Use ChromaDB Docker (recommended)
  chroma_host: "localhost"
  chroma_port: 8000
  search_k: 3             # Base number of results
```

---

## A/B Testing Framework

### Purpose

The A/B testing framework lets you empirically compare retrieval methods on your specific documents and queries.

### Metrics Tracked

| Metric | Description | What It Tells You |
|--------|-------------|-------------------|
| **Latency (ms)** | Time to retrieve results | Performance impact |
| **Top Score** | Highest relevance score | Best match quality |
| **Avg Score** | Average across results | Overall quality |
| **Score Variance** | Spread of scores | Consistency |
| **Num Results** | Actual results returned | Coverage |

### Test Variants

```yaml
ab_testing:
  default_variants:
    - "semantic"        # Pure vector search
    - "bm25"           # Pure keyword search
    - "hybrid"         # Combined (no reranking)
    - "hybrid_rerank"  # Combined + reranking
```

### How to Use A/B Testing

1. **Upload a Document**
   - A/B testing is enabled after uploading a document
   - Expand the "A/B Testing" panel on the main page

2. **Run Comparison**
   - Enter a test query representative of real usage
   - Click "Run Comparison" to automatically test all 4 retrieval methods:
     - Semantic (pure vector search)
     - BM25 (pure keyword search)
     - Hybrid (combined, no reranking)
     - Hybrid + Rerank (combined with reranking)

3. **View Results**
   - Results show average score and latency for each method
   - System recommends the best performing variant
   - Run multiple queries to get aggregate statistics

4. **Export Results**
   - Click "Export Results (CSV)" to download comparison data
   - Analyze offline to make informed decisions

### Interpreting A/B Results

**Scenario 1: Semantic consistently wins**
→ Your queries are conceptual, use `alpha: 0.7-1.0`

**Scenario 2: BM25 consistently wins**
→ Your queries are keyword-heavy, use `alpha: 0.0-0.3`

**Scenario 3: Mixed results**
→ Use hybrid with `alpha: 0.5` as a safe default

**Scenario 4: Reranking significantly improves scores**
→ Enable reranking in production (worth the latency cost)

---

## Conversation History

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│ Session 1: "machine_learning.pdf"                               │
├─────────────────────────────────────────────────────────────────┤
│ Q1: "What is supervised learning?"                              │
│ A1: "Supervised learning is..."                                 │
│                                                                 │
│ Q2: "How does it differ from unsupervised?"                     │
│     ↓ System detects follow-up                                  │
│     ↓ Expands query: "How does supervised learning differ..."   │
│ A2: "Unlike supervised learning..."                             │
│                                                                 │
│ Q3: "Give me examples"                                          │
│     ↓ Uses context from Q1 and Q2                               │
│ A3: "Examples of supervised vs unsupervised..."                 │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
conversation:
  enabled: true
  storage_dir: "./conversation_history"
  max_history: 50           # Max queries per session
  follow_up_optimization: true
  context_window: 3         # Recent Q&As for context
```

### Follow-up Query Optimization

The system detects follow-up questions by checking for:
- Short queries (< 5 words)
- Pronouns ("it", "they", "this")
- Continuity words ("also", "another", "more")

When detected, the query is expanded with context from recent conversation.

---

## Practical Examples

### Example 1: Technical Documentation Search

**Document:** API Reference Guide

**Query:** "authentication endpoint"

| Method | Likely Result |
|--------|---------------|
| BM25 | Exact matches for "authentication" and "endpoint" |
| Semantic | Might also find "login API", "OAuth flow" |
| Hybrid | Best of both - exact matches + related concepts |

**Recommended Config:**
```yaml
alpha: 0.4  # Slightly favor keywords for technical docs
```

---

### Example 2: Conceptual Questions

**Document:** Machine Learning Textbook

**Query:** "How do neural networks learn?"

| Method | Likely Result |
|--------|---------------|
| BM25 | Only finds chunks with exact phrase |
| Semantic | Finds "backpropagation", "gradient descent", "training" |
| Hybrid | Semantic wins here |

**Recommended Config:**
```yaml
alpha: 0.7  # Favor semantic for conceptual queries
```

---

### Example 3: Follow-up Conversation

**Session:**
```
Q1: "What is transfer learning?"
A1: [Explains transfer learning]

Q2: "What are its benefits?"
    → Optimized to: "What are the benefits of transfer learning?"
A2: [Lists benefits with proper context]
```

---

## Quick Reference: Configuration Cheatsheet

| Use Case | Alpha | Reranking | Chunk Size |
|----------|-------|-----------|------------|
| Technical docs | 0.3-0.4 | Optional | 800-1000 |
| General knowledge | 0.5 | Recommended | 1000 |
| Academic papers | 0.6-0.7 | Recommended | 1200-1500 |
| Legal documents | 0.4 | Recommended | 1500-2000 |
| Code documentation | 0.3 | Optional | 500-800 |

---

## Troubleshooting

### Poor Search Results?

1. **Check chunk size** - Too large chunks may dilute relevance
2. **Adjust alpha** - Try different BM25/semantic balances
3. **Enable reranking** - Often improves top results significantly
4. **Use A/B testing** - Empirically find the best config

### Slow Performance?

1. **Disable reranking** - Biggest latency impact
2. **Reduce search_k** - Fewer results = faster
3. **Lower fetch_k_multiplier** - Less reranking work
4. **Use ChromaDB Docker** - More stable than local SQLite

### Context Issues in Conversation?

1. **Increase context_window** - More history for context
2. **Check follow_up_optimization** - Ensure it's enabled
3. **Start new session** - If context is confusing the model
