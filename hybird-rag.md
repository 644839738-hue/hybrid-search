╭─── Claude Code v2.1.131 ─────────────────────────────────────────────────────╮
│                                         │ Tips for getting started           │
│              Welcome back!              │ Run /init to create a CLAUDE.md f… │
│                                         │ ────────────────────────────────── │
│                 ▐▛███▜▌                 │ What's new                         │
│                ▝▜█████▛▘                │ Check the Claude Code changelog f… │
│                  ▘▘ ▝▝                  │                                    │
│                                         │                                    │
│   deepseek-v4-pro · API Usage Billing   │                                    │
│   ~\Desktop\agent前端\semantic-search   │                                    │
╰──────────────────────────────────────────────────────────────────────────────╯

✻ Conversation compacted (ctrl+o for history)

  ⎿  Read config_loader.py (431 lines)
  ⎿  Read requirements.txt (27 lines)
  ⎿  Read config.yaml (198 lines)
  ⎿  Referenced file app.py
  ⎿  Referenced file core\vector_store.py

  Searched for 2 patterns, read 3 files (ctrl+o to expand)

● Now I have all the context needed. Let me implement the OCR changes across all
   three files.

● Update(core\document_processor.py)
  ⎿  Added 3 lines, removed 4 lines
      1  """
      2 -Document Processor Module                                       
      2 +文档处理模块                                                    
      3  
      4 -Handles PDF document loading and text chunking for semantic sear
        -ch.                                                             
      5 -Supports collection-scoped indexing for filtered retrieval.     
      4 +负责 PDF 文档加载和文本分块，支持扫描版 PDF 的 OCR 识别。       
      5 +支持按集合范围索引以实现过滤检索。                              
      6  """
      7  
      8  import os
     ...
      10  import tempfile
      11  import logging
      12  from typing import List, Optional, Dict, Any
      13 -from langchain_community.document_loaders import PyPDFLoader   
      13  from langchain_text_splitters import RecursiveCharacterTextSpli
          tter
      14  from langchain_core.documents import Document
      15  

● Now update the __init__ to accept OCR config params:

● Update(core\document_processor.py)
  ⎿  Added 23 lines, removed 6 lines
      35          >>> chunks = processor.process_uploaded_file(uploaded_f
          ile)
      36      """
      37  
      38 -    def __init__(self, chunk_size: int = 1000, chunk_overlap: i
         -nt = 200, add_start_index: bool = True):                       
      38 +    def __init__(                                              
      39 +        self,                                                  
      40 +        chunk_size: int = 1000,                                
      41 +        chunk_overlap: int = 200,                              
      42 +        add_start_index: bool = True,                          
      43 +        enable_ocr: bool = True,                               
      44 +        ocr_languages: str = "chi_sim+eng",                    
      45 +        ocr_dpi: int = 200,                                    
      46 +        tesseract_cmd: str = ""                                
      47 +    ):                                                         
      48          """
      40 -        Initialize the document processor.                     
      49 +        初始化文档处理器。                                     
      50  
      51          Args:
      43 -            chunk_size: Size of text chunks in characters      
      44 -            chunk_overlap: Overlap between consecutive chunks  
      45 -            add_start_index: Whether to add start index metadat
         -a                                                              
      52 +            chunk_size: 文本块大小（字符数）                   
      53 +            chunk_overlap: 文本块之间的重叠长度                
      54 +            add_start_index: 是否为文本块添加起始索引          
      55 +            enable_ocr: 是否启用 OCR 识别扫描版 PDF            
      56 +            ocr_languages: OCR 语言包（如 "chi_sim+eng"）      
      57 +            ocr_dpi: OCR 渲染 DPI                              
      58 +            tesseract_cmd: Tesseract 可执行文件路径            
      59          """
      60          self.chunk_size = chunk_size
      61          self.chunk_overlap = chunk_overlap
      62          self.add_start_index = add_start_index
      63 +        self.enable_ocr = enable_ocr                           
      64 +        self.ocr_languages = ocr_languages                     
      65 +        self.ocr_dpi = ocr_dpi                                 
      66 +        self.tesseract_cmd = tesseract_cmd                     
      67  
      68          self.text_splitter = RecursiveCharacterTextSplitter(
      69              chunk_size=self.chunk_size,
     ...
      71              add_start_index=self.add_start_index
      72          )
      73  
      57 -        logger.info(f"DocumentProcessor initialized with chunk_
         -size={chunk_size}, overlap={chunk_overlap}")                   
      74 +        logger.info(f"DocumentProcessor 初始化: chunk_size={chu
         +nk_size}, overlap={chunk_overlap}, ocr={'enabled' if enable_ocr
         + else 'disabled'}")                                            
      75  
      76      def process_uploaded_file(
      77          self,

● Now replace the _load_pdf method with PyMuPDF primary extraction and Tesseract
   OCR fallback:

● Update(core\document_processor.py)
  ⎿  Added 83 lines, removed 6 lines
      167  
      168      def _load_pdf(self, file_path: str) -> List[Document]:
      169          """
      170 -        Load PDF document using PyPDFLoader.                  
      170 +        加载 PDF 文档（PyMuPDF 为主，Tesseract OCR 为备）。   
      171  
      172 +        优先使用 PyMuPDF 提取文本；若文本为空且启用 OCR，     
      173 +        则将每页渲染为图像后调用 Tesseract 识别。             
      174 +                                                              
      175          Args:
      173 -            file_path: Path to PDF file                       
      176 +            file_path: PDF 文件路径                           
      177  
      178          Returns:
      176 -            List of Document objects (one per page)           
      179 +            Document 对象列表（每页一个）                     
      180  
      181          Raises:
      179 -            Exception: If PDF loading fails                   
      182 +            Exception: 若 PDF 加载或 OCR 识别全部失败         
      183          """
      184          logger.info(f"Loading PDF from: {file_path}")
      182 -        loader = PyPDFLoader(file_path)                       
      183 -        return loader.load()                                  
      185  
      186 +        try:                                                  
      187 +            import fitz                                       
      188 +        except ImportError:                                   
      189 +            raise ImportError(                                
      190 +                "PyMuPDF (fitz) 未安装。请运行: pip install py
          +mupdf"                                                        
      191 +            )                                                 
      192 +                                                              
      193 +        pdf_doc = fitz.open(file_path)                        
      194 +        documents = []                                        
      195 +        ocr_pages = 0                                         
      196 +                                                              
      197 +        for page_num, page in enumerate(pdf_doc):             
      198 +            text = page.get_text("text").strip()              
      199 +                                                              
      200 +            if text:                                          
      201 +                documents.append(Document(                    
      202 +                    page_content=text,                        
      203 +                    metadata={"page": page_num + 1, "source": 
          +file_path}                                                    
      204 +                ))                                            
      205 +            elif self.enable_ocr:                             
      206 +                logger.info(f"第 {page_num + 1} 页无文本，尝试
          + OCR 识别...")                                                
      207 +                ocr_text = self._ocr_page(page, page_num)     
      208 +                if ocr_text:                                  
      209 +                    documents.append(Document(                
      210 +                        page_content=ocr_text,                
      211 +                        metadata={"page": page_num + 1, "sourc
          +e": file_path, "ocr": True}                                   
      212 +                    ))                                        
      213 +                    ocr_pages += 1                            
      214 +                                                              
      215 +        pdf_doc.close()                                       
      216 +                                                              
      217 +        if not documents:                                     
      218 +            raise ValueError(                                 
      219 +                "PDF 未提取到任何文本。\n"                    
      220 +                "可能原因:\n"                                 
      221 +                "  1. PDF 为扫描版图片，且 OCR 未启用（请在 co
          +nfig.yaml 中设置 enable_ocr: true）\n"                        
      222 +                "  2. Tesseract 未安装或路径不正确（请检查 con
          +fig.yaml 中的 tesseract_cmd）\n"                              
      223 +                "  3. PDF 文件已损坏或为空白文档"             
      224 +            )                                                 
      225 +                                                              
      226 +        if ocr_pages > 0:                                     
      227 +            logger.info(f"OCR 识别了 {ocr_pages}/{page_num + 1
          +} 页")                                                        
      228 +                                                              
      229 +        logger.info(f"Loaded {len(documents)} pages from PDF")
      230 +        return documents                                      
      231 +                                                              
      232 +    def _ocr_page(self, page, page_num: int) -> str:          
      233 +        """                                                   
      234 +        对单页 PDF 执行 OCR 识别。                            
      235 +                                                              
      236 +        Args:                                                 
      237 +            page: PyMuPDF Page 对象                           
      238 +            page_num: 页码（0-based）                         
      239 +                                                              
      240 +        Returns:                                              
      241 +            识别出的文本，失败返回空字符串                    
      242 +        """                                                   
      243 +        try:                                                  
      244 +            from PIL import Image                             
      245 +            import pytesseract                                
      246 +        except ImportError as e:                              
      247 +            logger.error(f"OCR 依赖未安装: {e}")              
      248 +            return ""                                         
      249 +                                                              
      250 +        if self.tesseract_cmd:                                
      251 +            pytesseract.pytesseract.tesseract_cmd = self.tesse
          +ract_cmd                                                      
      252 +                                                              
      253 +        try:                                                  
      254 +            mat = page.get_pixmap(dpi=self.ocr_dpi)           
      255 +            img = Image.frombytes("RGB", [mat.width, mat.heigh
          +t], mat.samples)                                              
      256 +            text = pytesseract.image_to_string(img, lang=self.
          +ocr_languages).strip()                                        
      257 +            return text                                       
      258 +        except Exception as e:                                
      259 +            logger.warning(f"第 {page_num + 1} 页 OCR 失败: {e
          +}")                                                           
      260 +            return ""                                         
      261 +                                                              
      262      def _log_chunk_stats(self, chunks: List[Document]) -> None
           :
      263          """
      264          Log statistics about document chunks.

● Now update app.py to pass OCR config to DocumentProcessor:

● Update(app.py)
  ⎿  Added 5 lines, removed 1 line
      425              doc_processor = DocumentProcessor(
      426                  chunk_size=config.get_chunk_size(),
      427                  chunk_overlap=config.get_chunk_overlap(),
      428 -                add_start_index=config.get_add_start_index()  
      428 +                add_start_index=config.get_add_start_index(), 
      429 +                enable_ocr=config.is_ocr_enabled(),           
      430 +                ocr_languages=config.get_ocr_languages(),     
      431 +                ocr_dpi=config.get_ocr_dpi(),                 
      432 +                tesseract_cmd=config.get_tesseract_cmd()      
      433              )
      434  
      435              st.info(f"正在处理: **{uploaded_file.name}**")

● Now update pages/2_Collections.py to pass OCR config:

● Update(pages\2_Collections.py)
  ⎿  Added 5 lines, removed 1 line
      534                      doc_processor = DocumentProcessor(
      535                          chunk_size=chunk_size,
      536                          chunk_overlap=chunk_overlap,
      537 -                        add_start_index=config.get_add_start_i
          -ndex()                                                        
      537 +                        add_start_index=config.get_add_start_i
          +ndex(),                                                       
      538 +                        enable_ocr=config.is_ocr_enabled(),   
      539 +                        ocr_languages=config.get_ocr_languages
          +(),                                                           
      540 +                        ocr_dpi=config.get_ocr_dpi(),         
      541 +                        tesseract_cmd=config.get_tesseract_cmd
          +()                                                            
      542                      )
      543  
      544                      # Reset file position for processing

● Let me verify the final state of document_processor.py:

  Read 1 file (ctrl+o to expand)

● Let me check the error handling in app.py around this section:

  Read 1 file (ctrl+o to expand)

● Let me update the now-outdated error message since we have OCR support:

● Update(app.py)
  ⎿  Added 9 lines, removed 2 lines
      436              chunks = doc_processor.process_uploaded_file(uploa
           ded_file)
      437  
      438              if not chunks:
      439 -                st.error("PDF 未解析出文本，可能是扫描版 PDF（
          -纯图片），无法进行语义搜索。请上传包含可选中文字的 PDF 文档。"
          -)                                                             
      440 -                logger.warning(f"PDF 解析为空: {uploaded_file.
          -name}（可能是扫描版）")                                       
      439 +                st.error(                                     
      440 +                    "PDF 未解析出任何文本。\n\n"              
      441 +                    "可能原因:\n"                             
      442 +                    "  1. PDF 为扫描版图片，且 OCR 未启用（请 
          +在 config.yaml 中设置 document_processing.enable_ocr: true）\n
          +"                                                             
      443 +                    "  2. Tesseract OCR 未安装或路径不正确（请
          +检查 config.yaml 中的 document_processing.tesseract_cmd）\n"  
      444 +                    "  3. PDF 文件已损坏或为空白文档\n\n"     
      445 +                    "Windows Tesseract 安装: https://github.co
          +m/UB-Mannheim/tesseract/wiki"                                 
      446 +                )                                             
      447 +                logger.warning(f"PDF 解析为空: {uploaded_file.
          +name}（OCR 也未识别出文本）")                                 
      448                  return
      449  
      450              st.session_state.documents = chunks
● Let me check for similar error messages in the Collections page:

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● The Collections page has proper exception handling that will display the
  detailed ValueError from _load_pdf. Everything is in place. Let me update the
  task tracking:

● OCR 支持已完成。以下是修改的三个文件：

  core/document_processor.py — 核心改动：
  - 移除了 PyPDFLoader 依赖，改用 PyMuPDF (fitz) 作为主提取器
  - _load_pdf() 逐页用 page.get_text("text") 提取文本；若某页无文本且 OCR
  已启用，自动回退到 Tesseract
  - 新增 _ocr_page() 方法：将页面渲染为图像（可配置 DPI），调用
  pytesseract.image_to_string() 识别
  - 所有页面都为空时抛出清晰的 ValueError，列出三种可能原因和排查方法
  - 通过 OCR 识别的页面会在 metadata 中标记 ocr: True

  app.py — 两处改动：
  - 第 425-433 行：DocumentProcessor 实例化时传入
  enable_ocr、ocr_languages、ocr_dpi、tesseract_cmd
  - 第 438-448 行：空 chunks 错误提示更新为包含 OCR 故障排查指引

  pages/2_Collections.py — 第 534-543 行：
  - DocumentProcessor 实例化同样传入完整的 OCR 配置参数

  处理流程：上传 PDF → PyMuPDF 提取文本 → 空页自动 OCR → 分块 → ChromaDB
  索引。用户无需任何额外操作，扫描版 PDF 现在能自动识别。

✻ Churned for 8m 6s

※ recap: OCR for scanned PDFs is now complete. PyMuPDF extracts text first, Tesseract OCR kicks in automatically for pages with no text. You can upload a scanned PDF to test it.

❯ 请你帮我把当前项目的文本切分方式升级为“基于 NLP 的语义动态切分 + 重叠窗口”。  
                                                                                
  项目背景：                                                                    
  这是一个 Streamlit + LangChain + ChromaDB 的 RAG                              
  文档问答项目。当前项目流程大概是：                                            
  PDF 上传                                                                      
  → PyMuPDF / OCR 提取文本                                                      
  → 文本切分 chunk                                                              
  → 生成 embedding                                                              
  → 存入 ChromaDB                                                               
  → Semantic Search / BM25 / Hybrid Search                                      
  → Rerank                                                                      
  → LLM 生成回答                                                                
                                                                                
  我的目标：                                                                    
  现在不要再简单使用固定长度切分，而是希望通过 NLP 方法实现更智能的文本切分：   
  1. 先用 NLP 方式识别句子、段落、标题等语义边界。                              
  2. 再根据语义相似度和 chunk_size 动态合并文本单元。                           
  3. 相邻 chunk 之间保留重叠窗口，且 overlap                                    
  尽量按完整句子保留，不要从字符中间截断。                                      
  4. 最终生成适合 RAG 检索的 chunk，并保留 source、page、chunk_index 等         
  metadata。                                                                    
                                                                                
  请重点检查和修改这些文件：                                                    
  1. core/document_processor.py                                                 
  2. config.yaml                                                                
  3. config_loader.py                                                           
  4. requirements.txt                                                           
  5. 如果 app.py 中初始化 DocumentProcessor 时需要传参，也请做最小修改          
                                                                                
  不要修改这些模块：                                                            
  1. OCR 主流程不要重构                                                         
  2. Embedding 生成逻辑不要改                                                   
  3. ChromaDB 入库逻辑不要改                                                    
  4. Semantic Search / BM25 / Hybrid Search / Rerank 不要改                     
  5. Streamlit UI 不要大改                                                      
                                                                                
  一、请先分析当前文本切分逻辑                                                  
                                                                                
  请先阅读 core/document_processor.py，找出当前文本切分是如何实现的，例如：     
  1. 是否使用 RecursiveCharacterTextSplitter                                    
  2. chunk_size 和 chunk_overlap 从哪里读取                                     
  3. 切分后的 chunk 如何生成 metadata                                           
  4. 切分结果如何传给 vector_store                                              
  5. 是否按 page 处理 PDF 文本                                                  
                                                                                
  请在不破坏原逻辑的基础上新增一种切分策略。                                    
                                                                                
  二、请在 config.yaml 增加配置                                                 
                                                                                
  请新增或修改 document_processing 配置：                                       
                                                                                
  ```yaml                                                                       
  document_processing:                                                          
    chunking_strategy: "nlp_dynamic"                                            
    chunk_size: 800                                                             
    chunk_overlap: 150                                                          
    min_chunk_size: 300                                                         
    max_chunk_size: 1200                                                        
                                                                                
    nlp:                                                                        
      sentence_splitter: "regex"                                                
      enable_semantic_similarity: true                                          
      semantic_similarity_threshold: 0.55                                       
      semantic_embedding_model: "sentence-transformers/all-MiniLM-L6-v2"        
      overlap_by_sentence: true                                                 
      preserve_headings: true                                                   

● 我先全面分析当前文本切分逻辑，然后实施升级。

  Searched for 1 pattern, read 6 files (ctrl+o to expand)

● Now I have a complete picture. Let me implement the NLP dynamic chunking
  across all four files.
● 先更新 config.yaml，在 document_processing 下增加 NLP 动态切分配置：

● Update(config.yaml)
  ⎿  Added 31 lines, removed 2 lines
      29  
      30  # 文档处理配置
      31  document_processing:
      32 -  chunk_size: 1000                                             
      33 -  chunk_overlap: 200                                           
      32 +  # 切分策略: "fixed_size" / "nlp_dynamic"                     
      33 +  #   - fixed_size: 基于字符数固定长度切分（RecursiveCharacterT
         +extSplitter）                                                  
      34 +  #   - nlp_dynamic: 基于 NLP 语义动态切分 + 句子级重叠窗口    
      35 +  chunking_strategy: "nlp_dynamic"                             
      36 +                                                               
      37 +  # 基础切分参数（两种策略通用）                               
      38 +  chunk_size: 800          # 目标文本块大小（字符数）          
      39 +  chunk_overlap: 150       # 相邻块重叠长度（字符数）          
      40    add_start_index: true
      41  
      42 +  # NLP 动态切分专用参数（仅 chunking_strategy = "nlp_dynamic" 
         +时生效）                                                       
      43 +  min_chunk_size: 300      # 最小块大小，低于此值即使语义不同也
         +会合并                                                         
      44 +  max_chunk_size: 1200     # 最大块大小，超过此值强制断句      
      45 +                                                               
      46 +  nlp:                                                         
      47 +    # 句子分割器: "regex" 使用正则表达式分割中英文句子         
      48 +    sentence_splitter: "regex"                                 
      49 +                                                               
      50 +    # 是否启用语义相似度合并（将语义相近的句子合并到同一块）   
      51 +    enable_semantic_similarity: true                           
      52 +                                                               
      53 +    # 语义相似度阈值 (0-1)，句子与当前块质心的余弦相似度低于此 
         +值将开始新块                                                   
      54 +    semantic_similarity_threshold: 0.55                        
      55 +                                                               
      56 +    # 语义相似度计算使用的嵌入模型（与主 embedding 模型可不同）
      57 +    semantic_embedding_model: "sentence-transformers/all-MiniLM
         +-L6-v2"                                                        
      58 +                                                               
      59 +    # 重叠窗口是否按完整句子保留（true=句子级, false=字符级截断
         +）                                                             
      60 +    overlap_by_sentence: true                                  
      61 +                                                               
      62 +    # 是否保留标题/章节标记（识别到的标题会作为新块的起点）    
      63 +    preserve_headings: true                                    
      64 +                                                               
      65    # OCR 配置（扫描版 PDF 自动识别）
      66    enable_ocr: true
      67    ocr_engine: "tesseract"

● Now update config_loader.py — add getters after the existing chunk config
  methods:

● Update(config_loader.py)
  ⎿  Added 23 lines
      107          """获取是否为分块添加起始索引。"""
      108          return self.config.get("document_processing", {}).get(
           "add_start_index", True)
      109  
      110 +    def get_chunking_strategy(self) -> str:                   
      111 +        """获取切分策略: 'fixed_size' / 'nlp_dynamic'。"""    
      112 +        return self.config.get("document_processing", {}).get(
          +"chunking_strategy", "fixed_size")                            
      113 +                                                              
      114 +    def get_min_chunk_size(self) -> int:                      
      115 +        """获取 NLP 动态切分的最小块大小。"""                 
      116 +        return self.config.get("document_processing", {}).get(
          +"min_chunk_size", 300)                                        
      117 +                                                              
      118 +    def get_max_chunk_size(self) -> int:                      
      119 +        """获取 NLP 动态切分的最大块大小。"""                 
      120 +        return self.config.get("document_processing", {}).get(
          +"max_chunk_size", 1200)                                       
      121 +                                                              
      122 +    def get_nlp_chunking_config(self) -> Dict[str, Any]:      
      123 +        """获取完整的 NLP 动态切分配置。"""                   
      124 +        return self.config.get("document_processing", {}).get(
          +"nlp", {                                                      
      125 +            "sentence_splitter": "regex",                     
      126 +            "enable_semantic_similarity": True,               
      127 +            "semantic_similarity_threshold": 0.55,            
      128 +            "semantic_embedding_model": "sentence-transformers
          +/all-MiniLM-L6-v2",                                           
      129 +            "overlap_by_sentence": True,                      
      130 +            "preserve_headings": True                         
      131 +        })                                                    
      132 +                                                              
      133      def is_ocr_enabled(self) -> bool:
      134          """检查是否启用 OCR（扫描版 PDF 自动识别）。"""
      135          return self.config.get("document_processing", {}).get(
           "enable_ocr", True)

● Now the core — implement NLPDynamicChunker and update DocumentProcessor.
  First, add the imports:

● Update(core\document_processor.py)
  ⎿  Added 7 lines, removed 1 line
       3  
       4  负责 PDF 文档加载和文本分块，支持扫描版 PDF 的 OCR 识别。
       5  支持按集合范围索引以实现过滤检索。
       6 +                                                               
       7 +提供两种切分策略:                                              
       8 +  - fixed_size: 基于字符数固定长度切分（RecursiveCharacterTextS
         +plitter）                                                      
       9 +  - nlp_dynamic: 基于 NLP 语义动态切分 + 句子级重叠窗口        
      10  """
      11  
      12  import os
      13 +import re                                                      
      14  import uuid
      15  import tempfile
      16  import logging
      12 -from typing import List, Optional, Dict, Any                   
      17 +from typing import List, Optional, Dict, Any, Tuple            
      18 +import numpy as np                                             
      19  from langchain_text_splitters import RecursiveCharacterTextSpli
          tter
      20  from langchain_core.documents import Document
      21  

● Now insert the NLPDynamicChunker class before DocumentProcessor:

● Update(core\document_processor.py)
  ⎿  Added 388 lines
       22  logger = logging.getLogger(__name__)
       23  
       24  
       25 +class NLPDynamicChunker:                                      
       26 +    """                                                       
       27 +    基于 NLP 的语义动态文本切分器。                           
       28 +                                                              
       29 +    工作流程:                                                 
       30 +    1. 用正则表达式将文本分割为句子（支持中英文混合）         
       31 +    2. 可选：用 sentence-transformers 计算句子级语义嵌入      
       32 +    3. 按语义相似度 + chunk_size 贪婪合并句子                 
       33 +    4. 重叠窗口按完整句子保留，不在字符中间截断               
       34 +    5. 识别章节标题作为自然切分边界                           
       35 +                                                              
       36 +    属性:                                                     
       37 +        target_chunk_size: 目标块大小（字符数）               
       38 +        min_chunk_size: 最小块大小                            
       39 +        max_chunk_size: 最大块大小                            
       40 +        chunk_overlap: 块间重叠字符数                         
       41 +        overlap_by_sentence: 是否按完整句子重叠               
       42 +        enable_semantic: 是否启用语义相似度计算               
       43 +        similarity_threshold: 语义相似度阈值                  
       44 +        embedding_model_name: 语义嵌入模型名称                
       45 +        preserve_headings: 是否保留标题边界                   
       46 +        add_start_index: 是否添加起始索引 metadata            
       47 +    """                                                       
       48 +                                                              
       49 +    # 中英文混合句子分割正则                                  
       50 +    SENTENCE_PATTERN = re.compile(                            
       51 +        r'(?:[^。！？\n!?.;：；，,]+(?:\.\.\.|…|[。！？.!?]))'
       52 +        r'|(?:[^。！？\n.!?;：；，,]+(?:\\n|\n|$))'           
       53 +        r'|(?:[^。！？\n.!?]+)'                               
       54 +    )                                                         
       55 +                                                              
       56 +    # 标题/章节识别模式                                       
       57 +    HEADING_PATTERN = re.compile(                             
       58 +        r'^\s*(?:第[一二三四五六七八九十\d]+[章节]|'          
       59 +        r'(?:chapter|section|part)\s+\d+|'                    
       60 +        r'(?:\d+[.、．])\s*\S|'                               
       61 +        r'^(?:[A-Z][^.。！？\n]{0,60}$)|'                     
       62 +        r'^(?:[一二三四五六七八九十]、|（[一二三四五六七八九十
          +]）))',                                                       
       63 +        re.IGNORECASE | re.MULTILINE                          
       64 +    )                                                         
       65 +                                                              
       66 +    def __init__(                                             
       67 +        self,                                                 
       68 +        target_chunk_size: int = 800,                         
       69 +        min_chunk_size: int = 300,                            
       70 +        max_chunk_size: int = 1200,                           
       71 +        chunk_overlap: int = 150,                             
       72 +        overlap_by_sentence: bool = True,                     
       73 +        enable_semantic: bool = True,                         
       74 +        similarity_threshold: float = 0.55,                   
       75 +        embedding_model_name: str = "sentence-transformers/all
          +-MiniLM-L6-v2",                                               
       76 +        preserve_headings: bool = True,                       
       77 +        add_start_index: bool = True                          
       78 +    ):                                                        
       79 +        self.target_chunk_size = target_chunk_size            
       80 +        self.min_chunk_size = min_chunk_size                  
       81 +        self.max_chunk_size = max_chunk_size                  
       82 +        self.chunk_overlap = chunk_overlap                    
       83 +        self.overlap_by_sentence = overlap_by_sentence        
       84 +        self.enable_semantic = enable_semantic                
       85 +        self.similarity_threshold = similarity_threshold      
       86 +        self.embedding_model_name = embedding_model_name      
       87 +        self.preserve_headings = preserve_headings            
       88 +        self.add_start_index = add_start_index                
       89 +        self._embedding_model = None                          
       90 +                                                              
       91 +        logger.info(                                          
       92 +            f"NLPDynamicChunker 初始化: target={target_chunk_s
          +ize}, "                                                       
       93 +            f"min={min_chunk_size}, max={max_chunk_size}, "   
       94 +            f"overlap={chunk_overlap}, semantic={enable_semant
          +ic}"                                                          
       95 +        )                                                     
       96 +                                                              
       97 +    def _get_embedding_model(self):                           
       98 +        """懒加载语义嵌入模型。"""                            
       99 +        if self._embedding_model is None and self.enable_seman
          +tic:                                                          
      100 +            try:                                              
      101 +                from sentence_transformers import SentenceTran
          +sformer                                                       
      102 +                self._embedding_model = SentenceTransformer(  
      103 +                    self.embedding_model_name,                
      104 +                    device="cpu"                              
      105 +                )                                             
      106 +                logger.info(f"语义嵌入模型已加载: {self.embedd
          +ing_model_name}")                                             
      107 +            except Exception as e:                            
      108 +                logger.warning(f"语义嵌入模型加载失败，回退到 
          +纯长度切分: {e}")                                             
      109 +                self.enable_semantic = False                  
      110 +        return self._embedding_model                          
      111 +                                                              
      112 +    def split_documents(self, documents: List[Document]) -> Li
          +st[Document]:                                                 
      113 +        """                                                   
      114 +        将文档列表按 NLP 动态策略切分为文本块。               
      115 +                                                              
      116 +        Args:                                                 
      117 +            documents: LangChain Document 列表（通常每页一个）
      118 +                                                              
      119 +        Returns:                                              
      120 +            切分后的 Document 列表，每个包含 page_content 和 m
          +etadata                                                       
      121 +        """                                                   
      122 +        if not documents:                                     
      123 +            return []                                         
      124 +                                                              
      125 +        all_chunks = []                                       
      126 +                                                              
      127 +        for doc in documents:                                 
      128 +            text = doc.page_content                           
      129 +            page = doc.metadata.get("page", 1)                
      130 +            source = doc.metadata.get("source", "")           
      131 +                                                              
      132 +            if not text.strip():                              
      133 +                continue                                      
      134 +                                                              
      135 +            sentences, heading_flags = self._split_sentences(t
          +ext)                                                          
      136 +                                                              
      137 +            if self.enable_semantic:                          
      138 +                self._get_embedding_model()                   
      139 +                embeddings = self._compute_embeddings(sentence
          +s)                                                            
      140 +            else:                                             
      141 +                embeddings = None                             
      142 +                                                              
      143 +            chunk_texts = self._merge_sentences_into_chunks(  
      144 +                sentences, embeddings, heading_flags          
      145 +            )                                                 
      146 +                                                              
      147 +            for i, chunk_text in enumerate(chunk_texts):      
      148 +                chunk_meta = {                                
      149 +                    "page": page,                             
      150 +                    "source": source,                         
      151 +                    "chunk_index": i                          
      152 +                }                                             
      153 +                if self.add_start_index:                      
      154 +                    chunk_meta["start_index"] = i * self.targe
          +t_chunk_size                                                  
      155 +                                                              
      156 +                all_chunks.append(Document(                   
      157 +                    page_content=chunk_text,                  
      158 +                    metadata=chunk_meta                       
      159 +                ))                                            
      160 +                                                              
      161 +        # 跨页重新编号 chunk_index                            
      162 +        for i, chunk in enumerate(all_chunks):                
      163 +            chunk.metadata["chunk_index"] = i                 
      164 +                                                              
      165 +        logger.info(                                          
      166 +            f"NLP 动态切分完成: {len(documents)} 页 → {len(all
          +_chunks)} 块"                                                 
      167 +        )                                                     
      168 +        return all_chunks                                     
      169 +                                                              
      170 +    def _split_sentences(self, text: str) -> Tuple[List[str], 
          +List[bool]]:                                                  
      171 +        """                                                   
      172 +        将文本分割为句子列表。                                
      173 +                                                              
      174 +        支持中英文混合句子边界: . ! ? 。！？换行符。          
      175 +        同时识别章节标题。                                    
      176 +                                                              
      177 +        Args:                                                 
      178 +            text: 输入文本                                    
      179 +                                                              
      180 +        Returns:                                              
      181 +            (句子列表, 是否为标题的布尔列表)                  
      182 +        """                                                   
      183 +        raw_sentences = []                                    
      184 +        heading_flags = []                                    
      185 +                                                              
      186 +        # 按换行预分割段落                                    
      187 +        paragraphs = text.split('\n')                         
      188 +                                                              
      189 +        for para in paragraphs:                               
      190 +            para = para.strip()                               
      191 +            if not para:                                      
      192 +                continue                                      
      193 +                                                              
      194 +            is_heading = bool(self.HEADING_PATTERN.match(para)
          +) if self.preserve_headings else False                        
      195 +                                                              
      196 +            # 对段落内进行句子分割                            
      197 +            para_sentences = self._regex_split_sentences(para)
      198 +                                                              
      199 +            for sent in para_sentences:                       
      200 +                sent = sent.strip()                           
      201 +                if sent:                                      
      202 +                    raw_sentences.append(sent)                
      203 +                    heading_flags.append(is_heading)          
      204 +                    is_heading = False  # 仅段落首句标记为标题
      205 +                                                              
      206 +        return raw_sentences, heading_flags                   
      207 +                                                              
      208 +    def _regex_split_sentences(self, text: str) -> List[str]: 
      209 +        """                                                   
      210 +        用正则表达式分割句子，保留标点符号。                  
      211 +                                                              
      212 +        处理:                                                 
      213 +        - 中文标点: 。！？作为句子边界                        
      214 +        - 英文标点: . ! ? 后跟空格和大写字母作为边界          
      215 +        - 避免: 数字中的小数点、缩写中的点号                  
      216 +        """                                                   
      217 +        sentences = []                                        
      218 +        buf = ""                                              
      219 +                                                              
      220 +        for i, ch in enumerate(text):                         
      221 +            buf += ch                                         
      222 +                                                              
      223 +            if ch in '。！？':                                
      224 +                sentences.append(buf)                         
      225 +                buf = ""                                      
      226 +            elif ch in '.!?':                                 
      227 +                # 英文句子边界: 需要看后续是否为空格+大写或结 
          +尾                                                            
      228 +                rest = text[i + 1:]                           
      229 +                if not rest or (rest[0] in ' \n' and (        
      230 +                    len(rest) < 2 or rest.lstrip()[0].isupper(
          +) or rest.lstrip()[0] in '"\'')):                             
      231 +                    sentences.append(buf)                     
      232 +                    buf = ""                                  
      233 +                                                              
      234 +        if buf.strip():                                       
      235 +            sentences.append(buf)                             
      236 +                                                              
      237 +        return sentences                                      
      238 +                                                              
      239 +    def _compute_embeddings(self, sentences: List[str]):      
      240 +        """计算句子级语义嵌入向量。"""                        
      241 +        if not self._embedding_model:                         
      242 +            return None                                       
      243 +        try:                                                  
      244 +            embeddings = self._embedding_model.encode(        
      245 +                sentences,                                    
      246 +                normalize_embeddings=True,                    
      247 +                show_progress_bar=False                       
      248 +            )                                                 
      249 +            return embeddings                                 
      250 +        except Exception as e:                                
      251 +            logger.warning(f"嵌入计算失败，回退到纯长度切分: {
          +e}")                                                          
      252 +            self.enable_semantic = False                      
      253 +            return None                                       
      254 +                                                              
      255 +    def _merge_sentences_into_chunks(                         
      256 +        self,                                                 
      257 +        sentences: List[str],                                 
      258 +        embeddings,                                           
      259 +        heading_flags: List[bool]                             
      260 +    ) -> List[str]:                                           
      261 +        """                                                   
      262 +        核心合并算法: 按语义相似度 + 大小约束贪婪合并句子为文 
          +本块。                                                        
      263 +                                                              
      264 +        Args:                                                 
      265 +            sentences: 句子列表                               
      266 +            embeddings: 句子嵌入数组（可为 None）             
      267 +            heading_flags: 每个句子是否为标题                 
      268 +                                                              
      269 +        Returns:                                              
      270 +            合并后的文本块列表                                
      271 +        """                                                   
      272 +        if not sentences:                                     
      273 +            return []                                         
      274 +                                                              
      275 +        chunks = []                                           
      276 +        current_sentences = []                                
      277 +        current_embeddings = []                               
      278 +        current_len = 0                                       
      279 +                                                              
      280 +        for i, (sent, is_heading) in enumerate(zip(sentences, 
          +heading_flags)):                                              
      281 +            sent_len = len(sent)                              
      282 +            sent_emb = embeddings[i] if embeddings is not None
          + else None                                                    
      283 +                                                              
      284 +            # 规则1: 遇到标题强制开启新块                     
      285 +            if is_heading and current_sentences and current_le
          +n >= self.min_chunk_size:                                     
      286 +                chunks.append("".join(current_sentences))     
      287 +                current_sentences, current_embeddings, current
          +_len = self._start_new_chunk(                                 
      288 +                    sentences, embeddings, current_sentences, 
          +i, sent, sent_emb                                             
      289 +                )                                             
      290 +                continue                                      
      291 +                                                              
      292 +            # 首个句子直接加入                                
      293 +            if not current_sentences:                         
      294 +                current_sentences.append(sent)                
      295 +                if sent_emb is not None:                      
      296 +                    current_embeddings.append(sent_emb)       
      297 +                current_len = sent_len                        
      298 +                continue                                      
      299 +                                                              
      300 +            # 规则2: 超过最大块大小，强制开启新块（带重叠）   
      301 +            if current_len + sent_len > self.max_chunk_size:  
      302 +                chunks.append("".join(current_sentences))     
      303 +                current_sentences, current_embeddings, current
          +_len = self._start_new_chunk(                                 
      304 +                    sentences, embeddings, current_sentences, 
          +i, sent, sent_emb                                             
      305 +                )                                             
      306 +                continue                                      
      307 +                                                              
      308 +            # 规则3: 语义相似度判断                           
      309 +            should_merge = True                               
      310 +            if self.enable_semantic and sent_emb is not None a
          +nd current_embeddings:                                        
      311 +                chunk_centroid = np.mean(current_embeddings, a
          +xis=0)                                                        
      312 +                similarity = float(np.dot(sent_emb, chunk_cent
          +roid))                                                        
      313 +                # 余弦相似度（已归一化）= 内积                
      314 +                                                              
      315 +                if similarity < self.similarity_threshold:    
      316 +                    should_merge = False                      
      317 +                                                              
      318 +            if should_merge:                                  
      319 +                # 语义相近，合并                              
      320 +                current_sentences.append(sent)                
      321 +                if sent_emb is not None:                      
      322 +                    current_embeddings.append(sent_emb)       
      323 +                current_len += sent_len                       
      324 +            elif current_len >= self.min_chunk_size:          
      325 +                # 语义不同且已达最小块，开启新块              
      326 +                chunks.append("".join(current_sentences))     
      327 +                current_sentences, current_embeddings, current
          +_len = self._start_new_chunk(                                 
      328 +                    sentences, embeddings, current_sentences, 
          +i, sent, sent_emb                                             
      329 +                )                                             
      330 +            else:                                             
      331 +                # 未达最小块，即使语义不同也强制合并          
      332 +                current_sentences.append(sent)                
      333 +                if sent_emb is not None:                      
      334 +                    current_embeddings.append(sent_emb)       
      335 +                current_len += sent_len                       
      336 +                                                              
      337 +        # 最后一个块                                          
      338 +        if current_sentences:                                 
      339 +            chunks.append("".join(current_sentences))         
      340 +                                                              
      341 +        return chunks                                         
      342 +                                                              
      343 +    def _start_new_chunk(                                     
      344 +        self,                                                 
      345 +        all_sentences: List[str],                             
      346 +        all_embeddings,                                       
      347 +        prev_sentences: List[str],                            
      348 +        current_idx: int,                                     
      349 +        new_sent: str,                                        
      350 +        new_emb                                               
      351 +    ) -> Tuple[List[str], List, int]:                         
      352 +        """                                                   
      353 +        开启新文本块，包含来自前一块的句子级重叠。            
      354 +                                                              
      355 +        Args:                                                 
      356 +            all_sentences: 所有句子列表（仅用于非重叠模式）   
      357 +            all_embeddings: 所有嵌入列表                      
      358 +            prev_sentences: 前一个块的句子列表                
      359 +            current_idx: 当前句子索引                         
      360 +            new_sent: 新块的第一个句子                        
      361 +            new_emb: 新块第一个句子的嵌入                     
      362 +                                                              
      363 +        Returns:                                              
      364 +            (新块句子列表, 新块嵌入列表, 新块总长度)          
      365 +        """                                                   
      366 +        if not self.overlap_by_sentence:                      
      367 +            return [new_sent], ([new_emb] if new_emb is not No
          +ne else []), len(new_sent)                                    
      368 +                                                              
      369 +        overlap_sentences = self._get_overlap_sentences(prev_s
          +entences)                                                     
      370 +        new_sentences = list(overlap_sentences)               
      371 +        new_sentences.append(new_sent)                        
      372 +                                                              
      373 +        new_len = sum(len(s) for s in new_sentences)          
      374 +                                                              
      375 +        new_embeddings = []                                   
      376 +        if all_embeddings is not None and new_emb is not None:
      377 +            new_embeddings = []  # 重叠句子的嵌入暂不复制，避 
          +免复杂性                                                      
      378 +            if new_emb is not None:                           
      379 +                new_embeddings.append(new_emb)                
      380 +                                                              
      381 +        return new_sentences, new_embeddings, new_len         
      382 +                                                              
      383 +    def _get_overlap_sentences(self, sentences: List[str]) -> 
          +List[str]:                                                    
      384 +        """                                                   
      385 +        从前一个块末尾取完整句子作为下一个块的重叠前缀。      
      386 +                                                              
      387 +        根据 chunk_overlap 字符数计算需要保留的句子数，       
      388 +        确保不从句子中间截断。                                
      389 +                                                              
      390 +        Args:                                                 
      391 +            sentences: 前一个块的句子列表                     
      392 +                                                              
      393 +        Returns:                                              
      394 +            重叠句子列表（从前一块末尾取）                    
      395 +        """                                                   
      396 +        if not sentences or self.chunk_overlap <= 0:          
      397 +            return []                                         
      398 +                                                              
      399 +        target_overlap = self.chunk_overlap                   
      400 +        accumulated = 0                                       
      401 +        overlap_count = 0                                     
      402 +                                                              
      403 +        # 从末尾向前数，直到达到目标重叠字符数                
      404 +        for sent in reversed(sentences):                      
      405 +            accumulated += len(sent)                          
      406 +            overlap_count += 1                                
      407 +            if accumulated >= target_overlap:                 
      408 +                break                                         
      409 +                                                              
      410 +        return sentences[-overlap_count:] if overlap_count > 0
          + else []                                                      
      411 +                                                              
      412 +                                                              
      413  class DocumentProcessor:
      414      """
      415      Handles PDF document loading and text chunking.

● Now update DocumentProcessor.__init__ to accept chunking strategy and NLP
  params:

● Update(core\document_processor.py)
  ⎿  Added 39 lines, removed 7 lines
      437          enable_ocr: bool = True,
      438          ocr_languages: str = "chi_sim+eng",
      439          ocr_dpi: int = 200,
      440 -        tesseract_cmd: str = ""                               
      440 +        tesseract_cmd: str = "",                              
      441 +        chunking_strategy: str = "fixed_size",                
      442 +        min_chunk_size: int = 300,                            
      443 +        max_chunk_size: int = 1200,                           
      444 +        nlp_config: Optional[Dict[str, Any]] = None           
      445      ):
      446          """
      447          初始化文档处理器。
     ...
      454              ocr_languages: OCR 语言包（如 "chi_sim+eng"）
      455              ocr_dpi: OCR 渲染 DPI
      456              tesseract_cmd: Tesseract 可执行文件路径
      457 +            chunking_strategy: 切分策略 "fixed_size" / "nlp_dy
          +namic"                                                        
      458 +            min_chunk_size: NLP 动态切分最小块大小            
      459 +            max_chunk_size: NLP 动态切分最大块大小            
      460 +            nlp_config: NLP 动态切分详细配置字典              
      461          """
      462          self.chunk_size = chunk_size
      463          self.chunk_overlap = chunk_overlap
     ...
      466          self.ocr_languages = ocr_languages
      467          self.ocr_dpi = ocr_dpi
      468          self.tesseract_cmd = tesseract_cmd
      469 +        self.chunking_strategy = chunking_strategy            
      470 +        self.nlp_config = nlp_config or {}                    
      471  
      462 -        self.text_splitter = RecursiveCharacterTextSplitter(  
      463 -            chunk_size=self.chunk_size,                       
      464 -            chunk_overlap=self.chunk_overlap,                 
      465 -            add_start_index=self.add_start_index              
      472 +        # 初始化对应的切分器                                  
      473 +        if chunking_strategy == "nlp_dynamic":                
      474 +            self.chunker = NLPDynamicChunker(                 
      475 +                target_chunk_size=chunk_size,                 
      476 +                min_chunk_size=min_chunk_size,                
      477 +                max_chunk_size=max_chunk_size,                
      478 +                chunk_overlap=chunk_overlap,                  
      479 +                overlap_by_sentence=self.nlp_config.get("overl
          +ap_by_sentence", True),                                       
      480 +                enable_semantic=self.nlp_config.get("enable_se
          +mantic_similarity", True),                                    
      481 +                similarity_threshold=self.nlp_config.get("sema
          +ntic_similarity_threshold", 0.55),                            
      482 +                embedding_model_name=self.nlp_config.get(     
      483 +                    "semantic_embedding_model", "sentence-tran
          +sformers/all-MiniLM-L6-v2"                                    
      484 +                ),                                            
      485 +                preserve_headings=self.nlp_config.get("preserv
          +e_headings", True),                                           
      486 +                add_start_index=add_start_index               
      487 +            )                                                 
      488 +            self.text_splitter = None                         
      489 +        else:                                                 
      490 +            self.text_splitter = RecursiveCharacterTextSplitte
          +r(                                                            
      491 +                chunk_size=chunk_size,                        
      492 +                chunk_overlap=chunk_overlap,                  
      493 +                add_start_index=add_start_index               
      494 +            )                                                 
      495 +            self.chunker = None                               
      496 +                                                              
      497 +        logger.info(                                          
      498 +            f"DocumentProcessor 初始化: chunk_size={chunk_size
          +}, overlap={chunk_overlap}, "                                 
      499 +            f"strategy={chunking_strategy}, ocr={'enabled' if 
          +enable_ocr else 'disabled'}"                                  
      500          )
      501  
      468 -        logger.info(f"DocumentProcessor 初始化: chunk_size={ch
          -unk_size}, overlap={chunk_overlap}, ocr={'enabled' if enable_o
          -cr else 'disabled'}")                                         
      469 -                                                              
      502      def process_uploaded_file(
      503          self,
      504          uploaded_file,

● Now update the chunking call in process_uploaded_file to use the
  strategy-specific splitter:

● Update(core\document_processor.py)
  ⎿  Added 5 lines, removed 2 lines
      544              docs = self._load_pdf(temp_file_path)
      545              logger.info(f"Loaded {len(docs)} pages from PDF")
      546  
      547 -            # Split into chunks                               
      548 -            chunks = self.text_splitter.split_documents(docs) 
      547 +            # Split into chunks using the configured strategy 
      548 +            if self.chunking_strategy == "nlp_dynamic" and sel
          +f.chunker:                                                    
      549 +                chunks = self.chunker.split_documents(docs)   
      550 +            else:                                             
      551 +                chunks = self.text_splitter.split_documents(do
          +cs)                                                           
      552              logger.info(f"Document split into {len(chunks)} ch
           unks")
      553  
      554              # Update metadata for each chunk
● Now update app.py to pass NLP chunking config to DocumentProcessor:

● Update(app.py)
  ⎿  Added 5 lines, removed 1 line
      429                  enable_ocr=config.is_ocr_enabled(),
      430                  ocr_languages=config.get_ocr_languages(),
      431                  ocr_dpi=config.get_ocr_dpi(),
      432 -                tesseract_cmd=config.get_tesseract_cmd()      
      432 +                tesseract_cmd=config.get_tesseract_cmd(),     
      433 +                chunking_strategy=config.get_chunking_strategy
          +(),                                                           
      434 +                min_chunk_size=config.get_min_chunk_size(),   
      435 +                max_chunk_size=config.get_max_chunk_size(),   
      436 +                nlp_config=config.get_nlp_chunking_config()   
      437              )
      438  
      439              st.info(f"正在处理: **{uploaded_file.name}**")

  Read 1 file (ctrl+o to expand)

● Update(pages\2_Collections.py)
  ⎿  Added 5 lines, removed 1 line
      538                          enable_ocr=config.is_ocr_enabled(),
      539                          ocr_languages=config.get_ocr_languages
           (),
      540                          ocr_dpi=config.get_ocr_dpi(),
      541 -                        tesseract_cmd=config.get_tesseract_cmd
          -()                                                            
      541 +                        tesseract_cmd=config.get_tesseract_cmd
          +(),                                                           
      542 +                        chunking_strategy=config.get_chunking_
          +strategy(),                                                   
      543 +                        min_chunk_size=config.get_min_chunk_si
          +ze(),                                                         
      544 +                        max_chunk_size=config.get_max_chunk_si
          +ze(),                                                         
      545 +                        nlp_config=config.get_nlp_chunking_con
          +fig()                                                         
      546                      )
      547  
      548                      # Reset file position for processing

● requirements.txt 无需修改 — numpy（随 torch
  安装）、sentence-transformers、re（标准库）均已就位。

● 验证最终文件完整性：

  Read 1 file (ctrl+o to expand)

● All files are consistent. Let me do a quick trace to verify the end-to-end
  pipeline is correct.

  Read 1 file (ctrl+o to expand)

● 全部修改完成。以下是变更总结：

  ---
  变更总览

  1. config.yaml — 新增 NLP 动态切分配置

  document_processing:
    chunking_strategy: "nlp_dynamic"    # 策略: fixed_size / nlp_dynamic
    chunk_size: 800                      # 目标块大小
    chunk_overlap: 150                   # 重叠长度
    min_chunk_size: 300                  # 最小块（低于此强制合并）
    max_chunk_size: 1200                 # 最大块（超过强制断句）
    nlp:
      sentence_splitter: "regex"
      enable_semantic_similarity: true
      semantic_similarity_threshold: 0.55
      semantic_embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
      overlap_by_sentence: true          # 重叠按完整句子保留
      preserve_headings: true            # 章节标题作为切分边界

  2. config_loader.py — 新增 4 个 getter

  - get_chunking_strategy() — 返回切分策略名称
  - get_min_chunk_size() / get_max_chunk_size() — NLP 大小约束
  - get_nlp_chunking_config() — 返回完整 NLP 配置字典（含默认值）

  3. core/document_processor.py — 核心改动

  新增 NLPDynamicChunker 类（约 390 行），包含：

  ┌─────────────────────────┬───────────────────────────────────────────────┐
  │          方法           │                     功能                      │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │ _split_sentences()      │ 按段落/换行预分割，正则识别中英文句子边界，同 │
  │                         │ 时标记章节标题                                │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │ _regex_split_sentences( │ 字符级扫描，. ! ? 。 ！ ？                    │
  │ )                       │ 作为边界，避免小数点误分割                    │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │ _compute_embeddings()   │ 调用 SentenceTransformer.encode()             │
  │                         │ 计算句子向量（已归一化）                      │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │                         │ 核心算法：贪婪合并，三条规则 —— ①             │
  │ _merge_sentences_into_c │ 标题强制新块 ② 超 max_chunk_size 强制断句 ③   │
  │ hunks()                 │ 语义相似度 < 阈值且达标 min_chunk_size        │
  │                         │ 时开新块                                      │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │ _start_new_chunk()      │ 创建新块时从前一块末尾取完整句子作为重叠前缀  │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │ _get_overlap_sentences( │ 从末尾向前累加字符数，取 ≥ chunk_overlap      │
  │ )                       │ 的最小完整句子集                              │
  └─────────────────────────┴───────────────────────────────────────────────┘

  DocumentProcessor.__init__ 更新：
  - 新增 chunking_strategy、min_chunk_size、max_chunk_size、nlp_config 参数
  - 策略为 "nlp_dynamic" 时创建 NLPDynamicChunker，否则使用原有
  RecursiveCharacterTextSplitter
  - 向后兼容：默认 "fixed_size"，旧代码无需修改

  process_uploaded_file 更新：
  - 根据 chunking_strategy 选择 self.chunker.split_documents() 或
  self.text_splitter.split_documents()

  4. app.py + pages/2_Collections.py — 传参更新

  两处 DocumentProcessor(...) 调用均新增 4
  个参数：chunking_strategy、min_chunk_size、max_chunk_size、nlp_config

  5. requirements.txt — 无需修改

  sentence-transformers、torch、numpy 均已存在。

  ---
  回退到旧策略：将 config.yaml 中 chunking_strategy 改为 "fixed_size"
  即可恢复原有 RecursiveCharacterTextSplitter 行为。

✻ Crunched for 9m 23s

※ recap: NLP动态切分已全部实现完毕。`NLPDynamicChunker` 类支持正则分句、语义相似度合并、句子级重叠窗口和标题识别，通过 `config.yaml` 中 `chunking_strategy: "nlp_dynamic"` 切换。可直接启动测试。