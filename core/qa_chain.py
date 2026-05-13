"""
问答链模块

实现基于检索增强生成（RAG）的文档问答功能。
支持 OpenAI 和 DeepSeek（通过兼容 OpenAI 的 API）作为对话模型提供商。
"""

import os
import logging
from typing import List, Generator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


class QAChain:
    """
    基于检索增强生成的问答链。

    协调 RAG 流程:
    1. 检索相关文档文本块
    2. 将上下文和问题格式化为提示词
    3. 使用 LLM 生成回答
    4. 逐 token 流式输出回答

    支持多种对话模型提供商:
    - openai: 使用环境变量中的 OPENAI_API_KEY
    - deepseek: 使用环境变量中的 DEEPSEEK_API_KEY（兼容 OpenAI 的 API）

    属性:
        llm_model: 用于生成回答的 ChatOpenAI 模型
        retriever: 查找相关文本块的向量存储检索器
        system_prompt: 问答系统提示词模板

    示例:
        >>> qa = QAChain(
        ...     model_name="gpt-4o-mini",
        ...     retriever=retriever,
        ...     system_prompt="根据以下内容回答: {document}"
        ... )
        >>> for chunk in qa.stream_answer("什么是 AI？"):
        ...     print(chunk, end="")
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        retriever: VectorStoreRetriever = None,
        system_prompt: str = None,
        provider: str = "openai",
        base_url: str = ""
    ):
        """
        初始化问答链。

        Args:
            model_name: 对话模型名称
            temperature: 模型温度 (0.0 = 确定性输出)
            retriever: 向量存储检索器实例
            system_prompt: 系统提示词模板，需包含 {question} 和 {document} 占位符
            provider: 对话模型提供商 ("openai" / "deepseek")
            base_url: 兼容 OpenAI 的 API 基础地址（用于 deepseek）
        """
        self.model_name = model_name
        self.temperature = temperature
        self.retriever = retriever
        self.provider = provider

        # 根据提供商初始化 LLM
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "未找到 DEEPSEEK_API_KEY 环境变量。"
                    "请在 .env 文件中设置或执行 export DEEPSEEK_API_KEY=your_key"
                )
            self.llm_model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                base_url=base_url or "https://api.deepseek.com",
                api_key=api_key
            )
            logger.info(f"已初始化 DeepSeek 对话模型: model={model_name}, temp={temperature}")
        else:
            self.llm_model = ChatOpenAI(model=model_name, temperature=temperature)
            logger.info(f"已初始化 ChatOpenAI: model={model_name}, temp={temperature}")

        # 设置默认系统提示词
        self.system_prompt = system_prompt or """
        你是一个乐于助人的助手。
        请仅使用以下信息 {document} 来回答下面的问题 {question}。
        如果你无法回答问题，请直接说你无法回答该问题。
        """

        # 创建提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt)
        ])

    def retrieve_context(self, query: str) -> List[Document]:
        """
        检索与查询相关的文档文本块。

        Args:
            query: 用户问题

        Returns:
            相关 Document 对象列表

        Raises:
            ValueError: 检索器未设置时抛出

        示例:
            >>> docs = qa.retrieve_context("什么是机器学习？")
            >>> print(f"检索到 {len(docs)} 个文本块")
        """
        if not self.retriever:
            raise ValueError("检索器未配置")

        logger.info(f"正在检索上下文，查询: {query[:50]}...")
        docs = self.retriever.invoke(query)
        logger.info(f"检索到 {len(docs)} 个相关文本块")

        return docs

    def format_context(self, documents: List[Document]) -> str:
        """
        将检索到的文档格式化为单个上下文字符串。

        Args:
            documents: Document 对象列表

        Returns:
            拼接后的文档内容

        示例:
            >>> context = qa.format_context(docs)
            >>> print(f"上下文长度: {len(context)} 字符")
        """
        context = "\n\n".join([doc.page_content for doc in documents])
        logger.debug(f"已格式化上下文: {len(context)} 字符")
        return context

    def generate_answer(self, question: str, context: str) -> str:
        """
        使用 LLM 生成回答（非流式）。

        Args:
            question: 用户问题
            context: 检索到的文档上下文

        Returns:
            生成的回答文本

        示例:
            >>> answer = qa.generate_answer("什么是 AI？", context)
            >>> print(answer)
        """
        final_prompt = self.prompt_template.invoke({
            "question": question,
            "document": context
        })

        logger.info("正在生成回答（非流式）...")
        response = self.llm_model.invoke(final_prompt)
        answer = response.content

        logger.info(f"回答已生成: {len(answer)} 字符")
        return answer

    def stream_answer(self, question: str, context: str) -> Generator[str, None, None]:
        """
        流式生成回答，逐 token 输出。

        Args:
            question: 用户问题
            context: 检索到的文档上下文

        Yields:
            生成过程中的回答文本片段

        示例:
            >>> for chunk in qa.stream_answer("什么是 AI？", context):
            ...     print(chunk, end="", flush=True)
        """
        final_prompt = self.prompt_template.invoke({
            "question": question,
            "document": context
        })

        logger.debug(f"已生成最终提示词，上下文 {len(context)} 字符")
        logger.info("正在流式输出 LLM 回答...")

        for chunk in self.llm_model.stream(final_prompt):
            yield chunk.content

    def answer_question(
        self,
        question: str,
        stream: bool = True
    ) -> Generator[str, None, None] | str:
        """
        完整的 RAG 流程：检索、格式化、回答。

        这是结合检索与生成的主要方法。

        Args:
            question: 用户问题
            stream: 是否流式输出

        Returns:
            流式输出返回 Generator，非流式返回字符串

        Raises:
            ValueError: 未找到相关文档时抛出

        示例:
            >>> # 流式模式
            >>> for chunk in qa.answer_question("什么是 AI？"):
            ...     print(chunk, end="")
            >>>
            >>> # 非流式模式
            >>> answer = qa.answer_question("什么是 AI？", stream=False)
        """
        docs = self.retrieve_context(question)

        if not docs:
            logger.warning("未找到相关文档")
            raise ValueError("文档中未找到相关信息")

        context = self.format_context(docs)

        if stream:
            return self.stream_answer(question, context)
        else:
            return self.generate_answer(question, context)

    def update_retriever(self, retriever: VectorStoreRetriever) -> None:
        """
        更新检索器实例。

        Args:
            retriever: 新的检索器实例

        示例:
            >>> new_retriever = vector_store.get_retriever(search_k=5)
            >>> qa.update_retriever(new_retriever)
        """
        self.retriever = retriever
        logger.info("检索器已更新")
