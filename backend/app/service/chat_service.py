# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

import asyncio
import json
import os
from typing import Callable, List, Dict, Any, Optional, Generator
import numpy as np
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

from .document_service import DocumentService
from .web_search_service import WebSearchService
from .memory_service import get_memory_service
import logging
from service.llm_client import create_openai_client
from service.qwen_token_counter import get_qwen_token_counter

logger = logging.getLogger(__name__)


class ChatService:
    """Chat service that combines document retrieval and LLM generation"""
    
    def __init__(self, document_service: DocumentService, web_search_service: WebSearchService):
        """
        Initialize the ChatService.
        
        Args:
            document_service: Document service for knowledge base retrieval
            web_search_service: Web search service for internet search
        """
        self.document_service = document_service
        self.web_search_service = web_search_service
        self.openai_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.openai_model = os.environ.get("OPENAI_MODEL", "deepseek-r1")
        self.token_counter = get_qwen_token_counter()
    
    def retrieve_from_knowledge_base(self, question: str, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents from knowledge base.
        
        Args:
            question: User question
            dataset_id: Dataset ID
            
        Returns:
            List of retrieved documents
        """
        try:
            response = self.document_service.retrieve_documents(
                question=question,
                dataset_ids=[dataset_id]
            )
            
            if response.get("code") != 0:
                return []
                
            # 从响应中提取文档
            documents = []
            if "data" in response and "chunks" in response["data"]:
                for i, chunk in enumerate(response["data"]["chunks"]):
                    # 使用document_keyword作为title
                    title = chunk.get("document_keyword", None)
                    
                    doc = {
                        "id": i+1,
                        "content": chunk.get("content", ""),
                        "content_with_weight": f"{chunk.get('content', '')} (相关度: {chunk.get('score', 0):.2f})",
                        "source": "knowledge",
                        "title": title,
                        "weight": chunk.get("score", 1.0)
                    }
                    documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error retrieving from knowledge base: {str(e)}")
            return []
    
    def retrieve_from_web(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve information from web search.
        
        Args:
            question: User question
            
        Returns:
            List of search results formatted as documents
        """
        try:
            # 执行Web搜索
            search_results = self.web_search_service.search(query=question)
            
            if "error" in search_results and search_results["error"]:
                return []
            
            # 提取搜索结果（博查格式：name / url / snippet / summary / siteName / datePublished）
            formatted_results = self.web_search_service.extract_search_results(search_results)

            # 转换为下游统一的文档格式
            documents = []
            for i, result in enumerate(formatted_results):
                # 优先用长摘要，信息量比 snippet 大
                content = result.get("summary") or result.get("snippet") or ""
                if not content:
                    continue
                documents.append({
                    "id": i + 1,
                    "content": content,
                    "content_with_weight": content,
                    "source": "web",  # 类型标记，前端靠它给引用分组，勿改成站点名
                    "title": result.get("name") or None,
                    "link": result.get("url") or None,
                    "site_name": result.get("siteName", ""),
                    "date_published": result.get("datePublished", ""),
                    "weight": 1.0 - (i * 0.1),  # 根据位置降低权重
                })

            return documents
        except Exception as e:
            logger.error(f"Error retrieving from web: {str(e)}")
            return []
    
    def rerank_similarity(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """
        使用DashScope重排对文档进行相似度评分
        
        Args:
            query: 用户查询
            documents: 要评分的文档列表
            
        Returns:
            文档相似度分数列表
        """
        try:
            api_key = os.getenv("DASHSCOPE_API_KEY", self.openai_api_key)
            
            # 从文档中提取文本
            texts = [doc["content"] for doc in documents]
            
            # 创建节点列表
            nodes = [NodeWithScore(node=Node(text=text), score=1.0) for text in texts]
            
            # 初始化 DashScopeRerank
            dashscope_rerank = DashScopeRerank(top_n=len(texts), api_key=api_key)
            
            # 执行重排序
            results = dashscope_rerank.postprocess_nodes(nodes, query_str=query)
            
            # 提取分数
            scores = [res.score for res in results]
            scores = np.array(scores)
            
            return scores
        except Exception as e:
            logger.error(f"Error in rerank_similarity: {str(e)}")
            # 出错时返回原始权重
            return [doc.get("weight", 1.0) for doc in documents]
    
    def rerank_documents(self, question: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """在聊天路由的工作线程内复用会话RAG的精排与上下文预算。"""
        if not documents:
            return []
        from service.retrieval_service import rerank_knowledge_candidates
        audit = {}
        result = asyncio.run(rerank_knowledge_candidates(question, documents,
            counter=self.token_counter, audit=audit))
        logger.info("聊天精排: 候选%d条，返回%d条，状态%s", len(documents), len(result), audit.get("rerank_status"))
        return result
    
    def get_chat_completion(
        self,
        question: str,
        retrieved_content: List[Dict[str, Any]],
        history_messages: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        on_complete: Optional[Callable[[str, str], None]] = None,
    ) -> Generator[str, None, None]:
        """
        获取流式聊天完成结果，并按照指定格式输出。

        Args:
            question: 用户问题
            retrieved_content: 检索到的内容
            history_messages: 从 PostgreSQL 读取并截断后的短期历史
            user_id: 用户ID（用于检索长期记忆）
            on_complete: 模型正常结束后的持久化回调

        Returns:
            流式输出的生成器，每个元素为符合SSE格式的字符串
        """
        # 判断 contents 是否为空
        formatted_references = ""
        if retrieved_content:
            # 格式化参考内容，添加序号
            formatted_refs = []
            for i, ref in enumerate(retrieved_content):
                formatted_refs.append(f"[{i+1}] [{ref['source']}] {ref['content_with_weight']}")
            formatted_references = "\n".join(formatted_refs)

        history_messages = history_messages or []

        # 格式化历史对话
        if history_messages:
            # 注意：history_messages已经按时间顺序排列，最近的对话在后面
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_messages])
            history_context = f"\n\n历史对话（最近的对话内容更重要）：\n{history_text}"
        else:
            history_context = ""

        # 获取长期记忆上下文
        memory_context = ""
        if user_id:
            try:
                memory_service = get_memory_service()
                memory_context = memory_service.build_memory_context(
                    user_id=user_id,
                    current_query=question,
                    max_memories=3
                )
            except Exception as e:
                logger.error(f"获取长期记忆失败: {e}")

        prompt = f"""你是一个智能助手，负责根据用户的问题和提供的参考内容生成回答。请严格按照以下要求生成回答：
1. 回答必须基于提供的参考内容。
2. 在回答中，每一块内容都必须标注引用的来源，格式为：##编号$$。例如：##1$$ 表示引用自第1条参考内容。
3. 如果没有参考内容，请明确说明。
4. 注意保持与历史对话的连贯性。
5. 如果有相关历史记忆，请结合历史信息来回答。

{memory_context}
参考内容：
{formatted_references}
{history_context}

用户问题：{question}
"""

        logger.debug(prompt)

        try:
            # 初始化 OpenAI 客户端
            client = create_openai_client(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )

            # 创建聊天完成请求
            completion = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )

            # 处理流式响应
            model_answer = ""  # 用于存储大模型的回答
            think = ""  # 用于存储思考过程
            completed = False
            for chunk in completion:
                # print("原始 chunk 数据:", chunk)
                if not chunk.choices:
                    continue
                if chunk.choices[0].finish_reason is not None:
                    if on_complete and model_answer:
                        on_complete(model_answer, think)
                    # 模型回答结束后，返回检索内容
                    message = {
                        "documents": retrieved_content,
                    }
                    json_message = json.dumps(message)
                    yield f"event: message\ndata: {json_message}\n\n"
                    
                    # 最后发送 [DONE] 事件
                    yield "event: end\ndata: [DONE]\n\n"
                    completed = True
                    break
                else:
                    # 实时输出消息
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        model_answer += delta.content  # 累加大模型的回答
                        message = {
                            "role": "assistant",
                            "content": delta.content,
                            "thinking": False,
                        }
                        json_message = json.dumps(message)
                        yield f"event: message\ndata: {json_message}\n\n"
                    elif hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        think += delta.reasoning_content
                        message = {
                            "role": "assistant",
                            "content": delta.reasoning_content,
                            "thinking": True,
                        }
                        json_message = json.dumps(message)
                        yield f"event: message\ndata: {json_message}\n\n"

            # 少数兼容接口不会发送 finish_reason；流正常耗尽时仍需落库并结束 SSE。
            if not completed and model_answer:
                if on_complete:
                    on_complete(model_answer, think)
                yield f"event: message\ndata: {json.dumps({'documents': retrieved_content})}\n\n"
                yield "event: end\ndata: [DONE]\n\n"

        except Exception as e:
            # 发生错误时返回错误信息
            error_message = {
                "role": "error",
                "content": str(e)
            }
            json_error_message = json.dumps(error_message)
            yield f"event: error\ndata: {json_error_message}\n\n" 
