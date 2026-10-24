# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

from typing import List, Dict, Any
from .chat_service import ChatService
from .document_service import DocumentService
from .web_search_service import WebSearchService
from .policy_search_service import PolicySearchService
import json
import logging

logger = logging.getLogger(__name__)


class ChatServiceV2(ChatService):
    """Chat service that extends ChatService to use PolicySearchService for knowledge base retrieval"""
    
    def __init__(self, document_service: DocumentService, web_search_service: WebSearchService,
                 policy_search_service: PolicySearchService):
        """
        Initialize the ChatServiceV2.
        
        Args:
            document_service: Document service for knowledge base retrieval (fallback)
            web_search_service: Web search service for internet search
            policy_search_service: Policy search service for policy document retrieval
        """
        super().__init__(document_service, web_search_service)
        self.policy_search_service = policy_search_service
    
    def retrieve_from_policy_documents(self, question: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents from policy document index using hybrid search.
        
        Args:
            question: User question
            top_n: Number of results to retrieve
            
        Returns:
            List of retrieved documents
        """
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"政策文档检索请求: '{question}'")
            logger.info(f"{'='*50}")
            
            # 使用政策文档搜索服务进行混合检索
            response = self.policy_search_service.search(
                query=question,
                method="hybrid",
                top_n=top_n
            )
            
            logger.info(f"\n政策检索API响应:")
            logger.info(f"成功状态: {response.get('success', False)}")
            logger.info(f"检索方法: {response.get('method', 'unknown')}")
            logger.info(f"检索耗时: {response.get('took_ms', 0)}ms")
            logger.info(f"检索结果数: {response.get('total', 0)}")
            
            if not response.get("success", False):
                logger.error(f"检索失败: {response.get('message', '未知错误')}")
                return []
                
            # 从响应中提取文档
            documents = []
            # 用于去重的URL集合
            seen_urls = set()
            
            if "results" in response:
                logger.info(f"\n{'='*50}")
                logger.info(f"检索到的政策文档详情:")
                logger.info(f"{'='*50}")
                
                for i, result in enumerate(response["results"]):
                    # 检查URL是否已存在（去重逻辑）
                    detail_url = result.get("detail_url", "")
                    if detail_url and detail_url in seen_urls:
                        logger.warning(f"跳过重复的文档: {result.get('title', '无标题')} - {detail_url}")
                        continue
                    
                    # 记录已处理的URL
                    if detail_url:
                        seen_urls.add(detail_url)
                    
                    # 获取原始内容，而不是高亮的内容
                    original_content = result.get("content", "")
                    
                    # 处理高亮信息，作为展示内容
                    highlighted_content = original_content
                    if "highlights" in result and "content" in result["highlights"]:
                        highlighted_content = result["highlights"]["content"]
                    
                    logger.info(f"\n文档 #{i+1}:")
                    logger.info(f"标题: {result.get('title', '无标题')}")
                    logger.info(f"来源: {result.get('website', '未知来源')}")
                    logger.info(f"日期: {result.get('date', '未知日期')}")
                    logger.info(f"分数: {result.get('score', 0)}")
                    logger.info(f"链接: {detail_url}")
                    logger.info(f"内容预览: {highlighted_content[:200]}...")
                    
                    doc = {
                        "id": len(documents) + 1,  # 使用有序ID而不是原始索引i
                        "content": original_content,  # 使用原始内容
                        "content_with_weight": f"{highlighted_content} (相关度: {result.get('score', 0):.2f})",
                        "source": "knowledge",  # 将来源设置为knowledge
                        "title": result.get("title", "政策文档"),
                        "link": detail_url,
                        "weight": result.get("score", 1.0)
                    }
                    documents.append(doc)
            
            logger.info(f"\n{'='*50}")
            logger.info(f"共处理 {len(documents)} 个政策文档检索结果（去重后）")
            logger.info(f"{'='*50}\n")
            
            return documents
        except Exception as e:
            logger.error(f"Error retrieving from policy documents: {str(e)}")
            return [] 
