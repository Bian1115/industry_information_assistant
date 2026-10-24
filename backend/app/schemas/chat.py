# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class SessionCreate(BaseModel):
    """会话创建请求"""
    title: Optional[str] = Field(None, description="会话标题")
    session_type: str = Field("chat", description="会话类型: chat, deepsearch")


class SessionUpdate(BaseModel):
    """会话更新请求"""
    title: str = Field(..., description="新的会话标题")


class MessageResponse(BaseModel):
    """消息响应"""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="消息ID")
    session_id: str = Field(..., description="会话ID")
    turn_id: Optional[str] = Field(None, description="问答轮次ID")
    execution_mode: str = Field("chat", description="执行模式: chat, research")
    run_id: Optional[str] = Field(None, description="DeepResearch 运行ID")
    research_status: Optional[str] = Field(
        None, description="研究状态: running, completed, failed, cancelled"
    )
    role: str = Field(..., description="角色")
    content: str = Field(..., description="内容")
    thinking: Optional[str] = Field(None, description="思考过程")
    references_data: Optional[Dict[str, Any]] = Field(None, description="引用文档")
    image_results: Optional[List[Dict[str, Any]]] = Field(None, description="图片结果")
    created_at: datetime = Field(..., description="创建时间")

class SessionResponse(BaseModel):
    """会话响应"""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="会话ID")
    title: str = Field(..., description="会话标题")
    session_type: str = Field(..., description="会话类型")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    message_count: int = Field(0, description="消息数量")

class SessionWithMessagesResponse(SessionResponse):
    """带消息的会话响应"""
    messages: List[MessageResponse] = Field(default_factory=list, description="消息列表")


class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: UUID = Field(..., description="PostgreSQL 会话ID")
    turn_id: UUID = Field(..., description="客户端生成的幂等轮次ID")
    question: str = Field(..., description="用户问题")
    search_knowledge: bool = Field(True, description="是否搜索知识库")
    search_web: bool = Field(True, description="是否搜索网络")
    kb_ids: Optional[List[str]] = Field(
        None, description="指定检索的知识库ID列表；不传则检索当前用户的全部知识库"
    )



class RetrievedDocument(BaseModel):
    """检索到的文档"""
    id: Any = Field(..., description="文档ID")
    content: str = Field(..., description="文档内容")
    content_with_weight: str = Field(..., description="带权重的文档内容")
    source: str = Field(..., description="文档来源（knowledge或web）")
    title: Optional[str] = Field(None, description="文档标题")
    weight: float = Field(..., description="相关度分数")
    link: Optional[str] = Field(None, description="如果是网页，则为链接地址")
    
    
class ChatResponse(BaseModel):
    """聊天响应"""
    role: str = Field(..., description="角色（assistant或error）")
    content: str = Field(..., description="内容")
    thinking: Optional[bool] = Field(False, description="是否为思考过程")


# ========== 聊天附件相关 Schema ==========

class AttachmentResponse(BaseModel):
    """附件响应"""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="附件ID")
    session_id: str = Field(..., description="会话ID")
    message_id: Optional[str] = Field(None, description="消息ID")
    filename: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件类型")
    file_size: int = Field(..., description="文件大小（字节）")
    status: str = Field(..., description="处理状态: pending, processing, completed, failed")
    error_message: Optional[str] = Field(None, description="错误信息")
    created_at: datetime = Field(..., description="创建时间")

class AttachmentListResponse(BaseModel):
    """附件列表响应"""
    attachments: List[AttachmentResponse] = Field(default_factory=list, description="附件列表")
    total: int = Field(0, description="总数")


class ChatWithAttachmentsRequest(BaseModel):
    """带附件的聊天请求"""
    session_id: UUID = Field(..., description="PostgreSQL 会话ID")
    turn_id: UUID = Field(..., description="客户端生成的幂等轮次ID")
    question: str = Field(..., description="用户问题")
    attachment_ids: Optional[List[str]] = Field(None, description="附件ID列表")
    search_knowledge: bool = Field(True, description="是否搜索知识库")
    search_web: bool = Field(True, description="是否搜索网络")
    kb_ids: Optional[List[str]] = Field(
        None, description="指定检索的知识库ID列表；不传则检索当前用户的全部知识库"
    ) 
