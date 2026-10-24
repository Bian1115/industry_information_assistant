# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""知识库管理路由"""
import hashlib
import logging
import os
from typing import List
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from core.database import get_db
from core.ownership import owned_documents, require_owned_knowledge_base
from core.uploads import save_user_upload, upload_filename
from service import es_service
from service.milvus_service import get_milvus_service
from service.retrieval_service import kb_collection_name
from models.knowledge import KnowledgeBase, Document
from models.user import User
from router.auth_router import get_current_user_required
from schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    KnowledgeBaseWithDocuments,
    DocumentResponse,
    DocumentUploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-bases", tags=["知识库管理"])

# 文件上传目录
UPLOAD_DIR = "/tmp/knowledge_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 支持的文件类型
#
# 必须与阿里云 DocMind 实际支持的解析格式保持一致——文档解析完全依赖它。
# 名单若比 DocMind 宽，用户会收到"上传成功，正在后台处理"，然后文档静默变成
# failed 状态，界面上什么都搜不到；真正的原因只在后端日志里。这种"先成功后失败"
# 的体验最难排查，所以宁可在上传时就明确拒绝。
#
# 已移除：.txt .md .html .py .js .ts .json .yaml .yml .xml .csv .gif .webp
# 这些 DocMind 不解析。其中纯文本类（.txt/.md/.csv）其实不需要 DocMind，
# 本地读取切片即可，若要支持应单开一条解析分支，而不是放回本名单。
ALLOWED_EXTENSIONS = {
    # 文档类型
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    # 图片类型（DocMind 支持 OCR 解析）
    '.jpg', '.jpeg', '.png', '.bmp',
}


def get_file_extension(filename: str) -> str:
    """获取文件扩展名"""
    return os.path.splitext(filename)[1].lower()


def kb_to_response(kb: KnowledgeBase) -> KnowledgeBaseResponse:
    """将知识库模型转换为响应"""
    return KnowledgeBaseResponse(
        id=str(kb.id),
        name=kb.name,
        description=kb.description,
        document_count=kb.document_count or 0,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


def doc_to_response(doc: Document) -> DocumentResponse:
    """将文档模型转换为响应"""
    return DocumentResponse(
        id=str(doc.id),
        knowledge_base_id=str(doc.knowledge_base_id),
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        status=doc.status,
        chunk_count=doc.chunk_count or 0,
        error_message=doc.error_message,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


async def process_document(document_id: str, file_path: str, kb_id: str, db_session_factory, *, user_id: str):
    """后台处理文档：DocMind 解析 → 切片 → 向量化 → 写入 Milvus"""
    from service.docmind_service import process_document_with_docmind

    # 创建新的数据库会话
    db = db_session_factory()
    authorized_file = False
    try:
        # 获取文档记录
        try:
            kb = require_owned_knowledge_base(db, kb_id, user_id)
        except HTTPException:
            return
        doc = owned_documents(db, kb.id, user_id).filter(Document.id == UUID(document_id)).first()
        if not doc or doc.file_path != file_path:
            return
        authorized_file = True

        # 更新状态为处理中
        doc.status = "processing"
        db.commit()

        try:
            # Milvus 集合名由知识库 ID 生成（集合名不能含中文，且需在改名后保持稳定）
            index_name = kb_collection_name(str(doc.knowledge_base_id), allow_benchmark=False)

            # 使用 DocMind 处理文档
            result = process_document_with_docmind(
                file_path=file_path,
                file_name=doc.filename,
                index_name=index_name,
            )

            if result["success"]:
                doc.status = "completed"
                doc.chunk_count = result["document_count"]
                doc.error_message = None
            else:
                doc.status = "failed"
                doc.error_message = result["message"]

        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)

        db.commit()

    finally:
        db.close()
        # 清理临时文件
        if authorized_file and os.path.exists(file_path):
            os.remove(file_path)


@router.get("", response_model=List[KnowledgeBaseResponse])
async def get_knowledge_bases(
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """获取用户的知识库列表"""
    kbs = db.query(KnowledgeBase).filter(
        KnowledgeBase.user_id == current_user.id
    ).order_by(KnowledgeBase.updated_at.desc()).all()

    return [kb_to_response(kb) for kb in kbs]


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    kb_data: KnowledgeBaseCreate,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """创建知识库"""
    # 检查是否已存在同名知识库
    existing = db.query(KnowledgeBase).filter(
        KnowledgeBase.user_id == current_user.id,
        KnowledgeBase.name == kb_data.name
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="已存在同名知识库"
        )

    kb = KnowledgeBase(
        user_id=current_user.id,
        name=kb_data.name,
        description=kb_data.description,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)

    return kb_to_response(kb)


@router.get("/{kb_id}", response_model=KnowledgeBaseWithDocuments)
async def get_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """获取知识库详情（包含文档列表）"""
    try:
        kb_uuid = UUID(kb_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的知识库ID格式"
        )

    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_uuid,
        KnowledgeBase.user_id == current_user.id
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )

    documents = owned_documents(db, kb.id, current_user.id).order_by(Document.created_at.desc()).all()

    return KnowledgeBaseWithDocuments(
        id=str(kb.id),
        name=kb.name,
        description=kb.description,
        document_count=kb.document_count or 0,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
        documents=[doc_to_response(doc) for doc in documents],
    )


@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: str,
    kb_data: KnowledgeBaseUpdate,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """更新知识库"""
    try:
        kb_uuid = UUID(kb_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的知识库ID格式"
        )

    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_uuid,
        KnowledgeBase.user_id == current_user.id
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )

    if kb_data.name is not None:
        # 检查是否与其他知识库重名
        existing = db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == current_user.id,
            KnowledgeBase.name == kb_data.name,
            KnowledgeBase.id != kb_uuid
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="已存在同名知识库"
            )
        kb.name = kb_data.name

    if kb_data.description is not None:
        kb.description = kb_data.description

    db.commit()
    db.refresh(kb)

    return kb_to_response(kb)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """删除知识库"""
    try:
        kb_uuid = UUID(kb_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的知识库ID格式"
        )

    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_uuid,
        KnowledgeBase.user_id == current_user.id
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )

    # 先清理向量库与全文索引，再删数据库记录。
    # 顺序很重要：数据库记录是找到 kb_id 的唯一依据，先删就再也定位不到这些残留数据了。
    # 两个存储各自 try：任一清理失败都不应阻止知识库被删除，否则用户会陷入
    # “删不掉但又用不了”的状态；失败信息记日志，可事后手工清理。
    collection_name = kb_collection_name(str(kb.id), allow_benchmark=False)

    try:
        get_milvus_service().delete_collection(collection_name)
        logger.info(f"已删除 Milvus 集合: {collection_name}")
    except Exception as e:
        logger.error(f"删除 Milvus 集合 {collection_name} 失败: {e}")

    try:
        es_service.delete_index(collection_name)
    except Exception as e:
        logger.error(f"删除 ES 索引 {collection_name} 失败: {e}")

    db.delete(kb)
    db.commit()
    return None


@router.post("/{kb_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(
    kb_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """上传文档到知识库"""
    try:
        kb_uuid = UUID(kb_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的知识库ID格式"
        )

    # 验证知识库存在
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_uuid,
        KnowledgeBase.user_id == current_user.id
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )

    # 验证文件类型
    filename = upload_filename(file.filename)
    ext = get_file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型: {ext}，支持的类型: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 保存文件到临时目录
    document_id = uuid4()
    try:
        file_path, file_size = save_user_upload(file, UPLOAD_DIR, current_user.id, document_id, filename)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件保存失败: {str(e)}"
        )

    # 创建文档记录
    doc = Document(
        id=document_id,
        knowledge_base_id=kb_uuid,
        user_id=current_user.id,
        filename=filename,
        file_type=ext[1:] if ext else None,  # 去掉点
        file_size=file_size,
        file_path=file_path,
        status="pending",
    )
    db.add(doc)

    # 更新知识库文档计数
    kb.document_count = (kb.document_count or 0) + 1

    try:
        db.commit()
        db.refresh(doc)
    except Exception:
        db.rollback()
        os.unlink(file_path)
        raise

    # 获取数据库会话工厂
    from core.database import SessionLocal

    # 在后台处理文档
    background_tasks.add_task(
        process_document,
        str(doc.id),
        file_path,
        str(kb.id),
        SessionLocal,
        user_id=str(current_user.id),
    )

    return DocumentUploadResponse(
        id=str(doc.id),
        filename=doc.filename,
        process_status="pending",
        message="文档已上传，正在后台处理中"
    )


@router.get("/{kb_id}/documents", response_model=List[DocumentResponse])
async def get_documents(
    kb_id: str,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """获取知识库的文档列表"""
    try:
        kb_uuid = UUID(kb_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的知识库ID格式"
        )

    # 验证知识库存在
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_uuid,
        KnowledgeBase.user_id == current_user.id
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )

    documents = owned_documents(db, kb_uuid, current_user.id).order_by(Document.created_at.desc()).all()

    return [doc_to_response(doc) for doc in documents]


@router.get("/{kb_id}/documents/{doc_id}/chunks")
async def get_document_chunks(
    kb_id: str,
    doc_id: str,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """获取文档的所有切片"""
    from service.milvus_service import get_milvus_service

    try:
        kb_uuid = UUID(kb_id)
        doc_uuid = UUID(doc_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的ID格式"
        )

    # 验证知识库存在
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_uuid,
        KnowledgeBase.user_id == current_user.id
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )

    # 获取文档
    doc = owned_documents(db, kb_uuid, current_user.id).filter(
        Document.id == doc_uuid,
    ).first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    if doc.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文档尚未处理完成"
        )

    # 从 Milvus 获取切片
    collection_name = kb_collection_name(str(kb.id), allow_benchmark=False)
    logger.info(f"[get_document_chunks] 查询切片: collection={collection_name}, filename={doc.filename}")

    try:
        milvus = get_milvus_service()
        chunks = milvus.get_chunks_by_filename(collection_name, doc.filename)
        logger.info(f"[get_document_chunks] 找到 {len(chunks)} 个切片")
    except Exception as e:
        logger.error(f"[get_document_chunks] Milvus 查询失败: {e}")
        # 返回空结果而不是报错
        chunks = []

    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "index": chunk.get("chunk_index", i),
                "content": chunk.get("content", ""),
            }
            for i, chunk in enumerate(chunks)
        ]
    }


@router.delete("/{kb_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    kb_id: str,
    doc_id: str,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """删除文档"""
    try:
        kb_uuid = UUID(kb_id)
        doc_uuid = UUID(doc_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的ID格式"
        )

    # 验证知识库存在
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_uuid,
        KnowledgeBase.user_id == current_user.id
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在"
        )

    # 获取文档
    doc = owned_documents(db, kb_uuid, current_user.id).filter(
        Document.id == doc_uuid,
    ).first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    # 删除文件（如果存在）
    if doc.file_path and os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    # 清理该文档在向量库与全文索引中的切片。
    # 不做的话，文档虽然从列表消失，其内容仍会被检索到并出现在回答里。
    # doc_id 的取值规则必须与写入侧一致（docmind_service 用的是 md5(文件名)）。
    collection_name = kb_collection_name(str(kb.id), allow_benchmark=False)
    chunk_doc_id = hashlib.md5(doc.filename.encode()).hexdigest()

    try:
        get_milvus_service().delete_by_doc_id(collection_name, chunk_doc_id)
    except Exception as e:
        logger.error(f"删除 Milvus 切片失败 (doc={doc.filename}): {e}")

    try:
        es_service.delete_by_doc_id(collection_name, chunk_doc_id)
    except Exception as e:
        logger.error(f"删除 ES 切片失败 (doc={doc.filename}): {e}")

    # 更新知识库文档计数
    kb.document_count = max((kb.document_count or 0) - 1, 0)

    db.delete(doc)
    db.commit()
    return None
