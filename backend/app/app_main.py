# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from router import document_router, search_router, chat_router, research_router
from router.auth_router import router as auth_router
from router.auth_router import get_current_user_required
from router.session_router import router as session_router
from router.knowledge_router import router as knowledge_router
from router.attachment_router import router as attachment_router
from router.memory_router import router as memory_router
# 注册 ORM 元数据；历史业务模型仍保留，取消入口不等于删除历史数据表。
import models  # noqa: F401

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("应用启动中...")
    from core.security import validate_security_configuration
    validate_security_configuration()
    from config.llm_config import get_config as get_llm_config
    logger.info("DeepResearch 搜索后端：%s", get_llm_config().search_provider)

    # 启用时在启动日志中确认 SDK 初始化；凭据不会写入日志。
    from eval.runtime.langfuse_tracing import get_client as get_langfuse_client
    get_langfuse_client()

    # Qwen tokenizer 是所有 token 阈值与上下文预算的唯一计数依据。
    # 启动时预热，缺少依赖或模型文件时立即失败，禁止静默退回字符估算。
    from service.qwen_token_counter import get_qwen_token_counter

    get_qwen_token_counter().count_text("tokenizer startup check")
    logger.info("Qwen tokenizer 加载成功")

    # Redis 短期记忆是可重建缓存；不可用时记录告警并继续使用 PostgreSQL。
    from service.session_memory_service import get_session_memory_service

    if get_session_memory_service().ping():
        logger.info("Redis 会话工作记忆连接成功")
    else:
        logger.warning("Redis 会话工作记忆不可用，将降级为 PostgreSQL 上下文重建")

    # 仅启动记忆与语义上下文维护任务，不再启动行业资讯自动采集。
    try:
        from service.scheduler_service import init_scheduler
        await init_scheduler()
        logger.info("记忆维护调度器启动成功")
    except Exception as e:
        logger.error(f"定时任务调度器启动失败: {e}")

    # LangGraph 检查点是研究执行位置的权威来源，连接不可用时不应放行研究请求。
    # checkpoint-postgres 走 psycopg3 异步连接，不复用现有的 psycopg2 会话。
    # 建表是独立部署步骤（python -m service.deep_research_v2.checkpointer）。
    from service.deep_research_v2.checkpointer import checkpointer_lifespan

    async with checkpointer_lifespan(app):
        yield

    # 关闭时执行
    logger.info("应用关闭中...")
    try:
        from service.scheduler_service import get_scheduler_service
        scheduler = get_scheduler_service()
        scheduler.stop()
    except Exception as e:
        logger.error(f"定时任务调度器关闭失败: {e}")


app = FastAPI(
    title="深度研究助手 API",
    description="指定网站检索、证据整理与研究报告生成（单一 LangGraph 主链路）",
    version="3.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 注册路由
app.include_router(auth_router)
# 所有业务路由统一要求登录，防止新增接口遗漏身份认证；归属由资源查询继续校验。
private_dependencies = [Depends(get_current_user_required)]
app.include_router(session_router, dependencies=private_dependencies)
app.include_router(knowledge_router, dependencies=private_dependencies)
app.include_router(attachment_router, dependencies=private_dependencies)
app.include_router(memory_router, dependencies=private_dependencies)
app.include_router(document_router, dependencies=private_dependencies)
app.include_router(search_router, dependencies=private_dependencies)
app.include_router(chat_router, dependencies=private_dependencies)
app.include_router(research_router, dependencies=private_dependencies)

@app.get("/hello")
async def hello_world():
    """
    Simple hello world endpoint for network verification
    """
    return {
        "status": "success",
        "message": "Hello World! The API is working correctly."
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
