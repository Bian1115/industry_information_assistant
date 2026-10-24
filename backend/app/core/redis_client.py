# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""Redis 客户端"""
import os
import json
from typing import Optional, Any
import redis
import logging

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "1.0"))

# 创建 Redis 连接池
redis_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    db=REDIS_DB,
    decode_responses=True,
    max_connections=20,
    socket_connect_timeout=REDIS_SOCKET_TIMEOUT,
    socket_timeout=REDIS_SOCKET_TIMEOUT,
    health_check_interval=30,
)


def get_redis_client() -> redis.Redis:
    """获取 Redis 客户端"""
    return redis.Redis(connection_pool=redis_pool)


class RedisCache:
    """Redis 缓存工具类"""

    def __init__(self):
        self.client = get_redis_client()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """设置缓存，默认过期时间 1 小时"""
        try:
            self.client.setex(key, expire, json.dumps(value, ensure_ascii=False))
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """检查 key 是否存在"""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

# 全局缓存实例
cache = RedisCache()
