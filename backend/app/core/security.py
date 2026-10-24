# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""安全相关：密码哈希、JWT Token"""
import os
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID
from jose import JWTError, jwt
from pydantic import BaseModel

# JWT 配置
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))


def validate_security_configuration():
    """没有独立强密钥时拒绝启动和签发，防止使用公开默认密钥伪造用户身份。"""
    if not SECRET_KEY or len(SECRET_KEY) < 32 or SECRET_KEY in {
        "your-super-secret-key-change-in-production",
        "change-this-secret-key-in-production",
    }:
        raise RuntimeError("JWT_SECRET_KEY 必须配置为至少 32 字符的独立随机密钥")
    if ALGORITHM not in {"HS256", "HS384", "HS512"}:
        raise RuntimeError("JWT_ALGORITHM 必须使用 HS256、HS384 或 HS512")


class Token(BaseModel):
    """Token 响应模型"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token 数据模型"""
    user_id: Optional[str] = None
    username: Optional[str] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问 Token"""
    validate_security_configuration()
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """解码 Token"""
    try:
        validate_security_configuration()
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        if user_id is None:
            return None
        return TokenData(user_id=str(UUID(user_id)), username=username)
    except (JWTError, ValueError, TypeError, AttributeError, RuntimeError):
        return None
