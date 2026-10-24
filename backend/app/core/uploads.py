"""上传文件使用服务端生成的私有临时路径，客户端文件名仅用于展示。"""

import os
import shutil
import tempfile

from fastapi import HTTPException

from core.ownership import resource_uuid


def upload_filename(filename):
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in {".", ".."} or "\x00" in name or len(name) > 255:
        raise HTTPException(400, "无效的文件名")
    return name


def save_user_upload(file, directory, user_id, resource_id, filename):
    prefix = f"{resource_uuid(user_id).hex}_{resource_uuid(resource_id).hex}_"
    descriptor, path = tempfile.mkstemp(
        dir=directory, prefix=prefix, suffix=os.path.splitext(filename)[1].lower(),
    )
    try:
        with os.fdopen(descriptor, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return path, os.path.getsize(path)
    except Exception:
        os.unlink(path)
        raise
