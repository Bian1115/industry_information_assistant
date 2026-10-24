# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""统一使用 Qwen tokenizer 计算文本与消息 token。"""

from __future__ import annotations

import os
import threading
from typing import Any, Mapping, Sequence


DEFAULT_QWEN_TOKENIZER_MODEL = "Qwen/Qwen3-8B"


class QwenTokenCounter:
    """只加载 tokenizer，不加载模型权重。"""

    def __init__(
        self,
        model_name: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.model_name = (
            model_name
            or os.getenv("QWEN_TOKENIZER_MODEL", DEFAULT_QWEN_TOKENIZER_MODEL)
        ).strip()
        self.revision = (
            revision
            if revision is not None
            else os.getenv("QWEN_TOKENIZER_REVISION", "").strip() or None
        )
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._lock:
            if self._tokenizer is not None:
                return self._tokenizer
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen token 统计需要安装 transformers 和 tokenizers"
                ) from exc

            kwargs = {"use_fast": True}
            if self.revision:
                kwargs["revision"] = self.revision
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **kwargs,
            )
            return self._tokenizer

    def encode(self, text: str) -> list[int]:
        return list(
            self.tokenizer.encode(text or "", add_special_tokens=False)
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(
            self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def count_text(self, text: str) -> int:
        return len(self.encode(text))

    def count_messages(self, messages: Sequence[Mapping[str, Any]]) -> int:
        normalized = [
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
            for message in messages
        ]
        if not normalized:
            return 0
        token_ids = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=False,
        )
        return len(token_ids)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        token_ids = self.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        return self.decode(token_ids[:max_tokens])


_counter: QwenTokenCounter | None = None
_counter_lock = threading.Lock()


def get_qwen_token_counter() -> QwenTokenCounter:
    global _counter
    if _counter is not None:
        return _counter
    with _counter_lock:
        if _counter is None:
            _counter = QwenTokenCounter()
    return _counter
