"""Qwen tokenizer 统一计数服务的离线测试。"""

import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from service.qwen_token_counter import QwenTokenCounter


class FakeQwenTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return self.encode(rendered)


class QwenTokenCounterTests(unittest.TestCase):
    def setUp(self):
        self.counter = QwenTokenCounter("test-qwen")
        self.counter._tokenizer = FakeQwenTokenizer()

    def test_counts_text_with_qwen_tokenizer(self):
        self.assertEqual(self.counter.count_text("中国AI"), 4)

    def test_counts_complete_chat_template(self):
        plain = self.counter.count_text("你好")
        messages = self.counter.count_messages([
            {"role": "user", "content": "你好"},
        ])
        self.assertGreater(messages, plain)

    def test_truncates_on_token_boundary(self):
        self.assertEqual(self.counter.truncate_text("中国市场", 2), "中国")


if __name__ == "__main__":
    unittest.main()
