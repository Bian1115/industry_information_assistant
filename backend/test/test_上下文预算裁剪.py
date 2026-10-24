"""移除规则摘录后的离线预算裁剪回归。"""

import copy
import sys
import types
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))
service_package = types.ModuleType("service")
service_package.__path__ = [str(APP_DIR / "service")]
sys.modules.setdefault("service", service_package)

from service.context_compaction import compact_messages


class CharacterTokenCounter:
    """字符加角色开销，仅测裁剪机制，不模拟真实 Qwen 分词。"""

    @staticmethod
    def encode(text):
        return list(text or "")

    @staticmethod
    def decode(tokens):
        return "".join(tokens)

    @staticmethod
    def count_messages(messages):
        return sum(8 + len(item["content"]) for item in messages)


class ContextBudgetTests(unittest.TestCase):
    def setUp(self):
        self.counter = CharacterTokenCounter()
        self.messages = [
            {"role": role, "content": f"第{index}轮{role}"}
            for index in range(8)
            for role in ("user", "assistant")
        ]

    def test_old_turns_are_dropped_without_synthetic_messages(self):
        original = copy.deepcopy(self.messages)
        result = compact_messages(
            self.messages, token_limit=500, max_messages=6, token_counter=self.counter,
        )

        self.assertEqual(result.messages, original[-6:])
        self.assertEqual(result.summarized_messages, 0)
        self.assertEqual(self.messages, original)

    def test_recent_turns_use_budget_previously_reserved_for_excerpts(self):
        # 三轮恰好装满预算，第四轮放不下；不再只分配 60% 给近期原文。
        budget = self.counter.count_messages(self.messages[-6:])
        result = compact_messages(
            self.messages, token_limit=budget, max_messages=20, token_counter=self.counter,
        )

        self.assertEqual(result.messages, self.messages[-6:])
        self.assertEqual(result.compacted_tokens, budget)

    def test_existing_semantic_summary_survives_repeated_budget_fitting(self):
        summary = {
            "role": "system",
            "content": "[会话语义摘要：仅用于理解任务和指代，不是本轮研究事实证据]\n只看2025年",
        }
        first = compact_messages(
            [summary, *self.messages], token_limit=500, max_messages=7,
            token_counter=self.counter,
        )
        second = compact_messages(
            first.messages, token_limit=300, max_messages=4, token_counter=self.counter,
        )

        self.assertEqual(first.messages, [summary, *self.messages[-6:]])
        self.assertEqual(second.messages, [summary, *self.messages[-2:]])
        self.assertEqual(second.summarized_messages, 0)
        self.assertLessEqual(second.compacted_tokens, 300)

    def test_latest_oversized_turn_is_clipped_without_creating_summary(self):
        messages = [
            {"role": "user", "content": "用户问题" + "问" * 200},
            {"role": "assistant", "content": "助手回答" + "答" * 200},
        ]
        result = compact_messages(
            messages, token_limit=100, max_messages=20, token_counter=self.counter,
        )

        self.assertEqual([item["role"] for item in result.messages], ["user", "assistant"])
        self.assertTrue(result.messages[0]["content"].startswith("用户问题"))
        self.assertTrue(result.messages[1]["content"].startswith("助手回答"))
        self.assertLessEqual(result.compacted_tokens, 100)
        self.assertEqual(result.summarized_messages, 0)

    def test_tool_result_truncation_and_actual_token_count_are_preserved(self):
        messages = [
            {"role": "user", "content": "读取资料"},
            {"role": "tool", "content": "工具正文" * 500},
        ]
        result = compact_messages(
            messages, token_limit=500, max_messages=20, tool_result_chars=80,
            token_counter=self.counter,
        )

        self.assertEqual(result.pruned_tool_results, 1)
        self.assertEqual(result.compacted_tokens, self.counter.count_messages(result.messages))
        self.assertLess(result.compacted_tokens, result.original_tokens)
        self.assertIn("工具结果已裁剪", result.messages[-1]["content"])
        self.assertEqual(result.summarized_messages, 0)

    def test_zero_budget_does_not_claim_discarded_messages_were_summarized(self):
        for token_limit, max_messages in [(0, 20), (100, 0)]:
            with self.subTest(token_limit=token_limit, max_messages=max_messages):
                result = compact_messages(
                    self.messages, token_limit=token_limit, max_messages=max_messages,
                    token_counter=self.counter,
                )
                self.assertEqual(result.messages, [])
                self.assertEqual(result.summarized_messages, 0)


if __name__ == "__main__":
    unittest.main()
