"""LangChain 会话滚动摘要的离线机制回归；不访问真实模型。"""

import dataclasses
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from prompts.session_summary import DEFAULT_SUMMARY_PROMPT
from service.session_summarization import (
    SessionSummarizer,
    SessionSummaryConfig,
    summary_message,
)


class FakeTokenCounter:
    """每个字符视为一个 token，使阈值和压缩指标可以准确断言。"""

    def count_text(self, text):
        return len(text or "")

    def count_messages(self, messages):
        return sum(self.count_text(str(message.get("content", ""))) for message in messages)

    def truncate_text(self, text, max_tokens):
        return (text or "")[:max_tokens]

    def encode(self, text):
        return list((text or "").encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")


def config(**changes):
    if "context_window" in changes and "max_output_tokens" not in changes:
        changes["max_output_tokens"] = min(2000, max(1, changes["context_window"] // 4))
    return dataclasses.replace(SessionSummaryConfig.from_env({}), **changes)


def conversation(count):
    return [
        {
            "id": f"message-{index}",
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"第{index + 1}条消息",
        }
        for index in range(count)
    ]


class SessionSummaryConfigurationTests(unittest.TestCase):
    def test_defaults_match_user_requested_policy(self):
        value = SessionSummaryConfig.from_env({})

        self.assertEqual(value.keep_messages, 10)
        self.assertEqual(value.model, "qwen3.5-plus")
        self.assertEqual(value.context_window, 1_000_000)
        self.assertEqual(value.trigger_fraction, 0.85)
        self.assertEqual(value.trigger_tokens, 850_000)

    def test_all_public_configuration_values_can_be_overridden(self):
        value = SessionSummaryConfig.from_env({
            "SEMANTIC_CONTEXT_MODEL": "custom-summary-model",
            "SEMANTIC_CONTEXT_KEEP_MESSAGES": "5",
            "SEMANTIC_CONTEXT_CONTEXT_WINDOW": "2000",
            "SEMANTIC_CONTEXT_TRIGGER_FRACTION": "0.75",
            "SEMANTIC_CONTEXT_MAX_OUTPUT_TOKENS": "999",
            "SEMANTIC_CONTEXT_REQUEST_ATTEMPTS": "2",
        })

        self.assertEqual(value.model, "custom-summary-model")
        self.assertEqual(value.keep_messages, 5)
        self.assertEqual(value.context_window, 2000)
        self.assertEqual(value.trigger_tokens, 1500)
        self.assertEqual(value.max_output_tokens, 999)
        self.assertEqual(value.request_attempts, 2)

    def test_consumer_trigger_budget_does_not_reduce_summary_request_window(self):
        value = SessionSummaryConfig.from_env({
            "SEMANTIC_CONTEXT_CONTEXT_WINDOW": "1000000",
            "SEMANTIC_CONTEXT_TRIGGER_WINDOW": "65536",
        })
        self.assertEqual(value.context_window, 1_000_000)
        self.assertEqual(value.trigger_tokens, 55_705)
        self.assertNotEqual(value.fingerprint, SessionSummaryConfig.from_env({}).fingerprint)

    def test_summary_request_may_exceed_consumer_trigger_budget(self):
        seen = []
        service = SessionSummarizer(
            config=config(context_window=20000, trigger_context_window=100),
            token_counter=FakeTokenCounter(),
            completion=lambda prompt: seen.append(prompt) or "摘要",
        )
        messages = [{"role": "user", "content": "x" * 1000}]
        self.assertTrue(service.should_summarize(messages))
        summary, metrics = service.summarize(messages)
        self.assertEqual(summary, "摘要")
        self.assertIn("x" * 1000, seen[0])
        self.assertGreater(metrics["estimated_prompt_tokens"], 100)
        self.assertEqual(metrics["trigger_context_window"], 100)

    def test_invalid_configuration_fails_loudly(self):
        invalid_values = [
            ("SEMANTIC_CONTEXT_KEEP_MESSAGES", "0"),
            ("SEMANTIC_CONTEXT_KEEP_MESSAGES", "-1"),
            ("SEMANTIC_CONTEXT_KEEP_MESSAGES", "2.5"),
            ("SEMANTIC_CONTEXT_CONTEXT_WINDOW", "0"),
            ("SEMANTIC_CONTEXT_TRIGGER_WINDOW", "0"),
            ("SEMANTIC_CONTEXT_TRIGGER_WINDOW", "-1"),
            ("SEMANTIC_CONTEXT_TRIGGER_FRACTION", "0"),
            ("SEMANTIC_CONTEXT_TRIGGER_FRACTION", "1.01"),
            ("SEMANTIC_CONTEXT_TRIGGER_FRACTION", "nan"),
            ("SEMANTIC_CONTEXT_MAX_OUTPUT_TOKENS", "0"),
            ("SEMANTIC_CONTEXT_REQUEST_ATTEMPTS", "0"),
            ("SEMANTIC_CONTEXT_MODEL", "   "),
        ]
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                with self.assertRaises((ValueError, TypeError)):
                    SessionSummaryConfig.from_env({key: value})

    def test_config_fingerprint_changes_when_policy_changes(self):
        initial = config()

        for key, value in {
            "model": "another-model",
            "keep_messages": 6,
            "context_window": 2_000_000,
            "trigger_fraction": 0.8,
            "max_output_tokens": 777,
        }.items():
            with self.subTest(field=key):
                changed = dataclasses.replace(initial, **{key: value})
                self.assertNotEqual(initial.fingerprint, changed.fingerprint)


class SessionSummaryRetentionTests(unittest.TestCase):
    def setUp(self):
        self.counter = FakeTokenCounter()

    def summarizer(self, **changes):
        return SessionSummarizer(config=config(**changes), token_counter=self.counter)

    def test_trigger_uses_greater_than_or_equal_boundary(self):
        service = self.summarizer(context_window=100, trigger_fraction=0.85)

        self.assertFalse(service.should_summarize([{"role": "user", "content": "x" * 84}]))
        self.assertTrue(service.should_summarize([{"role": "user", "content": "x" * 85}]))
        self.assertTrue(service.should_summarize([{"role": "user", "content": "x" * 86}]))

    def test_pending_user_query_is_counted_before_model_call(self):
        service = self.summarizer(context_window=100, trigger_fraction=0.85)
        history = [{"role": "assistant", "content": "x" * 80}]

        self.assertFalse(service.should_summarize(history, current_query="yyyy"))
        self.assertTrue(service.should_summarize(history, current_query="yyyyy"))

    def test_threshold_reached_without_old_prefix_does_not_drop_recent_messages(self):
        service = self.summarizer(context_window=100, trigger_fraction=0.85)
        messages = [{"role": "user", "content": "x" * 100}]

        self.assertTrue(service.should_summarize(messages))
        prefix, tail = service.partition(messages)

        self.assertEqual(prefix, [])
        self.assertEqual(tail, messages)

    def test_keep_ten_means_ten_messages_not_ten_user_answer_pairs(self):
        messages = conversation(24)

        prefix, tail = self.summarizer().partition(messages)

        self.assertEqual(prefix, messages[:14])
        self.assertEqual(tail, messages[14:])
        self.assertEqual(len(tail), 10)
        self.assertEqual(sum(message["role"] == "user" for message in tail), 5)

    def test_configured_retention_changes_partition_without_altering_messages(self):
        messages = conversation(8)

        prefix, tail = self.summarizer(keep_messages=3).partition(messages)

        self.assertEqual(prefix, messages[:5])
        self.assertEqual(tail, messages[5:])
        self.assertEqual(messages, conversation(8))

    def test_cutoff_moves_back_to_keep_tool_call_and_results_together(self):
        messages = [
            {"role": "user", "content": "查市场规模"},
            {
                "role": "assistant", "content": "开始检索",
                "tool_calls": [
                    {"id": "call-a", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call-b", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "证据A", "tool_call_id": "call-a"},
            {"role": "tool", "content": "证据B", "tool_call_id": "call-b"},
            {"role": "assistant", "content": "已有结论"},
            {"role": "user", "content": "补充更多证据"},
        ]

        prefix, tail = self.summarizer(keep_messages=4).partition(messages)

        self.assertEqual(prefix, messages[:1])
        self.assertEqual(tail, messages[1:])
        self.assertEqual(len(tail), 5)

    def test_unmatched_tool_results_are_not_kept_as_orphans(self):
        messages = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧答复"},
            {"role": "tool", "content": "丢失调用的结果", "tool_call_id": "unknown"},
            {"role": "assistant", "content": "最近答复"},
            {"role": "user", "content": "最近问题"},
        ]

        prefix, tail = self.summarizer(keep_messages=3).partition(messages)

        self.assertEqual(prefix, messages[:3])
        self.assertEqual(tail, messages[3:])

    def test_all_consecutive_tool_ids_are_considered_for_safe_boundary(self):
        messages = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "检索", "tool_calls": [{"id": "valid", "name": "search", "args": {}}]},
            {"role": "tool", "content": "孤儿结果", "tool_call_id": "missing"},
            {"role": "tool", "content": "有效结果", "tool_call_id": "valid"},
            {"role": "assistant", "content": "最近答复"},
        ]

        prefix, tail = self.summarizer(keep_messages=3).partition(messages)

        self.assertEqual(prefix, messages[:1])
        self.assertEqual(tail, messages[1:])


class SessionSummaryGenerationTests(unittest.TestCase):
    def test_real_client_protocol_uses_configured_model_and_original_prompt(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="有效摘要"), finish_reason="stop")],
            usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 120, "completion_tokens": 12}),
        )
        client = Mock()
        client.chat.completions.create.return_value = response
        service = SessionSummarizer(config=config(), token_counter=FakeTokenCounter())
        with patch("service.session_summarization.create_openai_client", return_value=client) as factory:
            _, metrics = service.summarize([{"role": "user", "content": "研究中国储能"}])
        args = client.chat.completions.create.call_args.kwargs
        self.assertEqual(args["model"], "qwen3.5-plus")
        self.assertEqual([item["role"] for item in args["messages"]], ["user"])
        self.assertIn("## SESSION INTENT", args["messages"][0]["content"])
        self.assertEqual(args["max_tokens"], 2000)
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        self.assertEqual(metrics["prompt_tokens"], 120)

    def test_truncated_model_output_is_rejected_and_retried(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="不完整摘要"), finish_reason="length")],
            usage=None,
        )
        client = Mock()
        client.chat.completions.create.return_value = response
        service = SessionSummarizer(config=config(), token_counter=FakeTokenCounter())
        with patch("service.session_summarization.create_openai_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "截断"):
                service.summarize([{"role": "user", "content": "研究中国储能"}])
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def setUp(self):
        self.counter = FakeTokenCounter()

    def test_existing_summary_is_recursively_included_with_new_old_messages(self):
        prompts = []

        def complete(prompt):
            prompts.append(prompt)
            return "## SESSION INTENT\n继续研究储能\n## SUMMARY\n已更新范围"

        service = SessionSummarizer(
            config=config(keep_messages=2),
            token_counter=self.counter,
            completion=complete,
        )
        previous = summary_message("旧摘要：研究中国储能，排除预测值。")
        messages = [previous, *conversation(4)]
        prefix, tail = service.partition(messages)

        summary, metrics = service.summarize(prefix)

        self.assertEqual(tail, messages[-2:])
        self.assertEqual(len(prompts), 1)
        self.assertIn(previous["content"], prompts[0])
        self.assertIn(messages[1]["content"], prompts[0])
        self.assertIn(messages[2]["content"], prompts[0])
        self.assertNotIn(messages[-1]["content"], prompts[0])
        self.assertIn("## SESSION INTENT", summary)
        self.assertEqual(metrics["summary_input_tokens"], metrics["source_tokens"])

    def test_summary_wrapper_matches_langchain_human_message_content(self):
        message = summary_message("重要上下文")

        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"], "Here is a summary of the conversation to date:\n\n重要上下文")

    def test_summary_prompt_matches_optional_local_langchain_exactly(self):
        try:
            from langchain.agents.middleware.summarization import DEFAULT_SUMMARY_PROMPT as upstream
        except ImportError:
            self.skipTest("主测试环境未安装 LangChain；提示词来源由源码快照核对")

        self.assertEqual(DEFAULT_SUMMARY_PROMPT, upstream)

    def test_source_roles_and_xml_special_characters_are_unambiguous(self):
        captured = []
        service = SessionSummarizer(
            config=config(),
            token_counter=self.counter,
            completion=lambda prompt: captured.append(prompt) or "有效摘要",
        )

        service.summarize([{"role": "user", "content": "比较 A < B & B > C"}])

        self.assertIn('<message type="human">', captured[0])
        self.assertIn("A &lt; B &amp; B &gt; C", captured[0])

    def test_openai_tool_calls_keep_names_arguments_and_results_in_summary_input(self):
        prompts = []
        service = SessionSummarizer(
            config=config(),
            token_counter=self.counter,
            completion=lambda prompt: prompts.append(prompt) or "有效摘要",
        )
        source = [
            {"role": "user", "content": "查储能"},
            {
                "role": "assistant", "content": "开始搜索",
                "tool_calls": [{
                    "id": "call-market", "type": "function",
                    "function": {"name": "search", "arguments": '{"query":"储能规模"}'},
                }],
            },
            {"role": "tool", "content": "行业数据", "tool_call_id": "call-market"},
        ]

        service.summarize(source)

        self.assertIn('id="call-market"', prompts[0])
        self.assertIn('name="search"', prompts[0])
        self.assertIn("储能规模", prompts[0])
        self.assertIn("行业数据", prompts[0])

    def test_token_usage_and_compression_ratio_are_measured(self):
        source = [{"id": "old-1", "role": "user", "content": "x" * 100}]
        service = SessionSummarizer(
            config=config(),
            token_counter=self.counter,
            completion=lambda _: {"text": "s" * 10, "usage": {"prompt_tokens": 111, "completion_tokens": 10}},
        )

        summary, metrics = service.summarize(source)

        self.assertEqual(summary, "s" * 10)
        self.assertEqual(metrics["source_tokens"], 100)
        wrapped_summary_tokens = self.counter.count_messages([summary_message(summary)])
        self.assertEqual(metrics["summary_tokens"], wrapped_summary_tokens)
        self.assertEqual(metrics["prompt_tokens"], 111)
        self.assertEqual(metrics["completion_tokens"], 10)
        self.assertAlmostEqual(metrics["compression_ratio"], wrapped_summary_tokens / 100)
        self.assertEqual(metrics["model"], "qwen3.5-plus")
        self.assertTrue(metrics["prompt_sha256"])
        self.assertGreaterEqual(metrics["latency_ms"], 0)

    def test_summary_input_preserves_full_old_messages_by_default(self):
        prompts = []
        text = "FIRST-" + "x" * 5000 + "-LAST"
        service = SessionSummarizer(
            config=config(),
            token_counter=self.counter,
            completion=lambda prompt: prompts.append(prompt) or "有效摘要",
        )

        _, metrics = service.summarize([{"role": "user", "content": text}])

        self.assertIn(text, prompts[0])
        self.assertEqual(metrics["summary_input_tokens"], len(text))
        self.assertEqual(metrics["source_tokens"], len(text))

    def test_all_messages_above_four_thousand_tokens_reach_summary_model(self):
        prompts = []
        service = SessionSummarizer(
            config=config(),
            token_counter=self.counter,
            completion=lambda prompt: prompts.append(prompt) or "有效摘要",
        )
        source = [
            {"role": "user", "content": "EARLY-" + "a" * 2994},
            {"role": "assistant", "content": "b" * 1000},
            {"role": "user", "content": "RECENT-" + "c" * 993},
        ]

        _, metrics = service.summarize(source)

        self.assertGreater(metrics["summary_input_tokens"], 4000)
        self.assertEqual(metrics["summary_input_tokens"], metrics["source_tokens"])
        for message in source:
            self.assertIn(message["content"], prompts[0])

    def test_oversized_summary_request_fails_before_model_call_without_mutating_history(self):
        complete = Mock(return_value="不应生成摘要")
        service = SessionSummarizer(
            config=config(context_window=20000, max_output_tokens=128),
            token_counter=self.counter,
            completion=complete,
        )
        source = [{"role": "user", "content": "EARLY-" + "x" * 20000 + "-LAST"}]
        before = [dict(message) for message in source]

        with self.assertRaisesRegex(ValueError, "超过配置窗口"):
            service.summarize(source)

        complete.assert_not_called()
        self.assertEqual(source, before)

    def test_model_failure_propagates_without_mutating_source_history(self):
        messages = conversation(3)
        before = [dict(message) for message in messages]
        failure = RuntimeError("摘要服务不可用")

        def fail(_):
            raise failure

        service = SessionSummarizer(config=config(), token_counter=self.counter, completion=fail)

        with self.assertRaises(RuntimeError) as raised:
            service.summarize(messages)

        self.assertIn(str(failure), str(raised.exception))
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(messages, before)

    def test_empty_model_response_is_rejected_instead_of_becoming_summary(self):
        messages = conversation(3)
        for empty in ("", " \n\t", {"text": "", "usage": {}}):
            with self.subTest(output=repr(empty)):
                service = SessionSummarizer(
                    config=config(), token_counter=self.counter,
                    completion=lambda _, value=empty: value,
                )

                with self.assertRaises((RuntimeError, ValueError)):
                    service.summarize(messages)

                self.assertEqual(messages, conversation(3))


if __name__ == "__main__":
    unittest.main()
