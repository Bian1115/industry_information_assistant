"""集中式评估目录中的 LongMemEval-S 固定回归套件纯离线测试。"""

import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, call, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_longmemeval_regression.py"
SPEC = importlib.util.spec_from_file_location("run_longmemeval_regression", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_row(index, question_type, abstention=False):
    question_id = f"q{index:03d}" + ("_abs" if abstention else "")
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": f"question {index}",
        "answer": f"answer {index}",
        "question_date": "2024/01/02 (Tue) 03:04",
        "haystack_session_ids": [f"s{index}"],
        "haystack_dates": ["2024/01/01 (Mon) 01:02"],
        "haystack_sessions": [[{"role": "user", "content": "hello"}]],
        "answer_session_ids": [] if abstention else [f"s{index}"],
    }


class FixedSelectionTests(unittest.TestCase):
    def test_selection_is_stratified_and_independent_of_input_order(self):
        rows = []
        groups = {
            "single-session-user": 20,
            "single-session-assistant": 20,
            "single-session-preference": 20,
            "multi-session": 20,
            "knowledge-update": 20,
            "temporal-reasoning": 20,
        }
        index = 0
        for question_type, count in groups.items():
            for _ in range(count):
                rows.append(make_row(index, question_type))
                index += 1
        for _ in range(10):
            rows.append(make_row(index, "single-session-user", abstention=True))
            index += 1
        selection = {
            "seed": "fixed-test-seed",
            "quotas": {
                "abstention": 3,
                "single-session-user": 2,
                "single-session-assistant": 2,
                "single-session-preference": 2,
                "multi-session": 2,
                "knowledge-update": 2,
                "temporal-reasoning": 2,
            },
        }
        first = MODULE.select_fixed_questions(rows, selection)
        second = MODULE.select_fixed_questions(list(reversed(rows)), selection)
        self.assertEqual(
            [row["question_id"] for row in first],
            [row["question_id"] for row in second],
        )
        self.assertEqual(len(first), 15)
        self.assertEqual(sum(row["question_id"].endswith("_abs") for row in first), 3)

    def test_selection_fails_when_a_quota_cannot_be_met(self):
        rows = [make_row(1, "multi-session")]
        with self.assertRaisesRegex(ValueError, "无法抽取"):
            MODULE.select_fixed_questions(
                rows, {"seed": "x", "quotas": {"multi-session": 2}}
            )


class DatasetAndMetricTests(unittest.TestCase):
    def test_dashscope_connection_bypasses_proxy_without_dropping_existing_entries(self):
        environment = {
            "DASHSCOPE_BASE_URL": "https://dashscope.example.test/v1",
            "NO_PROXY": "127.0.0.1,localhost",
        }
        with patch.dict(MODULE.os.environ, environment, clear=True):
            hostname = MODULE.configure_dashscope_direct_connection()
            MODULE.configure_dashscope_direct_connection()
            uppercase = MODULE.os.environ["NO_PROXY"].split(",")
            lowercase = MODULE.os.environ["no_proxy"].split(",")

        self.assertEqual(hostname, "dashscope.example.test")
        self.assertEqual(uppercase, [
            "127.0.0.1", "localhost", "dashscope.example.test"
        ])
        self.assertEqual(lowercase, ["dashscope.example.test"])

    def test_runtime_requires_milvus_23_sdk(self):
        with patch.object(
            MODULE, "evaluation_runtime_config",
            return_value={"python": "3.13.0", "pymilvus": "3.0.1"},
        ):
            with self.assertRaisesRegex(RuntimeError, "PyMilvus 2.3.7"):
                MODULE.validate_evaluation_runtime()

    def test_dashscope_extraction_config_uses_runtime_model(self):
        with patch.dict(MODULE.os.environ, {
            "MEMORY_SUMMARY_PROVIDER": "dashscope",
            "DASHSCOPE_MODEL": "qwen-plus-test",
        }, clear=True):
            config = MODULE.memory_extraction_config()
        self.assertEqual(config["provider"], "dashscope")
        self.assertEqual(config["model"], "qwen-plus-test")

    def test_duplicate_source_session_ids_map_to_distinct_app_sessions(self):
        adapter = MODULE.ProjectMemoryAdapter.__new__(MODULE.ProjectMemoryAdapter)
        adapter.dataset_sha256 = "dataset-sha"
        first = adapter.session_id("q1", "duplicate", 0)
        second = adapter.session_id("q1", "duplicate", 1)
        self.assertNotEqual(first, second)

    def test_dataset_validation_rejects_misaligned_sessions(self):
        row = make_row(1, "multi-session")
        row["haystack_dates"].append("2024/01/02 (Tue) 01:02")
        with self.assertRaisesRegex(ValueError, "数量不一致"):
            MODULE.validate_dataset([row], 1)

    def test_date_parser_does_not_depend_on_weekday_locale(self):
        parsed = MODULE.parse_longmemeval_date("2024/01/02 (Tue) 03:04")
        self.assertEqual(parsed, datetime(2024, 1, 2, 3, 4))

    def test_retrieval_metrics_cover_any_all_recall_and_mrr(self):
        metrics = MODULE.score_retrieval(["a", "b"], ["x", "b", "z"])
        self.assertTrue(metrics["any_recall"])
        self.assertFalse(metrics["all_recall"])
        self.assertEqual(metrics["session_recall"], 0.5)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)

    def test_aggregate_excludes_abstention_from_retrieval(self):
        normal = {
            "question_id": "q1",
            "question_type": "multi-session",
            "abstention": False,
            "status": "ok",
            "retrieval": MODULE.score_retrieval(["a"], ["a"]),
            "judge_label": True,
            "ingestion": {
                "complete": True,
                "ready_sessions": 2,
                "total_sessions": 2,
                "failures": [],
            },
        }
        abstention = copy.deepcopy(normal)
        abstention.update(
            question_id="q2_abs",
            abstention=True,
            retrieval=None,
            judge_label=False,
        )
        metrics = MODULE.aggregate_metrics([normal, abstention])
        self.assertEqual(metrics["retrieval"]["evaluated_questions"], 1)
        self.assertEqual(metrics["retrieval"]["any_recall"], 1.0)
        self.assertEqual(metrics["qa_accuracy"], 0.5)
        self.assertEqual(metrics["ingestion"]["session_success_rate"], 1.0)

    def test_baseline_must_use_same_mode_and_top_k(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            baseline = Path(temporary_dir) / "baseline.json"
            MODULE.write_json(baseline, {
                "metadata": {
                    "dataset_sha256": "dataset-sha",
                    "mode": "retrieval",
                    "top_k": 10,
                },
                "metrics": {"retrieval": {"any_recall": 1.0}},
            })
            args = type("Args", (), {
                "min_any_recall": None,
                "min_all_recall": None,
                "min_qa_accuracy": None,
                "baseline": baseline,
                "dataset_sha256": "dataset-sha",
                "mode": "retrieval",
                "top_k": 5,
                "max_regression_drop": 0.03,
            })()
            with self.assertRaisesRegex(ValueError, "不同配置不能直接比较"):
                MODULE.check_gates(
                    {"retrieval": {"any_recall": 1.0, "all_recall": 1.0}}, args
                )

    def test_default_run_uses_unique_directory_and_requires_explicit_resume(self):
        suite = MODULE.read_json(MODULE.SUITE_PATH)
        args = MODULE.build_parser(suite).parse_args(["run"])
        self.assertIsNone(args.run_name)
        self.assertFalse(args.resume)
        self.assertEqual(args.context_mode, "current")
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.results_root, MODULE.DEFAULT_RESULTS_DIR)

    def test_context_modes_select_expected_memory_types(self):
        adapter = MODULE.ProjectMemoryAdapter.__new__(MODULE.ProjectMemoryAdapter)
        adapter.dataset_sha256 = "dataset-sha"
        adapter.provider = Mock()
        adapter.memory_service = Mock()
        app_session_id = str(adapter.session_id("q001", "s1", 0))
        adapter.provider.search.return_value = [types.SimpleNamespace(
            memory_id="memory-1",
            fragment_id="fact-1",
            memory_type="atomic_fact",
            content="User leads 4 engineers",
            score=0.9,
            session_id=app_session_id,
            metadata={"source_message_ids": ["m1"]},
        )]
        row = make_row(1, "single-session-user")
        row["question_id"] = "q001"
        row["haystack_session_ids"] = ["s1"]

        evidence = adapter.search(row, 5, "fact-only")

        options = adapter.provider.search.call_args.args[2]
        self.assertEqual(tuple(options.memory_types), ("atomic_fact",))
        self.assertEqual(evidence[0].source_session_id, "s1")
        self.assertEqual(evidence[0].metadata["source_message_ids"], ["m1"])
        adapter.memory_service._load_raw_evidence.assert_not_called()

    def test_hybrid_context_appends_raw_evidence_without_changing_session(self):
        adapter = MODULE.ProjectMemoryAdapter.__new__(MODULE.ProjectMemoryAdapter)
        adapter.dataset_sha256 = "dataset-sha"
        adapter.provider = Mock()
        adapter.memory_service = Mock()
        app_session_id = str(adapter.session_id("q001", "s1", 0))
        adapter.provider.search.return_value = [types.SimpleNamespace(
            memory_id="memory-1",
            fragment_id="fact-1",
            memory_type="atomic_fact",
            content="User's team size changed",
            score=0.9,
            session_id=app_session_id,
            metadata={},
        )]
        adapter.memory_service._load_raw_evidence.return_value = [{
            "memory_id": "memory-1",
            "message_id": "m1",
            "created_at": "2023-05-11T02:02:00",
            "role": "user",
            "content": "I lead 4 engineers.",
        }]
        row = make_row(1, "knowledge-update")
        row["question_id"] = "q001"
        row["haystack_session_ids"] = ["s1"]

        evidence = adapter.search(row, 5, "hybrid")

        self.assertIn("[Raw evidence]", evidence[0].content)
        self.assertIn("I lead 4 engineers.", evidence[0].content)
        self.assertEqual(evidence[0].source_session_id, "s1")
        self.assertEqual(tuple(adapter.provider.search.call_args.args[2].memory_types), (
            "atomic_fact", "preference"
        ))


class FullModeModelTests(unittest.TestCase):
    def test_full_mode_defaults_to_fixed_qwen_answer_and_claude_judge(self):
        with patch.dict(MODULE.os.environ, {}, clear=True):
            self.assertEqual(MODULE.full_model_config(), {
                "answer_provider": "dashscope",
                "answer_model": "qwen3.7-plus-2026-05-26",
                "answer_max_tokens": 1000,
                "answer_prompt_version": "longmemeval-rag-v1",
                "answer_content_filter_policy": "record-refusal-and-judge",
                "answer_request_attempts": 3,
                "judge_provider": "claude-relay",
                "judge_protocol": "anthropic-messages",
                "judge_model": "claude-opus-5",
                "judge_max_tokens": 1024,
                "judge_prompt_version": "longmemeval-adapted-official-v1",
                "judge_request_attempts": 3,
            })

    def test_answer_and_judge_use_separate_provider_clients(self):
        fake_service = types.ModuleType("service")
        fake_service.__path__ = []
        fake_llm_client = types.ModuleType("service.llm_client")
        create_client = Mock(return_value="qwen-client")
        fake_llm_client.create_openai_client = create_client
        environment = {
            "DASHSCOPE_API_KEY": "dashscope-test-key",
            "DASHSCOPE_BASE_URL": "https://dashscope.test/v1",
            "CLAUDE_RELAY_API_KEY": "claude-test-key",
            "CLAUDE_RELAY_BASE_URL": "https://claude.test/v1",
            "CLAUDE_RELAY_MODEL": "claude-opus-5",
        }
        row = make_row(1, "single-session-user")

        with (
            patch.dict(MODULE.os.environ, environment, clear=True),
            patch.dict(sys.modules, {
                "service": fake_service,
                "service.llm_client": fake_llm_client,
            }),
            patch.object(
                MODULE, "chat_completion", return_value="generated answer"
            ) as chat_completion,
            patch.object(
                MODULE, "claude_messages_completion", return_value="yes"
            ) as claude_completion,
        ):
            result = MODULE.answer_and_judge(row, [])

        self.assertEqual(
            result,
            (
                "generated answer",
                True,
                "qwen3.7-plus-2026-05-26",
                "claude-opus-5",
                None,
            ),
        )
        self.assertEqual(create_client.call_args_list, [
            call(api_key="dashscope-test-key", base_url="https://dashscope.test/v1"),
        ])
        self.assertEqual(chat_completion.call_args_list[0].args[:2], (
            "qwen-client", "qwen3.7-plus-2026-05-26"
        ))
        self.assertEqual(chat_completion.call_args_list[0].args[3], 1000)
        self.assertEqual(claude_completion.call_args.args[:3], (
            "claude-test-key", "https://claude.test/v1", "claude-opus-5"
        ))
        self.assertEqual(claude_completion.call_args.args[4], 1024)

    def test_claude_completion_uses_native_messages_without_temperature(self):
        if str(MODULE.APP_DIR) not in sys.path:
            sys.path.insert(0, str(MODULE.APP_DIR))
        import service.claude_client as claude_client

        response = Mock()
        response.is_error = False
        response.json.return_value = {
            "content": [{"type": "text", "text": "yes"}],
        }
        with patch.object(claude_client.httpx, "post", return_value=response) as post:
            result = MODULE.claude_messages_completion(
                "test-key",
                "https://claude.test/v1",
                "claude-opus-5",
                "Judge this answer",
                10,
            )

        self.assertEqual(result, "yes")
        self.assertEqual(post.call_args.args[0], "https://claude.test/v1/messages")
        request_json = post.call_args.kwargs["json"]
        self.assertEqual(request_json["model"], "claude-opus-5")
        self.assertEqual(request_json["max_tokens"], 10)
        self.assertNotIn("temperature", request_json)

    def test_judge_label_requires_exact_yes_or_no(self):
        self.assertTrue(MODULE.parse_judge_label(" yes\n"))
        self.assertFalse(MODULE.parse_judge_label("NO"))
        with self.assertRaisesRegex(RuntimeError, "只返回 yes/no"):
            MODULE.parse_judge_label("yes, because the answer is correct")
        self.assertTrue(MODULE.is_transient_llm_error(
            RuntimeError("Claude Judge 应只返回 yes/no，实际返回：No — wait. yes")
        ))

    def test_api_connection_error_is_transient(self):
        api_connection_error = type("APIConnectionError", (Exception,), {})
        self.assertTrue(MODULE.is_transient_llm_error(
            api_connection_error("Connection error.")
        ))


if __name__ == "__main__":
    unittest.main()
