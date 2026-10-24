"""确认旁路服务不可用时仍保存完整载荷，失败轨迹不被成功记录掩盖。"""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.runtime.langfuse_tracing import observation
from eval.scripts.run_deepresearch_bench import run_one


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_local_observations_keep_full_payload_and_identity(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "埋点.jsonl"
            async def capture(run_id):
                value = {"text": "全文" * 5000}
                with observation("parent", run_id=run_id, input=value):
                    await asyncio.sleep(0)
                    with observation("child") as child:
                        child.update(output=value)
                        value["text"] = "后续修改"
            with patch.dict(os.environ, {"EVAL_TRACE_LOCAL_FILE": str(path)}), patch(
                "eval.runtime.langfuse_tracing.get_client", return_value=None
            ):
                await asyncio.gather(capture("run-a"), capture("run-b"))
            events = [json.loads(line) for line in path.read_text().splitlines()]
            parents = {e["observation_id"]: e["run_id"] for e in events if e["name"] == "parent" and e["event"] == "start"}
            children = [e for e in events if e["name"] == "child" and e["event"] == "start"]
            self.assertEqual({e["run_id"] for e in children}, {"run-a", "run-b"})
            for event in children:
                self.assertEqual(parents[event["parent_id"]], event["run_id"])
            updates = [e for e in events if e["event"] == "update"]
            self.assertTrue(all(e["output"]["text"] == "全文" * 5000 for e in updates))

    async def test_error_after_report_preserves_raw_stream_and_partial_report(self):
        class Service:
            async def research(self, **kwargs):
                for event in [{"type": "research_complete", "data": {"final_report": "已生成正文"}},
                              {"type": "error", "data": {"content": "交付失败"}}]:
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        with tempfile.TemporaryDirectory() as root:
            record = await run_one(Service(), {"id": 1, "prompt": "原题"}, "采集回归", "20260930", 2, False, root)
            self.assertEqual(record["article"], "")
            self.assertIn("交付失败", record["_meta"]["error"])
            self.assertEqual((Path(root) / "未完成报告.md").read_text(), "已生成正文")
            self.assertEqual(len((Path(root) / "原始SSE轨迹.jsonl").read_text().splitlines()), 2)
            self.assertEqual(json.loads((Path(root) / "运行身份.json").read_text())["task"]["prompt"], "原题")


if __name__ == "__main__":
    unittest.main()
