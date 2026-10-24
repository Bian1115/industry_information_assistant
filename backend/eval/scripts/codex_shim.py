#!/usr/bin/env python3
# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
Codex Shim —— 把 codex 命令行伪装成 OpenAI API

## 解决什么问题

DeepResearch Bench 的官方评测脚本要调 OpenAI API 打分，但你没有 API key，
只有本机能跑的 codex 命令行。两边接口对不上：一边发 HTTP 请求，
一边只能敲命令。

这个 shim 夹在中间做转换（就像出国用的电源转换插头）：

    DRB 脚本 ──HTTP──> 本 shim ──subprocess──> codex exec ──> GPT
                          │                                    │
                          └──包装成 OpenAI 响应格式 <───────────┘

DRB 全程以为自己在调 OpenAI，不需要改它一行代码——因为它的
utils/api.py 读 OPENAI_BASE_URL 环境变量，地址不是写死的。

## 用法

    # 终端 A：起 shim
    python eval/scripts/codex_shim.py --port 8899

    # 终端 B：让 DRB 指向它
    export LLM_BACKEND=openai
    export OPENAI_BASE_URL=http://127.0.0.1:8899/v1
    export OPENAI_API_KEY=dummy          # shim 不校验
    export RACE_MODEL=gpt-5.5            # 名字随便，shim 忽略
    export JINA_API_KEY=<你的key>         # FACT 抓网页要用，见 jina.ai/reader
    bash run_benchmark.sh

    # 自检（不需要 DRB）
    python eval/scripts/codex_shim.py --selftest

## 重要限制

1. **分数不等同官方榜分数**。codex 是 agent 包装层而非裸模型端点，
   自带 system prompt，判官行为与直连 GPT-5.5 API 未必一致。
   官方提交排行榜要求提供临时 key 由他们复现，自跑分数不被认可。
2. **吞吐低**。每次请求起一个进程，RACE 每道题要调用多次。
3. **额度走 ChatGPT 订阅**，可能撞订阅侧速率限制。

定位是本地迭代用的评测手段，不是官方跑分。
"""

import os
import re
import sys
import json
import time
import uuid
import shutil
import logging
import argparse
import tempfile
import threading
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] shim: %(message)s",
)
logger = logging.getLogger("codex_shim")


# ---------------------------------------------------------------------------
# 配置（由 main() 填充）
# ---------------------------------------------------------------------------

class Config:
    codex_bin = "codex"
    model = None            # None 表示用 codex 默认模型
    timeout = 600           # 单次 codex 调用超时（秒）
    max_retries = 2         # codex 偶发 WebSocket 重连失败，需要重试
    concurrency = 2         # 同时最多几个 codex 进程
    verbose = False


CFG = Config()
_slots = None               # 并发信号量，main() 里初始化
_stats = {"requests": 0, "ok": 0, "failed": 0, "total_s": 0.0}
_stats_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 消息拼装
# ---------------------------------------------------------------------------

def flatten_messages(messages) -> str:
    """
    把 OpenAI 的 messages 数组压成一段纯文本喂给 codex。

    codex exec 只接受一段提示词，没有 role 概念，
    所以用标题把 system / user / assistant 分隔开，保留角色语义。
    """
    parts = []
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # content 可能是数组形式 [{"type":"text","text":"..."}]
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        content = (content or "").strip()
        if not content:
            continue

        if role == "system":
            parts.append(f"# 系统指令\n{content}")
        elif role == "assistant":
            parts.append(f"# 助手先前的回复\n{content}")
        else:
            parts.append(content)

    return "\n\n".join(parts)


JSON_INSTRUCTION = (
    "\n\n---\n"
    "【输出格式要求】只输出一个合法的 JSON 对象本身。"
    "不要加 markdown 代码块标记，不要加任何解释性文字、前言或后记。"
)


def strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 包裹，DRB 侧解析 JSON 时会更省事。"""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# 调用 codex
# ---------------------------------------------------------------------------

def call_codex(prompt: str, want_json: bool) -> str:
    """
    执行一次 codex exec，返回它的最终回复文本。

    两个关键点：
    1. 提示词走 stdin 而不是命令行参数——RACE 会把整篇报告塞进提示词，
       动辄几十 KB，超过命令行长度限制会直接失败。
       codex exec 在参数位给 "-" 时从 stdin 读取。
    2. 用 -o 把最终回复写到临时文件——codex 的 stdout 混着
       会话回显、重连告警、token 统计等噪音，直接解析 stdout 不可靠。
    """
    if want_json:
        prompt = prompt + JSON_INSTRUCTION

    last_error = None
    for attempt in range(1, CFG.max_retries + 2):
        out_fd, out_path = tempfile.mkstemp(prefix="codex_shim_", suffix=".txt")
        os.close(out_fd)
        try:
            cmd = [CFG.codex_bin, "exec", "--skip-git-repo-check", "-o", out_path]
            if CFG.model:
                cmd += ["-m", CFG.model]
            cmd.append("-")          # 提示词从 stdin 读

            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=CFG.timeout,
            )

            with open(out_path, "r", encoding="utf-8") as f:
                answer = f.read().strip()

            if answer:
                return strip_code_fence(answer) if want_json else answer

            # -o 文件为空：可能 codex 异常退出
            tail = (proc.stderr or proc.stdout or "")[-400:]
            last_error = f"codex 返回空内容 (exit={proc.returncode}) {tail}"

        except subprocess.TimeoutExpired:
            last_error = f"codex 调用超时 (>{CFG.timeout}s)"
        except FileNotFoundError:
            # 二进制不存在，重试没意义
            raise RuntimeError(f"找不到 codex 可执行文件: {CFG.codex_bin}")
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

        if attempt <= CFG.max_retries:
            wait = 2 ** attempt
            logger.warning(f"第 {attempt} 次失败（{last_error}），{wait}s 后重试")
            time.sleep(wait)

    raise RuntimeError(last_error or "codex 调用失败")


# ---------------------------------------------------------------------------
# OpenAI 响应封装
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    粗略估算 token 数。

    codex 不回传真实用量，但 OpenAI 响应格式里 usage 是必需字段，
    调用方（含 DRB）通常只做日志记录不做计费，估算值足够。
    中英混排按 1 token ≈ 2.5 字符估。
    """
    return max(1, int(len(text or "") / 2.5))


def build_response(model: str, content: str, prompt_text: str) -> dict:
    pt, ct = estimate_tokens(prompt_text), estimate_tokens(content)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
        "system_fingerprint": "codex-shim",
    }


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if CFG.verbose:
            logger.info(f"{self.address_string()} {fmt % args}")

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str, etype: str = "shim_error"):
        self._send(code, {"error": {"message": message, "type": etype, "code": code}})

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health"):
            with _stats_lock:
                s = dict(_stats)
            avg = s["total_s"] / s["ok"] if s["ok"] else 0
            self._send(200, {"status": "ok", "backend": "codex",
                             "model": CFG.model or "codex-default",
                             "stats": s, "avg_seconds": round(avg, 1)})
        elif path == "/v1/models":
            # 某些客户端启动时会探测模型列表
            self._send(200, {"object": "list", "data": [
                {"id": CFG.model or "codex-default", "object": "model",
                 "created": int(time.time()), "owned_by": "codex-shim"}
            ]})
        else:
            self._error(404, f"未实现的路径: {self.path}")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, f"未实现的路径: {self.path}")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._error(400, f"请求体解析失败: {e}", "invalid_request_error")
            return

        messages = payload.get("messages") or []
        if not messages:
            self._error(400, "messages 为空", "invalid_request_error")
            return

        prompt = flatten_messages(messages)
        want_json = (payload.get("response_format") or {}).get("type") == "json_object"
        model = payload.get("model") or CFG.model or "codex-default"

        with _stats_lock:
            _stats["requests"] += 1
            seq = _stats["requests"]

        logger.info(f"[#{seq}] 收到请求 model={model} 提示词 {len(prompt):,} 字符 "
                    f"json={want_json}")

        started = time.time()
        try:
            with _slots:                      # 限制同时运行的 codex 进程数
                content = call_codex(prompt, want_json)
        except Exception as e:
            with _stats_lock:
                _stats["failed"] += 1
            logger.error(f"[#{seq}] 失败: {e}")
            # 用 502 而非 500：让调用方知道是上游（codex）的问题，
            # 多数 OpenAI 客户端对 5xx 会自动重试
            self._error(502, str(e), "upstream_error")
            return

        elapsed = time.time() - started
        with _stats_lock:
            _stats["ok"] += 1
            _stats["total_s"] += elapsed

        logger.info(f"[#{seq}] 完成 {len(content):,} 字符 / {elapsed:.1f}s")
        self._send(200, build_response(model, content, prompt))


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest(port: int) -> bool:
    """不依赖 DRB，直接验证 shim 是否正确说 OpenAI 协议。"""
    import urllib.request

    base = f"http://127.0.0.1:{port}"
    print("=" * 62)
    print("Codex Shim 自检")
    print("=" * 62)

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def post(path, body, timeout=CFG.timeout + 30):
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer dummy"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    ok = True

    try:
        h = get("/health")
        print(f"[1] /health          ✅ {h['status']} backend={h['backend']}")
    except Exception as e:
        print(f"[1] /health          ❌ {e}")
        return False

    try:
        m = get("/v1/models")
        print(f"[2] /v1/models       ✅ {m['data'][0]['id']}")
    except Exception as e:
        print(f"[2] /v1/models       ❌ {e}")
        ok = False

    print("[3] /v1/chat/completions  调用 codex 中，请稍候…")
    try:
        r = post("/v1/chat/completions", {
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "你是一个严格的评分员。"},
                {"role": "user", "content": "给这句话打分：'今天天气不错'。"
                                            "只回 JSON：{\"score\": 一个1到10的整数}"},
            ],
            "response_format": {"type": "json_object"},
        })
        content = r["choices"][0]["message"]["content"]
        shape_ok = all(k in r for k in ("id", "object", "created", "choices", "usage"))
        print(f"    响应结构        {'✅' if shape_ok else '❌'} "
              f"id={r['id'][:20]}… usage={r['usage']['total_tokens']} tokens")
        print(f"    返回内容        {content[:80]}")
        try:
            parsed = json.loads(content)
            print(f"    JSON 可解析     ✅ {parsed}")
        except json.JSONDecodeError:
            print(f"    JSON 可解析     ⚠️  不是纯 JSON（判官提示词通常容错，可接受）")
        ok = ok and shape_ok
    except Exception as e:
        print(f"    ❌ {e}")
        ok = False

    print("=" * 62)
    print("✅ 自检通过，可以让 DRB 指向本 shim" if ok else "❌ 自检未通过")
    if ok:
        print(f"\n  export LLM_BACKEND=openai")
        print(f"  export OPENAI_BASE_URL={base}/v1")
        print(f"  export OPENAI_API_KEY=dummy")
    return ok


def main():
    parser = argparse.ArgumentParser(description="把 codex 命令行伪装成 OpenAI API")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", default="127.0.0.1",
                        help="默认只监听本机；对外暴露前请确认网络环境")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", help="传给 codex 的 -m 参数，不填用 codex 默认")
    parser.add_argument("--timeout", type=int, default=600, help="单次 codex 调用超时秒数")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2,
                        help="同时最多几个 codex 进程")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="对已运行的 shim 做自检（需另一终端已启动）")
    args = parser.parse_args()

    CFG.codex_bin = args.codex_bin
    CFG.model = args.model
    CFG.timeout = args.timeout
    CFG.max_retries = args.max_retries
    CFG.concurrency = args.concurrency
    CFG.verbose = args.verbose

    if args.selftest:
        sys.exit(0 if selftest(args.port) else 1)

    if not shutil.which(CFG.codex_bin):
        print(f"❌ 找不到 codex 可执行文件: {CFG.codex_bin}")
        print("   用 --codex-bin 指定完整路径")
        sys.exit(1)

    global _slots
    _slots = threading.Semaphore(CFG.concurrency)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    print("=" * 62)
    print("Codex Shim —— OpenAI 兼容接口")
    print("=" * 62)
    print(f"监听      : http://{args.host}:{args.port}/v1")
    print(f"后端      : {CFG.codex_bin}  模型={CFG.model or 'codex 默认'}")
    print(f"并发/超时 : {CFG.concurrency} 进程 / {CFG.timeout}s，重试 {CFG.max_retries} 次")
    print()
    print("让 DRB 指向本服务：")
    print(f"  export LLM_BACKEND=openai")
    print(f"  export OPENAI_BASE_URL=http://{args.host}:{args.port}/v1")
    print(f"  export OPENAI_API_KEY=dummy")
    print()
    print(f"自检: python {os.path.basename(__file__)} --selftest --port {args.port}")
    print("Ctrl-C 退出")
    print("=" * 62)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        with _stats_lock:
            s = dict(_stats)
        print(f"\n已退出。累计 {s['requests']} 次请求 / 成功 {s['ok']} / 失败 {s['failed']}")
        server.shutdown()


if __name__ == "__main__":
    main()
