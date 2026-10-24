"""可插拔的 LLM 代码沙箱。

主业务只依赖 :class:`CodeSandboxProvider`，不直接执行模型生成的代码。
未配置 provider 时默认拒绝执行，绝不回退到进程内 ``exec``。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


logger = logging.getLogger(__name__)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[输出已截断，原长度 {len(value)} 字符]"


@dataclass(frozen=True)
class SandboxLimits:
    """单次代码执行的资源上限。"""

    timeout_seconds: int = 30
    max_output_chars: int = 20_000
    max_chart_bytes: int = 5 * 1024 * 1024
    memory: str = "512m"
    cpus: float = 1.0
    pids: int = 64


@dataclass
class SandboxExecutionResult:
    success: bool
    output: str = ""
    error: Optional[str] = None
    charts: List[str] = field(default_factory=list)
    provider: str = "unknown"
    error_code: Optional[str] = None

    def to_legacy_dict(self) -> dict[str, Any]:
        """保持 CodeWizard 原有返回结构，避免影响报告生成链路。"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "charts": self.charts,
            "sandbox_provider": self.provider,
            "error_code": self.error_code,
        }


class CodeSandboxProvider(ABC):
    @abstractmethod
    async def execute(self, code: str) -> SandboxExecutionResult:
        """在主服务之外执行代码。"""


class DisabledSandboxProvider(CodeSandboxProvider):
    async def execute(self, code: str) -> SandboxExecutionResult:
        del code
        return SandboxExecutionResult(
            success=False,
            error=(
                "代码沙箱未配置。请设置 CODE_SANDBOX_PROVIDER=docker-gvisor；"
                "系统已拒绝在 FastAPI 进程内执行模型生成代码。"
            ),
            provider="disabled",
            error_code="sandbox_not_configured",
        )


class DockerGVisorSandboxProvider(CodeSandboxProvider):
    """通过 Docker CLI 启动一次性 gVisor 容器。

    ``runsc`` 不可用时直接失败，不会悄悄降级为普通 ``runc``。
    """

    def __init__(
        self,
        image: str,
        runtime: str,
        limits: SandboxLimits,
        docker_binary: str = "docker",
    ):
        self.image = image
        self.runtime = runtime
        self.limits = limits
        self.docker_binary = docker_binary

    def _build_command(self, workspace: Path, container_name: str) -> list[str]:
        command = [
            self.docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(self.limits.pids),
            "--memory",
            self.limits.memory,
            "--cpus",
            str(self.limits.cpus),
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--env",
            "HOME=/tmp",
            "--env",
            "MPLCONFIGDIR=/tmp/matplotlib",
            "--volume",
            f"{workspace}:/workspace:rw",
        ]
        if self.runtime:
            command.extend(["--runtime", self.runtime])
        command.append(self.image)
        return command

    async def execute(self, code: str) -> SandboxExecutionResult:
        container_name = f"industry-code-{uuid.uuid4().hex[:16]}"
        with tempfile.TemporaryDirectory(prefix="industry-sandbox-") as temp_dir:
            workspace = Path(temp_dir)
            os.chmod(workspace, 0o777)
            (workspace / "code.py").write_text(code, encoding="utf-8")
            command = self._build_command(workspace, container_name)

            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                return SandboxExecutionResult(
                    success=False,
                    error="找不到 Docker CLI，无法启动私有沙箱。",
                    provider="docker-gvisor",
                    error_code="sandbox_runtime_missing",
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.limits.timeout_seconds + 5,
                )
            except asyncio.TimeoutError:
                await self._force_remove(container_name)
                process.kill()
                await process.wait()
                return SandboxExecutionResult(
                    success=False,
                    error=f"代码执行超过 {self.limits.timeout_seconds} 秒，沙箱已销毁。",
                    provider="docker-gvisor",
                    error_code="sandbox_timeout",
                )
            except asyncio.CancelledError:
                await self._force_remove(container_name)
                process.kill()
                await process.wait()
                raise
            finally:
                await self._force_remove(container_name)

            docker_error = stderr.decode("utf-8", errors="replace")
            result_path = workspace / "result.json"
            if process.returncode != 0 or not result_path.is_file():
                detail = docker_error or stdout.decode("utf-8", errors="replace")
                return SandboxExecutionResult(
                    success=False,
                    error=_truncate(
                        detail or "沙箱没有返回执行结果，请检查镜像和 gVisor 配置。",
                        self.limits.max_output_chars,
                    ),
                    provider="docker-gvisor",
                    error_code="sandbox_runtime_failed",
                )

            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return SandboxExecutionResult(
                    success=False,
                    error=f"无法读取沙箱结果：{exc}",
                    provider="docker-gvisor",
                    error_code="sandbox_invalid_result",
                )

            charts: list[str] = []
            chart_path = workspace / "chart.png"
            if chart_path.is_file() and chart_path.stat().st_size <= self.limits.max_chart_bytes:
                charts.append(base64.b64encode(chart_path.read_bytes()).decode("ascii"))

            return SandboxExecutionResult(
                success=bool(payload.get("success")),
                output=_truncate(str(payload.get("output", "")), self.limits.max_output_chars),
                error=(
                    _truncate(str(payload["error"]), self.limits.max_output_chars)
                    if payload.get("error")
                    else None
                ),
                charts=charts,
                provider="docker-gvisor",
                error_code=None if payload.get("success") else "sandbox_user_code_failed",
            )

    async def _force_remove(self, container_name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                self.docker_binary,
                "rm",
                "--force",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=5)
        except (FileNotFoundError, asyncio.TimeoutError):
            logger.warning("Failed to force-remove sandbox %s", container_name)


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid %s; using %s", name, default)
        return default
    return max(minimum, min(value, maximum))


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid %s; using %s", name, default)
        return default
    return max(minimum, min(value, maximum))


def create_code_sandbox_provider() -> CodeSandboxProvider:
    """从环境变量创建 provider；默认 fail-closed。"""
    limits = SandboxLimits(
        timeout_seconds=_int_env("CODE_SANDBOX_TIMEOUT_SECONDS", 30, 1, 300),
        max_output_chars=_int_env("CODE_SANDBOX_MAX_OUTPUT_CHARS", 20_000, 1_000, 200_000),
        max_chart_bytes=_int_env("CODE_SANDBOX_MAX_CHART_BYTES", 5 * 1024 * 1024, 1_024, 20 * 1024 * 1024),
        memory=os.getenv("CODE_SANDBOX_MEMORY", "512m"),
        cpus=_float_env("CODE_SANDBOX_CPUS", 1.0, 0.1, 8.0),
        pids=_int_env("CODE_SANDBOX_PIDS", 64, 8, 512),
    )
    provider = os.getenv("CODE_SANDBOX_PROVIDER", "disabled").strip().lower()

    if provider in {"docker", "docker-gvisor", "gvisor"}:
        runtime = os.getenv("CODE_SANDBOX_DOCKER_RUNTIME", "runsc").strip()
        if not runtime:
            logger.error(
                "CODE_SANDBOX_DOCKER_RUNTIME is empty; refusing to fall back to runc"
            )
            return DisabledSandboxProvider()
        return DockerGVisorSandboxProvider(
            image=os.getenv("CODE_SANDBOX_DOCKER_IMAGE", "industry-code-sandbox:latest"),
            runtime=runtime,
            limits=limits,
            docker_binary=os.getenv("CODE_SANDBOX_DOCKER_BINARY", "docker"),
        )
    if provider != "disabled":
        logger.error("Unknown CODE_SANDBOX_PROVIDER=%s; code execution disabled", provider)
    return DisabledSandboxProvider()
