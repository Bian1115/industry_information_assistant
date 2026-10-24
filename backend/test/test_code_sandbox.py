"""CodeWizard 沙箱边界的单元测试，不调用 Docker、E2B 或真实 LLM。"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))
# service/__init__.py 会加载整个业务依赖树；本测试只加载沙箱模块。
service_package = types.ModuleType("service")
service_package.__path__ = [str(APP_DIR / "service")]
sys.modules.setdefault("service", service_package)

from service.code_sandbox import (
    DisabledSandboxProvider,
    DockerGVisorSandboxProvider,
    SandboxLimits,
    create_code_sandbox_provider,
)


class SandboxFactoryTests(unittest.TestCase):
    def test_default_is_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            provider = create_code_sandbox_provider()
        self.assertIsInstance(provider, DisabledSandboxProvider)

    def test_enabled_provider_is_docker_gvisor(self):
        with patch.dict(
            os.environ,
            {
                "CODE_SANDBOX_PROVIDER": "docker-gvisor",
                "CODE_SANDBOX_DOCKER_RUNTIME": "runsc",
            },
            clear=True,
        ):
            provider = create_code_sandbox_provider()

        self.assertIsInstance(provider, DockerGVisorSandboxProvider)
        self.assertEqual(provider.runtime, "runsc")

    def test_empty_runtime_does_not_downgrade_to_runc(self):
        with patch.dict(
            os.environ,
            {
                "CODE_SANDBOX_PROVIDER": "docker-gvisor",
                "CODE_SANDBOX_DOCKER_RUNTIME": "",
            },
            clear=True,
        ):
            provider = create_code_sandbox_provider()

        self.assertIsInstance(provider, DisabledSandboxProvider)

    def test_e2b_is_not_an_enabled_provider(self):
        with patch.dict(
            os.environ,
            {"CODE_SANDBOX_PROVIDER": "e2b"},
            clear=True,
        ):
            provider = create_code_sandbox_provider()

        self.assertIsInstance(provider, DisabledSandboxProvider)

    def test_docker_command_enforces_security_limits(self):
        provider = DockerGVisorSandboxProvider(
            image="industry-code-sandbox:test",
            runtime="runsc",
            limits=SandboxLimits(
                timeout_seconds=12,
                memory="256m",
                cpus=0.5,
                pids=32,
            ),
        )
        command = provider._build_command(Path("/tmp/sandbox-test"), "safe-name")
        rendered = " ".join(command)

        self.assertIn("--runtime runsc", rendered)
        self.assertIn("--network none", rendered)
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop ALL", rendered)
        self.assertIn("no-new-privileges:true", command)
        self.assertIn("--memory 256m", rendered)
        self.assertIn("--cpus 0.5", rendered)
        self.assertIn("--pids-limit 32", rendered)
        self.assertNotIn("docker.sock", rendered)


class DisabledProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_provider_refuses_execution(self):
        result = await DisabledSandboxProvider().execute("print('unsafe')")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "sandbox_not_configured")


class WizardWiringTests(unittest.TestCase):
    def test_wizard_has_no_in_process_executor(self):
        wizard_path = (
            APP_DIR / "service" / "deep_research_v2" / "agents" / "wizard.py"
        )
        source = wizard_path.read_text(encoding="utf-8")
        self.assertNotIn("def _execute_in_sandbox", source)
        self.assertNotIn("exec(code,", source)
        self.assertIn("await self.sandbox_provider.execute(code)", source)

    def test_wizard_explicitly_skips_when_sandbox_is_disabled(self):
        wizard_path = (
            APP_DIR / "service" / "deep_research_v2" / "agents" / "wizard.py"
        )
        source = wizard_path.read_text(encoding="utf-8")
        self.assertIn("def sandbox_available", source)
        self.assertIn('"reason": "sandbox_disabled"', source)
        skip_position = source.index("if not self.sandbox_available")
        analysis_position = source.index("await self._analyze_data(state)")
        self.assertLess(skip_position, analysis_position)


if __name__ == "__main__":
    unittest.main()
