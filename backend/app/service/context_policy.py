"""可复现的上下文消融配置；未实现能力拒绝执行，不自动改变实验组。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

BACKEND_DIR = Path(__file__).resolve().parents[2]
PRESET_DIR = BACKEND_DIR / "eval" / "configs" / "上下文压缩消融"
AGENTS = ("ChiefArchitect", "DeepScout", "DataAnalyst", "LeadWriter", "CriticMaster")
OUTPUT_LIMITS = dict(zip(AGENTS, (3000, 6000, 3000, 60000, 6000)))
# 新运行的部署默认预算；250k 按 250,000 token 计，已有冻结配置不变。
DEFAULT_CONTEXT_WINDOW = 250000
DEFAULT_MAX_INPUT = 65536
DEFAULT_SAFETY_TOKENS = 1024


def _environment_switch(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    if value not in {"1", "true", "on", "0", "false", "off"}:
        raise ValueError(f"{name}须为1/0、true/false或on/off")
    return value in {"1", "true", "on"}


def _environment_compression_enabled() -> bool:
    return _environment_switch("CONTEXT_COMPRESSION_ENABLED")


def context_compression_enabled() -> bool:
    """总开关沿用显式冻结文件；无文件时读取部署设置。"""
    path = os.getenv("CONTEXT_ABLATION_CONFIG", "").strip()
    return read_policy(path).compression_enabled if path else _environment_compression_enabled()


def session_context_compression_enabled() -> bool:
    """会话原文模式可单独关闭摘要及裁剪；总开关关闭仍关闭全部方法。"""
    path = os.getenv("CONTEXT_ABLATION_CONFIG", "").strip()
    if path:
        return read_policy(path).effective_session_compression_enabled
    return (_environment_compression_enabled()
            and _environment_switch("SESSION_CONTEXT_COMPRESSION_ENABLED"))


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class InputBudget(PolicyModel):
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW, gt=0)
    max_input: int = Field(default=DEFAULT_MAX_INPUT, gt=0)
    safety_tokens: int = Field(default=DEFAULT_SAFETY_TOKENS, ge=0)

    @model_validator(mode="after")
    def validate_window(self):
        if self.max_input > self.context_window or self.safety_tokens >= self.context_window:
            raise ValueError("输入上限不能超过窗口，安全余量必须小于窗口")
        return self

    def available(self, output_tokens: int) -> int:
        if output_tokens <= 0:
            raise ValueError("输出预留必须为正数")
        value = min(self.max_input, self.context_window - output_tokens - self.safety_tokens)
        if value <= 0:
            raise ValueError("扣除输出预留和安全余量后没有输入空间")
        return value


class NLISettings(PolicyModel):
    # 旧模型名称仅供历史快照读取；运行入口拒绝执行旧后端。
    model_id: Literal["qwen3-max", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"] = "qwen3-max"
    revision: str = "api-alias"
    model_dir: str = ""
    entailment_threshold: float = Field(default=0.85, gt=0.5, le=1)
    contradiction_threshold: float = Field(default=0.5, ge=0.5, le=1)
    review_confidence_threshold: float = Field(default=0.8, ge=0, le=1)
    max_length: int = Field(default=65536, ge=16, le=258048)
    output_tokens: int = Field(default=4096, ge=256, le=65536)
    safety_tokens: int = Field(default=1024, ge=0, le=32768)
    batch_size: int = Field(default=4, ge=1, le=32)
    max_pairs: int = Field(default=256, ge=1, le=1024)
    timeout_seconds: float = Field(default=300.0, gt=0, le=1800)
    aggregation: Literal["any_support_no_contradiction_v1"] = "any_support_no_contradiction_v1"


def runtime_nli_settings() -> NLISettings:
    """新运行使用Qwen核验；摘要与Writer冻结同一套独立预算。"""
    return NLISettings(
        max_length=int(os.getenv("AGENT_NLI_MAX_INPUT_TOKENS", "65536")),
        output_tokens=int(os.getenv("AGENT_NLI_MAX_OUTPUT_TOKENS", "4096")),
        safety_tokens=int(os.getenv("AGENT_NLI_SAFETY_TOKENS", "1024")),
        timeout_seconds=float(os.getenv("AGENT_NLI_TIMEOUT_SECONDS", "300")),
        entailment_threshold=float(os.getenv("AGENT_SUMMARY_NLI_ENTAILMENT", "0.85")),
        contradiction_threshold=float(os.getenv("AGENT_SUMMARY_NLI_CONTRADICTION", "0.5")),
        review_confidence_threshold=float(os.getenv("AGENT_SUMMARY_NLI_REVIEW_CONFIDENCE", "0.8")),
    )


class VerificationSettings(PolicyModel):
    rules_version: Literal["verification-v1", "verification-v2"] = "verification-v2"
    nli_support: bool = True
    coverage: bool = False
    key_fields: bool = True
    evidence_context: bool = True
    repair_attempts: int = Field(default=1, ge=0, le=1)
    max_facts: int = Field(default=64, ge=1, le=256)
    max_context_chars: int = Field(default=2400, ge=128, le=16000)
    max_evidence_windows: int = Field(default=8, ge=1, le=32)
    max_stage_calls: int = Field(default=128, ge=1, le=512)
    timeout_seconds: float = Field(default=180.0, gt=0, le=600)


class InputHistorySettings(PolicyModel):
    """旧输入只做原文引用替换，不生成模型摘要。"""
    version: Literal["input-view-v2"] = "input-view-v2"
    reference_materials: bool = True
    min_reference_tokens: int = Field(default=128, ge=1)


class LegacyInputHistorySettings(PolicyModel):
    """仅兼容读取旧配置及校验快照哈希，单条摘要参数不再参与执行。"""
    version: Literal["input-view-v1"] = "input-view-v1"
    reference_materials: bool = True
    summarize_long_inputs: bool = True
    summary_trigger_tokens: int = Field(default=4000, ge=1)
    min_reference_tokens: int = Field(default=128, ge=1)
    max_summaries_per_request: int = Field(default=2, ge=1, le=20)


class SummaryLifecycleSettings(PolicyModel):
    version: Literal["summary-first-v1"] = "summary-first-v1"
    claimify_enabled: bool = False
    atomic_citations: bool = False
    generation_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    claimify_timeout_seconds: float = Field(default=300.0, gt=0, le=600)


class AgentCompressionSettings(PolicyModel):
    """每个 Agent 的历史输入压缩；不改变原始记录保存和当前输入保护。"""
    enabled: bool = False
    reference_materials: bool = True
    artifact_offload: bool = True  # 是否允许外置；实际执行另查触发条件。
    semantic_summary: bool = True
    semantic_verification: bool = False

    @model_validator(mode="after")
    def validate_methods(self):
        if self.enabled and self.semantic_verification and not self.semantic_summary:
            raise ValueError("摘要核验要求该 Agent 开启摘要")
        return self


class ContextPolicy(PolicyModel):
    _legacy_nli_review_config: bool = PrivateAttr(default=False)
    schema_version: Literal[1] = 1
    policy_version: Literal["legacy-runtime-v1", "context-ablation-v1"] = "context-ablation-v1"
    group: str = Field(min_length=1, max_length=64)
    compression_enabled: bool = True
    session_compression_enabled: bool | None = None
    artifact_offload: bool = True  # 是否允许外置；不表示本次输入已经触发外置。
    early_offload: bool = True
    recent_offload: bool = True
    semantic_summary: bool = True
    pinned_contract: bool = True
    semantic_verification: bool = True
    nli: NLISettings | None = None
    verification: VerificationSettings | None = None
    input_history: InputHistorySettings | LegacyInputHistorySettings | None = None
    summary_lifecycle: SummaryLifecycleSettings | None = None
    early_tool_rounds: int = Field(default=3, ge=1, le=100)
    tool_content_tokens: int = Field(default=800, ge=1)
    preview_chars: int = Field(default=400, ge=1)
    soft_token_cap: int = Field(default=17000, ge=1)
    soft_ratio: float = Field(default=0.85, gt=0, lt=1)
    summary_ratio: float = Field(default=0.95, gt=0, le=1)
    replay_agents: list[str] = Field(default_factory=lambda: ["DeepScout"])
    agent_compression: dict[str, AgentCompressionSettings] | None = None
    stage_budgets: dict[str, InputBudget] = Field(
        default_factory=lambda: {name: InputBudget() for name in AGENTS})
    output_limits: dict[str, int] = Field(default_factory=lambda: dict(OUTPUT_LIMITS))
    summary_budget: InputBudget = Field(default_factory=InputBudget)
    summary_output_tokens: int = Field(default=3000, ge=1, le=3000)
    summary_max_chunks: int = Field(default=8, ge=1, description="旧增量策略批数上限；chunk-merge-v1按token组批，不使用此限制")
    summary_max_merge_levels: int = Field(default=4, ge=1, description="兼容旧快照；chunk-merge-v1只允许一次额外分批合并，不使用此字段")
    summary_strategy: Literal["incremental-v3", "split-citations-v4", "chunk-merge-v1", "claimify-v1"] = "claimify-v1"
    history_read_max_calls: int | None = Field(
        default=None, ge=0, le=20, description="None表示不限回读次数；0禁用；正整数限制单次Agent调用的回读次数")
    history_read_max_chars: int = Field(default=8000, ge=1, le=8000)

    @model_validator(mode="after")
    def validate_policy(self):
        if self.soft_ratio >= self.summary_ratio:
            raise ValueError("必须满足软触发比例 < 摘要比例 <= 1")
        if self.semantic_verification and not self.semantic_summary:
            raise ValueError("去摘要组不能启用摘要语义校验")
        if self.summary_lifecycle is not None:
            if self.summary_strategy != "claimify-v1":
                raise ValueError("先保存正文的生命周期仅支持claimify-v1正文策略")
            if not self.summary_lifecycle.claimify_enabled and self.semantic_verification:
                raise ValueError("关闭Claimify链路时不能启用其摘要语义核验")
            if self.summary_lifecycle.atomic_citations and not self.summary_lifecycle.claimify_enabled:
                raise ValueError("原子论断引用链路要求开启Claimify审计")
        if set(self.stage_budgets) != set(AGENTS):
            raise ValueError("必须显式配置五个阶段的预算")
        if set(self.output_limits) != set(AGENTS):
            raise ValueError("必须配置五个阶段的输出预留上限")
        for name, limit in self.output_limits.items():
            maximum = 65536 if name == "LeadWriter" else 16000
            if not 256 <= limit <= maximum:
                raise ValueError(f"{name}阶段输出上限须在256至{maximum}之间")
            self.thresholds(self.stage_budgets[name].available(limit))
        if len(set(self.replay_agents)) != len(self.replay_agents) or set(self.replay_agents) - set(AGENTS):
            raise ValueError("历史回放智能体配置无效")
        if self.agent_compression is not None:
            if set(self.agent_compression) != set(AGENTS):
                raise ValueError("逐 Agent 压缩配置必须包含且仅包含五个研究 Agent")
            for settings in self.agent_compression.values():
                if settings.enabled and (settings.reference_materials or settings.artifact_offload) and self.history_read_max_calls == 0:
                    raise ValueError("资料引用化或外置要求开启有界历史回读")
                if (settings.enabled and settings.semantic_verification and self.summary_lifecycle is not None
                        and not self.summary_lifecycle.claimify_enabled):
                    raise ValueError("逐 Agent 摘要核验要求开启 Claimify 链路")
        self.summary_budget.available(self.summary_output_tokens)
        return self

    def thresholds(self, budget: int) -> tuple[int, int]:
        soft = min(self.soft_token_cap, int(self.soft_ratio * budget))
        summary = int(self.summary_ratio * budget)
        if not 0 < soft < summary <= budget:
            raise ValueError("当前预算下触发阈值不满足 0 < T_soft < T_summary <= B")
        return soft, summary

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.config_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def config_payload(self) -> dict:
        data = self.model_dump()
        if "compression_enabled" not in self.model_fields_set:
            data.pop("compression_enabled")  # 老快照保持哈希和既有行为。
        if "session_compression_enabled" not in self.model_fields_set:
            data.pop("session_compression_enabled")
        if self.agent_compression is None and "agent_compression" not in self.model_fields_set:
            data.pop("agent_compression")  # 已有冻结运行不改变哈希或行为。
        if self.input_history is None and "input_history" not in self.model_fields_set:
            data.pop("input_history")  # 原有冻结快照保持原哈希与原行为。
        if self.summary_lifecycle is None and "summary_lifecycle" not in self.model_fields_set:
            data.pop("summary_lifecycle")  # 老快照继续使用原有提交协议与哈希。
        elif self.summary_lifecycle is not None and "atomic_citations" not in self.summary_lifecycle.model_fields_set:
            data['summary_lifecycle'].pop('atomic_citations')  # 旧生命周期快照保持原哈希。
        if self.verification is None:
            data.pop("verification")
        # 旧快照没有 nli 字段，不能因新增可选字段改变其哈希。
        if self.nli is None:
            data.pop("nli")
        elif self._legacy_nli_review_config and "review_confidence_threshold" not in self.nli.model_fields_set:
            # 恢复旧快照时保留历史哈希；复核标记使用固定默认值，不改变原放行条件。
            data["nli"].pop("review_confidence_threshold")
        if self.nli is not None and self.nli.model_id != "qwen3-max":
            for key in ("output_tokens", "safety_tokens"):
                if key not in self.nli.model_fields_set:
                    data["nli"].pop(key)  # 旧冻结快照保持哈希，不偷偷改用新模型。
        return data

    @property
    def effective_session_compression_enabled(self) -> bool:
        return self.compression_enabled and self.session_compression_enabled is not False

    def snapshot(self) -> dict:
        return {"config": self.config_payload(), "config_hash": self.fingerprint}

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "ContextPolicy":
        if set(snapshot) != {"config", "config_hash"}:
            raise ValueError("上下文策略快照字段不完整")
        policy = cls.model_validate(snapshot["config"])
        if policy.nli is not None and "review_confidence_threshold" not in snapshot["config"]["nli"]:
            policy._legacy_nli_review_config = True
        if policy.fingerprint != snapshot["config_hash"]:
            raise ValueError("上下文策略快照哈希不匹配")
        return policy

    def blockers(self) -> list[str]:
        if not self.compression_enabled:
            return []
        reasons = []
        nli_required = self.verification is None or self.verification.nli_support
        verification_enabled = (self.semantic_verification if self.agent_compression is None else
                                any(s.enabled and s.semantic_verification for s in self.agent_compression.values()))
        if verification_enabled and (self.summary_strategy not in {"split-citations-v4", "chunk-merge-v1", "claimify-v1"}
                                          or (nli_required and self.nli is None)):
            reasons.append("新增核验需要引用分离策略及启用NLI时的冻结模型配置")
        return reasons

    def require_supported(self) -> None:
        reasons = self.blockers()
        if reasons:
            raise UnsupportedContextPolicy(f"实验组 {self.group} 尚不可执行：" + "；".join(reasons))

    def with_agent_compression(self, overrides: dict) -> "ContextPolicy":
        if not isinstance(overrides, dict) or set(overrides) - set(AGENTS):
            raise ValueError("逐 Agent 压缩配置包含未知 Agent")
        settings = {}
        for name in AGENTS:
            if self.agent_compression is not None:
                settings[name] = self.agent_compression[name].model_dump()
            else:
                settings[name] = dict(enabled=name in self.replay_agents,
                    reference_materials=bool(self.input_history and self.input_history.reference_materials),
                    artifact_offload=self.artifact_offload, semantic_summary=self.semantic_summary,
                    semantic_verification=self.semantic_verification)
        for name, value in overrides.items():
            patch = (value.model_dump(exclude_unset=True) if isinstance(value, AgentCompressionSettings)
                     else AgentCompressionSettings.model_validate(value).model_dump(exclude_unset=True))
            settings[name].update(patch)
        payload = self.config_payload()
        payload["agent_compression"] = settings
        result = ContextPolicy.model_validate(payload)
        result._legacy_nli_review_config = self._legacy_nli_review_config
        result.require_supported()
        return result


class AgentPolicyView:
    """执行时按 Agent 选择方法，配置哈希/快照仍指向整个冻结策略。"""
    def __init__(self, policy, name):
        self.policy, self.name = policy, name
        settings = (policy.agent_compression[name] if policy.agent_compression is not None else
                    AgentCompressionSettings(enabled=name in policy.replay_agents,
                        reference_materials=bool(policy.input_history and policy.input_history.reference_materials),
                        artifact_offload=policy.artifact_offload, semantic_summary=policy.semantic_summary,
                        semantic_verification=policy.semantic_verification))
        self.replay_agents = [name] if settings.enabled else []
        methods_enabled = settings.enabled and policy.compression_enabled
        # 有效功能开关；具体记录是否外置，由输入组装时的触发条件决定。
        self.artifact_offload = methods_enabled and settings.artifact_offload
        self.semantic_summary = methods_enabled and settings.semantic_summary
        self.semantic_verification = self.semantic_summary and settings.semantic_verification
        self.history_read_max_calls = policy.history_read_max_calls if settings.enabled else 0
        self.input_history = ((policy.input_history or InputHistorySettings()).model_copy(update={"reference_materials": True})
                              if methods_enabled and settings.reference_materials else None)
        self.early_offload = self.artifact_offload and policy.early_offload
        self.recent_offload = self.artifact_offload and policy.recent_offload

    def __getattr__(self, key):
        return getattr(self.policy, key)


class UnsupportedContextPolicy(ValueError):
    pass


def legacy_policy() -> ContextPolicy:
    """创建运行时冻结部署配置；已有快照不随新默认改变。"""
    budget = InputBudget(
        context_window=int(os.getenv("AGENT_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW))),
        max_input=int(os.getenv("AGENT_CONTEXT_MAX_INPUT", str(DEFAULT_MAX_INPUT))),
        safety_tokens=int(os.getenv("AGENT_CONTEXT_SAFETY_TOKENS", str(DEFAULT_SAFETY_TOKENS))),
    )
    from service.llm_client import default_max_output_tokens
    strategy = os.getenv("AGENT_SUMMARY_STRATEGY", "claimify-v1")
    lifecycle = SummaryLifecycleSettings(
        claimify_enabled=os.getenv("AGENT_SUMMARY_CLAIMIFY_ENABLED", "0") == "1",
        **({'atomic_citations': os.getenv('AGENT_SUMMARY_ATOMIC_CITATIONS_ENABLED') == '1'}
           if 'AGENT_SUMMARY_ATOMIC_CITATIONS_ENABLED' in os.environ else {}),
        generation_timeout_seconds=float(os.getenv("AGENT_SUMMARY_GENERATION_TIMEOUT_SECONDS", "120")),
        claimify_timeout_seconds=float(os.getenv("AGENT_SUMMARY_CLAIMIFY_TIMEOUT_SECONDS", "300")),
    ) if strategy == "claimify-v1" else None
    summary_enabled = os.getenv("AGENT_HISTORY_SUMMARY_ENABLED", "1") == "1"
    audit_enabled = lifecycle is None or lifecycle.claimify_enabled
    checks_enabled = summary_enabled and audit_enabled and strategy in {"split-citations-v4", "chunk-merge-v1", "claimify-v1"} and os.getenv("AGENT_SUMMARY_CHECKS_ENABLED", "0") == "1"
    nli_enabled = summary_enabled and audit_enabled and strategy in {"split-citations-v4", "chunk-merge-v1", "claimify-v1"} and os.getenv("AGENT_SUMMARY_NLI_ENABLED", "1") == "1"
    return ContextPolicy(
        policy_version="legacy-runtime-v1", group="原系统基线",
        compression_enabled=_environment_compression_enabled(),
        session_compression_enabled=_environment_switch("SESSION_CONTEXT_COMPRESSION_ENABLED"),
        semantic_summary=summary_enabled,
        semantic_verification=nli_enabled or checks_enabled, summary_strategy=strategy,
        summary_lifecycle=lifecycle,
        input_history=InputHistorySettings(
            min_reference_tokens=int(os.getenv("AGENT_INPUT_HISTORY_REFERENCE_TOKENS", "128")),
        ) if os.getenv("AGENT_INPUT_HISTORY_ENABLED", "1") == "1" else None,
        verification=VerificationSettings(
            nli_support=nli_enabled,
            coverage=os.getenv("AGENT_SUMMARY_COVERAGE_ENABLED", "0") == "1",
            key_fields=os.getenv("AGENT_SUMMARY_FIELDS_ENABLED", "1") == "1",
            evidence_context=os.getenv("AGENT_SUMMARY_EVIDENCE_CONTEXT", "1") == "1",
        ) if checks_enabled else None,
        nli=runtime_nli_settings() if nli_enabled else None,
        replay_agents=[x.strip() for x in os.getenv("AGENT_HISTORY_REPLAY", "LeadWriter").split(",") if x.strip()],
        stage_budgets={name: budget for name in AGENTS}, summary_budget=budget,
        output_limits={name: default_max_output_tokens(name) for name in AGENTS},
        summary_output_tokens=min(3000, int(os.getenv("AGENT_SUMMARY_OUTPUT_TOKENS", "3000"))),
        summary_max_chunks=int(os.getenv("AGENT_SUMMARY_MAX_CHUNKS", "8")),
    )


def read_policy(path: str | Path) -> ContextPolicy:
    """文件内数值优先且完整，不与环境变量隐式合并。"""
    path = Path(path)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    value = json.loads(path.read_text())
    if isinstance(value, dict) and set(value) == {"config", "config_hash"}:
        return ContextPolicy.from_snapshot(value)
    if value == {"baseline": "capture-current-runtime", "group": "原系统基线"}:
        return legacy_policy().model_copy(update={"group": "当前运行捕获（非改造前基线）"})
    return ContextPolicy.model_validate(value)


def selected_policy() -> ContextPolicy:
    path = os.getenv("CONTEXT_ABLATION_CONFIG", "").strip()
    policy = read_policy(path) if path else legacy_policy()
    # 显式消融文件优先；旧预设和旧冻结快照不受新环境设置影响。
    overrides = os.getenv("AGENT_COMPRESSION_CONFIG", "").strip()
    if not path and overrides:
        policy = policy.with_agent_compression(json.loads(overrides))
    policy.require_supported()
    return policy


def policy_for_scope(scope, agent_name=None):
    snapshot = getattr(scope, "context_policy", None)
    policy = ContextPolicy.from_snapshot(snapshot) if snapshot else legacy_policy()
    policy.require_supported()
    if agent_name in AGENTS and (policy.agent_compression is not None or not policy.compression_enabled):
        return AgentPolicyView(policy, agent_name)
    return policy


def storage_context_key(scope) -> str:
    """run、agent、子任务之外，再按实验配置隔离所有历史及派生状态。"""
    snapshot = getattr(scope, "context_policy", None)
    if not snapshot:
        return scope.context_key
    policy = ContextPolicy.from_snapshot(snapshot)
    if policy.policy_version == "legacy-runtime-v1":
        return scope.context_key
    subtask = hashlib.sha256(scope.context_key.encode()).hexdigest()[:32]
    return f"exp:{policy.fingerprint}:{subtask}"
