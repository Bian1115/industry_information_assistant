# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""研究来源范围：默认全网搜索，可配置域名限制。

模型提供关键词，网站范围固定保存在运行状态里；空列表表示全网。
指定域名时使用 site 前置检索，并对结果、待读取网页和引用进行本地过滤。
全网模式允许各网站的 HTTP/HTTPS 来源，读取器仍负责访问安全检查。
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, List, Optional
from urllib.parse import urlsplit


# 模型在关键词里自带的 site:/-site: 条件一律清除，避免与后端范围冲突。
SITE_FILTER = re.compile(r"(?i)(?<!\S)-?site:\S+")

# 纯域名，不接受协议、路径、端口、用户信息、通配符或 IP 字面量。
# 顶级标签必须是字母，因此 1.2.3.4 这类 IP 会被拒绝：按 IP 授权来源范围
# 无法表达"某个网站"，也绕不过反向代理和共享主机。
DOMAIN_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?!-)[a-z0-9-]{1,63}(?<!-)"
    r"(?:\.(?!-)[a-z0-9-]{1,63}(?<!-))*"
    r"\.[a-z]{2,63}$"
)

ALLOWED_SCHEMES = {"http", "https"}


class SourceScopeError(ValueError):
    """来源范围配置或使用不合法。"""


def normalize_domain(value: Any) -> str:
    """把配置值收敛成可比较的纯域名，例如 ``who.int``。

    允许运维在配置里写成 ``https://www.who.int/`` 这类顺手的形式，但结果
    必须是一个真实域名；空值、通配符、IP、端口和路径都视为配置错误。
    """
    raw = str(value or "").strip().lower().rstrip(".")
    if not raw:
        raise SourceScopeError("允许搜索的网站不能为空")
    if "://" in raw:
        raw = urlsplit(raw).hostname or ""
    else:
        # 没有协议时 urlsplit 会把整串当成 path，这里先手工剥掉路径和端口。
        raw = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        if "@" in raw:
            raise SourceScopeError(f"允许搜索的网站不能包含用户信息: {value!r}")
        raw = raw.split(":", 1)[0]
    raw = raw.strip().rstrip(".")
    if raw.startswith("*."):
        # 子域已由 belongs_to 覆盖，不需要也不接受通配符写法。
        raise SourceScopeError(f"允许搜索的网站不支持通配符写法: {value!r}")
    # who.int 通过子域匹配已覆盖 www.who.int，反过来不成立。这里不代为去掉
    # www 前缀：gov.cn 这类后缀被自动放宽后，来源范围会从一个网站悄悄扩成
    # 整个后缀下的所有站点。要求运维显式写裸域名，范围只能由配置决定。
    if raw.startswith("www."):
        raise SourceScopeError(
            f"请配置裸域名而不是 www 形式（子域已自动覆盖）: {value!r}"
        )
    if not DOMAIN_PATTERN.match(raw):
        raise SourceScopeError(f"不是合法的网站域名: {value!r}")
    return raw


def normalize_allowed_domains(values: Optional[Iterable[Any]]) -> List[str]:
    """规范化并去重，保持配置顺序，便于逐站搜索的结果可复现。"""
    result: List[str] = []
    for value in values or ():
        domain = normalize_domain(value)
        if domain not in result:
            result.append(domain)
    return result


def configured_allowed_domains() -> List[str]:
    """读取可选的部署范围；未配置或留空时使用全网搜索。"""
    raw = os.getenv("RESEARCH_ALLOWED_DOMAINS", "")
    domains = normalize_allowed_domains(
        part for part in re.split(r"[,\s;]+", raw) if part.strip()
    )
    return domains


def belongs_to(url: Any, domain: str) -> bool:
    """检查结果是否属于指定网站（含子域）。"""
    try:
        parsed = urlsplit(str(url or ""))
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if parsed.scheme not in ALLOWED_SCHEMES or not host:
        return False
    return host == domain or host.endswith("." + domain)


def url_in_scope(url: Any, domains: Iterable[str]) -> bool:
    """URL 是否落在来源范围内；空范围允许各网站的 HTTP/HTTPS URL。"""
    domains = list(domains or ())
    if not domains:
        try:
            parsed = urlsplit(str(url or ""))
            return parsed.scheme in ALLOWED_SCHEMES and bool(parsed.hostname)
        except ValueError:
            return False
    return any(belongs_to(url, domain) for domain in domains)


def strip_site_filters(query: Any) -> str:
    """清除模型自带的 site/-site 条件，只保留主题关键词。"""
    return re.sub(r"\s+", " ", SITE_FILTER.sub("", str(query or ""))).strip()


def scoped_query(domain: str, keywords: str) -> str:
    """site 统一前置；实测前置检索命中率显著高于后置。"""
    return f"site:{domain} {keywords}".strip()


def require_allowed_domains(state: Any) -> List[str]:
    """读取运行范围；空列表表示全网，兼容未记录范围的最小状态。"""
    return normalize_allowed_domains((state or {}).get("allowed_domains"))


def filter_results_in_scope(
    results: Iterable[Any], domains: Iterable[str]
) -> List[Any]:
    """按 URL 过滤任意结果列表，同一 URL 只保留一条。"""
    domains = list(domains or ())
    accepted: dict = {}
    for item in results or ():
        url = str((item or {}).get("url") or "").strip()
        if url and url not in accepted and url_in_scope(url, domains):
            accepted[url] = item
    return list(accepted.values())
