# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""面向 DeepResearch 的分层网页/PDF 正文读取服务。

读取顺序：
1. httpx 下载 + trafilatura 提取静态 HTML；
2. PyPDF2 提取文本型 PDF；
3. 静态正文不足时，可选使用本地 Crawl4AI（Python 包或 sidecar）。

服务不依赖任何新增付费 API。httpx 默认继承 HTTP(S)_PROXY，Crawl4AI
使用 ``WEB_READER_BROWSER_PROXY`` 显式配置浏览器代理。
"""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import tempfile
import time
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from service.web_document import QUALITY_VERSION, assess_body, content_hash
from service.web_document_structure import limit_body_preserving_tables
from service.web_content_selection import enrich_html_body

logger = logging.getLogger(__name__)
HTML_EXTRACTION_VERSION = "semantic-body-source-tables-v5"

WEB_READER_FAILURE_CATEGORIES = (
    "http_403",
    "http_429",
    "timeout",
    "dynamic_page",
    "content_too_short",
    "pdf_no_text_layer",
    "http_404",
    "network_error",
    "http_other",
    "invalid_url",
    "response_too_large",
    "redirect_error",
    "out_of_scope",
    "extraction_error",
    "body_quality_rejected",
    "disabled",
    "unknown",
)

try:
    import trafilatura
except ImportError:  # pragma: no cover - requirements.txt 已包含
    trafilatura = None

try:
    from bs4 import UnicodeDammit
except ImportError:  # pragma: no cover - requirements.txt 已包含
    UnicodeDammit = None

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover - requirements.txt 已包含
    PdfReader = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class WebReadResult:
    url: str
    final_url: str
    title: str = ""
    content: str = ""
    published_at: str = ""
    content_type: str = ""
    extractor: str = ""
    fetch_status: str = "failed"
    failure_category: Optional[str] = None
    http_status: Optional[int] = None
    recovery_action: str = ""
    attempts: int = 1
    error: Optional[str] = None
    from_cache: bool = False
    fetched_at: float = 0.0
    body_quality: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.fetch_status == "success" and bool(self.content.strip())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UnsafeURL(ValueError):
    """URL 指向本机、内网或不支持的协议。"""


class WebReaderService:
    """静态优先、浏览器兜底的统一正文读取服务。"""

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36 "
        "IndustryResearchReader/1.0"
    )

    def __init__(self) -> None:
        self.enabled = _env_bool("WEB_READER_ENABLED", True)
        self.timeout = _env_int("WEB_READER_TIMEOUT_SECONDS", 15)
        self.max_bytes = _env_int("WEB_READER_MAX_BYTES", 10 * 1024 * 1024)
        self.max_chars = _env_int("WEB_READER_MAX_CHARS", 30000)
        self.min_chars = _env_int("WEB_READER_MIN_CHARS", 500)
        self.max_redirects = _env_int("WEB_READER_MAX_REDIRECTS", 5)
        self.cache_ttl = _env_int("WEB_READER_CACHE_TTL_SECONDS", 7 * 86400)
        self.failure_ttl = _env_int("WEB_READER_FAILURE_TTL_SECONDS", 3600)
        self.crawl4ai_endpoint = os.getenv("CRAWL4AI_ENDPOINT", "").rstrip("/")
        self.ocr_endpoint = os.getenv("WEB_READER_OCR_ENDPOINT", "").rstrip("/")
        self.retry_attempts = max(1, _env_int("WEB_READER_RETRY_ATTEMPTS", 2))
        self.browser_proxy = os.getenv("WEB_READER_BROWSER_PROXY", "").strip()
        cache_default = os.path.join(tempfile.gettempdir(), "industry-web-reader-cache")
        self.cache_dir = Path(os.getenv("WEB_READER_CACHE_DIR", cache_default))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, WebReadResult] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        # 由调用方按单次研究注入的来源范围（纯域名列表）。为空表示不限制；
        # DeepResearch 主链路总会注入，因此那条路径上跳转也出不了范围。
        self.allowed_domains: Optional[List[str]] = None

    @staticmethod
    def normalize_url(url: str) -> str:
        raw = str(url or "").strip()
        parts = urlsplit(raw)
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            raise UnsafeURL("仅支持带域名的 http/https URL")
        if parts.username or parts.password:
            raise UnsafeURL("URL 不允许包含用户名或密码")
        netloc = parts.hostname.lower()
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))

    @staticmethod
    def _ip_is_unsafe(value: str) -> bool:
        ip = ipaddress.ip_address(value)
        return any((
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        ))

    def _in_allowed_scope(self, url: str) -> bool:
        """URL 是否落在调用方注入的来源范围内（含子域）。"""
        domains = self.allowed_domains
        if not domains:
            return True
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
        return any(
            host == domain or host.endswith("." + domain) for domain in domains
        )

    async def validate_url(self, url: str) -> str:
        normalized = self.normalize_url(url)
        host = urlsplit(normalized).hostname or ""
        if host.lower() == "localhost":
            raise UnsafeURL("禁止访问 localhost")
        # 每一跳都在这里过一次：首个 URL 和每个跳转目标都要重新判断，
        # 不允许因为起点合规就自动跟随到范围外的页面。
        if not self._in_allowed_scope(normalized):
            raise UnsafeURL(f"URL 超出本次研究允许的来源范围: {host}")
        try:
            if self._ip_is_unsafe(host):
                raise UnsafeURL("禁止访问内网或本机 IP")
            return normalized
        except ValueError:
            pass

        def resolve() -> Iterable[str]:
            return {
                item[4][0]
                for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            }

        try:
            addresses = await asyncio.to_thread(resolve)
        except socket.gaierror as exc:
            raise UnsafeURL(f"域名解析失败: {exc}") from exc
        if not addresses or any(self._ip_is_unsafe(address) for address in addresses):
            raise UnsafeURL("域名解析到内网、本机或保留地址")
        return normalized

    def _cache_key(self, url: str) -> str:
        # 新提取策略不复用旧 BeautifulSoup / raw_markdown 正文缓存。
        return hashlib.sha256(f"{HTML_EXTRACTION_VERSION}\n{url}".encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _save_html_snapshot(
        self, body: bytes, url: str, content_type: str, kind: str = "http_response"
    ) -> Dict[str, Any]:
        """保存提取前原始字节；浏览器 DOM 以 UTF-8 保存，引用不携带整页 HTML。"""
        digest = hashlib.sha256(body).hexdigest()
        prefix = "响应原始HTML" if kind == "http_response" else "浏览器DOM"
        directory = self.cache_dir.resolve() / "原始HTML"
        path = directory / f"{prefix}-{self._cache_key(url)}-{digest}.html"
        snapshot = {
            "kind": kind, "source_url": url, "content_type": content_type,
            "sha256": digest, "bytes": len(body), "path": "",
        }
        temporary = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(body)
            temporary.replace(path)
            snapshot["path"] = str(path)
        except OSError as exc:
            snapshot["error"] = str(exc)
            logger.warning("原始 HTML 保存失败 %s: %s", url, exc)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return snapshot

    def _sanitize_content(self, result: WebReadResult) -> WebReadResult:
        removed = result.content.count("\x00")
        if removed:
            raw_hash = content_hash(result.content)
            result.content = result.content.replace("\x00", "")
            result.body_quality = dict(result.body_quality) if isinstance(result.body_quality, dict) else {}
            result.body_quality["text_sanitization"] = {
                "version": "remove-nul-v1", "removed_null_chars": removed,
                "raw_content_sha256": raw_hash,
            }
            logger.warning("资料正文空字符清洗 source=%s removed=%d raw_sha256=%s",
                           result.final_url or result.url, removed, raw_hash)
        return result

    def _validate_body(self, result: WebReadResult, page_title: str = "") -> WebReadResult:
        # 覆盖旧缓存和浏览器/OCR 兜底路径，先清洗再核对正文哈希。
        self._sanitize_content(result)
        if (result.fetch_status == "success" and isinstance(result.body_quality, dict)
                and result.body_quality.get("text_sanitization") and result.content.strip()
                and len(result.content.strip()) < self.min_chars):
            result.fetch_status = "failed"
            result.failure_category = "content_too_short"
            result.error = f"清洗后正文过短：{len(result.content.strip())} 字符，最低要求 {self.min_chars}"
        # HTTP 200的短验证码页也应记录明确拒绝原因；正常短文仍保持读取失败。
        check_short_body = bool(result.content.strip() and result.failure_category in {
            "content_too_short", "dynamic_page"})
        if not result.success and not check_short_body:
            if result.fetch_status == "success":
                result.fetch_status = "failed"
                result.failure_category = "body_quality_rejected"
                result.error = "empty_body"
            return result
        if not isinstance(result.body_quality, dict):
            result.body_quality = {}
        if (result.body_quality.get("quality_version") != QUALITY_VERSION
                or result.body_quality.get("content_sha256") != content_hash(result.content)):
            sanitation = result.body_quality.get("text_sanitization")
            snapshots = result.body_quality.get("html_snapshots")
            page_title = page_title or result.body_quality.get("page_title", "")
            result.content, result.body_quality = assess_body(result.content, page_title or result.title)
            if sanitation:
                result.body_quality["text_sanitization"] = sanitation
            if snapshots:
                result.body_quality["html_snapshots"] = snapshots
            if page_title:
                result.body_quality["page_title"] = page_title
                target = re.sub(r"\W", "", result.title.lower())
                actual = re.sub(r"\W", "", page_title.lower())
                target_terms = set(re.findall(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2}", target))
                actual_terms = set(re.findall(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2}", actual))
                if (result.body_quality["status"] == "passed" and target_terms and actual_terms
                        and target not in actual and actual not in target and not target_terms & actual_terms):
                    result.body_quality["status"] = "uncertain"
                    result.body_quality["reasons"].append("target_title_mismatch")
            result.body_quality["content_sha256"] = content_hash(result.content)
        if result.body_quality.get("status") == "rejected":
            result.fetch_status = "failed"
            result.failure_category = "body_quality_rejected"
            result.error = ", ".join(result.body_quality.get("reasons", []))
            logger.warning("网页正文准入拒绝 source=%s reasons=%s",
                           result.final_url or result.url, result.error)
        return result

    def _cache_expired(self, result: WebReadResult, now: float) -> bool:
        ttl = self.cache_ttl if result.success else self.failure_ttl
        return now >= result.fetched_at + ttl

    def _delete_expired_cache_file(self, path: Path, now: float) -> bool:
        """以磁盘版本判断过期，避免旧内存缓存删除其他读取器刚刷新的文件。"""
        try:
            if path.is_symlink():
                return False
            with path.open(encoding="utf-8") as stream:
                original_stat = os.fstat(stream.fileno())
                result = WebReadResult(**json.load(stream))
            if not self._cache_expired(result, now):
                return False
            if path.stat() != original_stat:
                return False
            path.unlink()
            return True
        except (OSError, ValueError, TypeError) as exc:
            logger.debug("网页过期缓存删除跳过: %s", exc)
            return False

    def cleanup_expired_cache(self) -> int:
        """仅清理读取器拥有的过期缓存；研究来源快照和评测归档不在此目录。"""
        now = time.time()
        for key, result in list(self._memory_cache.items()):
            if self._cache_expired(result, now):
                self._memory_cache.pop(key, None)
        removed = 0
        for path in self.cache_dir.glob("*.json"):
            if re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
                removed += self._delete_expired_cache_file(path, now)
        # 原始 HTML 与成功缓存采用相同保留期；同内容再次抓取会刷新文件时间。
        for path in (self.cache_dir / "原始HTML").glob("*.html"):
            if not re.fullmatch(r"(?:响应原始HTML|浏览器DOM)-[0-9a-f]{64}-[0-9a-f]{64}\.html", path.name):
                continue
            try:
                stat = path.stat()
                if not path.is_symlink() and now >= stat.st_mtime + self.cache_ttl and path.stat() == stat:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.debug("原始 HTML 过期清理跳过 %s: %s", path, exc)
        return removed

    def _load_cache(self, key: str) -> Optional[WebReadResult]:
        cached = self._memory_cache.get(key)
        if cached is None:
            path = self._cache_path(key)
            if not path.exists():
                return None
            try:
                cached = WebReadResult(**json.loads(path.read_text("utf-8")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return None
            self._memory_cache[key] = cached
        now = time.time()
        if self._cache_expired(cached, now):
            self._memory_cache.pop(key, None)
            self._delete_expired_cache_file(self._cache_path(key), now)
            return None
        result = WebReadResult(**cached.to_dict())
        result.from_cache = True
        return self._validate_body(result)

    def _save_cache(self, key: str, result: WebReadResult) -> None:
        self._memory_cache[key] = result
        path = self._cache_path(key)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError as exc:
            logger.debug("网页读取缓存写入失败（忽略）: %s", exc)

    async def read_url(self, url: str, title: str = "") -> WebReadResult:
        if not self.enabled:
            return WebReadResult(
                url=url, final_url=url, title=title,
                fetch_status="disabled", failure_category="disabled",
                error="WEB_READER_ENABLED=false",
                fetched_at=time.time(),
            )
        try:
            normalized = await self.validate_url(url)
        except (UnsafeURL, ValueError) as exc:
            return WebReadResult(
                url=url, final_url=url, title=title,
                fetch_status="rejected", failure_category="invalid_url",
                error=str(exc), fetched_at=time.time(),
            )

        key = self._cache_key(normalized)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._load_cache(key)
            if cached is not None:
                return cached
            result = await self._read_uncached(normalized, title)
            self._save_cache(key, result)
            return result

    async def _read_uncached(self, url: str, title: str) -> WebReadResult:
        result = WebReadResult(
            url=url, final_url=url, title=title, fetched_at=time.time()
        )
        html = ""
        page_title = ""
        is_pdf = False
        try:
            final_url, content_type, body, attempts = await self._download_with_retry(url)
            result.attempts = attempts
            result.final_url = final_url
            result.content_type = content_type
            is_pdf = "pdf" in content_type or final_url.lower().endswith(".pdf")
            if is_pdf:
                result.content = self._extract_pdf(body)
                result.extractor = "pypdf2"
            else:
                result.body_quality["html_snapshots"] = [
                    self._save_html_snapshot(body, final_url, content_type)
                ]
                html = self._decode_html(body, content_type)
                extracted = self._extract_html(html, final_url)
                page_title = extracted.get("title", "")
                result.title = result.title or extracted.get("title", "")
                result.published_at = extracted.get("published_at", "")
                result.content = extracted.get("content", "")
                result.extractor = extracted.get("extractor", "")

            if is_pdf and not result.content.strip():
                ocr_result = await self._read_with_ocr(final_url, body, title)
                if ocr_result:
                    result = ocr_result
                    result.url = url
                    result.final_url = final_url
                    result.recovery_action = "ocr"
                    result.fetched_at = time.time()

            if len(result.content.strip()) < self.min_chars and not is_pdf:
                browser_result = await self._read_with_crawl4ai(final_url, title)
                snapshots = list(result.body_quality.get("html_snapshots", []))
                if browser_result:
                    snapshots.extend(browser_result.body_quality.get("html_snapshots", []))
                if browser_result and len(browser_result.content) > len(result.content):
                    result = browser_result
                    result.url = url
                    result.final_url = final_url
                    result.recovery_action = "browser_render"
                    result.fetched_at = time.time()
                    page_title = ""
                if snapshots:
                    result.body_quality["html_snapshots"] = snapshots

            self._sanitize_content(result)
            result.content = limit_body_preserving_tables(result.content.strip(), self.max_chars)
            if len(result.content) >= self.min_chars:
                result.fetch_status = "success"
                result.failure_category = None
                result.error = None
            elif is_pdf and not result.content:
                result.failure_category = "pdf_no_text_layer"
                result.error = "PDF 未提取到文本层，可能是扫描件"
            elif not is_pdf and self._looks_dynamic_page(html, result.content):
                result.failure_category = "dynamic_page"
                result.error = "页面依赖 JavaScript 渲染，未提取到足够正文"
            else:
                result.failure_category = "content_too_short"
                result.error = (
                    f"正文过短：{len(result.content)} 字符，最低要求 {self.min_chars}"
                )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            # 403/部分5xx通常是静态客户端被拦或页面依赖浏览器。直接走浏览器
            # 兜底一次；404/无效URL不重试，避免浪费时间。
            if status == 403 or status >= 500:
                browser_result = await self._read_with_crawl4ai(url, title)
                if browser_result:
                    result.body_quality["html_snapshots"] = browser_result.body_quality.get("html_snapshots", [])
                if browser_result and len(browser_result.content.strip()) >= self.min_chars:
                    browser_result.url = url
                    browser_result.final_url = url
                    browser_result.http_status = status
                    browser_result.recovery_action = "browser_after_http_error"
                    return self._validate_body(browser_result)
            result.http_status = status
            result.failure_category = (
                f"http_{status}" if status in {403, 404, 429} else "http_other"
            )
            result.error = f"HTTP {status}: {exc.response.reason_phrase}"
        except httpx.TimeoutException as exc:
            browser_result = await self._read_with_crawl4ai(url, title)
            if browser_result:
                result.body_quality["html_snapshots"] = browser_result.body_quality.get("html_snapshots", [])
            if browser_result and len(browser_result.content.strip()) >= self.min_chars:
                browser_result.url = url
                browser_result.recovery_action = "browser_after_timeout"
                return self._validate_body(browser_result)
            result.failure_category = "timeout"
            result.error = f"请求超时: {str(exc)[:500]}"
        except httpx.RequestError as exc:
            result.failure_category = "network_error"
            result.error = f"网络请求失败: {str(exc)[:500]}"
        except Exception as exc:  # 读取失败必须降级，不能终止研究主流程
            if isinstance(exc, UnsafeURL) and "来源范围" in str(exc):
                # 跳转目标越出来源范围：记为明确的范围拒绝，不当成解析失败，
                # 便于统计"有多少候选被跳转带出了指定网站"。
                result.failure_category = "out_of_scope"
            elif isinstance(exc, ValueError) and "响应超过" in str(exc):
                result.failure_category = "response_too_large"
            elif isinstance(exc, ValueError) and "重定向" in str(exc):
                result.failure_category = "redirect_error"
            else:
                result.failure_category = "extraction_error"
            result.error = f"{type(exc).__name__}: {str(exc)[:500]}"
            logger.info("网页读取失败 %s: %s", url, result.error)
        return self._validate_body(result, page_title)

    async def _download_with_retry(self, url: str) -> tuple[str, str, bytes, int]:
        """只重试有恢复价值的传输失败；404/403等交给分类路由。"""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                final_url, content_type, body = await self._download(url)
                return final_url, content_type, body, attempt
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                retryable = status == 429 or status >= 500
                if not retryable or attempt >= self.retry_attempts:
                    raise
            except (httpx.TimeoutException, httpx.RequestError):
                if attempt >= self.retry_attempts:
                    raise
            await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError("网页重试循环未返回")

    def _looks_dynamic_page(self, html: str, extracted_content: str) -> bool:
        """识别只有前端壳、需要 JavaScript 才能得到正文的页面。"""
        if len(extracted_content.strip()) >= self.min_chars:
            return False
        lowered = html.lower()
        explicit_markers = (
            "enable javascript",
            "please enable javascript",
            "javascript is required",
            "you need to enable javascript",
            "__next_data__",
            "webpackjsonp",
            "window.__nuxt__",
        )
        if any(marker in lowered for marker in explicit_markers):
            return True
        app_shell = re.search(
            r'<(?:div|main)[^>]+id=["\'](?:root|app|__next|nuxt)["\'][^>]*>\s*</',
            lowered,
        )
        return bool(app_shell and lowered.count("<script") >= 2)

    async def _download(self, url: str) -> tuple[str, str, bytes]:
        timeout = httpx.Timeout(self.timeout, connect=min(self.timeout, 10))
        current = url
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=True,
            headers={"User-Agent": self.USER_AGENT, "Accept": "text/html,application/pdf,*/*"},
        ) as client:
            for _ in range(self.max_redirects + 1):
                current = await self.validate_url(current)
                async with client.stream("GET", current) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise httpx.HTTPStatusError(
                                "重定向缺少 Location", request=response.request,
                                response=response,
                            )
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    chunks: List[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > self.max_bytes:
                            raise ValueError(f"响应超过 {self.max_bytes} bytes")
                        chunks.append(chunk)
                    content_type = response.headers.get("content-type", "")
                    return str(response.url), content_type, b"".join(chunks)
            raise ValueError(f"重定向超过 {self.max_redirects} 次")

    @staticmethod
    def _decode_html(body: bytes, content_type: str) -> str:
        # 必须在严格解码失败后才尝试下一个声明；先 errors="replace" 会把
        # GB2312 等原始字节不可逆地变成 U+FFFD，正文提取器无法再恢复。
        candidates: List[str] = []
        for marker, charset in (
            (codecs.BOM_UTF32_LE, "utf-32"),
            (codecs.BOM_UTF32_BE, "utf-32"),
            (codecs.BOM_UTF8, "utf-8-sig"),
            (codecs.BOM_UTF16_LE, "utf-16"),
            (codecs.BOM_UTF16_BE, "utf-16"),
        ):
            if body.startswith(marker):
                candidates.append(charset)
                break
        header_charset = re.search(
            r"\bcharset\s*=\s*[\"']?\s*([a-zA-Z0-9._:-]+)", content_type, re.I
        )
        if header_charset:
            candidates.append(header_charset.group(1))
        # 未声明 HTTP 编码时，完整有效的 UTF-8 比过期的 legacy meta 更可信。
        candidates.append("utf-8")
        head = re.sub(rb"<!--.*?-->", b"", body[:8192], flags=re.S)
        for tag in re.findall(rb"<meta\b[^>]*>", head, flags=re.I):
            declared = re.search(
                rb"\bcharset\s*=\s*[\"']?\s*([a-zA-Z0-9._:-]+)", tag, re.I
            )
            if declared:
                candidates.append(declared.group(1).decode("ascii"))
        attempted = set()
        for candidate in candidates:
            try:
                charset = codecs.lookup(candidate).name
                # 中文旧网页常标 GB2312，但实际含 GBK/GB18030 扩展字符。
                if charset in {"gb2312", "gbk"}:
                    charset = "gb18030"
                if charset in attempted:
                    continue
                attempted.add(charset)
                return body.decode(charset, errors="strict")
            except (LookupError, UnicodeError):
                continue
        if UnicodeDammit is not None:
            detected = UnicodeDammit(body, is_html=True)
            if detected.unicode_markup is not None:
                return detected.unicode_markup
        # 截断或损坏的响应仍可返回部分内容；这是所有严格候选失败后的末级兜底。
        return body.decode("utf-8", errors="replace")

    def _extract_html(self, html: str, url: str) -> Dict[str, Any]:
        # Trafilatura读取原始HTML；随后按语义正文范围和源表补足遗漏。
        if trafilatura is None:
            raise RuntimeError("未安装 Trafilatura，无法提取 HTML 正文")
        content = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=True,
            output_format="markdown", no_fallback=False, favor_precision=True,
        ) or ""
        content, selection = enrich_html_body(html, content)
        metadata = trafilatura.extract_metadata(html, default_url=url, extensive=False)
        return {
            "title": str(getattr(metadata, "title", "") or ""),
            "published_at": str(getattr(metadata, "date", "") or ""),
            "content": limit_body_preserving_tables(content, self.max_chars),
            "extractor": "trafilatura+semantic-dom" if selection["profile"] not in {"article", "blocked_page"} or any(
                selection[key] for key in ["replaced_tables", "recovered_tables", "flat_tables_restored"]
            ) else "trafilatura",
            "content_selection": selection,
        }

    def _extract_pdf(self, body: bytes) -> str:
        if PdfReader is None:
            return ""
        reader = PdfReader(BytesIO(body))
        parts: List[str] = []
        for page_number, page in enumerate(reader.pages[:100], start=1):
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(f"[第{page_number}页]\n{text}")
            if sum(len(item) for item in parts) >= self.max_chars:
                break
        return "\n\n".join(parts)[:self.max_chars]

    async def _read_with_ocr(
        self, url: str, body: bytes, title: str
    ) -> Optional[WebReadResult]:
        """扫描PDF的可选OCR边车；未配置时明确保留 pdf_no_text_layer。"""
        if not self.ocr_endpoint:
            return None
        try:
            async with httpx.AsyncClient(timeout=max(self.timeout * 4, 60)) as client:
                response = await client.post(
                    f"{self.ocr_endpoint}/extract",
                    files={"file": ("document.pdf", body, "application/pdf")},
                    data={"source_url": url},
                )
                response.raise_for_status()
                data = response.json()
            content = str(data.get("text") or data.get("content") or "").strip()
            if len(content) < self.min_chars:
                return None
            return WebReadResult(
                url=url, final_url=url, title=title, content=limit_body_preserving_tables(content, self.max_chars),
                extractor="ocr-sidecar", fetch_status="success", fetched_at=time.time(),
            )
        except Exception as exc:
            logger.info("OCR sidecar 读取失败 %s: %s", url, exc)
            return None

    async def _read_with_crawl4ai(
        self, url: str, title: str
    ) -> Optional[WebReadResult]:
        if self.crawl4ai_endpoint:
            try:
                payload: Dict[str, Any] = {"urls": [url]}
                if self.browser_proxy:
                    payload["browser_config"] = {"proxy": self.browser_proxy}
                async with httpx.AsyncClient(timeout=max(self.timeout * 2, 30)) as client:
                    response = await client.post(
                        f"{self.crawl4ai_endpoint}/crawl", json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
                item = self._crawl4ai_item(data)
                return self._extract_browser_html(
                    str(item.get("html") or ""), url, title, "crawl4ai-sidecar"
                )
            except Exception as exc:
                logger.info("Crawl4AI sidecar 读取失败 %s: %s", url, exc)
                return None

        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        except ImportError:
            return None
        try:
            browser_kwargs: Dict[str, Any] = {"headless": True}
            if self.browser_proxy:
                browser_kwargs["proxy_config"] = {"server": self.browser_proxy}
            async with AsyncWebCrawler(config=BrowserConfig(**browser_kwargs)) as crawler:
                crawl_result = await crawler.arun(
                    url=url, config=CrawlerRunConfig(page_timeout=self.timeout * 1000)
                )
            return self._extract_browser_html(
                str(getattr(crawl_result, "html", "") or ""), url, title, "crawl4ai"
            )
        except Exception as exc:
            logger.info("Crawl4AI 本地读取失败 %s: %s", url, exc)
            return None

    def _extract_browser_html(
        self, html: str, url: str, title: str, browser: str
    ) -> Optional[WebReadResult]:
        if not html.strip():
            logger.info("%s 未返回原始 DOM，无法使用 Trafilatura 提取 %s", browser, url)
            return None
        body = html.encode("utf-8")
        if len(body) > self.max_bytes:
            logger.info("%s DOM 超过 %s bytes，跳过提取 %s", browser, self.max_bytes, url)
            return None
        snapshot = self._save_html_snapshot(body, url, "text/html; charset=utf-8", "browser_dom")
        try:
            extracted = self._extract_html(html, url)
        except Exception as exc:
            logger.info("浏览器 DOM 正文提取失败 %s: %s", url, exc)
            return WebReadResult(
                url=url, final_url=url, title=title, extractor=f"{browser}+trafilatura",
                failure_category="extraction_error", error=str(exc), fetched_at=time.time(),
                body_quality={"html_snapshots": [snapshot]},
            )
        return WebReadResult(
            url=url, final_url=url, title=title or extracted["title"],
            published_at=extracted["published_at"], content=extracted["content"],
            content_type="text/html; charset=utf-8", extractor=f"{browser}+trafilatura",
            fetch_status="success" if extracted["content"].strip() else "failed",
            fetched_at=time.time(),
            body_quality={"page_title": extracted["title"], "html_snapshots": [snapshot]},
        )

    @staticmethod
    def _crawl4ai_item(data: Any) -> Dict[str, Any]:
        candidates = data.get("results", data) if isinstance(data, dict) else data
        if isinstance(candidates, list) and candidates:
            candidates = candidates[0]
        if not isinstance(candidates, dict):
            return {}
        return candidates

    async def read_many(
        self, items: List[Dict[str, Any]], limit: int = 3, concurrency: int = 2
    ) -> List[WebReadResult]:
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            url = str(item.get("url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            unique.append(item)
            if len(unique) >= limit:
                break
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def read(item: Dict[str, Any]) -> WebReadResult:
            async with semaphore:
                return await self.read_url(
                    str(item.get("url", "")), str(item.get("title", ""))
                )

        return await asyncio.gather(*(read(item) for item in unique))
