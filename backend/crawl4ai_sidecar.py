"""独立 Crawl4AI HTTP sidecar。

使用单独虚拟环境运行，避免 Playwright/LiteLLM 等依赖污染主后端。
接口形状与官方 Crawl4AI Docker API 保持项目所需的最小兼容面。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig


class CrawlRequest(BaseModel):
    urls: List[str] = Field(min_length=1, max_length=8)
    browser_config: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Industry Crawl4AI Sidecar", docs_url=None, redoc_url=None)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/crawl")
async def crawl(request: CrawlRequest) -> Dict[str, Any]:
    browser_kwargs: Dict[str, Any] = {"headless": True}
    proxy = request.browser_config.get("proxy")
    if proxy:
        browser_kwargs["proxy_config"] = {"server": str(proxy)}
    timeout_ms = max(5_000, int(os.getenv("CRAWL4AI_PAGE_TIMEOUT_MS", "30000")))
    results = []
    try:
        async with AsyncWebCrawler(config=BrowserConfig(**browser_kwargs)) as crawler:
            for url in request.urls:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(page_timeout=timeout_ms),
                )
                markdown = getattr(result, "markdown", "")
                if hasattr(markdown, "raw_markdown"):
                    markdown = markdown.raw_markdown
                results.append({
                    "url": url,
                    "success": bool(getattr(result, "success", markdown)),
                    # 主后端从原始浏览器 DOM 统一提取正文；Markdown 仅保留兼容。
                    "html": str(getattr(result, "html", "") or ""),
                    "markdown": {"raw_markdown": str(markdown or "")},
                    "error": str(getattr(result, "error_message", "") or ""),
                })
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"crawl failed: {exc}") from exc
    return {"results": results}
