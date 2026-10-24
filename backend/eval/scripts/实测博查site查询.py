"""直接检验博查 query 中的 site:；只保存评测文件，不调用业务工作流。"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[3]
TZ = ZoneInfo("Asia/Shanghai")
ENDPOINT = "https://api.bocha.cn/v1/web-search"


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def belongs_to(url, domain):
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return host == domain or host.endswith("." + domain)


def main():
    directory = ROOT / "backend/eval/data" / (
        "博查site语法实测-" + datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    )
    directory.mkdir(parents=True, exist_ok=False)
    key = os.environ.get("BOCHA_API_KEY") or dotenv_values(
        ROOT / "backend/.env", interpolate=False
    ).get("BOCHA_API_KEY")

    def save(name, value):
        content = json.dumps(value, ensure_ascii=False, indent=2)
        if key:
            content = content.replace(key, "[REDACTED]")
        (directory / name).write_text(content + "\n", encoding="utf-8")

    cases = []
    for label, keywords, domain in [
        ("政府网站", "人工智能", "gov.cn"),
        ("世卫组织", "diabetes", "who.int"),
        ("医学文献", "type 2 diabetes", "pubmed.ncbi.nlm.nih.gov"),
    ]:
        for variant, query in [
            ("无域名限制", keywords),
            ("site前置", f"site:{domain} {keywords}"),
            ("site后置", f"{keywords} site:{domain}"),
        ]:
            cases.append(dict(label=label, variant=variant, query=query,
                              domain=domain, negative_control=False))
    for variant, query in [
        ("site前置", "site:bocha-site-probe.invalid 人工智能"),
        ("site后置", "diabetes site:bocha-site-probe.invalid"),
    ]:
        cases.append(dict(label="不存在域名", variant=variant, query=query,
                          domain="bocha-site-probe.invalid", negative_control=True))

    manifest = dict(
        started_at=now(), endpoint=ENDPOINT, planned_requests=len(cases),
        parameters=dict(count=10, freshness="noLimit", summary=True),
        local_filter_applied=False, local_cache_used=False, auto_retry=False,
        credential_source="BOCHA_API_KEY 环境变量或 backend/.env（不保存值）",
        hostname_rule="目标域名本身或以 '.' + 目标域名结尾的子域名",
        cases=cases, results=[], status="running",
    )
    save("运行清单.json", manifest)
    print(json.dumps({"output_directory": str(directory)}, ensure_ascii=False), flush=True)
    if not key:
        manifest.update(status="blocked_missing_credential", finished_at=now())
        save("运行清单.json", manifest)
        print("未找到 BOCHA_API_KEY，未发送请求。", flush=True)
        return

    # 优先沿用 HTTPS 代理，避免同时初始化不用于本次 HTTPS 请求的 SOCKS 代理。
    https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    try:
        client = httpx.Client(timeout=httpx.Timeout(30, connect=10), trust_env=True,
                              proxy=https_proxy, follow_redirects=False)
    except Exception as exc:
        manifest.update(status="blocked_client_initialization", finished_at=now(),
                        error_type=type(exc).__name__)
        save("运行清单.json", manifest)
        print("HTTP 客户端初始化失败，未发送请求。", flush=True)
        return
    with client:
        for index, case in enumerate(cases, start=1):
            filename = f"{index:02d}-{case['label']}-{case['variant']}-原始响应.json"
            payload = dict(query=case["query"], **manifest["parameters"])
            record = dict(index=index, **case, started_at=now(), request=payload)
            started = time.monotonic()
            stop = False
            try:
                response = client.post(ENDPOINT, headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                }, json=payload)
                record["http_status"] = response.status_code
                try:
                    body = response.json()
                except ValueError:
                    body = {"unparsed_response": response.text}
                record["response_body"] = body
                code = str(body.get("code", "200")) if isinstance(body, dict) else "invalid"
                record["api_code"] = code
                success = response.status_code == 200 and code == "200"
                data = body.get("data", body) if isinstance(body, dict) else {}
                data = data if isinstance(data, dict) else {}
                pages = data.get("webPages") or {}
                rows = pages.get("value") or [] if isinstance(pages, dict) else []
                if not isinstance(rows, list):
                    success = False
                    rows = []
                urls = [row.get("url", "") if isinstance(row, dict) else "" for row in rows]
                matching = [url for url in urls if belongs_to(url, case["domain"])]
                unexpected = [url for url in urls if not belongs_to(url, case["domain"])]
                record.update(
                    success=success, result_count=len(urls), unique_url_count=len(set(urls)),
                    matching_count=len(matching), other_count=len(unexpected),
                    matching_ratio=len(matching) / len(urls) if urls else None,
                    urls=urls, other_urls=unexpected,
                    negative_control_passed=success and not urls if case["negative_control"] else None,
                    estimated_matches=pages.get("totalEstimatedMatches") if isinstance(pages, dict) else None,
                )
                if response.status_code in (401, 402, 403, 429) or code in ("401", "402", "403", "429"):
                    stop = True
                    record["stop_reason"] = "认证、配额或限流错误，停止后续请求"
            except httpx.RequestError as exc:
                record.update(success=False, error_type=type(exc).__name__,
                              stop_reason="连接或请求失败，停止后续请求；未记录可能含敏感信息的异常文本")
                stop = True
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["finished_at"] = now()
            save(filename, record)
            summary = {k: v for k, v in record.items() if k not in ("response_body", "request")}
            summary["raw_file"] = filename
            manifest["results"].append(summary)
            save("运行清单.json", manifest)
            print(json.dumps({k: summary.get(k) for k in (
                "index", "label", "variant", "success", "http_status", "result_count",
                "matching_count", "other_count", "elapsed_seconds", "error_type"
            )}, ensure_ascii=False), flush=True)
            if stop:
                manifest["status"] = "stopped_on_error"
                break
            if index < len(cases):
                time.sleep(1)
    if manifest["status"] == "running":
        manifest["status"] = "completed"
    manifest["finished_at"] = now()
    save("运行清单.json", manifest)
    print(json.dumps({"status": manifest["status"], "attempted_requests": len(manifest["results"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
