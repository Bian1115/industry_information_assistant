# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""联网搜索服务

支持博查(Bocha)与 Serper 两个供应商，**统一采用博查的数据格式**作为内部标准：

    {"data": {"webPages": {"value": [
        {"name", "url", "snippet", "summary", "siteName", "datePublished"}
    ]}}}

之所以选博查格式而不是 Serper 格式，是因为它多带了 siteName（来源站点）和
datePublished（发布时间）两个字段——行业资讯类问答里，信息的新旧和出处很重要。
Serper 的响应会在 _search_serper 内部被翻译成上面这个形状，因此上层代码
只需要认识一套字段名。

默认优先用博查：项目里 BOCHA_API_KEY 已配置且跑通（行业资讯采集在用），
而 Serper 走 google.serper.dev，国内网络常不可达。
"""

import http.client
import json
import logging
import os
from typing import Dict, Any, List
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _empty_result() -> Dict[str, Any]:
    """空的博查格式响应"""
    return {"data": {"webPages": {"value": []}}}


class WebSearchService:
    """联网搜索服务（对外统一输出博查格式）"""

    def __init__(self, api_key: str = None, provider: str = None):
        """
        Args:
            api_key: 搜索服务的 API Key。不传则按 provider 从环境变量读取。
            provider: "bocha" 或 "serper"。不传则自动选择——
                      有博查 key 就用博查，否则退回 Serper。
        """
        bocha_key = os.environ.get('BOCHA_API_KEY', '')
        serper_key = os.environ.get('SERPER_API_KEY', '')

        # 显式传入的 key 一律按博查处理（当前调用方只会传博查 key）
        self.provider = provider or ('bocha' if (api_key or bocha_key) else 'serper')

        if self.provider == 'bocha':
            self.api_key = api_key or bocha_key
            self.host = "api.bochaai.com"
            self.headers = {
                'Authorization': f"Bearer {self.api_key}",
                'Content-Type': 'application/json'
            }
        else:
            self.api_key = api_key or serper_key
            self.host = "google.serper.dev"
            self.headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

        if not self.api_key:
            logger.warning(f"WebSearchService: {self.provider} 未配置 API Key，联网搜索不可用")

    # ------------------------------------------------------------------ #
    # 对外入口
    # ------------------------------------------------------------------ #

    def search(self,
              query: str,
              gl: str = "cn",
              hl: str = "zh-cn",
              autocorrect: bool = True,
              page: int = 1,
              search_type: str = "search") -> Dict[str, Any]:
        """执行联网搜索，统一返回博查格式

        Args:
            query: 查询文本
            gl / hl / autocorrect / search_type: 仅 Serper 使用，博查会忽略
            page: 页码，两者通用

        Returns:
            博查格式的结果字典；失败时为 {"error": True, "message": ...}
        """
        if not self.api_key:
            return {"error": True, "message": f"{self.provider} API Key 未配置"}

        if self.provider == 'bocha':
            return self._search_bocha(query=query, page=page)
        return self._search_serper(
            query=query, gl=gl, hl=hl,
            autocorrect=autocorrect, page=page, search_type=search_type,
        )

    def extract_search_results(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从博查格式的响应中抽出网页列表

        Args:
            search_results: search() 的返回值

        Returns:
            条目列表，字段沿用博查命名：
            name / url / snippet / summary / siteName / datePublished
        """
        if not search_results or search_results.get("error"):
            return []

        value_list = (
            search_results.get("data", {})
                          .get("webPages", {})
                          .get("value", [])
        )
        if not isinstance(value_list, list):
            logger.warning(f"webPages.value 不是列表: {type(value_list)}")
            return []

        results = []
        for item in value_list:
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            summary = item.get("summary", "")
            if not url or not (snippet or summary):
                continue
            results.append({
                "name": item.get("name", ""),
                "url": url,
                "snippet": snippet,
                "summary": summary,
                "siteName": item.get("siteName", ""),
                "datePublished": item.get("datePublished", ""),
            })
        return results

    # ------------------------------------------------------------------ #
    # 供应商实现
    # ------------------------------------------------------------------ #

    def _search_bocha(self, query: str, count: int = 10, page: int = 1) -> Dict[str, Any]:
        """调用博查 web-search 接口。响应本身就是内部标准格式，原样返回。

        供保留的通用网页研究流程使用；医学文献流程使用专用数据库适配器。
        """
        conn = http.client.HTTPSConnection(self.host, timeout=30)
        payload = json.dumps({
            "query": query,
            "summary": True,
            "count": count,
            "page": page,
        })

        try:
            conn.request("POST", "/v1/web-search", payload, self.headers)
            res = conn.getresponse()
            raw = res.read()

            if res.status != 200:
                logger.error(f"博查搜索失败: HTTP {res.status} {raw[:200]}")
                return {"error": True, "message": f"Bocha search failed: HTTP {res.status}"}

            data = json.loads(raw.decode("utf-8"))
            count_got = len(data.get("data", {}).get("webPages", {}).get("value", []) or [])
            logger.info(f"博查搜索 '{query}' 返回 {count_got} 条结果")
            return data

        except Exception as e:
            logger.error(f"博查搜索异常: {e}")
            return {"error": True, "message": f"Bocha search failed: {str(e)}"}
        finally:
            conn.close()

    def _search_serper(self, query: str, gl: str, hl: str,
                       autocorrect: bool, page: int, search_type: str) -> Dict[str, Any]:
        """调用 Serper (google.serper.dev)，并把响应翻译成博查格式"""
        conn = http.client.HTTPSConnection(self.host)
        payload = json.dumps({
            "q": query,
            "gl": gl,
            "hl": hl,
            "autocorrect": autocorrect,
            "page": page,
            "type": search_type
        })

        try:
            conn.request("POST", "/search", payload, self.headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))

            if data.get("error"):
                return {"error": True, "message": data.get("message", "Serper search failed")}

            return self._serper_to_bocha(data)

        except Exception as e:
            logger.error(f"Serper 搜索异常: {e}")
            return {"error": True, "message": f"Search failed: {str(e)}"}
        finally:
            conn.close()

    @staticmethod
    def _serper_to_bocha(data: Dict[str, Any]) -> Dict[str, Any]:
        """把 Serper 响应翻译成博查格式

        Serper 没有 siteName，用链接的域名代替；发布时间取它可选的 date 字段。
        知识面板(knowledgeGraph)作为第一条网页结果并入，避免信息丢失。
        """
        value: List[Dict[str, Any]] = []

        kg = data.get("knowledgeGraph")
        if isinstance(kg, dict) and kg.get("description"):
            link = kg.get("descriptionLink", "")
            value.append({
                "name": kg.get("title", ""),
                "url": link,
                "snippet": kg.get("description", ""),
                "summary": kg.get("description", ""),
                "siteName": kg.get("descriptionSource", "") or _host_of(link),
                "datePublished": "",
            })

        for item in data.get("organic", []) or []:
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            value.append({
                "name": item.get("title", ""),
                "url": link,
                "snippet": snippet,
                "summary": snippet,          # Serper 没有长摘要，用 snippet 顶上
                "siteName": _host_of(link),
                "datePublished": item.get("date", ""),
            })

        return {"data": {"webPages": {"value": value}}}


def _host_of(url: str) -> str:
    """从 URL 里取域名，作为来源站点名的兜底"""
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""
