#!/usr/bin/env python3
"""离线统计导出工作历史中的 Crawl4AI 采用比例，保留逐条来源且拒绝覆盖。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "backend/eval/data"
FIVE = DATA / "通用研究轨迹加载成功核对与MinerU候选分析-20261017/逐条返回加载核对明细.jsonl"
TOOL = "v2_web_fulltext_read"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def norm(url):
    value = urlsplit(str(url).strip())
    return urlunsplit((value.scheme.lower(), value.netloc.lower(), value.path or "/", value.query, ""))


def rate(n, d):
    return {"分子": n, "分母": d, "比例": n / d if d else None}


def pct(value):
    return f'{value["分子"]}/{value["分母"]}＝{value["比例"] * 100:.2f}%' if value["分母"] else "无分母"


def summarize(rows):
    fresh = [r for r in rows if r["from_cache"] is False]
    adopted = [r for r in rows if r["crawl4ai_selected"]]
    fresh_adopted = [r for r in fresh if r["crawl4ai_selected"]]
    successful = [r for r in rows if r["fetch_status"] == "success"]
    used_batches = {(r["run_id"], r["call_id"]) for r in adopted}
    all_batches = {(r["run_id"], r["call_id"]) for r in rows}
    return {
        "读取返回条目数": len(rows),
        "有返回的工具批次数": len(all_batches),
        "原始记录归属分组数": len({r["run_id"] for r in rows}),
        "关联已知运行编号数": len({run_id for r in rows for run_id in r.get("associated_run_ids", [])}),
        "Crawl4AI结果采用比例": rate(len(adopted), len(rows)),
        "非缓存返回中Crawl4AI采用比例": rate(len(fresh_adopted), len(fresh)),
        "成功返回中Crawl4AI结果占比": rate(sum(r["crawl4ai_selected"] for r in successful), len(successful)),
        "Crawl4AI已采用结果的原始成功率": rate(sum(r["fetch_status"] == "success" for r in adopted), len(adopted)),
        "至少一条采用Crawl4AI的批次比例": rate(len(used_batches), len(all_batches)),
        "Crawl4AI采用结果的缓存数": sum(r["from_cache"] is True for r in adopted),
        "缓存返回数": sum(r["from_cache"] is True for r in rows),
        "缓存标记未知数": sum(r["from_cache"] is None for r in rows),
        "非缓存Crawl4AI采用但原始状态非成功数": sum(r["fetch_status"] != "success" for r in fresh_adopted),
        "读取器分布": dict(sorted(Counter(r["extractor"] or "未记录" for r in rows).items())),
        "原始状态分布": dict(sorted(Counter(r["fetch_status"] or "未记录" for r in rows).items())),
        "Crawl4AI采用结果原始状态分布": dict(sorted(Counter(r["fetch_status"] or "未记录" for r in adopted).items())),
        "浏览器恢复标记数": sum(r["recovery_action"].startswith("browser_") for r in rows),
        "浏览器恢复标记但未采用Crawl4AI数": sum(r["recovery_action"].startswith("browser_") and not r["crawl4ai_selected"] for r in rows),
    }


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--date", default="2026-10-31")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError("统计目录已经存在；请使用新的中文目录，禁止覆盖历史结果。")

    histories = sorted(p for p in DATA.rglob("*.json") if p.name == "智能体原始历史.json"
                       or p.name.endswith("工作历史.json"))
    inventory, calls, raw_records, conflicts, unsupported = [], {}, {}, [], []
    seen_copies = 0
    for index, path in enumerate(histories):
        raw_bytes = path.read_bytes()
        records = json.loads(raw_bytes)
        item = {"路径": relative(path), "字节数": len(raw_bytes), "SHA256": digest(raw_bytes),
                "原始读取结果记录数": 0, "读取工具相关记录数": 0}
        inventory.append(item)
        record_container = "root"
        if isinstance(records, dict) and isinstance(records.get("messages"), list):
            records = records["messages"]
            record_container = "messages"
        if not isinstance(records, list):
            unsupported.append({"path": relative(path), "type": type(records).__name__})
            continue
        for offset, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            payload = record.get("payload") or {}
            if not isinstance(payload, dict) or payload.get("tool_name") != TOOL:
                continue
            item["读取工具相关记录数"] += 1
            key = (TOOL, str(record.get("call_id") or ""))
            if not key[1]:
                raise ValueError(f"读取事件缺少 call_id：{path}:{offset}")
            run_id = str(record.get("run_id") or "")
            call = calls.setdefault(key, {"call_id": key[1], "kinds": set(), "sources": set(),
                                         "run_ids": set(), "context_ids": set()})
            if run_id:
                call["run_ids"].add(run_id)
            if record.get("context_id"):
                call["context_ids"].add(record["context_id"])
            call["kinds"].add(record.get("kind"))
            call["sources"].add(relative(path))
            if record.get("kind") != "tool_raw_result":
                continue
            item["原始读取结果记录数"] += 1
            pointer = {"history_path": relative(path), "record_index": offset, "seq": record.get("seq"),
                       "record_container": record_container,
                       "source_id": record.get("source_id") or record.get("id")}
            payload_hash = digest(canonical(payload))
            if key in raw_records:
                seen_copies += 1
                if raw_records[key]["payload_sha256"] != payload_hash:
                    conflicts.append({"run_id": key[0], "call_id": key[1], "other": pointer,
                                      "first": raw_records[key]["sources"][0]})
                raw_records[key]["sources"].append(pointer)
                continue
            values = payload.get("value")
            if not isinstance(values, list):
                raise TypeError(f"未知读取工具结果结构：{path}:{offset}")
            entries = []
            for position, value in enumerate(values):
                body = str(value.get("content") or "")
                extractor = str(value.get("extractor") or "")
                entry = {
                    "run_id": run_id, "call_id": key[1], "result_index": position,
                    "created_at": record.get("created_at"), "url": value.get("url"),
                    "final_url": value.get("final_url"), "extractor": extractor,
                    "crawl4ai_selected": "crawl4ai" in extractor.lower(),
                    "from_cache": value.get("from_cache"), "fetch_status": value.get("fetch_status"),
                    "failure_category": value.get("failure_category"), "http_status": value.get("http_status"),
                    "recovery_action": str(value.get("recovery_action") or ""), "error": value.get("error"),
                    "fetched_at": value.get("fetched_at"), "attempts": value.get("attempts"),
                    "content_chars": len(body), "content_sha256": digest(body.encode()),
                    "content_type": value.get("content_type"),
                    "body_quality_status": (value.get("body_quality") or {}).get("status"),
                    "text_sanitization": (value.get("body_quality") or {}).get("text_sanitization"),
                    "source": pointer, "raw_value_sha256": digest(canonical(value)),
                }
                entries.append(entry)
            raw_records[key] = {"run_id": run_id, "call_id": key[1], "created_at": record.get("created_at"),
                                "payload_sha256": payload_hash, "sources": [pointer], "entries": entries}
        if (index + 1) % 100 == 0:
            print(f"已读取 {index + 1}/{len(histories)} 份历史，已识别 {len(raw_records)} 个独立正文读取批次", flush=True)
    if conflicts:
        raise ValueError(f"同一 run_id/call_id 存在冲突结果：{conflicts[:3]}")
    for key, call in calls.items():
        # 历史恢复导出可能把同一原始调用关联到新run；payload完全一致才允许合并。
        # 保留全部归属，不把恢复导入当作再次联网。主归属仅用于分组展示。
        original_run = raw_records.get(key, {}).get("run_id")
        call["run_id"] = original_run or next(iter(sorted(call["run_ids"])), "未知运行_上下文_" + "_".join(sorted(call["context_ids"])))
        if key in raw_records:
            raw_records[key]["run_id"] = call["run_id"]
            raw_records[key]["associated_run_ids"] = sorted(call["run_ids"])
            for row in raw_records[key]["entries"]:
                row["run_id"] = call["run_id"]
                row["associated_run_ids"] = sorted(call["run_ids"])
    rows = [row for key in sorted(raw_records) for row in raw_records[key]["entries"]]
    row_lookup = {(r["run_id"], r["call_id"], r["result_index"]): r for r in rows}
    assert len(row_lookup) == len(rows)
    assert all(urlsplit(str(r["url"])).scheme in {"http", "https"} for r in rows)
    assert all(r["from_cache"] in {True, False, None} for r in rows)

    # 固定五题索引只选择原始事件，不把此前分析文件当成新增读取。
    five_bytes = FIVE.read_bytes()
    selected = [json.loads(line) for line in five_bytes.decode().splitlines() if line.strip()]
    pointer_lookup = {}
    for key, record in raw_records.items():
        for source in record["sources"]:
            pointer_lookup[(source["history_path"], key[1])] = key
    five_rows = []
    for selected_row in selected:
        key = pointer_lookup[(relative(selected_row["history_path"]), selected_row["call_id"])]
        row = row_lookup[(raw_records[key]["run_id"], key[1], selected_row["raw_result_index"])]
        assert norm(row["url"]) == norm(selected_row["url"])
        assert row["content_sha256"] == selected_row["content_sha256"]
        assert row["extractor"] == selected_row["extractor"]
        assert row["from_cache"] == selected_row["from_cache"]
        five_rows.append({**row, "topic": selected_row["topic"], "page_id": selected_row["page_id"],
                          "historical_content_usable": selected_row["content_usable"]})
    assert len(five_rows) == 302
    five_unique = list({(r["topic"], norm(r["url"])): r for r in five_rows}.values())
    assert len(five_unique) == 289

    # 五题的源码快照完全相同，才按其确定分支推断进入浏览器函数的下界。
    frozen_manifest = []
    for path in sorted({Path(r["history_path"]).parent.parent for r in selected}):
        source = path / "运行工作副本/backend/app/service/web_reader_service.py"
        blob = source.read_bytes()
        frozen_manifest.append({"路径": relative(source), "字节数": len(blob), "SHA256": digest(blob)})
    assert len({r["SHA256"] for r in frozen_manifest}) == 1
    inference_rows = []
    for row in five_rows:
        reason = None
        if row["from_cache"] is not False:
            reason = "缓存返回，不推断本次触发浏览器"
        elif row["crawl4ai_selected"]:
            reason = "直接记录：非缓存结果采用Crawl4AI"
        elif row["http_status"] == 403 or (row["http_status"] or 0) >= 500:
            reason = "代码路径推断：HTTP403或5xx分支调用浏览器函数"
        elif row["failure_category"] == "timeout":
            reason = "代码路径推断：静态超时分支调用浏览器函数"
        elif (row["failure_category"] in {"dynamic_page", "content_too_short"}
              and row["extractor"] in {"trafilatura", "beautifulsoup"}
              and row["content_chars"] < 500 and not row["text_sanitization"]):
            reason = "代码路径推断：非PDF静态正文不足500字符分支调用浏览器函数"
        else:
            reason = "现有最终结果不增加浏览器调用下界"
        inference_rows.append({**row, "推断依据": reason,
                               "计入浏览器函数调用下界": reason.startswith(("直接记录：", "代码路径推断："))})
    inferred = [r for r in inference_rows if r["计入浏览器函数调用下界"]]

    per_run = []
    groups = defaultdict(list)
    for row in rows:
        groups[row["run_id"]].append(row)
    for run, items in sorted(groups.items()):
        sources = sorted({s["history_path"] for record in raw_records.values()
                          if record["run_id"] == run for s in record["sources"]})
        per_run.append({"run_id": run, "source_paths": sources,
                        "first_read_at": min(str(r["created_at"]) for r in items),
                        "last_read_at": max(str(r["created_at"]) for r in items), **summarize(items)})
    per_topic = [{"题号": topic, **summarize([r for r in five_rows if r["topic"] == topic])}
                 for topic in [35, 31, 46, 16, 49]]

    # 控制台错误仅作为确有失败调用的证据，缺少调用ID，不能直接与返回条目相加。
    log_rows, log_manifest = [], []
    for path in sorted(DATA.rglob("*.log")):
        blob = path.read_bytes()
        found = []
        for number, line in enumerate(blob.decode(errors="replace").splitlines(), 1):
            if "Crawl4AI sidecar 读取失败" in line or "Crawl4AI 本地读取失败" in line:
                found.append({"path": relative(path), "line": number, "text": line})
        if found:
            log_rows.extend(found)
            log_manifest.append({"路径": relative(path), "字节数": len(blob), "SHA256": digest(blob)})

    dates = sorted(datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00")) for r in rows)
    no_raw = [{**call, "kinds": sorted(call["kinds"]), "sources": sorted(call["sources"]),
               "run_ids": sorted(call["run_ids"]), "context_ids": sorted(call["context_ids"])}
              for key, call in sorted(calls.items()) if key not in raw_records]
    result = {
        "评估日期": args.date, "时区": "Asia/Singapore",
        "范围": "backend/eval/data 下所有 *工作历史.json 和 智能体原始历史.json 的原始正文读取结果；包含试跑、恢复及消融，跨导出去重",
        "数据审计": {"扫描历史文件数": len(histories), "包含读取事件的文件数": sum(i["读取工具相关记录数"] > 0 for i in inventory),
                     "扫描字节数": sum(i["字节数"] for i in inventory), "重复原始结果副本数": seen_copies,
                     "冲突结果数": len(conflicts), "未支持结构": unsupported,
                     "有读取事件的关联已知运行编号数": len({run_id for call in calls.values() for run_id in call["run_ids"]}),
                     "缺少运行编号的读取调用数": sum(not call["run_ids"] for call in calls.values()),
                     "跨运行归属复用的调用编号数": sum(len(call["run_ids"]) > 1 for call in calls.values()),
                     "读取工具调用编号数": len(calls), "有原始结果的批次数": len(raw_records),
                     "空结果批次数": sum(not r["entries"] for r in raw_records.values()),
                     "未保存原始结果的调用编号数": len(no_raw),
                     "最早返回记录时间": dates[0].isoformat(), "最晚返回记录时间": dates[-1].isoformat(),
                     "固定五题逐条核对数": len(five_rows), "Crawl4AI失败控制台行数": len(log_rows)},
        "全部已导出轨迹": summarize(rows),
        "固定五题按读取次数": summarize(five_rows),
        "固定五题按题与URL去重": summarize(five_unique),
        "固定五题逐题": per_topic,
        "固定五题浏览器函数调用下界推断": {
            "边界": "间接推断进入_read_with_crawl4ai的下界，不等同于Crawl4AI服务端实际执行或成功；不外推到其他源码版本",
            "占非缓存读取比例": rate(len(inferred), sum(r["from_cache"] is False for r in five_rows)),
            "依据分布": dict(Counter(r["推断依据"] for r in inferred)),
            "历史源码指纹": frozen_manifest,
        },
        "新网络请求": 0, "新模型调用": 0,
    }
    config = {
        "评估日期": args.date, "输入根目录": relative(DATA),
        "历史文件选择": "文件名等于智能体原始历史.json或以工作历史.json结尾，包括累计历史、最终历史、恢复历史",
        "事件选择": "kind=tool_raw_result且payload.tool_name=v2_web_fulltext_read",
        "重复导出去重": "tool_name+call_id（每次ToolRuntime执行生成的UUID），逐份校验规范化原始payload的SHA256一致；同一调用被恢复归档关联到多个run_id时保留全部归属，不能再算一次联网；缺少run_id的旧导出以context_id分组且不冒充已知运行；每个value[]元素算一次URL读取返回",
        "分子": "extractor包含crawl4ai（不区分大小写）；不要求fetch_status=success",
        "主分母": "全部去重后的URL返回条目，包括失败、缓存及重复访问URL",
        "实际调用下界": "仅from_cache=false且extractor标记Crawl4AI的条目可直接确认本次非缓存读取采用Crawl4AI；失败及未采用的尝试可能漏记",
        "缓存": "缓存返回仅复用既有结果，不视为此次发生Crawl4AI网络调用；未知缓存字段不计入非缓存分母",
        "未完成调用": "仅输入/派发/错误而无原始结果时单列，不根据入参候选列表臆造URL读取次数",
        "日志": "另保存Crawl4AI失败日志作为漏记证据；因缺少run_id/call_id，不与原始返回直接相加",
        "排除": "不统计正文重抓/公开网页基准/单元测试/伪工具结果/缓存文件/检查点和工具输出副本，不从纯模型文本中匹配crawl4ai",
        "边界": "不声称覆盖数据库未导出记录；不以当前代码推定所有历史版本的具体浏览器调用次数；不进行正文质量重评",
    }
    out.mkdir(parents=True)
    write_json(out / "统计配置.json", config)
    write_json(out / "指标汇总.json", result)
    write_json(out / "逐运行统计.json", per_run)
    write_json(out / "来源文件清单与指纹.json", inventory + log_manifest + [
        {"路径": relative(FIVE), "字节数": len(five_bytes), "SHA256": digest(five_bytes)},
        {"路径": relative(__file__), "SHA256": digest(Path(__file__).read_bytes())},
    ] + frozen_manifest)
    write_json(out / "缺少原始返回的调用.json", no_raw)
    write_jsonl(out / "逐次正文读取明细.jsonl", rows)
    write_jsonl(out / "固定五题读取明细.jsonl", five_rows)
    write_jsonl(out / "固定五题浏览器调用下界推断明细.jsonl", inference_rows)
    write_jsonl(out / "Crawl4AI失败控制台证据.jsonl", log_rows)
    write_jsonl(out / "去重批次与原始来源.jsonl", [{k: v for k, v in r.items() if k != "entries"}
                                                     for r in raw_records.values()])
    print(json.dumps({"输出目录": str(out), "数据审计": result["数据审计"],
                      "全部已导出轨迹": result["全部已导出轨迹"],
                      "固定五题浏览器函数调用下界推断": result["固定五题浏览器函数调用下界推断"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
