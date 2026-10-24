"""为主代理审读提供匿名正文、去流程元数据来源及可追溯原文窗口；不评分。"""
import argparse
import hashlib
import json
from pathlib import Path
import re

FIELDS = {"id", "document_id", "url", "source_url", "title", "source_name", "date", "publish_date",
          "content", "full_content", "original_sentences", "sentence_index", "body_quality"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--label", choices=["报告甲", "报告乙"])
    parser.add_argument("--url")
    parser.add_argument("--query")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=12000)
    args = parser.parse_args(); root = args.directory.resolve()
    view = root / "匿名审读材料"
    counts = {}
    for report in sorted((root / "盲评材料").glob("题目*/报告*.md")):
        destination = view / report.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / report.name).write_bytes(report.read_bytes())
        source = report.with_name(report.stem + "来源.json")
        docs = json.loads(source.read_text())
        sanitized = [{key: value for key, value in doc.items() if key in FIELDS} for doc in docs]
        (destination / source.name).write_text(json.dumps(sanitized, ensure_ascii=False, indent=2))
        counts[str(report.relative_to(root / "盲评材料"))] = {"report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "sources_count": len(sanitized), "report_chars": len(report.read_text())}
    (view / "材料清单.json").parent.mkdir(parents=True, exist_ok=True)
    (view / "材料清单.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2))
    if args.task_id is None:
        print(json.dumps({name: {key: value for key, value in item.items() if key != "report_sha256"}
                          for name, item in counts.items()}, ensure_ascii=False))
        return
    docs = json.loads((view / f"题目{args.task_id}" / (args.label + "来源.json")).read_text())
    selected = [doc for doc in docs if args.url is None or args.url in {doc.get("url"), doc.get("source_url")}]
    if not args.query and args.url is None:
        print(json.dumps([{key: doc.get(key) for key in ("document_id", "title", "url", "date")} |
            {"content_chars": len(doc.get("full_content") or doc.get("content") or "")} for doc in selected], ensure_ascii=False, indent=2))
        return
    remaining = args.max_chars
    for doc in selected:
        text = doc.get("full_content") or doc.get("content") or ""
        if args.query:
            positions = [match.start() for match in re.finditer(args.query, text, re.I)]
            ranges = []
            for position in positions:
                left, right = max(0, position - 450), min(len(text), position + 850)
                if ranges and left <= ranges[-1][1]:
                    ranges[-1] = (ranges[-1][0], max(right, ranges[-1][1]))
                else:
                    ranges.append((left, right))
        else:
            ranges = [(args.offset, len(text))]
        for left, right in ranges:
            if remaining <= 0:
                break
            right = min(right, left + remaining)
            value = {"title": doc.get("title"), "url": doc.get("url") or doc.get("source_url"),
                "document_id": doc.get("document_id"), "start": left, "end": right,
                "total_chars": len(text), "excerpt": text[left:right]}
            print(json.dumps(value, ensure_ascii=False))
            remaining -= right - left


if __name__ == "__main__":
    main()
