"""同一运行的评分口径复核：叶子表格、多行多列子集、适用关系分母。"""
from pathlib import Path
import importlib.util
import json
import shutil
import signal
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "backend/eval/data/网页读取链路公开基准评估-20261018-首轮"
spec = importlib.util.spec_from_file_location("网页评分实现", ROOT / "backend/eval/scripts/评估公开网页读取质量.py")
metrics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metrics
spec.loader.exec_module(metrics)
signal.signal(signal.SIGALRM, metrics.alarm_handler)


def main():
    archive = RUN / "首次评分归档"
    archive.mkdir(exist_ok=False)
    for name in ["逐页指标汇总.json", "核心指标与分层汇总.json", "失败样例候选.json", "指标机制验证.json"]:
        shutil.copyfile(RUN / name, archive / name)
    rows = json.loads((RUN / "逐页指标汇总.json").read_text())
    network = metrics.install_network_guard()
    metrics.dump(RUN / "指标机制验证.json", metrics.validate_metrics())
    mode_files = {"unlimited_pre_gate": "放宽上限正文.txt", "production_pre_gate": "生产门禁前正文.txt",
                  "production_cleaned": "生产清理后正文.txt", "accepted_output": "最终可接纳正文.txt"}
    for i, r in enumerate(rows):
        if r["dataset"] != "WebMainBench545":
            continue
        folder = RUN / "逐页结果" / r["id"]
        old = archive / r["id"]
        old.mkdir(); shutil.copyfile(folder / "页面结果.json", old / "页面结果.json")
        tables = metrics.extract_tables((folder / "参考正文.txt").read_text())
        r["reference_tables_detected"] = len(tables)
        r["reference_table_error"] = None
        memo = {}
        for mode, filename in mode_files.items():
            text = (folder / filename).read_text()
            digest = metrics.sha(text.encode())
            if digest not in memo:
                memo[digest] = metrics.table_scores(text, tables)
            r["scores"][mode]["table"] = memo[digest]
        r["table_protocol_revision"] = "2026-10-18-v1.1"
        metrics.dump(folder / "页面结果.json", r)
        if (i + 1) % 100 == 0:
            print("已复核表格", i + 1, flush=True)
    metrics.dump(RUN / "逐页指标汇总.json", rows)
    metrics.dump(RUN / "同次运行评分口径复核记录.json", {
        "复核日期": "2026-10-18", "完成时间": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "运行": "同一批745页；不重新提取，不调用API，只对保存的参考与预测重计表格指标",
        "变更": ["嵌套table只取叶子表，避免布局容器覆盖内部表格", "另报至少2行2列且4个非空单元格的子集，不把单格文章布局等同于数据表",
                 "行列关系仅在参考存在该类关系时参与宏平均，无关系页面不获得默认满分"],
        "未变更": ["输入HTML", "当前业务代码", "正文预测", "正文Precision/Recall/F1", "质量门禁结果", "WCXB片段指标"],
        "首次结果存档": "首次评分归档", "复核网络尝试": network, "新增付费调用": 0})
    shutil.copyfile(ROOT / "backend/eval/scripts/评估公开网页读取质量.py", RUN / "补充复评评分实现.py")
    shutil.copyfile(__file__, RUN / "补充复评脚本.py")
    print("表格口径复核完成；网络尝试", len(network), flush=True)


if __name__ == "__main__":
    main()
