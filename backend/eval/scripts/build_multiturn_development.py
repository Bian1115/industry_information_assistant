#!/usr/bin/env python3
"""将固定官方原题与人工编写的多轮脚本合并；不调用模型，不生成事实答案。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from eval.multiturn import Task

REVISION = "214741e474a5347ea7f869ab79d27e7c8734ede7"
SOURCE_SHA256 = "62b197a264a027fd4e2e795ff322a82dd6d566f5e350596ccc7572539ec987d1"
SOURCE_URL = f"https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/{REVISION}/data/prompt_data/query.jsonl"
AS_OF = "2026-09-27"

# 改写的是研究请求；不把原题中的预设关系当作已经被证实的事实。
SCRIPTS = {
    8: {
        "title":"机器学习材料配比研究",
        "queries":[
            "我想了解机器学习或深度学习如何优化材料元素配比以改善性能。请以2026年9月27日为资料截止日，先梳理2020年以来的主要技术路线及适用材料，并说明值得继续研究的问题。整个讨论请区分计算预测、实验验证和产业化应用；涉及准确度时保留数据集、测试划分和指标名称，不把不同测试条件下的分数直接排名。较早的奠基工作可作背景。",
            "请进一步调查这一方向的活跃课题组、代表论文和使用的数据集，解释各自解决什么问题。对有公开信息的研究，整理材料对象、数据规模、输入输出、验证方法和来源；缺失项明确说明。",
            "接下来比较前面路线在小样本、跨材料体系和多目标优化中的表现，补充相关研究。哪些提升经过独立实验验证，哪些仍只在数据集上成立？请把论据对应到具体来源。",
            "请专门寻找失败案例、负面结果或与前面结论冲突的证据，检查数据泄漏、外推失效、合成可行性和评价条件差异，并据此修正此前的适用边界。",
            "结合前面调研，写成一份完整研究报告，覆盖课题组、方法、数据库、模型表现、挑战、可行性，以及距离大规模产业应用还有哪些障碍。对未来判断说明依据和不确定性。",
        ],
        "anchor":"区分计算预测、实验验证、产业化；跨研究指标保留数据集和测试条件，不能直接排名。",
        "evidence":"最终报告能准确复用早期论文的方法、数据规模、验证条件和来源，数值没有串到另一研究。",
        "conflict":"保留外推或验证失败证据，结论不将数据集成绩直接等同于产业效果。",
    },
    25: {
        "title":"中性粒细胞与脑缺血",
        "queries":[
            "请调研中性粒细胞在脑缺血急性期和慢性期的功能变化，以2026年9月27日为资料截止日，重点考察2020年以来研究，较早工作仅作背景。先给出阶段划分和主要争议。整个讨论请区分动物实验与人体证据，保留各研究自己的时间窗，不把不同研究的急性期定义混用；机制假说和已经观察到的结果分开表述。",
            "先深入急性期：不同中性粒细胞状态或亚群在何时出现，与哪些细胞相互作用？请查找原始研究，整理物种或人群、缺血模型、采样时间、检测方法、观察结果与来源。",
            "再研究恢复或慢性阶段，比较亚群及其细胞互作如何变化，哪些变化关联不同功能结局？请与前面的研究逐项对照，说明哪些可以比较、哪些条件并不一致。",
            "请主动寻找中性粒细胞作用方向相反或干预结果不一致的研究，分析是否与阶段、亚群、实验方法或样本有关，并修正之前过于笼统的判断。",
            "综合前面材料形成研究综述，解释阶段变化和细胞互作怎样可能导向不同临床结局，区分已知证据、尚待验证的机制预测，并提出后续研究问题和验证设计。",
        ],
        "anchor":"保留研究自己的时间窗，区分动物/人体证据和观测/机制预测。",
        "evidence":"早期研究的物种、人群、采样时间和效应方向在最终综述中仍与原始来源对应。",
        "conflict":"急性期与恢复期、不同亚群的相反结果不能被压缩为单一有害或有益结论。",
    },
    26: {
        "title":"T细胞代谢与命运分岔",
        "queries":[
            "我想研究慢性抗原刺激下CD8+ T细胞的线粒体融合/裂变、表观遗传变化与终末耗竭或组织驻留记忆细胞命运之间的联系。以2026年9月27日为资料截止日，请先检查这些联系各自的证据基础，并提出调研框架。整个讨论分别处理肿瘤微环境与HIV感染，不预设m6A或组蛋白乳酸化已经被证明是统一的因果通路，也不把相关性自动写成命运转换机制。",
            "请先聚焦肿瘤环境，整理线粒体动力学与这两类细胞状态相关的原始研究。逐项记录研究对象、干预、测量方法、时间点和效应方向，说明哪些结果支持因果推断、哪些仅为相关观察。",
            "请再检查HIV或持续感染场景中的相应证据，重点寻找与前面一致和不同的机制；同时核查m6A及组蛋白乳酸化在这些具体场景里有哪些直接证据，哪些只是外推。",
            "如果要建立代谢—表观遗传互作的定量模型，可以使用哪些变量、观测数据和约束？请检查可辨识性、数据缺口和反证，不能把缺少测量的参数编成已知值。",
            "根据前面证据写出研究报告和可执行的建模方案，说明能够检验哪些命运分岔假说、如何验证与证伪，以及哪些问题暂时不能定量回答。",
        ],
        "anchor":"分别讨论肿瘤与HIV，相关性不能自动当因果，m6A和乳酸化联系需逐项核实。",
        "evidence":"早期研究的干预、对象、时间点、效应方向和证据强度在综合建模方案中不被混合。",
        "conflict":"保留机制缺证、相反证据与参数不可辨识性，不用猜测补成统一已证实网络。",
    },
    27: {
        "title":"AI与人类心理咨询协作",
        "queries":[
            "请研究AI心理咨询与人类心理咨询如何协作，以2026年9月27日为资料截止日，先梳理成人非紧急心理支持中的服务模式与现有证据。整个讨论明确区分一般心理支持、筛查和临床治疗；危机或紧急情况仅讨论识别、升级和转介，不把AI独立处理危机纳入建议。效果数字要保留人群、对照条件和随访时间。",
            "请进一步比较AI独立支持、人类咨询及人机协作的研究证据，整理研究设计、样本、比较对象、结局指标、随访和失访情况。对缺乏直接比较的模式，明确证据不能支持哪些结论。",
            "请调研实际协作流程：分诊、咨询前准备、咨询间支持和人工复核分别适合怎样分工？将流程建议与前面证据对应，并补充隐私、知情同意、责任和可及性方面的研究。",
            "请寻找负面结果、偏差和可能受到伤害的群体，检查短期改善是否维持、不同人群能否外推，以及何时必须升级到人工服务。必要时调整前面的分工方案。",
            "结合前面材料形成一份协作方案与研究报告，说明实施条件、证据局限、人工升级流程和后续评价设计，不把尚未验证的收益写成既成事实。",
        ],
        "anchor":"仅成人非紧急支持；区分支持/筛查/治疗；危机只讨论识别升级转介。",
        "evidence":"保留早期研究的人群、对照、随访、失访及结局指标，不用短期效果替代长期有效性。",
        "conflict":"整合负面研究、群体差异与隐私风险，不能只保留收益和方便性。",
    },
    28: {
        "title":"大模型与药物影响评估",
        "queries":[
            "请调研现阶段大模型能否结合多组学信息模拟药物对机体的影响，以及这一方向可能怎样发展。资料截止2026年9月27日，先梳理主要模型路线与应用层次。整个讨论分别区分分子、细胞、组织和整体人体层面的证据；基准预测、实验验证与临床效用分开，不把虚拟细胞的结果直接外推为人体疗效。",
            "请深入代表性模型及数据，整理模型对象、训练数据模态、扰动设置、评价任务和公开验证结果。保留可核实的数字与来源；对数据是否存在训练测试重叠或分布外测试，明确说明。",
            "请比较这些路线如何处理剂量、时间、细胞状态和个体异质性，以及多组学融合是否带来可验证提升。哪些能力已经有证据，哪些只是模型设想？",
            "请寻找失败案例和外推边界，特别关注跨细胞类型、跨个体、多药联用和跨尺度预测；检查此前结论是否过度概括，并说明证据空白。",
            "整合成研究报告，回答现阶段可以评估哪些药物影响、哪些尚不能可靠评估，以及未来需要的数据、验证体系和研究路线。所有展望应与现有证据和缺口对应。",
        ],
        "anchor":"分开分子/细胞/组织/人体层次，区分预测/实验/临床，禁止把虚拟细胞直接外推为人体疗效。",
        "evidence":"模型的数据模态、扰动设置、数值指标与独立验证条件能在最终报告准确复用。",
        "conflict":"保留跨对象、剂量时间和多药联用的失败或缺证结果，不能只留下平均基准优势。",
    },
}


def build(source):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("输入原题文件与固定官方版本不一致，拒绝生成")
    originals = {r["id"]:r for r in map(json.loads, raw.decode().splitlines())}
    tasks = []
    objectives = ["建立研究范围与早期约束", "累积第一组可追溯证据", "累积另一维度并跨轮比较", "寻找反证并修正边界", "整合历史形成报告", "更新篇幅要求且保留早期约束"]
    final = "请将上一版报告正文控制在3000字以内，保留此前约定的研究范围、证据边界和引用；参考文献单独列出。"
    for source_id, spec in SCRIPTS.items():
        original = originals[source_id]
        queries = spec["queries"] + [final]
        task = Task.model_validate({
            "schema_version":1, "task_id":f"drb-mt-{source_id:03}", "split":"development",
            "source":{"benchmark":"DeepResearch Bench", **original,
                      "prompt_sha256":hashlib.sha256(original["prompt"].encode()).hexdigest(),
                      "revision":REVISION, "url":SOURCE_URL, "file_sha256":SOURCE_SHA256},
            "as_of":AS_OF,
            "rewrite_notes":"保留原题核心研究目标；添加明确截止日、证据边界和第6轮篇幅上限用于记忆测试。只改写问题，没有提供事实答案。医学题为研究任务，不是个体诊疗建议。",
            "turns":[{"turn_id":i,"user_query":q,"depends_on":list(range(1,i)),"objective":objectives[i-1]} for i,q in enumerate(queries,1)],
            "probes":[
                {"probe_id":"早期范围","introduced_turn":1,"check_turns":[3,5,6],"kind":"constraint","criterion":spec["anchor"],"annotation_status":"user_constraint"},
                {"probe_id":"早期证据","introduced_turn":2,"check_turns":[5,6],"kind":"evidence","criterion":spec["evidence"],"annotation_status":"pending_source_annotation"},
                {"probe_id":"反证保留","introduced_turn":4,"check_turns":[5,6],"kind":"conflict","criterion":spec["conflict"],"annotation_status":"pending_source_annotation"},
                {"probe_id":"篇幅更新","introduced_turn":6,"check_turns":[6],"kind":"revision","criterion":"最终正文非空白Unicode字符不超过3000，参考文献段不计；标点、英文和数字逐字符计。篇幅更新不能覆盖早期范围约束。","annotation_status":"user_constraint"},
            ],
        })
        tasks.append(task)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BACKEND/"eval/data/deepresearch_bench/query.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = build(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    name = "DeepResearch多轮开发集-5题-20260927"
    files = {args.output/f"{name}.jsonl":"".join(t.model_dump_json()+"\n" for t in tasks)}
    lines = ["# DeepResearch多轮开发集（5题）", "", "这是开发集，含30轮固定用户请求；未执行研究，不包含人工事实真值。", "",
             f"官方原题版本：`{REVISION}`；原始文件SHA256：`{SOURCE_SHA256}`。", f"原题下载地址：{SOURCE_URL}", "",
             "选取4条Health与1条Science & Technology；属于医疗侧重的目的性开发抽样，不代表整个DeepResearch Bench分布。候选5题、测试20题尚未选定。"]
    for t in tasks:
        lines += ["",f"## 原题{t.source.id}：{SCRIPTS[t.source.id]['title']}","",f"原题：{t.source.prompt}",""]
        for turn in t.turns:
            lines += [f"### 第{turn.turn_id}轮", "", turn.user_query, "", f"评测意图（不传给模型）：{turn.objective}", ""]
        lines += ["检查项（不传给模型）："] + [f"- {p.probe_id}：{p.criterion}（{p.annotation_status}）" for p in t.probes]
    files[args.output/f"{name}.md"] = "\n".join(lines)+"\n"
    files[args.output/"原题来源与划分清单-20260927.json"] = json.dumps({
        "source_url":SOURCE_URL,"source_revision":REVISION,"source_file_sha256":SOURCE_SHA256,
        "split_plan":{"development":5,"selection":5,"test":20},
        "assigned":{"development":[t.source.id for t in tasks],"selection":[],"test":[]},
        "source_selection":"4条医疗＋1条研究方法，按任务划分；其余集合未抽样", "constructed_by":"assistant_manual_rewrite",
        "human_review_status":"pending", "facts_annotation_status":"pending_source_snapshot",
        "official_leaderboard_comparable":False,
    },ensure_ascii=False,indent=2)
    files[args.output/"多轮任务数据结构-20260927.json"] = json.dumps(Task.model_json_schema(),ensure_ascii=False,indent=2)
    if any(path.exists() for path in files):
        raise FileExistsError("目标文件已存在，拒绝覆盖")
    for path, text in files.items():
        path.write_text(text)
    print(json.dumps({"tasks":len(tasks),"turns":sum(len(t.turns) for t in tasks),"files":[str(p) for p in files]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
