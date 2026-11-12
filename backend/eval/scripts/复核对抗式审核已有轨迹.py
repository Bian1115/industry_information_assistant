"""提取历史审核的当轮输入/输出，并复算独立标注；不调用模型或改变运行状态。"""
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'backend/eval/data/Critic职责分离通用研究五题消融-20261017-231745'
OUT = ROOT / 'backend/eval/data/对抗式审核意见复核-20261030'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def extract():
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / '复核口径.json', {
        '评估日期': '2026-10-30', '审核基准日期': '2026-10-17',
        '对象': '最新有完整审核输入输出的两题六轮；旧批次56条仅复算既有标注，分开报告',
        '评审者': '本轮Codex主代理，单一AI审读者；非人类金标，非盲评',
        '判定': {'正确': '主要问题、关键依据和纠正方向成立；轻微严重度/措辞另列',
                 '部分正确': '真实问题夹带实质性错误前提、范围、验收或建议',
                 '错误': '主要指控不成立，或无依据增加用户硬性要求',
                 '无法判断': '现有证据不足以判断主要指控'},
        '复查判断': 'issue_reviews单列；依当轮正文和原验收条件判断resolved是否成立，新增内容不自动代表实质修复',
        '指标': '严格正确率=正确/全部；至少部分有效=(正确+部分正确)/全部；错误率=错误/全部；无法判断保留分母并另列可判定口径',
        '边界': ['不存在独立穷举的问题金标，不计算召回率、F1或TN型accuracy',
                 '新问题与旧问题复查分母分开；程序合并遗留问题不算模型再次指控',
                 '来源不足不等于现实虚假；每轮核对当轮报告，不用最终报告反驳首轮',
                 '当轮版本为三维十分制；当前工作区四维百分制提示词无本批实测'],
    })
    rounds = []
    for q in [35, 31]:
        path = SOURCE / f'题目{q}/开启审核/智能体原始历史.json'
        history = json.loads(path.read_text())
        inputs = {h['call_id']: h for h in history if h.get('node') == 'review' and h['kind'] == 'llm_input'}
        requests = {h['call_id']: h for h in history if h.get('node') == 'review' and h['kind'] == 'llm_request'}
        outputs = [h for h in history if h.get('node') == 'review' and h['kind'] == 'llm_output']
        for n, output in enumerate(outputs, 1):
            cid = output['call_id']
            prompt = inputs[cid]['payload']['content']
            report = prompt.split('### 待审核的最终报告\n', 1)[1].split('\n### 使用的数据点', 1)[0].strip()
            result = json.loads(output['payload']['content'])
            actual_messages = requests[cid]['payload']['request']['messages']
            report_sent_intact = any(isinstance(m.get('content'), str) and report in m['content']
                                     for m in actual_messages)
            assert report_sent_intact, (q,n,'实际请求未包含完整当轮报告')
            folder = OUT / f'题目{q}/第{n}次审核'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / '待审报告.md').write_text(report + '\n', encoding='utf-8')
            save(folder / '模型原始输入.json', inputs[cid]['payload'])
            save(folder / '模型原始输出.json', result)
            save(folder / '模型实际请求.json', requests[cid]['payload'])
            matches = []
            cpfile = SOURCE / f'题目{q}/开启审核/完整检查点历史.json'
            for cp in json.loads(cpfile.read_text()):
                state = cp['checkpoint'].get('channel_values', {}).get('research', {})
                if state.get('final_report', '').strip() == report:
                    matches.append({'checkpoint_id': cp['checkpoint']['id'], 'ts': cp['checkpoint']['ts'],
                                    'iteration': state.get('iteration'), 'phase': state.get('phase'),
                                    'feedback_count': len(state.get('critic_feedback', []))})
            endpoint = SOURCE / f'题目{q}' / ('公共首稿' if n == 1 else '开启审核') / '研究报告.md'
            rec = {'question': q, 'round': n, 'call_id': cid, 'source': str(path.relative_to(ROOT)),
                   'report_path': str((folder/'待审报告.md').relative_to(ROOT)),
                   'report_sha256': sha(report), 'checkpoint_matches': matches,
                   'report_sent_intact':report_sent_intact,
                   'endpoint_matches': endpoint.read_text().strip() == report if n in [1,3] else None,
                   'model': requests[cid]['payload']['request']['model'], 'result': result}
            assert matches, (q,n,'未找到当轮检查点')
            if n in [1,3]:
                assert rec['endpoint_matches'], (q,n,'首尾报告不一致')
            rounds.append(rec)
    save(OUT / '六轮原始意见与报告对齐.json', rounds)
    print(json.dumps([{'题目':r['question'],'轮次':r['round'],'新问题':len(r['result']['issues']),
                       '复查':len(r['result']['issue_reviews']), '检查点匹配':len(r['checkpoint_matches'])}
                      for r in rounds],ensure_ascii=False))


def summarize():
    rounds = json.loads((OUT / '六轮原始意见与报告对齐.json').read_text())
    annotations = json.loads((OUT / '独立逐条标注.json').read_text())
    labels = ['正确', '部分正确', '错误', '无法判断']

    def stats(rows):
        counts = Counter(r['label'] for r in rows)
        n = len(rows)
        return {'n': n, 'counts': {k: counts[k] for k in labels},
                'strict_correct_rate': counts['正确']/n if n else None,
                'at_least_partial_rate': (counts['正确']+counts['部分正确'])/n if n else None,
                'incorrect_rate': counts['错误']/n if n else None,
                'decidable_rate': (n-counts['无法判断'])/n if n else None}

    def grouped(rows, key):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        return {k:stats(v) for k,v in sorted(groups.items())}

    enriched = {}
    for category, field, id_field in [('新问题','issues','id'),('旧问题复查','issue_reviews','issue_id')]:
        originals = {}
        for r in rounds:
            for issue in r['result'][field]:
                key = f"{r['question']}-R{r['round']}-{issue[id_field]}"
                assert key not in originals
                originals[key] = (r, issue)
        rows = annotations[category]
        assert len(rows) == len(originals)
        assert {x['key'] for x in rows} == set(originals)
        for row in rows:
            r, issue = originals[row['key']]
            row.update(question=r['question'], round=r['round'], original=issue,
                       report_path=r['report_path'], report_sha256=r['report_sha256'])
            report = (ROOT/r['report_path']).read_text()
            lines = report.splitlines()
            assert row['label'] in labels
            assert all(1 <= n <= len(lines) for n in row['lines'])
            row['excerpts'] = [{'line':n,'text':lines[n-1]} for n in row['lines']]
            if category == '新问题':
                row.update(severity=issue['severity'], issue_type=issue['issue_type'],
                           requires_new_search=issue['requires_new_search'])
            else:
                row['exact_quote_match'] = issue['report_quote'] in report
                history = json.loads((SOURCE/f"题目{r['question']}/开启审核/完整检查点历史.json").read_text())
                actual = []
                for cp in history:
                    state = cp['checkpoint']['channel_values'].get('research', {})
                    if state.get('final_report','').strip() != report.strip():
                        continue
                    for record in state.get('critic_feedback', []):
                        review = record.get('last_review', {})
                        if record['id'] == issue['issue_id'] and review.get('iteration') == r['round']-1:
                            actual.append(review.get('effective_status'))
                assert actual and len(set(actual)) == 1, row['key']
                row['program_effective_status'] = actual[0]
                assert (actual[0]=='resolved') == row['exact_quote_match']
        enriched[category] = rows
    save(OUT/'逐条判定与原文证据.json', enriched)
    new, reviews = enriched['新问题'], enriched['旧问题复查']
    summary = {'评估日期':'2026-10-30', 'new_issues':stats(new), 'rechecks':stats(reviews),
               'new_by_question':grouped(new,'question'), 'new_by_round':grouped(new,'round'),
               'new_by_severity':grouped(new,'severity'), 'new_by_type':grouped(new,'issue_type'),
               'new_by_search':grouped(new,'requires_new_search'),
               'rechecks_by_question':grouped(reviews,'question'), 'rechecks_by_round':grouped(reviews,'round'),
               'macro_question_strict_correct_rate':sum(stats([x for x in new if x['question']==q])['strict_correct_rate'] for q in [35,31])/2,
               'recheck_unique_question_issue_pairs':len({(r['question'],r['original']['issue_id']) for r in reviews}),
               'program_closure':{'model_resolved':len(reviews),
                    'program_accepted':sum(r['program_effective_status']=='resolved' for r in reviews),
                    'program_rejected':sum(r['program_effective_status']!='resolved' for r in reviews),
                    'correct_rechecks':sum(r['label']=='正确' for r in reviews),
                    'correct_rechecks_rejected':sum(r['label']=='正确' and r['program_effective_status']!='resolved' for r in reviews)},
               'raw_model_pass_rounds':sum(r['result']['overall_assessment']['verdict']=='pass' for r in rounds),
               'scope_limits':['2个独立题，6轮相关报告，非随机抽样；不估总体准确率或显著性',
                               '无独立穷举问题金标，recall/F1/含真负例的accuracy不可计算',
                               '没有运行当前四维百分制模型，也未续跑历史中断任务']}
    oldfile = ROOT/'backend/eval/data/Critic五十六条意见正确率复核-20261011-201238/逐条判定.json'
    old = json.loads(oldfile.read_text())
    mapping = {'correct':'正确','partial':'部分正确','incorrect':'错误','uncertain':'无法判断','undetermined':'无法判断'}
    historical = [dict(x, label=mapping[x['finding_correctness']]) for x in old]
    summary['historical_recomputed'] = {'source':str(oldfile.relative_to(ROOT)),
        'sha256':sha(oldfile.read_text()), 'assessment_date':'2026-10-11',
        'method':'复算既有56条标签，不宣称本轮重新审读全部56条', 'overall':stats(historical),
        'by_question':grouped(historical,'task_id'), 'by_round':grouped(historical,'round'),
        'missing_existing_citation_errors':sum(x['label']=='错误' and x['issue_pattern']=='漏读已有引用' for x in historical)}
    save(OUT/'指标汇总.json', summary)

    selected = []
    patterns = [r'60\s*%?\s*[-–—~～至]\s*80\s*[%％]',r'1\s*%?\s*[-–—~～至]\s*3\s*[%％]',
                r'30\s*%?\s*[-–—~～至]\s*50\s*[%％]',r'10\s*%?\s*[-–—~～至]\s*15\s*[%％]']
    source_checks = []
    for q in [35,31]:
        path = SOURCE/f'题目{q}/开启审核/完整来源正文.json'
        sources = json.loads(path.read_text())
        needed = {s for cat in enriched.values() for row in cat if row['question']==q for s in row.get('source_ids',[])}
        for s in sources:
            if s['id'] in needed:
                selected.append({'question':q,'source_path':str(path.relative_to(ROOT)), 'id':s['id'],
                                 'url':s['url'],'title':s['title'],'content':s.get('content',''),
                                 'sha256':sha(s.get('content',''))})
        if q == 35:
            source_checks = [{'source_id':s['id'],'chars':len(s.get('content','')),
                'matched_ratio_patterns':[p for p in patterns if re.search(p,s.get('content',''))]} for s in sources]
    save(OUT/'引用来源定向复查.json', {'selected_sources':selected,'污水21份保存正文比例扫描':source_checks,
        '边界':'数值正则检索用于定位，不等于语义穷举；正文未含比例不排除原站图片、附件、未保存材料中出现，也不证明现实数值必假。已直接阅读全文核对报告所指的1058字行业页面。'})
    with zipfile.ZipFile(SOURCE/'运行源码快照.zip') as archive:
        text = archive.read('backend/app/service/deep_research_v2/issue_workflow.py').decode()
        (OUT/'历史问题复查实现快照.py').write_text(text)
    save(OUT/'外部一手来源核对.json', {
        'date':'2026-10-30','issue':'31-R3-issue_8','url':'https://museum.bunka.go.jp/law/',
        'publisher':'日本文化厅','title':'法改正の概要',
        'verified_summary':'官方说明2023-04-01转入新登记制度；调整设置主体限制，并将数字档案制作公开纳入博物馆事业。支持要求更新日本现状资料。',
        'temporal_boundary':'仅核对原审核日期2026-10-17以前已发生的2023年改革，不把2026-10-30之后事件倒灌历史报告',
        'additional_official_source':'https://www.bunka.go.jp/seisaku/bijutsukan_hakubutsukan/shinko/kankei_horei/93697301.html'})
    prior_fix = ROOT/'backend/eval/data/复审摘录取消逐字匹配回归-20261018-005033/交付核对记录.json'
    current_critic = ROOT/'backend/app/service/deep_research_v2/agents/critic.py'
    current_prompt = ROOT/'backend/app/service/deep_research_v2/prompts/对抗式审核提示词.txt'
    save(OUT/'版本与既有修复边界.json', {
        '历史逐字匹配缺陷':'属于2026-10-17冻结运行；本次未将其标为当前未修复缺陷',
        '已记录修复':{'path':str(prior_fix.relative_to(ROOT)), 'record':json.loads(prior_fix.read_text())},
        '当前审核源码':{'path':str(current_critic.relative_to(ROOT)), 'sha256':sha(current_critic.read_text()),
                       '含旧issue_reviews协议':'"issue_reviews"' in current_critic.read_text()},
        '当前提示词':{'path':str(current_prompt.relative_to(ROOT)), 'sha256':sha(current_prompt.read_text()),
                       '口径':'四维0–100分；当前process以每轮问题替换，不复用历史issue_reviews协议'},
        '现有新版本验证':'backend/eval/data/对抗式审核百分制与路由验证-20261030/首轮验证记录.json说明模型和搜索均为替身；不计入真实模型意见正确率',
        '下一步边界':'如验证当前新机制，应使用当前输入输出合同重放相同冻结报告，而非重加旧接口兼容层；本次未调用外部模型生成意见。'})
    appendix = ['# 对抗式审核逐条复核证据附表','', '评估日期：2026-10-30。判定口径与指标见主报告。']
    for category, rows in enriched.items():
        appendix += ['', f'## {category}']
        for row in rows:
            original = row['original']
            links = '、'.join(f"[第{n}行]({ROOT/row['report_path']}:{n})" for n in row['lines'])
            appendix += ['',f"### {row['key']}：{row['label']}", '',
                         f"原意见：{original.get('description') or original.get('reason')}",'',
                         f"判定：{row['reason']}",'',f"当轮正文：{links}。"]
            if row.get('note'):
                appendix += ['',f"边界：{row['note']}"]
            if row.get('source_ids'):
                appendix += ['',f"保存来源ID：{'、'.join(row['source_ids'])}；全文见本目录引用来源定向复查.json。"]
            if row.get('external_evidence'):
                appendix += ['',f"补充一手依据：[日本文化厅]({row['external_evidence']})。"]
            if category == '旧问题复查':
                appendix += ['',f"程序最终处理：{row['program_effective_status']}；引用片段逐字匹配：{row['exact_quote_match']}。"]
    (OUT/'逐条复核证据附表.md').write_text('\n'.join(appendix)+'\n')
    checks = {'new_count':len(new),'recheck_count':len(reviews),'all_annotations_covered':True,
              'rounds_with_checkpoint_match':sum(bool(r['checkpoint_matches']) for r in rounds),
              'reports_sent_intact_in_actual_requests':sum(r['report_sent_intact'] for r in rounds),
              'first_last_reports_verified':sum(r['endpoint_matches'] is True for r in rounds),
              'all_line_references_in_range':True,'program_status_verified_against_checkpoints':True,
              'historical_count':len(old)}
    save(OUT/'完整性核对.json', checks)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    extract()
    if (OUT / '独立逐条标注.json').exists():
        summarize()
