"""核查保留清单指纹、再暴露漏读及质量评审中的共同章节问题。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('evaluation',Path(__file__).with_name('评估通用研究历史压缩与后续写作.py'))
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)

def main(out):
    reappeared, disproved, quality, annotations = [], [], [], []
    for topic in ev.SOURCES:
        d=out/f'题目{topic}';sample=ev.load(d/'冻结样本.json')
        common='\n\n'.join(ev.render_chapter_task(t['content'],sample['titles'],sample['fixed_chapters'])[0] for t in sample['next_tasks'])
        points=ev.load(d/'冻结关键信息标注.json')['checkpoints']
        retention={r['id']:r for r in ev.load(d/'信息保留逐项结果.json')['coverage']}
        gold_hash=hashlib.sha256((d/'冻结关键信息标注.json').read_bytes()).hexdigest()
        assert gold_hash==ev.load(d/'压缩统计.json')['annotation_sha256']
        annotations.append({'topic':topic,'unchanged':True,'sha256':gold_hash})
        rows=[]
        for p in points:
            match=ev.locate(p['source_span']['quote'],common)
            rows.append({'topic':topic,'id':p['id'],'retention':retention[p['id']]['status'],
                         'source_quote_reappears':bool(match),'common_input_span':match})
        ev.save(d/'共同输入原标注引句核对.json',{'method':'冻结标注完整原文引句在后续共同输入中定位；未命中不等于语义缺失。','common_input_sha256':ev.sha(common),'items':rows})
        reappeared+=rows
        exposure_path=d/'共同输入信息再暴露.json'
        if exposure_path.exists():
            direct={r['id']:r for r in rows}
            for r in ev.load(exposure_path)['items']:
                if r['status']=='absent' and direct[r['id']]['source_quote_reappears']:
                    disproved.append({'topic':topic,'id':r['id'],'judge_reason':r['reason'],'actual_match':direct[r['id']]['common_input_span']})
        for path in sorted(d.glob('匿名质量评审[0-9].json')):
            raw=ev.load(path);issues=[]
            reports={group:(d/group/'完整报告.md').read_text() for group in ['原文组','压缩组']}
            continued={group:(d/group/'本轮新增内容.md').read_text() for group in reports}
            for candidate in raw['result']['candidates']:
                group=raw['mapping'][candidate['id']]
                for item in candidate['issues']:
                    quote=item['report_quote']
                    both=all(ev.locate(quote,r) for r in reports.values())
                    in_new=bool(ev.locate(quote,continued[group]))
                    issues.append({'group':group,'report_quote':quote,'appears_in_both_reports':both,
                                   'appears_in_this_group_new_content':in_new,'reason':item['reason']})
            quality.append({'topic':topic,'model':raw['model'],'candidate_mapping':raw['mapping'],
                            'winner_id':raw['result']['winner'],'issues':issues,
                            'shared_only_issue_count':sum(x['appears_in_both_reports'] and not x['appears_in_this_group_new_content'] for x in issues)})
    ev.save(out/'共同输入原标注引句核对汇总.json',reappeared)
    ev.save(out/'再暴露裁判漏读证据.json',{'method':'只列模型absent被完整原句定位直接反驳的反例。','counterexamples':disproved,'main_report_uses_model_reexposure_rates':False})
    ev.save(out/'质量评审共同章节问题核查.json',quality)
    ev.save(out/'评审后冻结标注指纹核查.json',annotations)
    print(json.dumps({'unchanged_annotation_sets':len(annotations),'disproved_absent':len(disproved),
        'quality_reviews':[{'topic':x['topic'],'model':x['model'],'shared_only_issues':x['shared_only_issue_count']} for x in quality]},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
