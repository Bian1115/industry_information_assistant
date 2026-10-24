"""校验同输入配对、冻结代码、免费运行审计，并独立重算代表样例。"""
import ast
import gzip
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'backend/eval/data/网页正文与表格优化对照评估-20261018-首轮'
VERSIONS = ['优化前', '候选初版', '候选收紧版', '候选表格净化版', '优化后']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(name, data):
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    spec = importlib.util.spec_from_file_location('优化评估核验计分', REPO / 'backend/eval/scripts/评估公开网页读取质量.py')
    metric = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metric
    spec.loader.exec_module(metric)
    attempts = metric.install_network_guard()
    signal.signal(signal.SIGALRM, metric.alarm_handler)
    before, after = [json.loads((ROOT / v / '逐页指标汇总.json').read_text()) for v in ['优化前', '优化后']]
    old = {r['id']: r for r in before}
    assert len(old) == len(after) == 745
    checks = []
    for row in after:
        sid = row['id']
        previous = old[sid]
        assert previous['input'] == row['input']
        for label in ['优化前', '优化后']:
            folder = ROOT / label
            data = gzip.decompress((folder / row['input']['clean_html_path']).read_bytes())
            ref = (folder / '逐页结果' / sid / '参考正文.txt').read_bytes()
            assert sha(data) == row['input']['clean_html_sha256']
            assert sha(ref) == row['input']['reference_sha256']
            assert not re.search(rb'\scc-[\w-]+\s*=', data, re.I)
        checks.append({'id': sid, 'clean_html_sha256': sha(data), 'reference_sha256': sha(ref), 'matched': True})
    function_names = ['norm', 'tokens', 'overlap', 'render', 'visible', 'text_scores', 'official_words',
                      'snippets', 'clean_annotations', 'table_record', 'extract_tables', 'table_teds', 'table_scores']
    frozen = {}
    for label in ['优化前', '优化后']:
        tree = ast.parse((ROOT / label / '冻结评估脚本.py').read_text())
        frozen[label] = {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                         if isinstance(n, ast.FunctionDef) and n.name in function_names}
    assert frozen['优化前'] == frozen['优化后'] and len(frozen['优化前']) == len(function_names)
    cases = ['WebMainBench-0aa9f0da-dc0a-436b-bb63-692843f02ef4',
             'WebMainBench-34b822a2-f7d6-4b69-b6af-133a1ca21326',
             'WebMainBench-8ec3baea-f43f-4f57-bde4-b67507bd56c1',
             'WebMainBench-670bfc19-ef03-4389-8a76-5881d9476f49', 'WCXB-0556', 'WCXB-4317']
    rescored = []
    for label, rows in [('优化前', before), ('优化后', after)]:
        index = {r['id']: r for r in rows}
        for sid in cases:
            folder = ROOT / label / '逐页结果' / sid
            pred, ref = [(folder / name).read_text() for name in ['生产门禁前正文.txt', '参考正文.txt']]
            escaped = __import__('html').escape(ref) if sid.startswith('WCXB-') else ref
            text = metric.text_scores(pred, escaped)
            assert text == index[sid]['scores']['production_pre_gate']['text']
            if sid.startswith('WebMainBench-'):
                tables = metric.table_scores(pred, metric.extract_tables(ref))
                assert tables == index[sid]['scores']['production_pre_gate']['table']
            rescored.append({'version': label, 'id': sid, 'matched': True})
    dump('计分器与样例重算核验.json', {'日期': '2026-10-18', '同运行补充': '最终 v5 再核验；包括表单包裹论坛样例',
                                  '关键评分函数AST一致': True, '函数': function_names,
                                  '独立重算配对样例': rescored, '网络尝试': attempts, '费用': 0})
    source_checks = []
    for name in ['web_content_selection.py', 'web_reader_service.py', 'web_document.py', 'web_document_structure.py']:
        current = (REPO / 'backend/app/service' / name).read_bytes()
        saved = (ROOT / '优化后/冻结代码/service' / name).read_bytes()
        assert current == saved
        compile(current, name, 'exec')
        source_checks.append({'path': 'backend/app/service/' + name, 'sha256': sha(current), 'matches_final_frozen': True})
    audits = {}
    for label in VERSIONS:
        value = json.loads((ROOT / label / '网络与费用审计.json').read_text())
        assert value['completed_pages'] == 745 and not value['network_connection_attempts']
        assert all(value[k] == 0 for k in ['paid_api_calls', 'browser_executions', 'ocr_executions', 'llm_scoring_calls'])
        audits[label] = value
    testlog = (ROOT / '论坛表单修正后最终回归日志.txt').read_text()
    assert re.search(r'Ran 35 tests', testlog) and re.search(r'^OK$', testlog, re.M)
    for name in ['比较正文与表格优化结果.py', '生成正文与表格优化评估报告.py', '核验正文与表格优化评估.py']:
        file = REPO / 'backend/eval/scripts' / name
        compile(file.read_text(), str(file), 'exec')
        shutil.copyfile(file, ROOT / ('冻结' + name))
    app_metadata = subprocess.run([str(REPO / 'backend/.venv/bin/python'), '-c',
        'import json,sys;from importlib.metadata import version;print(json.dumps({"python":sys.version,"markdownify":version("markdownify"),"trafilatura":version("trafilatura")}))'],
        check=True, capture_output=True, text=True)
    assert not attempts
    dump('最终输入源码与费用核验.json', {
        '日期': '2026-10-18', '最终策略': 'semantic-body-source-tables-v5', '配对总页数': 745,
        '质量有效配对页': sum(bool(r['reference_chars']) for r in after), '所有输入与参考重算校验一致': True,
        '已剥离cc标注属性': True, '逐页校验': checks, '源码核验': source_checks,
        '评测环境新增依赖': {'markdownify': version('markdownify')}, '应用环境': json.loads(app_metadata.stdout),
        '全部版本审计': audits, '总提取页次': sum(a['completed_pages'] for a in audits.values()),
        '付费API调用': 0, '新增外部服务费用': 0, '费用边界': '本地开源计算；不核算电费、存储、已有订阅或网络接入费用',
        '回归': {'passed': 35, 'log': '论坛表单修正后最终回归日志.txt'}, '核验过程网络尝试': attempts})
    print(json.dumps({'配对页数': len(checks), '重算样例': len(rescored), '测试通过': 35, '费用': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
