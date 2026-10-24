import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Alert, App, ConfigProvider, Segmented } from 'antd'
import MedicalScopeForm from '../src/pages/chat/component/medical-scope-form'
import MedicalWorkbench from '../src/pages/chat/component/medical-workbench'
import { defaultMedicalScope, type MedicalBundle } from '../src/api/medical'

// 开发期组件夹具，不连接 API，不进入生产路由或构建入口。
const fixture: MedicalBundle = {
  protocol: { ...defaultMedicalScope(), disease: '组件测试疾病', synonyms: ['example disease'], population: '模拟人群', start_date: '2023-01-01' },
  status: 'partial',
  papers: [
    { paper_id: 'fixture:1', title: '组件测试文献：研究设计、主要终点与较长标题的展示效果', source: '模拟数据', journal: '组件验证资料', publication_date: '2026-01-01', authors: ['测试作者'], content_level: 'abstract', access_status: 'abstract_available', abstract: '这段文字仅用于验证摘要排版、换行和原文定位，不代表任何疾病的真实医学结论。', sections: [{ section_id: 'abstract', title: '模拟摘要', text: '这段文字仅用于验证摘要排版、换行和原文定位，不代表任何疾病的真实医学结论。' }] },
    { paper_id: 'fixture:2', title: '组件测试文献：待确认的预印本', source: '模拟数据', publication_date: '2026-02-01', content_level: 'metadata', is_preprint: true, sections: [] },
  ],
  screening: [{ paper_id: 'fixture:1', decision: 'include', reason: '仅作为纳入卡片布局测试' }, { paper_id: 'fixture:2', decision: 'uncertain', reason: '缺少摘要，需人工确认（组件夹具）' }],
  evidence: [{ evidence_id: 'fixture:e1', paper_id: 'fixture:1', study_design: '模拟研究设计', population: '模拟人群', sample_size: '120（模拟）', intervention: '方案 A（模拟）', comparator: '方案 B（模拟）', outcome: '模拟终点', effect: '未报告', confidence_interval: null, quote: '这段文字仅用于验证摘要排版、换行和原文定位，不代表任何疾病的真实医学结论。', section_id: 'abstract', verification_status: 'quote_matched', limitations: '全部数据均为组件测试夹具。' }],
  search_log: [{ source: '模拟数据', query: 'example disease', status: '组件夹具', count: 2 }],
  stats: { papers_count: 2, included_count: 1, full_text_count: 0, evidence_count: 1, llm_calls: 0 },
  limitations: ['此页未执行检索、模型推理、后端持久化、审核或导出，不构成系统端到端验证。'],
  review: { status: '组件验证', issues: ['需要真实后端集成验证'] },
  changes: { status: 'initial', added: [], note: '组件夹具，无真实版本比较。' },
  final_report: '# 研究进展报告 · 组件测试\n\n这是一份用于验证 Markdown 排版的模拟报告。\n\n| 展示内容 | 状态 |\n| --- | --- |\n| 文献表格 | 组件验证 |\n| 真实检索与结论 | 未执行 |\n\n## 结论与局限\n\n此页只验证界面，不提供疾病结论。<script>alert("不应执行")</script>',
}

function Preview() {
  const [tab, setTab] = useState('表单')
  const [submitted, setSubmitted] = useState('')
  return <ConfigProvider theme={{ token: { colorPrimary: '#17786e', borderRadius: 8, fontFamily: 'system-ui, sans-serif' } }}><App>
    <Alert type="warning" showIcon banner message="开发组件验证：模拟数据，不连接真实后端。此页面不代表医学效果或后端操作通过。" />
    <div style={{ padding: 20 }}><Segmented value={tab} onChange={setTab} options={['表单', '结果工作台', '空结果', '中断结果']} /></div>
    {tab === '表单' ? <main className="medical-start" style={{ paddingTop: 8 }}><h1>疾病研究范围表单</h1><div className="medical-start__card"><MedicalScopeForm initialValue={{ ...defaultMedicalScope(), disease: '组件测试疾病' }} onSubmit={(scope, question) => setSubmitted(JSON.stringify({ scope, question }, null, 2))} submitLabel="验证表单提交（不请求后端）" /></div>{submitted && <pre style={{ whiteSpace: 'pre-wrap' }} aria-label="本地表单提交结果">{submitted}</pre>}</main> : <MedicalWorkbench detail={{ stepId: 'writing', stepType: 'writing', title: '研究报告', streamingReport: tab === '空结果' ? '' : fixture.final_report }} bundle={tab === '空结果' ? { papers: [], limitations: ['组件测试：无检索结果。'] } : fixture} runId={null} status={tab === '中断结果' ? 'failed' : 'completed'} onBundleChange={() => {}} onRefresh={() => {}} />}
  </App></ConfigProvider>
}

createRoot(document.getElementById('root')!).render(<Preview />)
