import { Alert, Button, Collapse, Descriptions, Dropdown, Empty, Space, Tabs, Tag, message } from 'antd'
import { DownloadOutlined, ReloadOutlined } from '@ant-design/icons'
import { useState } from 'react'
import Markdown from '@/components/markdown'
import { getMedicalResearch, type MedicalBundle } from '@/api/medical'
import type { ResearchDetailData } from './research-detail'
import './medical-workbench.scss'

const rows = (value: unknown): Record<string, unknown>[] => Array.isArray(value) ? value.filter(item => item && typeof item === 'object') : []
const text = (value: unknown) => value == null ? '' : typeof value === 'string' ? value : JSON.stringify(value)
const safeLink = (value: unknown) => /^https?:\/\//i.test(text(value)) ? text(value) : undefined
const labels: Record<string, string> = { running: '研究进行中', completed: '研究完成', failed: '研究中断', paused: '研究已暂停', cancelled: '研究已暂停', idle: '等待研究' }

export default function GraphWorkbench({ bundle, runId, status, detail, error, title, onRefresh, onResume }: {
  bundle: MedicalBundle | null; runId: string | null; status: string; detail?: ResearchDetailData; error: string; title: string
  onRefresh: () => Promise<void>; onResume: () => void
}) {
  const [exporting, setExporting] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const state = bundle?.graph_state
  const sources = rows(state?.draft?.raw_sources)
  const facts = rows(state?.draft?.facts)
  const references = rows(state?.draft?.references)
  const queries = rows(state?.draft?.search_budget?.queries)
  const attempts = rows(state?.draft?.web_reader_attempts)
  const running = status === 'running'
  const counts = state?.progress?.counts || { queries: queries.length, sources: sources.length, facts: facts.length, references: references.length }
  async function refresh() {
    setRefreshing(true)
    try { await onRefresh() }
    finally { setRefreshing(false) }
  }
  async function download(format: string) {
    if (!runId) return
    setExporting(true)
    try {
      const latest = await getMedicalResearch(runId)
      if (latest.pipeline !== 'graph') throw new Error('研究类型已变化，请刷新')
      const body = format === 'json' ? JSON.stringify(latest.graph_state, null, 2) : latest.final_report
      if (!body) { message.info('本次尚未生成报告，可导出 JSON 研究记录'); return }
      const url = URL.createObjectURL(new Blob([body], { type: format === 'json' ? 'application/json' : 'text/markdown;charset=utf-8' }))
      const link = document.createElement('a')
      link.href = url
      link.download = `研究结果-${runId}.${format === 'json' ? 'json' : 'md'}`
      document.body.appendChild(link)
      link.click()
      link.remove()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch { message.error('导出失败，请重试') }
    finally { setExporting(false) }
  }
  const reportText = detail?.streamingReport || ''
  return <section className="medical-workbench" aria-label="研究结果">
    <div className="medical-workbench__header"><div><div className="medical-start__eyebrow">研究工作台</div><h2 title={state?.original_query || state?.query}>{state?.resolved_query || state?.query || title || '研究结果'}</h2><Tag color={running ? 'processing' : status === 'failed' ? 'error' : 'default'}>{labels[status] || status}</Tag></div>
      <Space wrap>{(state?.delivery_pending || ['failed', 'paused', 'cancelled'].includes(status)) && <Button type="primary" disabled={!state?.resumable} onClick={onResume}>{state?.delivery_pending ? '完成交付' : '继续本次研究'}</Button>}<Button icon={<ReloadOutlined />} loading={refreshing} disabled={!runId} onClick={() => void refresh()}>刷新</Button><Dropdown disabled={!runId} menu={{ items: [{ key: 'markdown', label: '报告 · Markdown' }, { key: 'json', label: '研究记录 · JSON' }], onClick: ({ key }) => void download(key) }}><Button loading={exporting} icon={<DownloadOutlined />}>导出研究</Button></Dropdown></Space>
    </div>
    {error && <Alert style={{ marginTop: 16 }} type="error" showIcon message={error} action={<Button onClick={onRefresh}>重新读取</Button>} />}
    {['failed', 'paused', 'cancelled'].includes(status) && <Alert style={{ marginTop: 16 }} type="warning" showIcon message="研究未完整完成，以下展示已保存的资料与报告。" />}
    {state && !state.exists && <Alert style={{ marginTop: 16 }} type="info" message="本次研究尚未保存执行检查点，无法断点继续。请新建研究重试。" />}
    <div className="medical-workbench__metrics">{[['检索请求', counts.queries], ['已读取来源', counts.sources], ['整理事实', counts.facts], ['报告引用', counts.references]].map(([label, count]) => <div className="medical-workbench__metric" key={label}><strong>{count}</strong><span>{label}</span></div>)}</div>
    {bundle?.fetched_at && <p className="medical-workbench__muted">最近刷新：{new Date(bundle.fetched_at).toLocaleTimeString()}{running ? ' · 每 5 秒自动刷新' : ''}。{state?.progress ? '计数包含当前阶段进度，资料明细在阶段完成后更新。' : running ? '当前显示已保存阶段的计数，等待新的进度记录。' : ''}</p>}
    <Tabs items={[
      { key: 'report', label: '研究报告', children: reportText ? <Markdown value={reportText} /> : <Empty description={running ? '正在研究，报告会随执行更新' : '尚未生成报告'} /> },
      { key: 'sources', label: `来源资料 ${sources.length}`, children: sources.length ? <Collapse items={sources.map((source, i) => ({ key: text(source.id) || String(i), label: text(source.title) || text(source.url), children: <><a href={safeLink(source.url || source.source_url)} target="_blank" rel="noopener noreferrer">打开来源</a><p>{text(source.published_at)}</p><div style={{ whiteSpace: 'pre-wrap' }}>{text(source.content)}</div></> }))} /> : <Empty description="尚无已保存的来源正文；请查看读取记录" /> },
      { key: 'facts', label: `整理事实 ${facts.length}`, children: facts.length ? facts.map((fact, i) => <article className="medical-paper" key={text(fact.id) || i}><p>{text(fact.content)}</p><span className="medical-workbench__muted">来源：{text(sources.find(source => source.id === fact.source_id)?.title || fact.source_id) || '未关联'}</span></article>) : <Empty description="尚无已保存的事实" /> },
      { key: 'log', label: '检索与读取记录', children: <><Descriptions column={1} items={[{ key: 'run', label: '研究编号', children: runId }, { key: 'domains', label: '检索网站', children: state?.allowed_domains ? state.allowed_domains.join('、') || '全网搜索' : '未记录' }, { key: 'phase', label: '当前阶段', children: state?.phase || '未记录' }]} /><Collapse items={[{ key: 'queries', label: `检索请求 ${queries.length}`, children: <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(queries, null, 2)}</pre> }, { key: 'reads', label: `网页读取 ${attempts.length}`, children: <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(attempts, null, 2)}</pre> }]} /></> },
    ]} />
  </section>
}
