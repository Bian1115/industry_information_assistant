import { Alert, Button, Collapse, Descriptions, Drawer, Dropdown, Empty, Input, Modal, Select, Space, Spin, Table, Tabs, Tag, Typography, message } from 'antd'
import { DownloadOutlined, FileSearchOutlined, ReloadOutlined } from '@ant-design/icons'
import { useMemo, useRef, useState } from 'react'
import Markdown from '@/components/markdown'
import { exportMedicalResearch, getMedicalPaper, scopeFromBundle, type Evidence, type ExportFormat, type MedicalBundle, type Paper, type ScreeningRecord } from '@/api/medical'
import type { ResearchDetailData } from './research-detail'
import './medical-workbench.scss'

const text = (value: unknown): string => value == null || value === '' ? '未报告' : Array.isArray(value) ? value.map(text).join('；') : typeof value === 'object' ? JSON.stringify(value) : String(value)
const decisionLabels: Record<string, string> = { include: '纳入', exclude: '排除', uncertain: '待确认' }
const sourceLabels: Record<string, string> = { pubmed: 'PubMed', europe_pmc: 'Europe PMC', clinical_trials: 'ClinicalTrials.gov' }
const accessStatusLabels: Record<string, string> = { full_text_available: '已获取全文', abstract_available: '可用摘要', metadata_only: '仅题录可用', registry_available: '可用试验注册记录', open_access: '开放获取', unavailable: '不可获取', failed: '获取失败', paywalled: '需要订阅或授权' }
const accessLabels: Record<string, string> = { abstract: '仅摘要', abstract_only: '仅摘要', full_text: '全文', fulltext: '全文', metadata: '仅题录', metadata_only: '仅题录', trial_registry: '试验注册', registry: '试验注册' }
const verificationLabels: Record<string, string> = { verified: '已核验', quote_verified: '原文匹配', quote_matched: '原文匹配', quote_not_found: '引句未匹配', missing_quote: '缺少原文引句', supported: '有支持', unverified: '未核验', not_evaluated: '未核验', pending: '待核验', failed: '核验未通过', unsupported: '支持不足', human_corrected: '人工纠正' }
const correctionFields: Array<[keyof Evidence, string]> = [['study_design', '研究设计'], ['population', '人群'], ['sample_size', '样本量'], ['intervention', '干预'], ['comparator', '对照'], ['outcome', '结局'], ['effect', '效应值'], ['confidence_interval', '置信区间'], ['follow_up', '随访'], ['safety', '安全性'], ['limitations', '局限']]
const safeUrl = (url?: string) => url && /^https?:\/\//i.test(url) ? url : undefined
const paperLink = (paper: Paper) => safeUrl(paper.url) || (paper.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(paper.pmid)}/` : paper.doi ? `https://doi.org/${encodeURIComponent(paper.doi)}` : undefined)

interface Props {
  bundle: MedicalBundle | null
  runId: string | null
  status: string
  detail?: ResearchDetailData
  error?: string
  onBundleChange: (bundle: MedicalBundle) => void
  onRefresh: () => void
  onUpdate?: () => void
  onResume?: () => void
}

// 历史疾病研究记录为只读视图：不再回写 bundle，因此不消费 onBundleChange。
export default function MedicalWorkbench({ bundle, runId, status, detail, error, onRefresh, onUpdate, onResume }: Props) {
  const [filter, setFilter] = useState('all')
  const [query, setQuery] = useState('')
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null)
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null)
  const [paperLoading, setPaperLoading] = useState(false)
  const [paperError, setPaperError] = useState('')
  const paperRequestRef = useRef(0)
  const [reviewTarget, setReviewTarget] = useState<Paper | null>(null)
  const [decision, setDecision] = useState('include')
  const [reason, setReason] = useState('')
  const [saving] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [correction, setCorrection] = useState<Evidence | null>(null)
  const [correctionValues, setCorrectionValues] = useState<Record<string, string>>({})
  const [correctionReason, setCorrectionReason] = useState('')
  const papers = bundle?.papers || []
  const evidence = bundle?.evidence || []
  const screening = useMemo(() => new Map((bundle?.screening || []).map(row => [row.paper_id, row])), [bundle?.screening])
  const scope = bundle ? scopeFromBundle(bundle) : null
  const completed = status === 'completed'
  const running = status === 'running'
  const stale = Boolean(bundle?.review?.report_stale)
  const filtered = papers.filter(paper => (filter === 'all' || (screening.get(paper.paper_id)?.decision || 'uncertain') === filter)
    && (!query.trim() || [paper.title, paper.abstract, paper.pmid, paper.doi].some(value => value?.toLowerCase().includes(query.trim().toLowerCase()))))
  const labels: Record<string, string> = { running: '检索与整理中', completed: '研究已完成 · 待专业复核', failed: '研究中断', cancelled: '已停止', paused: '已暂停', idle: '等待开始' }

  async function download(format: ExportFormat) {
    if (!runId) return
    setExporting(true)
    try { await exportMedicalResearch(runId, format, scope?.disease || '疾病') }
    catch { message.error('导出失败，请稍后重试') }
    finally { setExporting(false) }
  }
  // 人工纳排入口已随独立医疗编排下线：历史记录只读，保存动作不再可用。
  async function saveReview() {
    message.info('历史疾病研究记录为只读，人工纳排入口已下线')
    setReviewTarget(null)
  }
  function beginReview(paper: Paper, record?: ScreeningRecord) {
    setReviewTarget(paper)
    setDecision(record?.decision || 'uncertain')
    setReason(record?.reason || '')
  }
  function beginCorrection(row: Evidence) {
    setCorrection(row)
    setCorrectionValues(Object.fromEntries(correctionFields.map(([key]) => [key, row[key] == null ? '' : text(row[key])])))
    setCorrectionReason('')
  }
  // 证据纠正入口同上：历史记录只读。
  async function saveCorrection() {
    message.info('历史疾病研究记录为只读，证据纠正入口已下线')
    setCorrection(null)
  }
  async function openPaper(paper: Paper, row: Evidence | null = null) {
    const requestId = ++paperRequestRef.current
    setSelectedPaper(paper)
    setSelectedEvidence(row)
    setPaperError('')
    if (!runId) return
    setPaperLoading(true)
    try {
      const detailed = await getMedicalPaper(runId, paper.paper_id)
      if (paperRequestRef.current === requestId) setSelectedPaper(detailed)
    } catch {
      if (paperRequestRef.current === requestId) setPaperError('原文段落读取失败，可点击来源阅读全文。')
    } finally {
      if (paperRequestRef.current === requestId) setPaperLoading(false)
    }
  }
  const currentReport = detail?.streamingReport || ''
  const section = selectedPaper?.sections?.find(item => item.section_id === selectedEvidence?.section_id)
  const exportItems = [
    { key: 'docx', label: '报告 · Word' }, { key: 'markdown', label: '报告 · Markdown' }, { key: 'html', label: '报告 · HTML（可浏览器打印为 PDF）' },
    { key: 'csv', label: '证据表 · CSV' }, { key: 'bibtex', label: '参考文献 · BibTeX' },
    { key: 'ris', label: '参考文献 · RIS' }, { key: 'json', label: '完整研究记录 · JSON' },
  ]

  return <section className="medical-workbench" aria-label="医学文献研究结果">
    <div className="medical-workbench__header"><div><div className="medical-start__eyebrow">医学证据工作台</div><h2>{scope?.disease || '疾病文献研究'}</h2>
      <Space wrap><Tag color={running ? 'processing' : completed ? 'success' : status === 'failed' ? 'error' : 'default'}>{labels[status] || status}</Tag>{scope?.end_date && <span className="medical-workbench__muted">检索截止：{scope.end_date}</span>}</Space></div>
      <Space wrap>{onResume && ['paused', 'cancelled', 'failed'].includes(status) && <Button type="primary" onClick={onResume}>继续本次研究</Button>}<Button icon={<ReloadOutlined />} onClick={onRefresh} disabled={!runId || running}>刷新</Button><Dropdown menu={{ items: exportItems, onClick: ({ key }) => void download(key as ExportFormat) }} disabled={!bundle || !runId}><Button icon={<DownloadOutlined />} loading={exporting}>导出研究</Button></Dropdown></Space>
    </div>
    {running && <div className="medical-workbench__muted" style={{ marginTop: 14 }}><Spin size="small" /> 文献、筛选和证据会随研究逐步更新。</div>}
    {Boolean(bundle?.agent_steps?.length) && <div aria-label="研究协作进度" style={{ marginTop: 16 }}>
      <Typography.Text strong>研究协作进度</Typography.Text>
      <Space wrap style={{ display: 'flex', marginTop: 8 }}>{bundle?.agent_steps?.map(step => {
        const stepStatus = step.status === 'running' && !running ? (completed ? 'completed' : 'paused') : step.status
        const stepLabels: Record<string, string> = { running: '进行中', completed: '已完成', failed: '中断', cancelled: '已暂停', paused: '已暂停' }
        return <Tag key={step.type} color={stepStatus === 'running' ? 'processing' : stepStatus === 'completed' ? 'success' : 'default'}>{step.title} · {stepLabels[stepStatus] || stepStatus}</Tag>
      })}</Space>
    </div>}
    {error && <Alert style={{ marginTop: 16 }} type="error" showIcon message={error} action={<Button size="small" onClick={onRefresh}>重试</Button>} />}
    {['failed', 'cancelled', 'paused'].includes(status) && <Alert style={{ marginTop: 16 }} type="warning" showIcon message="本次研究未完整完成，以下为已保存的中间结果。" />}
    {completed && bundle?.status === 'partial' && <Alert style={{ marginTop: 16 }} type="warning" showIcon message="本次仅形成部分研究结果，证据不足或审核尚未完成。请查看局限与核验记录。" />}
    {stale && <Alert style={{ marginTop: 16 }} type="warning" showIcon message="纳排记录已变更，当前报告尚未反映这些变更。" description="请保留本次审核记录，并在重新生成报告后检查结论和引用。" />}
    <div className="medical-workbench__metrics">{[
      ['检索文献', papers.length], ['纳入文献', Array.from(screening.values()).filter(row => row.decision === 'include').length],
      ['获取全文', papers.filter(paper => ['full_text', 'fulltext'].includes(paper.content_level || '')).length], ['提取证据', evidence.length],
    ].map(([label, count]) => <div className="medical-workbench__metric" key={label}><strong>{count}</strong><span>{label}</span></div>)}</div>
    <Tabs items={[
      { key: 'papers', label: `文献清单 ${papers.length}`, children: <>
        <Space wrap style={{ marginBottom: 8 }}><Input.Search aria-label="筛选文献" placeholder="搜索标题、摘要、PMID 或 DOI" allowClear value={query} onChange={event => setQuery(event.target.value)} style={{ width: 280 }} /><Select aria-label="纳排状态" value={filter} onChange={setFilter} style={{ width: 125 }} options={[{ value: 'all', label: '全部状态' }, ...Object.entries(decisionLabels).map(([value, label]) => ({ value, label }))]} /></Space>
        {!filtered.length && <Empty description={papers.length ? '没有符合筛选条件的文献' : running ? '正在检索文献' : '暂无文献记录，请查看检索日志与局限'} />}
        {filtered.map(paper => { const record = screening.get(paper.paper_id); return <article className="medical-paper" key={paper.paper_id}>
          <h3><a href={paperLink(paper)} target="_blank" rel="noopener noreferrer">{paper.title || '无标题'}</a></h3>
          <Space wrap size={[6, 8]}><Tag color={record?.decision === 'include' ? 'green' : record?.decision === 'exclude' ? 'default' : 'orange'}>{decisionLabels[record?.decision || 'uncertain'] || record?.decision}</Tag><Tag>{accessLabels[paper.content_level || ''] || paper.content_level || '内容级别未知'}</Tag><Tag>{sourceLabels[paper.source || ''] || paper.source || '来源待确认'}</Tag>{paper.is_preprint && <Tag color="orange">预印本</Tag>}{paper.retraction_status && !['none', 'not_retracted', 'unknown'].includes(paper.retraction_status) && <Tag color="red">撤回 / 更正：{paper.retraction_status}</Tag>}</Space>
          <p className="medical-workbench__muted">{[paper.journal, paper.publication_date, paper.pmid && `PMID: ${paper.pmid}`, paper.doi && `DOI: ${paper.doi}`].filter(Boolean).join(' · ')}</p>
          {record?.reason && <p className="medical-workbench__muted">筛选依据：{record.reason}</p>}
          <Space wrap><Button size="small" icon={<FileSearchOutlined />} onClick={() => void openPaper(paper)}>查看摘要与原文</Button><Button size="small" disabled={!runId || running} onClick={() => beginReview(paper, record)}>人工审核纳排</Button></Space>
        </article> })}
      </> },
      { key: 'evidence', label: `证据表 ${evidence.length}`, children: <><Alert type="info" showIcon message="点击证据查看来源原文。缺失字段保留“未报告”，不同研究的终点与人群需要分别解读。" style={{ marginBottom: 16 }} /><Table<Evidence> rowKey="evidence_id" size="small" dataSource={evidence} scroll={{ x: 1100 }} pagination={{ pageSize: 10 }} columns={[
        { title: '研究', key: 'paper', width: 230, render: (_, row) => { const paper = papers.find(item => item.paper_id === row.paper_id); return <Button type="link" style={{ whiteSpace: 'normal', height: 'auto', textAlign: 'left', padding: 0 }} onClick={() => void openPaper(paper || { paper_id: row.paper_id, title: row.paper_id }, row)}>{paper?.title || row.paper_id}</Button> } },
        { title: '设计 / 人群 / 样本', key: 'design', width: 180, render: (_, row) => <>{text(row.study_design)}<br />{text(row.population)}<br />N = {text(row.sample_size)}</> },
        { title: '干预 / 对照', key: 'comparison', width: 180, render: (_, row) => <>{text(row.intervention)}<br />对照：{text(row.comparator)}</> },
        { title: '结局 / 效应', key: 'effect', width: 210, render: (_, row) => <>{text(row.outcome)}<br />{text(row.effect)}<br />置信区间：{text(row.confidence_interval)}</> },
        { title: '随访 / 安全性', key: 'safety', width: 160, render: (_, row) => <>{text(row.follow_up)}<br />{text(row.safety)}</> },
        { title: '核验', dataIndex: 'verification_status', width: 125, render: value => verificationLabels[value] || text(value) },
        { title: '人工复核', key: 'correct', width: 95, render: (_, row) => <Button size="small" disabled={!runId || running} onClick={() => beginCorrection(row)}>纠正字段</Button> },
      ]} /></> },
      { key: 'report', label: '研究进展报告', children: currentReport ? <div className="medical-workbench__report"><Markdown value={currentReport} /></div> : <Empty description={running ? '文献整理完成后生成研究报告' : '本次尚未生成报告'} /> },
      { key: 'log', label: '检索记录与范围', children: <>
        <Descriptions column={1} size="small" bordered items={[
          { key: 'run', label: '研究编号', children: runId ? <Typography.Text copyable>{runId}</Typography.Text> : '未分配' },
          { key: 'range', label: '时间范围', children: `${scope?.start_date || '不限'} — ${scope?.end_date || '未设置'}` },
          { key: 'population', label: '人群', children: scope?.population || '未限定' },
          { key: 'synonyms', label: '同义词', children: scope?.synonyms?.join('；') || '未指定' },
          { key: 'include', label: '纳入条件', children: scope?.inclusion_criteria?.join('；') || '未指定' },
          { key: 'exclude', label: '排除条件', children: scope?.exclusion_criteria?.join('；') || '未指定' },
          { key: 'source', label: '来源', children: scope?.sources?.map(source => sourceLabels[source] || source).join('、') || '未记录' },
        ]} />
        <Collapse style={{ marginTop: 16 }} items={(bundle?.search_log || []).map((entry, index) => ({ key: String(index), label: `${sourceLabels[String(entry.source)] || entry.source || '检索'} · ${entry.status || entry.event || `记录 ${index + 1}`}`, children: <Descriptions size="small" column={1} items={Object.entries(entry).map(([key, value]) => ({ key, label: ({ query: '检索式', source: '来源', count: '记录数', result_count: '结果数', status: '状态', error: '失败原因', started_at: '开始时间', elapsed_seconds: '耗时（秒）' } as Record<string, string>)[key] || key, children: <span style={{ overflowWrap: 'anywhere' }}>{text(value)}</span> }))} /> }))} />
        {!bundle?.search_log?.length && <p className="medical-workbench__muted">暂无检索日志。</p>}
        {bundle?.stats && <p className="medical-workbench__muted">{Object.entries(bundle.stats).filter(([, value]) => typeof value === 'number').map(([key, value]) => `${({ llm_calls: '模型调用', search_requests: '检索请求', papers_count: '文献数', included_count: '纳入数', full_text_count: '全文数', evidence_count: '证据数' } as Record<string, string>)[key] || key}：${value}`).join(' · ')}</p>}
      </> },
      { key: 'review', label: '局限与更新', children: <>
        <Alert type="info" showIcon message="本工具整理研究证据，交付前需要专业复核。摘要证据、预印本和试验注册不会自动等同于成熟临床结论。" />
        {(bundle?.limitations || []).map((limitation, index) => <p key={index}>{index + 1}. {text(limitation)}</p>)}
        {bundle?.review && <Collapse style={{ marginTop: 16 }} items={[{ key: 'review', label: '核验与审核记录', children: <Descriptions column={1} items={Object.entries(bundle.review).map(([key, value]) => ({ key, label: ({ status: '状态', issues: '发现的问题', report_stale: '报告待重新生成', human_reviews: '人工审核' } as Record<string, string>)[key] || key, children: text(value) }))} /> }]} />}
        {bundle?.changes && Object.keys(bundle.changes).length > 0 && <Collapse style={{ marginTop: 16 }} items={[{ key: 'changes', label: '与上次研究比较', children: <Descriptions column={1} items={Object.entries(bundle.changes).map(([key, value]) => ({ key, label: ({ added: '新增', removed: '本轮未检出', changed: '发生变化', updated: '内容更新', status: '比较状态', note: '说明', previous_run_id: '上次研究编号' } as Record<string, string>)[key] || key, children: text(value) }))} /> }]} />}
        {onUpdate && completed && <Button style={{ marginTop: 18 }} onClick={onUpdate}>{stale ? '按审核后的记录重新生成研究' : '以本次为基线，创建研究更新'}</Button>}
      </> },
    ]} />
    <Drawer width={700} open={Boolean(selectedPaper)} title={selectedPaper?.title} onClose={() => { paperRequestRef.current++; setSelectedPaper(null); setSelectedEvidence(null); setPaperLoading(false) }}>
      {paperLoading && <p><Spin size="small" /> 正在读取保存的原文段落…</p>}
      {paperError && <Alert type="warning" showIcon message={paperError} />}
      {selectedPaper && <><Space wrap><Tag>{accessLabels[selectedPaper.content_level || ''] || selectedPaper.content_level || '内容级别未知'}</Tag><a href={paperLink(selectedPaper)} target="_blank" rel="noopener noreferrer">打开来源</a></Space>
        <p className="medical-workbench__muted">{selectedPaper.authors?.join('；') || '作者未记录'}<br />{selectedPaper.journal} · {selectedPaper.publication_date}<br />获取状态：{accessStatusLabels[selectedPaper.access_status || ''] || selectedPaper.access_status || '未记录'}</p>
        {selectedEvidence && <><h3>支持证据原文</h3><blockquote className="medical-evidence-quote">{selectedEvidence.quote || '未提供原文引句'}</blockquote><p>原文位置：{selectedEvidence.section_id || '未定位'} · {verificationLabels[selectedEvidence.verification_status || ''] || selectedEvidence.verification_status || '未核验'}</p><p>研究局限：{text(selectedEvidence.limitations)}</p>{section && <div className="medical-paper__abstract"><strong>{section.title || section.section_id}</strong><p>{section.text || section.content}</p></div>}</>}
        <h3>摘要</h3><p className="medical-paper__abstract">{selectedPaper.abstract || '来源未提供摘要'}</p>
        <Collapse items={(selectedPaper.sections || []).map((item, index) => ({ key: item.section_id || String(index), label: `${item.title || '原文段落'} · ${item.section_id || index + 1}`, children: <p className="medical-paper__abstract">{item.text || item.content || '内容为空'}</p> }))} />
      </>}
    </Drawer>
    <Modal title="人工审核文献纳排" open={Boolean(reviewTarget)} onCancel={() => setReviewTarget(null)} onOk={() => void saveReview()} confirmLoading={saving} okButtonProps={{ disabled: !reason.trim() }} okText="保存审核记录" cancelText="取消">
      <p>{reviewTarget?.title}</p><Select aria-label="人工纳排决定" value={decision} onChange={setDecision} options={Object.entries(decisionLabels).map(([value, label]) => ({ value, label }))} style={{ width: '100%', marginBottom: 14 }} /><Input.TextArea aria-label="纳排理由" maxLength={4000} showCount value={reason} onChange={event => setReason(event.target.value)} placeholder="请说明纳入、排除或待确认的依据" rows={4} /><p className="medical-workbench__muted">变更会保存到研究记录；现有报告将标记为需要重新生成。</p>
    </Modal>
    <Drawer title="纠正结构化证据" width={700} open={Boolean(correction)} onClose={() => setCorrection(null)} extra={<Button type="primary" loading={saving} disabled={!correctionReason.trim()} onClick={() => void saveCorrection()}>保存纠正</Button>}>
      <Alert type="info" showIcon message="依据原文纠正提取字段，并说明理由。修改会保留审核记录，当前报告将标记为待更新。" />
      <blockquote className="medical-evidence-quote">{correction?.quote || '该条证据未提供原文引句，请先核对来源。'}</blockquote>
      <div className="medical-form-grid">{correctionFields.map(([key, label]) => <label key={key} style={{ display: 'block', marginBottom: 14 }}>{label}<Input.TextArea aria-label={`纠正${label}`} maxLength={6000} value={correctionValues[key] || ''} onChange={event => setCorrectionValues(values => ({ ...values, [key]: event.target.value }))} autoSize={{ minRows: 2, maxRows: 5 }} placeholder="未报告可留空" /></label>)}</div>
      <label>纠正理由（必填）<Input.TextArea aria-label="证据纠正理由" maxLength={4000} showCount rows={3} value={correctionReason} onChange={event => setCorrectionReason(event.target.value)} placeholder="说明原字段的问题及原文依据" /></label>
    </Drawer>
  </section>
}
