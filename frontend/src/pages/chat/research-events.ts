/** DeepResearch SSE v1 协议：解析、兼容旧事件和纯 reducer。 */

// 兼容层最终要交给尚未完成类型化的旧 UI 分支，迁移完成后可收紧为 JsonValue。
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type JsonObject = Record<string, any>

/** 报告审核分数与已有引用核验分开呈现，缺少核验结果不视为通过。 */
export function formatResearchReview(content: JsonObject): { passed: boolean; text: string } {
  const score = content.quality_score
  const passed = typeof content.verdict === 'string'
    ? content.verdict === 'pass'
    : typeof content.passed === 'boolean' ? content.passed : typeof score === 'number' && score >= 7
  const lines = [
    `🔍 审核结果: 报告审核评分 ${typeof score === 'number' ? `${score}/10` : '未评分'}`,
    passed ? '✅ 审核通过' : '⚠️ 需要修订',
  ]
  const dimensions = content.quality_dimension_scores || {}
  const labels = [
    ['logical_coherence', '整体逻辑'],
    ['requirement_coverage', '要求覆盖'],
    ['freshness', '时效性'],
  ]
  const scores = labels
    .filter(([key]) => typeof dimensions[key] === 'number')
    .map(([key, label]) => `${label} ${dimensions[key]}/10`)
  if (scores.length) lines.push(scores.join('；'))

  const audit = content.citation_audit || {}
  const statusLabels: Record<string, string> = {
    passed: '通过', failed: '未通过', not_evaluated: '未核验',
    disabled: '核验已关闭', no_citations: '没有可核验引用',
  }
  const count = (value: unknown) => typeof value === 'number' && Number.isInteger(value) && value >= 0 ? String(value) : '—'
  const trace = audit.traceability || {}
  const support = audit.support || {}
  lines.push(`引用溯源（URL）：${statusLabels[trace.status] || '未核验'}；可追溯 ${count(trace.citations_verified)}/${count(trace.citations_total)}`)
  lines.push(`引用支持（论断—引用对）：${statusLabels[support.status] || '未核验'}；支持 ${count(support.supported)}/${count(support.citations_total)}；不支持 ${count(support.unsupported)}；未确认 ${count(support.unknown)}`)
  lines.push('核验范围：报告中的引用 URL，以及已抽取的行内论断—引用对；不代表全篇事实均已核验。')
  return { passed, text: lines.join('\n') }
}

export interface ResearchEventBase<TType extends string, TData extends JsonObject> {
  version: 1
  event_id: string
  type: TType
  session_id: string
  run_id: string
  turn_id?: string | null
  seq: number
  timestamp: string
  data: TData
}

export type ResearchEvent =
  | ResearchEventBase<'research_start', { query?: string; search_web?: boolean; search_local?: boolean }>
  | ResearchEventBase<'session_clarification', { question: string; content?: string }>
  | ResearchEventBase<'research_complete', { final_report?: string; references?: JsonObject[] }>
  | ResearchEventBase<'research_cancelled', { message?: string }>
  | ResearchEventBase<'error', { content?: string; error_code?: string }>
  | ResearchEventBase<'phase', { phase?: string; content?: unknown }>
  | ResearchEventBase<'research_step', { content?: JsonObject }>
  | ResearchEventBase<string, JsonObject>

export interface ResearchProtocolState {
  runId: string | null
  lastSeq: number
  status: 'idle' | 'running' | 'completed' | 'cancelled' | 'failed' | 'clarification'
  eventCount: number
}

export interface ResearchProtocolAdvance {
  state: ResearchProtocolState
  accepted: boolean
  reason: 'accepted' | 'duplicate' | 'gap' | 'run_mismatch'
  gapAfterSeq?: number
}

export interface ResearchRecoveryProjection {
  react_steps: API.ReactStep[]
  research_steps: JsonObject[]
  search_results: JsonObject[]
  charts: JsonObject[]
  knowledge_graph: JsonObject | null
  streaming_report: string
  references: JsonObject[]
  medical_research: JsonObject | null
  phase: string
  status: ResearchProtocolState['status']
}

export const initialResearchProtocolState: ResearchProtocolState = {
  runId: null,
  lastSeq: 0,
  status: 'idle',
  eventCount: 0,
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asObjects(value: unknown): JsonObject[] {
  return Array.isArray(value) ? value.filter(isObject) : []
}

function eventContent(event: ResearchEvent): JsonObject {
  const data: JsonObject = event.data
  return isObject(data.content) ? data.content : data
}

function upsertStep(steps: JsonObject[], next: JsonObject): JsonObject[] {
  const type = String(next.step_type || next.type || '')
  if (!type) return steps
  const index = steps.findIndex(step => String(step.type || step.step_type || '') === type)
  const normalized = { ...next, type }
  if (index < 0) return [...steps, normalized]
  return steps.map((step, current) => current === index ? { ...step, ...normalized } : step)
}

function mergeResults(current: JsonObject[], incoming: JsonObject[]): JsonObject[] {
  const byKey = new Map<string, JsonObject>()
  for (const result of [...current, ...incoming]) {
    const key = String(result.id || result.url || result.source_url || result.title || JSON.stringify(result))
    byKey.set(key, result)
  }
  return Array.from(byKey.values())
}

export function createRecoveryProjection(checkpoint: {
  status?: string
  phase?: string
  final_report?: string | null
  ui_state_json?: JsonObject
}): ResearchRecoveryProjection {
  const ui = checkpoint.ui_state_json || {}
  const status = checkpoint.status === 'completed' ? 'completed'
    : checkpoint.status === 'failed' ? 'failed'
      : ['paused', 'cancelled'].includes(checkpoint.status || '') ? 'cancelled' : 'running'
  return {
    react_steps: [],
    research_steps: asObjects(ui.research_steps),
    search_results: asObjects(ui.search_results),
    charts: asObjects(ui.charts),
    knowledge_graph: isObject(ui.knowledge_graph) ? ui.knowledge_graph : null,
    streaming_report: String(ui.streaming_report || checkpoint.final_report || ''),
    references: asObjects(ui.references),
    medical_research: isObject(ui.medical_research) ? ui.medical_research : null,
    phase: String(checkpoint.phase || ''),
    status,
  }
}

/**
 * 把图状态里的业务底稿投影成页面快照。
 *
 * 页面恢复的底稿来自当前检查点（资料与报告），进度事件另按 seq 重放：
 * 两者的游标职责分开，不能用事件序号推断执行位置，也不能用检查点冒充进度。
 */
export function projectionFromRunState(state: {
  run_status?: string
  status?: string
  phase?: string
  pending_nodes?: string[]
  draft?: {
    facts?: unknown
    references?: unknown
    knowledge_graph?: unknown
    final_report?: string
  }
}): ResearchRecoveryProjection {
  const draft = state.draft || {}
  // 图仍有待执行节点就是"可继续"，没有则按业务终态展示。
  const running = Array.isArray(state.pending_nodes) && state.pending_nodes.length > 0
  const runStatus = state.run_status === 'paused' ? 'cancelled' : state.run_status
  const status: ResearchProtocolState['status'] =
    runStatus === 'failed' || runStatus === 'cancelled' || runStatus === 'completed' || runStatus === 'running' || runStatus === 'idle'
      ? runStatus : running ? 'running' : state.status === 'completed' ? 'completed' : 'cancelled'
  return {
    react_steps: [],
    research_steps: [],
    search_results: asObjects(draft.facts).map(fact => ({
      id: String(fact.id || ''),
      title: String(fact.source_name || fact.content || '').slice(0, 80),
      url: String(fact.source_url || ''),
      snippet: String(fact.content || '').slice(0, 200),
      source: String(fact.source_type || 'web'),
    })),
    charts: [],
    knowledge_graph: isObject(draft.knowledge_graph) ? draft.knowledge_graph : null,
    streaming_report: String(draft.final_report || ''),
    references: asObjects(draft.references),
    medical_research: null,
    phase: String(state.phase || ''),
    status,
  }
}

/** 将一个账本事件确定性地投影到 Checkpoint UI 快照上。 */
export function reduceRecoveryProjection(
  projection: ResearchRecoveryProjection,
  event: ResearchEvent,
): ResearchRecoveryProjection {
  const data: JsonObject = event.data
  const content = eventContent(event)
  const next = { ...projection }
  const displayStep = recoveryDisplayStep(event, projection.react_steps.length + 1)
  if (displayStep) next.react_steps = [...projection.react_steps, displayStep]

  if (event.type === 'research_start' || event.type === 'research_resumed') {
    next.status = 'running'
  } else if (event.type === 'research_step') {
    next.research_steps = upsertStep(projection.research_steps, content)
  } else if ((event.type === 'phase' || event.type === 'phase_change')) {
    const phase = String(data.phase || content.phase || '')
    next.phase = phase
    if (phase) {
      next.research_steps = upsertStep(projection.research_steps, {
        type: phase,
        title: phase,
        status: 'running',
      })
    }
  } else if (event.type === 'search_results') {
    const incoming = asObjects(content.results || data.results)
    next.search_results = content.isIncremental || data.isIncremental
      ? mergeResults(projection.search_results, incoming)
      : incoming
  } else if (event.type === 'literature_update') {
    next.medical_research = content
  } else if (event.type === 'knowledge_graph') {
    const graph = isObject(content.graph) ? content.graph : content
    next.knowledge_graph = graph
  } else if (event.type === 'charts') {
    next.charts = asObjects(content.charts || data.charts)
  } else if (event.type === 'chart') {
    next.charts = mergeResults(projection.charts, [{ id: event.event_id, ...content }])
  } else if (event.type === 'section_content') {
    const section = String(content.content || '')
    const title = String(content.section_title || '')
    if (section) {
      const prefix = title ? `## ${title}\n\n` : ''
      next.streaming_report = projection.streaming_report
        ? `${projection.streaming_report}\n\n${prefix}${section}`
        : `${prefix}${section}`
    }
  } else if (event.type === 'report_chunk') {
    next.streaming_report = projection.streaming_report + String(content.chunk || '')
  } else if (event.type === 'report_draft') {
    const report = typeof data.content === 'string'
      ? data.content
      : String(content.content || '')
    if (report) next.streaming_report = report
  } else if (event.type === 'session_clarification') {
    next.status = 'clarification'
    next.streaming_report = String(data.question || data.content || '')
  } else if (event.type === 'research_complete') {
    next.streaming_report = String(data.final_report || projection.streaming_report)
    next.references = asObjects(data.references)
    next.status = 'completed'
    if (isObject(data.medical_research)) next.medical_research = data.medical_research
    next.research_steps = projection.research_steps.map(step => ({ ...step, status: 'completed' }))
  } else if (event.type === 'research_cancelled') {
    next.status = 'cancelled'
  } else if (event.type === 'research_paused') {
    next.status = 'cancelled'
  } else if (event.type === 'error') {
    next.status = 'failed'
  }

  return next
}

export const researchPhaseLabels: Record<string, string> = {
  routing: '追问路由', user_revising: '按要求修订',
  init: '规划阶段', planning: '规划阶段', researching: '搜索阶段', searching: '搜索阶段',
  analyzing: '分析阶段', writing: '写作阶段', reviewing: '审核阶段',
  re_researching: '补充搜索', rewriting: '重写阶段', revising: '修订阶段',
}

/** 首跑使用 reactSteps 渲染左侧；回放及续跑也必须恢复同一展示数据。 */
function recoveryDisplayStep(event: ResearchEvent, step: number): API.ReactStep | null {
  const data: JsonObject = event.data
  const content = eventContent(event)
  const describe = (value: unknown): string => typeof value === 'string' ? value
    : value == null ? '' : JSON.stringify(value, null, 2)
  const body = describe(content.content || content.message || (typeof data.content === 'string' ? data.content : ''))
  const base = { step, timestamp: Date.parse(event.timestamp) || 0 }
  switch (event.type) {
    case 'research_start':
      return { ...base, type: 'plan', content: `开始研究：${data.query || ''}` }
    case 'research_resumed':
      return { ...base, type: 'thought', content: '已从检查点继续本次研究' }
    case 'phase':
    case 'phase_change': {
      const phase = String(data.phase || content.phase || '')
      return { ...base, type: 'thought', content: `${researchPhaseLabels[phase] || phase}${body ? `：${body}` : ''}` }
    }
    case 'thought':
      return { ...base, type: 'thought', content: body }
    case 'action': {
      const tool = String(content.tool || data.tool || '')
      return { ...base, type: 'action', tool: tool || undefined,
        content: body || `调用工具：${tool}${content.section ? ` · ${content.section}` : ''}`,
        params: isObject(data.params) ? data.params : content }
    }
    case 'observation':
      return { ...base, type: 'observation', content: body || describe(content), success: content.success !== false }
    case 'research_step':
      return { ...base, type: 'thought', content: `${content.title || content.step_type || '研究阶段'}${content.status === 'completed' ? '：已完成' : content.status === 'running' ? '：进行中' : ''}` }
    case 'section_draft':
      return { ...base, type: 'observation', content: `章节「${content.section_title || '未命名'}」撰写完成` }
    case 'session_clarification':
      return { ...base, type: 'observation', content: String(data.question || data.content || '') }
    case 'research_complete':
      return { ...base, type: 'observation', content: '本次研究已完成', success: true }
    case 'research_paused':
    case 'research_cancelled':
      return { ...base, type: 'observation', content: body || '研究已暂停' }
    case 'error':
      return { ...base, type: 'observation', content: `研究中断：${body || '执行失败'}`, success: false }
    default:
      return null
  }
}

export function parseResearchSseData(text: string): ResearchEvent | null {
  if (!text || text === '[DONE]') return null
  const parsed: unknown = JSON.parse(text)
  if (!isObject(parsed) || typeof parsed.type !== 'string') {
    throw new Error('研究事件缺少 type')
  }

  if (parsed.version === 1 && isObject(parsed.data)) {
    if (typeof parsed.seq !== 'number' || typeof parsed.run_id !== 'string') {
      throw new Error('研究事件缺少 seq 或 run_id')
    }
    return parsed as ResearchEvent
  }

  // 兼容尚未升级的后端或历史录制文件。
  const { type, timestamp, ...data } = parsed
  return {
    version: 1,
    event_id: `legacy-${Date.now()}-${Math.random()}`,
    type,
    session_id: String(parsed.session_id || ''),
    run_id: String(parsed.run_id || parsed.turn_id || parsed.session_id || 'legacy'),
    turn_id: parsed.turn_id ? String(parsed.turn_id) : null,
    seq: Number(parsed.seq || 0),
    timestamp: String(timestamp || new Date().toISOString()),
    data,
  }
}

/** 按 SSE 行读取流，自动处理网络分片；返回时不负责关闭 reader。 */
export async function consumeResearchSse(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onEvent: (event: ResearchEvent) => void | Promise<void>,
): Promise<void> {
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const consumeLine = async (line: string) => {
    if (!line.startsWith('data:')) return
    const event = parseResearchSseData(line.slice(5).trim())
    if (event) await onEvent(event)
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    let newline = buffer.indexOf('\n')
    while (newline >= 0) {
      const line = buffer.slice(0, newline).replace(/\r$/, '')
      buffer = buffer.slice(newline + 1)
      await consumeLine(line)
      newline = buffer.indexOf('\n')
    }
    if (done) {
      if (buffer) await consumeLine(buffer.replace(/\r$/, ''))
      return
    }
  }
}

/** 供旧 UI 分支逐步迁移；协议元数据不会被 data 中的同名字段覆盖。 */
export function flattenResearchEvent(event: ResearchEvent): JsonObject {
  return {
    ...event.data,
    version: event.version,
    event_id: event.event_id,
    type: event.type,
    session_id: event.session_id,
    run_id: event.run_id,
    turn_id: event.turn_id,
    seq: event.seq,
    timestamp: event.timestamp,
  }
}

export function advanceResearchProtocol(
  state: ResearchProtocolState,
  event: ResearchEvent,
): ResearchProtocolAdvance {
  if (state.runId && state.runId !== event.run_id && event.type !== 'research_start') {
    return { state, accepted: false, reason: 'run_mismatch' }
  }

  const sameRun = state.runId === event.run_id
  if (state.runId === event.run_id && event.seq > 0 && event.seq <= state.lastSeq) {
    return { state, accepted: false, reason: 'duplicate' }
  }
  if (sameRun && event.seq > state.lastSeq + 1) {
    return {
      state,
      accepted: false,
      reason: 'gap',
      gapAfterSeq: state.lastSeq,
    }
  }

  const base = sameRun || !state.runId ? state : initialResearchProtocolState
  let status = base.status
  if (event.type === 'research_start' || event.type === 'research_resumed') status = 'running'
  if (event.type === 'research_complete') status = 'completed'
  if (event.type === 'session_clarification') status = 'clarification'
  if (event.type === 'research_cancelled') status = 'cancelled'
  if (event.type === 'error') status = 'failed'
  return { state: {
    runId: event.run_id,
    lastSeq: Math.max(sameRun ? base.lastSeq : 0, event.seq),
    status,
    eventCount: base.eventCount + 1,
  }, accepted: true, reason: 'accepted' }
}

export function reduceResearchProtocol(
  state: ResearchProtocolState,
  event: ResearchEvent,
): ResearchProtocolState {
  return advanceResearchProtocol(state, event).state
}
