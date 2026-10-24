/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

import { AxiosRequestConfig } from 'axios'
import { request } from './request'

// ============ 新的会话管理 API ============

export interface Session {
  id: string
  title: string
  session_type: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface Message {
  id: string
  session_id: string
  turn_id?: string | null
  execution_mode: 'chat' | 'research'
  run_id?: string | null
  research_status?: 'running' | 'completed' | 'failed' | 'cancelled' | null
  role: 'user' | 'assistant' | 'system'
  content: string
  thinking?: string
  references_data?: Record<string, unknown>
  image_results?: Array<Record<string, unknown>>
  created_at: string
}

export interface SessionWithMessages extends Session {
  messages: Message[]
}

export interface CreateSessionParams {
  title?: string
  session_type?: 'chat' | 'deepsearch'
}

export interface UpdateSessionParams {
  title: string
}

/**
 * 获取会话列表
 */
export function getSessions(params?: { limit?: number; offset?: number; session_type?: string }) {
  return request.get<Session[]>('/sessions', { params, loading: false })
}

/**
 * 创建新会话
 */
export function createSession(params?: CreateSessionParams) {
  return request.post<Session>('/sessions', params || {}, { loading: false })
}

/**
 * 获取会话详情（包含消息）
 */
export function getSession(sessionId: string) {
  return request.get<SessionWithMessages>(`/sessions/${sessionId}`, { loading: false })
}

/**
 * 更新会话标题
 */
export function updateSession(sessionId: string, params: UpdateSessionParams) {
  return request.put<Session>(`/sessions/${sessionId}`, params, { loading: false })
}

/**
 * 删除会话
 */
export function deleteSession(sessionId: string) {
  return request.delete(`/sessions/${sessionId}`, { loading: false })
}

/**
 * 获取会话消息列表
 */
export function getMessages(sessionId: string, params?: { limit?: number; offset?: number }) {
  return request.get<Message[]>(`/sessions/${sessionId}/messages`, { params })
}

export function chat(
  params: {
    session_id: string
    turn_id: string
    question: string
    search_knowledge: boolean
    search_web: boolean
    /** 限定检索的知识库 ID；不传或空数组表示检索用户的全部知识库 */
    kb_ids?: string[]
  },
  options?: AxiosRequestConfig,
) {
  return request.post<ReadableStream>('/chat/completion', params, {
    headers: {
      Accept: 'text/event-stream',
    },
    responseType: 'stream',
    adapter: 'fetch',
    loading: false,
    ...options,
  })
}

export function deepsearch(
  params: {
    query: string
    session_id: string
    turn_id: string
    /**
     * 可选：在后端配置的来源范围内进一步收窄本次研究的网站。
     * 不传表示使用后端配置的全部允许网站；传入范围外的域名会被后端拒绝。
     */
    allowed_domains?: string[]
    /** 假设机制开关；未传使用后端默认值，单次运行固定。 */
    hypothesis_enabled?: boolean
    agent_compression?: Partial<Record<'ChiefArchitect' | 'DeepScout' | 'DataAnalyst' | 'LeadWriter' | 'CriticMaster', {
      enabled?: boolean
      reference_materials?: boolean
      artifact_offload?: boolean
      semantic_summary?: boolean
      semantic_verification?: boolean
    }>>
  },
  options?: AxiosRequestConfig,
) {
  return request.post<ReadableStream>('/research/stream', params, {
    headers: {
      Accept: 'text/event-stream',
    },
    responseType: 'stream',
    adapter: 'fetch',
    loading: false,
    ...options,
  })
}

// ============ 附件 API ============

export interface Attachment {
  id: string
  session_id: string
  message_id?: string
  filename: string
  file_type: string
  file_size: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  error_message?: string
  created_at: string
}

export interface AttachmentListResponse {
  attachments: Attachment[]
  total: number
}

/**
 * 上传附件
 */
export function uploadAttachment(sessionId: string, file: File) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('session_id', sessionId)
  return request.post<Attachment>('/attachments', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
}

/**
 * 获取附件详情
 */
export function getAttachment(attachmentId: string) {
  return request.get<Attachment>(`/attachments/${attachmentId}`)
}

/**
 * 获取会话的所有附件
 */
export function getSessionAttachments(sessionId: string) {
  return request.get<AttachmentListResponse>(`/attachments/session/${sessionId}`)
}

/**
 * 删除附件
 */
export function deleteAttachment(attachmentId: string) {
  return request.delete(`/attachments/${attachmentId}`)
}

/**
 * 带附件的聊天
 */
export function chatWithAttachments(
  params: {
    session_id: string
    turn_id: string
    question: string
    attachment_ids?: string[]
    search_knowledge: boolean
    search_web: boolean
    /** 限定检索的知识库 ID；不传或空数组表示检索用户的全部知识库 */
    kb_ids?: string[]
  },
  options?: AxiosRequestConfig,
) {
  return request.post<ReadableStream>('/chat/completion/v3', params, {
    headers: {
      Accept: 'text/event-stream',
    },
    responseType: 'stream',
    adapter: 'fetch',
    loading: false,
    ...options,
  })
}

// ============ 研究控制 API ============

/**
 * 取消正在进行的研究任务
 */
export function cancelResearch(sessionId: string, runId: string) {
  const query = `?run_id=${encodeURIComponent(runId)}`
  return request.post<{ success: boolean; message: string; run_id?: string }>(
    `/research/cancel/${sessionId}${query}`,
    {},
    { loading: false },
  )
}

// ============ 研究检查点 API ============

export interface ResearchStep {
  type: string
  status: 'pending' | 'running' | 'completed'
  stats?: Record<string, number>
}

export interface ResearchUIState {
  research_steps: ResearchStep[]
  search_results: unknown[]
  charts: unknown[]
  knowledge_graph: unknown | null
  streaming_report: string
  references?: unknown[]
}

export interface ResearchCheckpoint {
  id: string
  session_id: string
  run_id: string | null
  last_event_seq: number
  user_id: string | null
  query: string
  phase: string
  iteration: number
  status: 'running' | 'paused' | 'completed' | 'failed'
  error_message: string | null
  final_report: string | null
  state_json?: Record<string, unknown>
  ui_state_json?: ResearchUIState
  created_at: string
  updated_at: string
}

export interface ResearchRunState {
  progress?: { counts: { queries: number; sources: number; facts: number; references: number }; basis: 'in_progress'; node: string; attempt: number } | null
  pipeline?: 'graph' | 'legacy'
  run_status?: string
  error?: string
  query?: string
  original_query?: string
  resolved_query?: string
  resumable?: boolean
  delivery_ready?: boolean
  delivery_pending?: boolean
  saved_result_nodes?: string[]
  run_id: string
  exists: boolean
  graph_version: string
  pending_nodes: string[]
  phase: string
  status: string
  completion_reason: string
  iteration: number
  quality_verdict: string
  /** 独立报告论断与引用核验审计；开关关闭时为 not_executed。 */
  citation_verification: Record<string, unknown>
  hypothesis_enabled: boolean
  allowed_domains: string[]
  final_report: string
  outline: unknown[]
  draft: {
    raw_sources?: Record<string, unknown>[]
    search_budget?: Record<string, unknown>
    web_reader_attempts?: Record<string, unknown>[]
    outline: unknown[]
    facts: unknown[]
    references: unknown[]
    knowledge_graph: unknown
    final_report: string
    quality_gates: Record<string, unknown>
    node_attempts: Record<string, number>
  }
}

/**
 * 读取一次研究的图状态：业务底稿 + 执行位置。
 *
 * 页面恢复以这里的资料和报告为底稿；进度事件另按 seq 用 getResearchEvents
 * 重放。两者游标职责分开，不要用事件序号推断执行位置。
 */
export function getResearchRunState(runId: string) {
  return request.get<{ success: boolean; state: ResearchRunState }>(
    `/research/runs/${encodeURIComponent(runId)}/state`,
    // 页面资料读取与恢复状态读取是独立消费者，不能互相中止请求。
    { loading: false, cancelRepeat: false },
  )
}

/**
 * 读取历史研究快照（LangGraph 接管执行之前写入的完整快照，只读）。
 * 这些运行不能续跑，只能查看和导出。
 */
export function getLegacyResearchCheckpoint(sessionId: string, runId?: string) {
  return request.get<{ success: boolean; read_only: boolean; checkpoint?: ResearchCheckpoint; message?: string }>(
    `/research/legacy/checkpoint/${sessionId}`,
    { params: runId ? { run_id: runId } : undefined, loading: false }
  )
}

export interface ResearchEventsResponse {
  run_id: string
  events: Array<Record<string, unknown>>
  last_seq: number
  has_more: boolean
}

/** 读取 Checkpoint 游标之后的追加式事件。 */
export function getResearchEvents(runId: string, afterSeq: number, limit = 2000) {
  return request.get<ResearchEventsResponse>(
    `/research/runs/${encodeURIComponent(runId)}/events`,
    { params: { after_seq: afterSeq, limit }, loading: false },
  )
}

/**
 * 从图检查点续接一次技术中断的研究；服务端会先补发游标之后的事件。
 * 旧版本运行没有本图可用的检查点，会返回 legacy_run_not_resumable，
 * 需要由用户明确重新发起，而不是假装断点恢复。
 */
export function resumeResearch(
  runId: string,
  afterSeq: number,
  options?: AxiosRequestConfig,
) {
  return request.post<ReadableStream>(
    `/research/resume/${encodeURIComponent(runId)}?after_seq=${afterSeq}`,
    {},
    {
      headers: { Accept: 'text/event-stream' },
      responseType: 'stream',
      adapter: 'fetch',
      loading: false,
      ...options,
    },
  )
}

/**
 * 获取历史研究快照列表（只读，不能据此续跑）。
 */
export function listLegacyResearchCheckpoints(params?: { status?: string; limit?: number }) {
  return request.get<{ success: boolean; read_only: boolean; checkpoints: ResearchCheckpoint[]; total: number }>(
    '/research/legacy/checkpoints',
    { params }
  )
}
