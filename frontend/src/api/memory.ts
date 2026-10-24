/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

import { request } from './request'

export interface Memory {
  id: string
  session_id?: string | null
  summary: string
  key_insights?: Record<string, unknown> | null
  token_count?: number | null
  provider: string
  index_status: string
  created_at: string
}

export interface MemoryListResponse {
  memories: Memory[]
  total: number
}

export interface MemorySearchResult {
  id: string
  fragment_id: string
  session_id?: string | null
  memory_type: string
  content: string
  score: number
}

export interface MemoryOperation {
  operation_id: string
  operation_type: 'add' | 'update' | 'delete' | 'reset_user'
  status: 'pending' | 'running' | 'retry' | 'succeeded' | 'dead'
  attempts: number
  memory_id?: string | null
  last_error?: string | null
  created_at: string
  updated_at: string
}

/**
 * 获取记忆列表
 */
export function getMemories(params?: { limit?: number; offset?: number }) {
  return request.get<MemoryListResponse>('/memories', { params })
}

/**
 * 获取单个记忆详情
 */
export function getMemory(memoryId: string) {
  return request.get<Memory>(`/memories/${memoryId}`)
}

/**
 * 搜索记忆
 */
export function searchMemories(params: { query: string; top_k?: number }) {
  return request.post<MemorySearchResult[]>('/memories/search', params)
}

/**
 * 从会话创建记忆
 */
export function createMemory(sessionId: string) {
  return request.post<{ operation_id: string; status: string }>(
    '/memories/create',
    { session_id: sessionId },
  )
}

/**
 * 查询异步记忆任务状态
 */
export function getMemoryOperation(operationId: string) {
  return request.get<MemoryOperation>(`/memories/jobs/${operationId}`)
}

/**
 * 删除记忆
 */
export function deleteMemory(memoryId: string) {
  return request.delete(`/memories/${memoryId}`)
}

/**
 * 获取记忆上下文
 */
export function getMemoryContext(query: string) {
  return request.get<{ context: string }>(`/memories/context/${encodeURIComponent(query)}`)
}
