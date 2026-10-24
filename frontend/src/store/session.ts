/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

import { proxy } from 'valtio'
import {
  Session,
  SessionWithMessages,
  getSessions,
  createSession,
  getSession,
  updateSession,
  deleteSession,
} from '@/api/session'

interface SessionState {
  sessions: Session[]
  currentSession: SessionWithMessages | null
  loading: boolean
  error: string | null
}

export const sessionState = proxy<SessionState>({
  sessions: [],
  currentSession: null,
  loading: false,
  error: null,
})

// 账号切换后，旧账号尚未返回的请求也不能重新写入会话缓存。
let sessionGeneration = 0

function requireCurrentGeneration(generation: number) {
  if (generation !== sessionGeneration) throw new Error('账号已切换，请重新操作')
}

export const sessionActions = {
  reset() {
    sessionGeneration += 1
    sessionState.sessions = []
    sessionState.currentSession = null
    sessionState.loading = false
    sessionState.error = null
  },

  async fetchSessions() {
    const generation = sessionGeneration
    sessionState.loading = true
    sessionState.error = null
    try {
      const response = await getSessions({ limit: 50 })
      if (generation !== sessionGeneration) return
      // 兼容 axios 响应格式和直接返回数据
      const sessions = (response as any).data || response
      sessionState.sessions = Array.isArray(sessions) ? sessions : []
    } catch (err) {
      if (generation === sessionGeneration) sessionState.error = (err as Error).message || '获取会话列表失败'
    } finally {
      if (generation === sessionGeneration) sessionState.loading = false
    }
  },

  async createNewSession(title?: string, sessionType: 'chat' | 'deepsearch' = 'chat') {
    const generation = sessionGeneration
    try {
      const response = await createSession({ title, session_type: sessionType })
      requireCurrentGeneration(generation)
      const newSession = ((response as any).data || response) as Session
      sessionState.sessions.unshift(newSession)
      return newSession
    } catch (err) {
      if (generation === sessionGeneration) sessionState.error = (err as Error).message || '创建会话失败'
      throw err
    }
  },

  async loadSession(sessionId: string) {
    const generation = sessionGeneration
    sessionState.loading = true
    sessionState.error = null
    try {
      const response = await getSession(sessionId)
      requireCurrentGeneration(generation)
      const session = ((response as any).data || response) as SessionWithMessages
      sessionState.currentSession = session
      return session
    } catch (err) {
      if (generation === sessionGeneration) sessionState.error = (err as Error).message || '加载会话失败'
      throw err
    } finally {
      if (generation === sessionGeneration) sessionState.loading = false
    }
  },

  async renameSession(sessionId: string, title: string) {
    const generation = sessionGeneration
    try {
      const response = await updateSession(sessionId, { title })
      requireCurrentGeneration(generation)
      const updatedSession = ((response as any).data || response) as Session
      const index = sessionState.sessions.findIndex((s) => s.id === sessionId)
      if (index !== -1) {
        sessionState.sessions[index] = updatedSession
      }
      if (sessionState.currentSession?.id === sessionId) {
        sessionState.currentSession.title = title
      }
      return updatedSession
    } catch (err) {
      if (generation === sessionGeneration) sessionState.error = (err as Error).message || '重命名会话失败'
      throw err
    }
  },

  async removeSession(sessionId: string) {
    const generation = sessionGeneration
    try {
      await deleteSession(sessionId)
      if (generation !== sessionGeneration) return
      sessionState.sessions = sessionState.sessions.filter((s) => s.id !== sessionId)
      if (sessionState.currentSession?.id === sessionId) {
        sessionState.currentSession = null
      }
    } catch (err) {
      if (generation === sessionGeneration) sessionState.error = (err as Error).message || '删除会话失败'
      throw err
    }
  },

  setCurrentSession(session: SessionWithMessages | null) {
    sessionState.currentSession = session
  },

  clearError() {
    sessionState.error = null
  },
}
