/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

import type { MedicalScope } from '@/api/medical'
import { PageTransportKey } from '@/utils'

export type ChatEnterData = {
  sessionId?: string
  message: string
  attachmentIds?: string[]
  researchScope?: MedicalScope
  previousRunId?: string
  executionMode?: 'chat' | 'research'
}

export const transportToChatEnter = Symbol() as PageTransportKey<{
  data: ChatEnterData
}>

let id = 0

export const createChatId = () => {
  return ++id
}

export function createChatIdText(id: number) {
  return `chat-item-${id}`
}
