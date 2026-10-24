/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

import storage from './storage'
import proxyWithPersist, { PersistStrategy } from './valtio-persist'

// 搜索模式类型
export type SearchMode = 'web' | 'local'
export type ExecutionMode = 'chat' | 'research'

const state = proxyWithPersist({
  name: 'device',
  version: 3,
  getStorage: () => storage,
  persistStrategies: {
    searchModes: PersistStrategy.SingleFile,
    executionMode: PersistStrategy.SingleFile,
    selectedKbIds: PersistStrategy.SingleFile,
  },
  migrations: {
    1: () => { state.searchModes = ['web'] },
    2: () => { state.executionMode = 'research' },
    3: () => { state.executionMode = 'research'; state.searchModes = ['web'] },
  },

  initialState: {
    chatting: false,
    // 回答方式与资料来源相互独立。
    executionMode: 'research' as ExecutionMode,
    // 资料来源: 'web' = 网络, 'local' = 本地知识库
    searchModes: ['web'] as SearchMode[],
    // 本次问答限定检索的知识库 ID；空数组 = 不限定，后端会检索该用户的全部知识库
    selectedKbIds: [] as string[],
  },
})

const actions = {
  setChatting(chatting: boolean) {
    state.chatting = chatting
  },
  setSearchModes(modes: SearchMode[]) {
    state.searchModes = modes
  },
  setExecutionMode(mode: ExecutionMode) {
    state.executionMode = mode
  },
  toggleSearchMode(mode: SearchMode) {
    const currentModes = state.searchModes as SearchMode[]
    if (currentModes.includes(mode)) {
      state.searchModes = currentModes.filter(m => m !== mode)
    } else {
      state.searchModes = [...currentModes, mode]
    }
  },
  setSelectedKbIds(ids: string[]) {
    state.selectedKbIds = ids
  },
  toggleKb(kbId: string) {
    const current = state.selectedKbIds as string[]
    state.selectedKbIds = current.includes(kbId)
      ? current.filter(id => id !== kbId)
      : [...current, kbId]
  },
  clearSelectedKbs() {
    state.selectedKbIds = []
  },
  // 兼容旧代码
  get useDeepsearch() {
    return state.executionMode === 'research'
  },
  get useLocalKb() {
    return (state.searchModes as SearchMode[]).includes('local')
  },
}

export const deviceState = state
export const deviceActions = actions
