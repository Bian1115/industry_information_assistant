import { ChatRole, ChatType } from '@/configs'
import { researchPhaseLabels, type ResearchRecoveryProjection } from './research-events'

/** 按 run 补齐助手消息；历史会话只有用户问题时也必须能显示续跑进度。 */
export function applyRecoveryToMessages(
  list: API.ChatItem[],
  runId: string,
  query: string,
  projection: ResearchRecoveryProjection,
  createId: () => number,
): API.ChatItem {
  if (list.length === 0 && query) {
    list.push({ id: createId(), role: ChatRole.User, type: ChatType.Normal, content: query, runId })
  }
  let index = list.findIndex(item => item.role === ChatRole.Assistant && item.runId === runId)
  if (index < 0) {
    const question = list.findIndex(item => item.role === ChatRole.User && item.runId === runId)
    index = question < 0 ? list.length : question + 1
    list.splice(index, 0, { id: createId(), role: ChatRole.Assistant, type: ChatType.Deepsearch, runId })
  }
  // 插入后从列表重新获取，以便对 Valtio proxy 的更新能触发重绘。
  const assistant = list[index]
  assistant.type = ChatType.Deepsearch
  assistant.reactMode = true
  assistant.reactSteps = projection.react_steps.length ? projection.react_steps : [{
    step: 1, type: 'thought', content: projection.phase
      ? `已保存阶段：${researchPhaseLabels[projection.phase] || projection.phase}`
      : '正在读取研究进度',
  }]
  assistant.researchStatus = projection.status === 'idle' ? undefined : projection.status
  assistant.loading = projection.status === 'running'
  assistant.content = projection.streaming_report
  assistant.charts = projection.charts as API.ChartConfig[]
  assistant.reference = projection.references.map((ref, i) => ({
    id: i + 1, title: String(ref.title || ref.source_name || '来源'),
    link: String(ref.link || ref.url || ref.source_url || ''),
    content: String(ref.content || ref.summary || ''),
    source: ref.source_type === 'local' ? 'knowledge' : 'web',
  }))
  return assistant
}
