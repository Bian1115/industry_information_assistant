import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'
import { proxy, snapshot, subscribe } from 'valtio/vanilla'

const compile = source => ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
}).outputText
const url = code => `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`
const eventUrl = url(compile(await readFile(new URL('../src/pages/chat/research-events.ts', import.meta.url), 'utf8')))
const enumUrl = url(compile(await readFile(new URL('../src/configs/enum.ts', import.meta.url), 'utf8')))
const {
  createRecoveryProjection, projectionFromRunState, reduceRecoveryProjection,
  advanceResearchProtocol, initialResearchProtocolState, consumeResearchSse,
} = await import(eventUrl)
const { ChatRole, ChatType } = await import(enumUrl)
const recovery = compile(await readFile(new URL('../src/pages/chat/recovery-messages.ts', import.meta.url), 'utf8'))
  .replace("'@/configs'", JSON.stringify(enumUrl)).replace("'./research-events'", JSON.stringify(eventUrl))
const { applyRecoveryToMessages } = await import(url(recovery))
let id = 10
const createId = () => ++id
const event = (type, data = {}, seq = 1, run_id = 'run-1') => ({
  version: 1, event_id: `${run_id}:${seq}`, type, data, seq, run_id,
  session_id: 'session-1', timestamp: '2026-09-25T07:00:00Z',
})
const question = runId => ({ id: createId(), role: ChatRole.User, type: ChatType.Deepsearch, runId, content: '研究问题' })

function replay(events, list = proxy([question('run-1')])) {
  let projection = projectionFromRunState({ run_status: 'failed', phase: 'writing', pending_nodes: ['write'] })
  let protocol = { ...initialResearchProtocolState, runId: 'run-1' }
  const apply = e => {
    const advance = advanceResearchProtocol(protocol, e)
    if (!advance.accepted) return advance.reason
    protocol = advance.state
    projection = reduceRecoveryProjection(projection, e)
    applyRecoveryToMessages(list, 'run-1', '研究问题', projection, createId)
    return advance.reason
  }
  events.forEach(apply)
  return { list, apply, get projection() { return projection }, get protocol() { return protocol } }
}

test('只有用户问题的历史会话续跑时补建助手并显示当前写作阶段', () => {
  const r = replay([
    event('research_resumed'),
    event('phase', { phase: 'writing', content: '开始撰写报告' }, 2),
    event('action', { content: { tool: 'write_section', section: '系统比较' } }, 3),
    event('thought', { content: { content: '正在比较研究结果' } }, 4),
    event('observation', { content: { content: '章节已生成' } }, 5),
  ])
  const assistant = snapshot(r.list)[1]
  assert.equal(r.list.length, 2)
  assert.equal(assistant.role, ChatRole.Assistant)
  assert.equal(assistant.type, ChatType.Deepsearch)
  assert.equal(assistant.loading, true)
  assert.equal(assistant.researchStatus, 'running')
  assert.match(assistant.reactSteps[1].content, /写作阶段.*开始撰写报告/)
  assert.match(assistant.reactSteps[2].content, /write_section.*系统比较/)
  assert.equal(assistant.reactSteps[2].params.section, '系统比较')
  assert.equal(assistant.reactSteps[3].content, '正在比较研究结果')
  assert.equal(assistant.reactSteps[4].content, '章节已生成')
})

test('同一事件回放和SSE重复投递不重复卡片，其他run不会污染本研究', () => {
  const start = event('research_resumed')
  const r = replay([start])
  const originalId = r.list[1].id
  assert.equal(r.apply(start), 'duplicate')
  assert.equal(r.apply(event('phase', { phase: 'reviewing' }, 2, 'other')), 'run_mismatch')
  r.apply(event('phase', { phase: 'reviewing' }, 2))
  assert.equal(r.list.length, 2)
  assert.equal(r.list[1].id, originalId)
  assert.equal(r.list[1].reactSteps.length, 2)
  assert.match(r.list[1].reactSteps[1].content, /审核阶段/)
})

test('恢复旧研究时按run插入正确位置，保留其他版本的报告与状态', () => {
  const later = { id: createId(), role: ChatRole.Assistant, runId: 'run-2', content: '新版报告', researchStatus: 'completed' }
  const list = proxy([question('run-1'), question('run-2'), later])
  replay([event('research_resumed')], list)
  assert.deepEqual(snapshot(list).map(x => [x.runId, x.role]), [
    ['run-1', ChatRole.User], ['run-1', ChatRole.Assistant],
    ['run-2', ChatRole.User], ['run-2', ChatRole.Assistant],
  ])
  assert.equal(list[3].content, '新版报告')
  assert.equal(list[3].researchStatus, 'completed')
})

test('已有助手消息原位更新并通过真实Valtio订阅触发界面更新', async () => {
  const list = proxy([question('run-1'), { id: createId(), role: ChatRole.Assistant, runId: 'run-1', reactSteps: [] }])
  const oldId = list[1].id
  let changes = 0
  const unsubscribe = subscribe(list, () => { changes++ })
  try {
    replay([event('research_resumed'), event('phase', { phase: 'writing' }, 2)], list)
    await Promise.resolve()
    assert.ok(changes > 0)
    assert.equal(list.length, 2)
    assert.equal(list[1].id, oldId)
    assert.equal(snapshot(list)[1].reactSteps.length, 2)
  } finally { unsubscribe() }
})

test('无事件快照也显示已保存阶段，空会话补齐原问题且失败态不假装运行', () => {
  const list = proxy([])
  const projection = projectionFromRunState({ run_status: 'failed', phase: 'writing', pending_nodes: ['write'] })
  applyRecoveryToMessages(list, 'run-1', '原研究问题', projection, createId)
  assert.equal(list[0].content, '原研究问题')
  assert.equal(list[1].loading, false)
  assert.equal(list[1].researchStatus, 'failed')
  assert.match(list[1].reactSteps[0].content, /已保存阶段：写作阶段/)
})

test('失败、暂停和完成事件立即终止加载并保留进度过程', () => {
  for (const [type, expected] of [['error', 'failed'], ['research_cancelled', 'cancelled'], ['research_paused', 'cancelled'], ['research_complete', 'completed']]) {
    const r = replay([event('research_resumed'), event(type, { content: '正文为空', final_report: '最终报告' }, 2)])
    assert.equal(r.list[1].loading, false)
    assert.equal(r.list[1].researchStatus, expected)
    assert.equal(r.list[1].reactSteps.length, 2)
    if (type === 'error') assert.match(r.list[1].reactSteps[1].content, /研究中断：正文为空/)
    if (type === 'research_complete') assert.equal(r.list[1].content, '最终报告')
  }
})

test('真实SSE分片解析经恢复投影写入消息，兼容phase_change及嵌套中文', async () => {
  const values = [event('research_resumed'), event('phase_change', { content: { phase: 'revising', message: '修订研究报告' } }, 2)]
  const bytes = new TextEncoder().encode(values.map(e => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n')
  const stream = new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7))
    controller.close()
  } })
  const r = replay([])
  await consumeResearchSse(stream.getReader(), r.apply)
  assert.equal(r.protocol.lastSeq, 2)
  assert.match(r.list[1].reactSteps[1].content, /修订阶段：修订研究报告/)
})

test('旧快照无步骤字段时仍能继续消费新事件', () => {
  const initial = createRecoveryProjection({ status: 'paused', final_report: '已有草稿' })
  const next = reduceRecoveryProjection(initial, event('research_resumed'))
  assert.equal(next.status, 'running')
  assert.equal(next.streaming_report, '已有草稿')
  assert.equal(initial.react_steps.length, 0)
  assert.equal(next.react_steps.length, 1)
})
