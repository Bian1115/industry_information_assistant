import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

const source = await readFile(new URL('../src/pages/chat/research-events.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText
const { createRecoveryProjection, reduceRecoveryProjection, advanceResearchProtocol, initialResearchProtocolState, consumeResearchSse } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)

function event(type, data, seq = 1) {
  return { version: 1, event_id: `event-${seq}`, type, session_id: 'session-1', run_id: 'run-1', seq, timestamp: '2026-09-14T00:00:00Z', data }
}

test('医学文献事件与分块报告可从检查点继续恢复', () => {
  const old = { protocol: { disease: '测试疾病' }, papers: [], screening: [], evidence: [] }
  let state = createRecoveryProjection({ status: 'running', ui_state_json: { medical_research: old } })
  const updated = { ...old, papers: [{ paper_id: 'pmid:1', title: '真实来源的结构示例' }], limitations: ['仅摘要'] }
  state = reduceRecoveryProjection(state, event('literature_update', { content: updated }))
  state = reduceRecoveryProjection(state, event('phase_change', { content: { phase: 'writing', message: '撰写报告' } }, 2))
  state = reduceRecoveryProjection(state, event('report_chunk', { content: { chunk: '# 研究进展\n' } }, 3))
  state = reduceRecoveryProjection(state, event('report_chunk', { content: { chunk: '带原文引用的结果。' } }, 4))
  assert.equal(state.medical_research.papers[0].paper_id, 'pmid:1')
  assert.equal(state.phase, 'writing')
  assert.equal(state.streaming_report, '# 研究进展\n带原文引用的结果。')
  const final = { ...updated, review: { status: 'needs_human_review' } }
  state = reduceRecoveryProjection(state, event('research_complete', { final_report: '最终报告', medical_research: final }, 5))
  assert.equal(state.status, 'completed')
  assert.deepEqual(state.medical_research, final)
  assert.equal(state.streaming_report, '最终报告')
})

test('重复和缺口事件不会静默覆盖医学研究状态', () => {
  const started = advanceResearchProtocol(initialResearchProtocolState, event('research_start', {}))
  assert.equal(started.accepted, true)
  assert.equal(advanceResearchProtocol(started.state, event('literature_update', { content: {} }, 1)).reason, 'duplicate')
  const gap = advanceResearchProtocol(started.state, event('literature_update', { content: {} }, 3))
  assert.equal(gap.reason, 'gap')
  assert.equal(gap.gapAfterSeq, 1)
  const mismatch = { ...event('literature_update', { content: {} }, 2), run_id: 'other-run' }
  assert.equal(advanceResearchProtocol(started.state, mismatch).reason, 'run_mismatch')
})

test('UTF-8 网络分片不会截断中文文献与最终事件', async () => {
  const update = event('literature_update', { content: { papers: [{ paper_id: 'p1', title: '疾病研究进展' }] } })
  const complete = event('research_complete', { final_report: '完成' }, 2)
  const bytes = new TextEncoder().encode(`data: ${JSON.stringify(update)}\n\ndata: ${JSON.stringify(complete)}`)
  const stream = new ReadableStream({ start(controller) { for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7)); controller.close() } })
  const events = []
  await consumeResearchSse(stream.getReader(), value => { events.push(value) })
  assert.equal(events.length, 2)
  assert.equal(events[0].data.content.papers[0].title, '疾病研究进展')
  assert.equal(events[1].data.final_report, '完成')
})

test('中断事件保留已收集文献并明确失败状态', () => {
  const state = createRecoveryProjection({ status: 'running', ui_state_json: { medical_research: { papers: [{ paper_id: 'p1' }] } } })
  const failed = reduceRecoveryProjection(state, event('error', { content: '来源服务失败' }))
  assert.equal(failed.status, 'failed')
  assert.equal(failed.medical_research.papers.length, 1)
})

test('打开主动暂停的历史研究不会自动变成执行状态', () => {
  const state = createRecoveryProjection({ status: 'paused', ui_state_json: { medical_research: { papers: [{ paper_id: 'p1' }] } } })
  assert.equal(state.status, 'cancelled')
  assert.equal(state.medical_research.papers.length, 1)
})

test('医疗多智能体的协作步骤可经快照和后续事件恢复', () => {
  const analyst = { type: 'medical_extraction', step_type: 'medical_extraction', agent: 'DataAnalyst', title: '证据提取', status: 'running' }
  let projection = createRecoveryProjection({ status: 'running', ui_state_json: {
    research_steps: [analyst], medical_research: { agent_steps: [analyst], papers: [{ paper_id: 'p1' }] },
  } })
  const completed = { ...analyst, status: 'completed' }
  projection = reduceRecoveryProjection(projection, event('research_step', { content: completed }))
  projection = reduceRecoveryProjection(projection, event('literature_update', {
    content: { agent_steps: [completed], papers: [{ paper_id: 'p1' }] },
  }, 2))
  assert.equal(projection.research_steps.length, 1)
  assert.equal(projection.research_steps[0].status, 'completed')
  assert.equal(projection.medical_research.agent_steps[0].agent, 'DataAnalyst')
  assert.equal(projection.medical_research.papers[0].paper_id, 'p1')
})
