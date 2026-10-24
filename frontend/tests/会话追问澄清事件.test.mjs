import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'

const source = await readFile(new URL('../src/pages/chat/research-events.ts', import.meta.url), 'utf8')
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText
const events = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`)

test('澄清事件显示问题，协议不宣称研究完成', () => {
  const event = events.parseResearchSseData(JSON.stringify({ type: 'session_clarification', question: '请明确是哪份报告。', run_id: 'turn' }))
  const advance = events.advanceResearchProtocol(events.initialResearchProtocolState, event)
  assert.equal(advance.accepted, true)
  assert.equal(advance.state.status, 'clarification')
  const initial = events.projectionFromRunState({ run_status: 'running' })
  const projection = events.reduceRecoveryProjection(initial, event)
  assert.equal(projection.status, 'clarification')
  assert.equal(projection.streaming_report, '请明确是哪份报告。')
})
