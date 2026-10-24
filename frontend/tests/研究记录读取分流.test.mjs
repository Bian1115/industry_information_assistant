import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

const compile = source => ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText
let handler
let calls
globalThis.__researchReadRequest = { get: async (url, config) => { calls.push(url); return handler(url, config) } }
const source = compile(await readFile(new URL('../src/api/medical.ts', import.meta.url), 'utf8'))
  .replace(/import \{ request \} from ['"]\.\/request['"];?/, 'const request = globalThis.__researchReadRequest;')
const { getMedicalResearch } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

test('新图结果、未保存检查点的新运行均不访问历史接口', async () => {
  for (const exists of [true, false]) {
    calls = []
    handler = async () => ({ data: { state: { pipeline: 'graph', exists, run_status: 'failed', status: 'running', draft: { final_report: '已保存报告' } } } })
    const result = await getMedicalResearch('run-1')
    assert.equal(result.pipeline, 'graph')
    assert.equal(result.run_status, 'failed')
    assert.equal(result.final_report, '已保存报告')
    assert.deepEqual(calls, ['/research/runs/run-1/state'])
  }
})

test('旧医疗记录在新接口404或明确标记legacy时仍可读取', async () => {
  for (const missing of [true, false]) {
    calls = []
    handler = async url => {
      if (url.endsWith('/state')) {
        if (missing) throw { response: { status: 404 } }
        return { data: { state: { exists: false, pipeline: 'legacy' } } }
      }
      return { data: { papers: [{ paper_id: 'p1' }], final_report: '历史报告' } }
    }
    const result = await getMedicalResearch('old')
    assert.equal(result.pipeline, 'legacy')
    assert.equal(result.papers[0].paper_id, 'p1')
    assert.equal(calls.length, 2)
  }
})

test('权限及服务器错误不能触发历史降级或被吞掉', async () => {
  for (const status of [401, 403, 500]) {
    calls = []
    const error = { response: { status } }
    handler = async () => { throw error }
    await assert.rejects(getMedicalResearch('run-1'), value => value === error)
    assert.equal(calls.length, 1)
  }
})

const events = compile(await readFile(new URL('../src/pages/chat/research-events.ts', import.meta.url), 'utf8'))
const { projectionFromRunState } = await import(`data:text/javascript;base64,${Buffer.from(events).toString('base64')}`)
test('有待执行节点的失败、暂停运行不会仅因刷新而标记运行中', () => {
  for (const [run_status, expected] of [['failed', 'failed'], ['paused', 'cancelled'], ['completed', 'completed']]) {
    assert.equal(projectionFromRunState({ run_status, status: 'running', pending_nodes: ['scout'], draft: {} }).status, expected)
  }
})
