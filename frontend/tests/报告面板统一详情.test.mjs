import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

// 执行页面真实回调、SSE解析器和工作台组件，替换网络与React/antd渲染边界。
// 本文件验证状态到组件的更新，不声称测量浏览器DOM或像素首显。
const page = await readFile(new URL('../src/pages/chat/index.tsx', import.meta.url), 'utf8')
const tree = ts.createSourceFile('index.tsx', page, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
const compile = (source, module = ts.ModuleKind.CommonJS) => ts.transpileModule(source, {
  fileName: 'test.tsx', compilerOptions: {
    module, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.React, jsxFactory: 'jsx', jsxFragmentFactory: 'Fragment',
  },
}).outputText
const events = await import(`data:text/javascript;base64,${Buffer.from(compile(
  await readFile(new URL('../src/pages/chat/research-events.ts', import.meta.url), 'utf8'),
  ts.ModuleKind.ESNext,
)).toString('base64')}`)

function initializer(name) {
  let found
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === name) found = node.initializer
    ts.forEachChild(node, visit)
  }
  visit(tree)
  assert.ok(found, `找不到页面变量 ${name}`)
  return found
}

const jsx = (type, props, ...children) => typeof type === 'function'
  ? type({ ...props, children })
  : { type, props: { ...props, children } }

async function component(path) {
  const exports = {}
  vm.runInNewContext(compile(await readFile(new URL(path, import.meta.url), 'utf8')), {
    exports, jsx, Fragment: 'Fragment',
    require(id) {
      if (id === 'react') return {
        useState: value => [value, () => {}], useRef: value => ({ current: value }), useMemo: fn => fn(),
      }
      if (id === '@/components/markdown') return { default: 'Markdown' }
      if (id === '@/api/medical') return { scopeFromBundle: () => null }
      if (id === 'antd' || id === '@ant-design/icons') return new Proxy({}, { get: (_, key) => key })
      return {}
    },
  })
  return exports.default
}
const GraphWorkbench = await component('../src/pages/chat/component/graph-workbench.tsx')
const MedicalWorkbench = await component('../src/pages/chat/component/medical-workbench.tsx')

function reportText(node) {
  if (!node || typeof node !== 'object') return ''
  if (node.type === 'Markdown') return node.props.value
  return Object.values(node).map(reportText).filter(Boolean).join('\n')
}

function fixture() {
  let controller
  const stream = new ReadableStream({ start(value) { controller = value } })
  const target = { type: 'deepsearch', role: 'assistant', content: '', reactSteps: [] }
  const context = vm.createContext({
    ...events, jsx, Fragment: 'Fragment', GraphWorkbench, MedicalWorkbench,
    console: { log() {}, debug() {}, warn() {} }, Date, Map, Set, JSON, TextDecoder,
    id: 'session', ChatType: { Deepsearch: 'deepsearch' },
    scopeRef: { current: {} }, activeMedicalRunRef: { current: null },
    resultRequestRef: { current: 0 }, researchStopRequestedRef: { current: false },
    researchDetailsRef: { current: new Map() }, researchStepsRef: { current: [] },
    researchProtocolRef: { current: events.initialResearchProtocolState },
    currentSessionIdRef: { current: 'session' }, currentRunIdRef: { current: null },
    readerRef: { current: null }, scopeFromBundle: () => null,
    medicalRunId: null, medicalBundle: null, medicalStatus: 'idle', medicalError: '',
    researchSteps: [], selectedResearchDetail: null, researchDataVersion: 0,
    currentChatItem: target, selectedStepDetail: null, isDeepResearchMode: true,
    aggregatedResearchData: null, list: [], loading: false, disposed: false,
    handleResearchStepClick() {}, navigate() {}, scrollToBottom() {}, createChatId: () => 'assistant',
    setResolvedQueries() {}, applyRecoveryToMessages: () => target,
    chat: { list: [target] },
    api: { session: { deepsearch: async () => ({ data: stream }) } },
  })
  for (const [setter, key] of Object.entries({
    setMedicalRunId: 'medicalRunId', setMedicalBundle: 'medicalBundle',
    setMedicalStatus: 'medicalStatus', setMedicalError: 'medicalError',
    setResearchSteps: 'researchSteps', setSelectedResearchDetail: 'selectedResearchDetail',
    setResearchDataVersion: 'researchDataVersion', setCurrentChatItem: 'currentChatItem',
  })) context[setter] = value => { context[key] = typeof value === 'function' ? value(context[key]) : value }
  for (const name of ['resetResearchDetails', 'updateReportDetail', 'applyMedicalBundle', 'loadMedicalRun', 'sendChat']) {
    context[name] = vm.runInContext(compile(`(${initializer(name).arguments[0].getText(tree)})`), context)
  }
  for (const name of ['normalizeStepType', 'applyProjection']) {
    context[name] = vm.runInContext(compile(`(${initializer(name).getText(tree)})`), context)
  }
  let memo
  context.useMemo = (fn, deps) => {
    if (!memo || deps.some((value, index) => !Object.is(value, memo.deps[index]))) memo = { deps, value: fn() }
    return memo.value
  }
  let seq = 0
  return {
    context, target,
    start: () => context.sendChat(target, '测试报告', 'run', []),
    push(type, data) {
      controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({
        version: 1, event_id: `event-${++seq}`, session_id: 'session', run_id: 'run', seq,
        type, timestamp: '2026-10-20T00:00:00Z', data,
      })}\n\n`))
    },
    close: () => controller.close(),
    detail: () => context.researchDetailsRef.current.get('writing'),
    render: () => reportText(vm.runInContext(compile(initializer('rightPanelContent').getText(tree)), context)),
  }
}

async function until(check) {
  for (let turn = 0; turn < 40; turn++) {
    if (check()) return
    await new Promise(resolve => setImmediate(resolve))
  }
  assert.ok(check(), '事件未按预期进入页面状态')
}
const section = (id, text) => ({ content: { section_id: id, section_title: `章节${id}`, content: text } })

test('首次发起在完成事件之前逐章显示，完成后用最终正文替换同一detail', async () => {
  const f = fixture()
  const pending = f.start()
  f.push('research_start', { query: '测试报告' })
  f.push('section_content', section('1', '第一章正文'))
  await until(() => f.detail()?.streamingReport.includes('第一章正文'))
  assert.equal(f.context.medicalStatus, 'running')
  assert.equal(f.target.loading, true)
  assert.match(f.render(), /第一章正文/)
  f.push('section_content', section('2', '第二章正文'))
  await until(() => f.detail()?.sections.length === 2)
  assert.match(f.render(), /第一章正文[\s\S]*第二章正文/)
  f.push('research_complete', { final_report: '审核后的最终报告', references: [] })
  f.close()
  await pending
  assert.equal(f.context.medicalStatus, 'completed')
  assert.equal(f.detail().streamingReport, '审核后的最终报告')
  assert.equal(f.render(), '审核后的最终报告')
})

test('同章节以新seq再次到达时按ID更新，正文不叠加旧版本', async () => {
  const f = fixture()
  const pending = f.start()
  f.push('research_start', {})
  f.push('section_content', section('1', '章节旧内容'))
  f.push('section_content', section('1', '章节新内容'))
  await until(() => f.detail()?.streamingReport.includes('章节新内容'))
  assert.equal(f.detail().sections.length, 1)
  assert.doesNotMatch(f.render(), /章节旧内容/)
  assert.equal(f.render().match(/章节新内容/g).length, 1)
  f.close()
  await pending
})

test('report_chunk只更新detail时，报告面板也会重新计算', async () => {
  const f = fixture()
  const pending = f.start()
  f.push('research_start', {})
  f.push('report_chunk', { content: { chunk: '第一段' } })
  await until(() => f.detail()?.streamingReport === '第一段')
  assert.equal(f.render(), '第一段')
  f.push('report_chunk', { content: { chunk: '第二段' } })
  await until(() => f.detail()?.streamingReport === '第一段第二段')
  assert.equal(f.render(), '第一段第二段')
  f.close()
  await pending
})

test('没有提前章节的最终交付也会创建写作detail', async () => {
  const f = fixture()
  const pending = f.start()
  f.push('research_start', {})
  f.push('research_complete', { final_report: '直接交付报告' })
  f.close()
  await pending
  assert.equal(f.detail().streamingReport, '直接交付报告')
  assert.equal(f.render(), '直接交付报告')
})

test('结果刷新写入detail，运行中的旧快照不会清空已收到正文', () => {
  const f = fixture()
  f.context.updateReportDetail('正在生成的章节正文')
  f.context.applyMedicalBundle({ pipeline: 'graph', run_status: 'running', final_report: '' })
  assert.equal(f.render(), '正在生成的章节正文')
  f.context.applyMedicalBundle({ pipeline: 'graph', run_status: 'completed', final_report: '最终正文' })
  assert.equal(f.render(), '最终正文')
})

test('恢复投影缺少写作步骤时仍创建写作detail并驱动报告面板', () => {
  const f = fixture()
  const projection = events.createRecoveryProjection({ status: 'paused', final_report: '已保存的部分报告' })
  projection.research_steps = [{ type: 'planning', title: '规划', status: 'completed' }]
  f.context.applyProjection({ run_id: 'run', query: '测试报告' }, projection)
  assert.equal(f.detail().streamingReport, '已保存的部分报告')
  assert.equal(f.render(), '已保存的部分报告')
  f.context.applyProjection({ run_id: 'run' }, { ...projection, streaming_report: '恢复后的报告' })
  assert.equal(f.render(), '恢复后的报告')
})

test('切换研究版本先清除上一个detail，读取完成后显示新版本', async () => {
  const f = fixture()
  f.context.activeMedicalRunRef.current = 'previous'
  f.context.updateReportDetail('上一份报告')
  let resolve
  f.context.getMedicalResearch = () => new Promise(done => { resolve = done })
  const pending = f.context.loadMedicalRun('next')
  assert.equal(f.render(), '')
  resolve({ pipeline: 'graph', run_status: 'completed', final_report: '新的报告' })
  await pending
  assert.equal(f.render(), '新的报告')
})

test('两个工作台都读取detail，空detail不会绕过统一状态显示bundle中的报告', () => {
  const props = { bundle: { final_report: 'bundle正文' }, status: 'completed', runId: 'run' }
  for (const Workbench of [GraphWorkbench, MedicalWorkbench]) {
    assert.equal(reportText(Workbench(props)), '')
    assert.equal(reportText(Workbench({ ...props, detail: { streamingReport: 'detail正文' } })), 'detail正文')
  }
})
