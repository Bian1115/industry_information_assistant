import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import vm from 'node:vm'
import test from 'node:test'
import { randomUUID } from 'node:crypto'
import ts from 'typescript'

// 执行页面真实的两个发送回调，替换网络与 React 状态边界，避免复制发送逻辑。
const source = await readFile(new URL('../src/pages/chat/index.tsx', import.meta.url), 'utf8')
const tree = ts.createSourceFile('index.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
function callback(name, context) {
  let expression
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === name &&
        node.initializer && ts.isCallExpression(node.initializer) &&
        node.initializer.expression.getText(tree) === 'useCallback') {
      expression = node.initializer.arguments[0].getText(tree)
    }
    ts.forEachChild(node, visit)
  }
  visit(tree)
  assert.ok(expression, `找不到页面回调 ${name}`)
  const compiled = ts.transpileModule(`(${expression})`, {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText
  return vm.runInNewContext(compiled, context)
}

function fixture({ mode = 'research', loading = false, disease = '', reject = false } = {}) {
  const calls = []
  const scopePrompts = []
  const context = {
    id: 'session-existing',
    chat: { list: [{ role: 'assistant', content: '上一轮已完成的报告', runId: 'previous-run' }] },
    ChatRole: { User: 'user', Assistant: 'assistant' },
    ChatType: { Normal: 'normal', Deepsearch: 'deepsearch' },
    loadingRef: { current: loading },
    deviceState: { executionMode: mode, selectedKbIds: [], searchModes: ['web'] },
    scopeRef: { current: { disease } },
    researchStopRequestedRef: { current: false },
    activeMedicalRunRef: { current: null },
    currentSessionIdRef: { current: 'session-existing' },
    crypto: { randomUUID }, createChatId: randomUUID,
    scrollToBottom() {}, setCurrentChatItem() {}, setMedicalRunId() {},
    setMedicalStatus() {}, setMedicalError() {}, resetResearchDetails() {},
    setMedicalBundle() {}, setAttachments() {},
    setPendingMedicalQuestion: value => scopePrompts.push(value),
    setScopeEditorOpen: value => scopePrompts.push(value),
    api: { session: Object.fromEntries(['deepsearch', 'chat', 'chatWithAttachments'].map(method => [
      method, async payload => {
        calls.push({ method, payload })
        if (reject) throw new Error('测试网络失败')
        // 到达网络边界即结束；本测试不声称验证 SSE 或模型输出。
        return { data: { getReader: () => null } }
      },
    ])) },
  }
  context.sendChat = callback('sendChat', context)
  return { context, calls, scopePrompts, send: callback('send', context) }
}

test('重新打开无旧疾病字段的图研究后，3000字追问发送到同一会话', async () => {
  const f = fixture()
  await f.send('请将字数控制在3000字')
  assert.equal(f.calls.length, 1, '追问必须到达请求边界，不能只展开顶部表单')
  assert.equal(f.calls[0].method, 'deepsearch')
  assert.equal(f.calls[0].payload.session_id, 'session-existing')
  assert.equal(f.calls[0].payload.query, '请将字数控制在3000字')
  assert.equal(f.context.chat.list[0].content, '上一轮已完成的报告')
  assert.equal(f.context.chat.list[1].content, '请将字数控制在3000字')
  assert.equal(f.context.chat.list[2].loading, false)
  assert.deepEqual(f.scopePrompts, [])
})

test('连续追问分别生成新的轮次，已有范围也能正常发送', async () => {
  const f = fixture({ disease: '已有主题' })
  await f.send('请将字数控制在3000字')
  await f.send('再保留主要局限')
  assert.equal(f.calls.length, 2)
  assert.notEqual(f.calls[0].payload.turn_id, f.calls[1].payload.turn_id)
  assert.equal(f.calls[1].payload.session_id, f.calls[0].payload.session_id)
  assert.equal(f.context.chat.list.length, 5)
})

test('正在生成和空消息不发送新请求', async () => {
  const busy = fixture({ loading: true })
  await busy.send('请将字数控制在3000字')
  const empty = fixture()
  await empty.send('')
  assert.equal(busy.calls.length, 0)
  assert.equal(empty.calls.length, 0)
  assert.equal(busy.context.chat.list.length, 1)
  assert.equal(empty.context.chat.list.length, 1)
})

test('快速问答与附件请求保持原路径', async () => {
  const f = fixture({ mode: 'chat' })
  await f.send('解释上一份报告')
  await f.send('', ['attachment-1'])
  assert.equal(f.calls[0].method, 'chat')
  assert.equal(f.calls[1].method, 'chatWithAttachments')
  assert.equal(f.calls[1].payload.question, '请分析附件内容')
  assert.deepEqual(f.calls[1].payload.attachment_ids, ['attachment-1'])
})

test('第二轮请求失败后保留错误反馈并解除加载', async () => {
  const f = fixture({ reject: true })
  await f.send('请将字数控制在3000字')
  assert.equal(f.calls.length, 1)
  const target = f.context.chat.list.at(-1)
  assert.equal(target.researchStatus, 'failed')
  assert.equal(target.loading, false)
  assert.match(target.content, /请求中断/)
})
