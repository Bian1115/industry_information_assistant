/* 运行真实 auth/session/device store，替换网络 API 验证账号切换与延迟响应。 */
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const esbuild = require('esbuild')

async function main() {
  const data = new Map()
  global.localStorage = {
    getItem: key => data.get(key) ?? null,
    setItem: (key, value) => data.set(key, value),
    removeItem: key => data.delete(key),
  }
  global.window = { localStorage: global.localStorage }
  global.__isolationApi = {}
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), '用户隔离状态-'))
  const root = path.resolve(__dirname, '..')
  try {
    const output = path.join(directory, 'stores.cjs')
    await esbuild.build({
      stdin: { contents: `export {authActions, authState} from './src/store/auth'; export {sessionActions, sessionState} from './src/store/session'; export {deviceState, deviceActions} from './src/store/device';`, resolveDir: root, loader: 'ts' },
      bundle: true, platform: 'node', format: 'cjs', outfile: output,
      plugins: [{ name: '替换网络', setup(build) {
        build.onResolve({ filter: /^@\/api\/session$/ }, () => ({ path: '会话API', namespace: 'test' }))
        build.onLoad({ filter: /.*/, namespace: 'test' }, () => ({ contents: ['getSessions', 'createSession', 'getSession', 'updateSession', 'deleteSession'].map(name => `export const ${name}=(...args)=>globalThis.__isolationApi.${name}(...args);`).join('\n'), loader: 'js' }))
      } }],
    })
    const { authActions, sessionActions, sessionState, deviceActions, deviceState } = require(output)
    await new Promise(resolve => setImmediate(resolve))
    const a = { id: 'a', username: '甲', email: 'a@example.invalid', is_active: true, created_at: '' }
    const b = { ...a, id: 'b', username: '乙' }
    const old = { id: 'old', title: '甲私有会话', messages: [] }
    const current = { id: 'current', title: '乙私有会话', messages: [] }
    const records = []
    async function test(name, fn) {
      try { await fn(); records.push({ name, status: 'passed' }) }
      catch (error) { records.push({ name, status: 'failed', error: error.stack }); process.exitCode = 1 }
    }
    function deferred() {
      let resolve
      const promise = new Promise(done => { resolve = done })
      return { promise, resolve }
    }
    await test('登录同步更新身份并清空上一账号会话和知识库选择', async () => {
      sessionState.sessions = [old]
      sessionState.currentSession = old
      deviceActions.setSelectedKbIds(['private-a'])
      deviceActions.setChatting(true)
      authActions.login('token-b', b)
      assert.equal(JSON.parse(data.get('auth')).token, 'token-b')
      assert.equal(sessionState.currentSession, null)
      assert.equal(sessionState.sessions.length, 0)
      assert.equal(deviceState.selectedKbIds.length, 0)
      assert.equal(deviceState.chatting, false)
    })
    await test('登出清除身份及私有缓存', async () => {
      sessionState.sessions = [old]
      sessionState.currentSession = old
      deviceActions.setSelectedKbIds(['private-a'])
      authActions.logout()
      assert.equal(data.get('auth'), undefined)
      assert.equal(sessionState.currentSession, null)
      assert.equal(sessionState.sessions.length, 0)
      assert.equal(deviceState.selectedKbIds.length, 0)
    })
    await test('旧账号延迟列表不能覆盖新账号缓存', async () => {
      authActions.login('token-a', a)
      const pending = deferred()
      global.__isolationApi.getSessions = () => pending.promise
      const request = sessionActions.fetchSessions()
      authActions.login('token-b', b)
      sessionState.sessions = [current]
      pending.resolve({ data: [old] })
      await request
      assert.deepEqual(sessionState.sessions.map(row => row.id), ['current'])
      assert.equal(sessionState.loading, false)
    })
    for (const [api, action, args] of [
      ['getSession', 'loadSession', ['old']],
      ['createSession', 'createNewSession', ['甲新会话']],
      ['updateSession', 'renameSession', ['old', '甲重命名']],
    ]) {
      await test(`旧账号延迟${action}不能写入新账号缓存`, async () => {
        authActions.login('token-a', a)
        const pending = deferred()
        global.__isolationApi[api] = () => pending.promise
        const request = sessionActions[action](...args)
        // 提前连接拒绝处理，避免 Node 将预期拒绝误报为未处理异常。
        const assertion = assert.rejects(request, /账号已切换/)
        authActions.login('token-b', b)
        sessionState.sessions = [current]
        sessionState.currentSession = current
        pending.resolve({ data: old })
        await assertion
        assert.deepEqual(sessionState.sessions.map(row => row.id), ['current'])
        assert.equal(sessionState.currentSession.id, 'current')
        assert.equal(sessionState.error, null)
      })
    }
    await test('旧账号延迟删除不能修改新账号缓存', async () => {
      authActions.login('token-a', a)
      const pending = deferred()
      global.__isolationApi.deleteSession = () => pending.promise
      const request = sessionActions.removeSession('old')
      authActions.login('token-b', b)
      sessionState.sessions = [current]
      pending.resolve({})
      await request
      assert.deepEqual(sessionState.sessions.map(row => row.id), ['current'])
    })
    await test('同账号会话操作继续正常写入缓存', async () => {
      authActions.login('token-b', b)
      global.__isolationApi.getSessions = async () => ({ data: [current] })
      await sessionActions.fetchSessions()
      global.__isolationApi.getSession = async () => ({ data: current })
      await sessionActions.loadSession('current')
      assert.deepEqual(sessionState.sessions.map(row => row.id), ['current'])
      assert.equal(sessionState.currentSession.id, 'current')
    })
    const result = { date: '2026-10-09', scope: '真实前端 store，网络替身；未运行浏览器', total: records.length, passed: records.filter(row => row.status === 'passed').length, records }
    const destination = process.argv[2]
    if (destination) fs.writeFileSync(destination, JSON.stringify(result, null, 2) + '\n')
    console.log(JSON.stringify({ total: result.total, passed: result.passed, destination }))
  } finally { fs.rmSync(directory, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
