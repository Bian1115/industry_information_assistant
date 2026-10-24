import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import axios from 'axios'
import ts from 'typescript'

const compile = source => ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
}).outputText
const load = source => import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)
const { repeatPlugin } = await load(compile(await readFile(
  new URL('../src/api/request/plugins/repeat.ts', import.meta.url), 'utf8',
)))
const { installPlugins } = await load(compile(await readFile(
  new URL('../src/api/request/plugins/plugin.ts', import.meta.url), 'utf8',
)))
// 使用实际 Axios 和项目去重拦截器，仅替换网络传输。
const request = axios.create({ cancelRepeat: true })
installPlugins(request, [repeatPlugin])
globalThis.__parallelResearchRequest = request
async function loadApi(path) {
  return load(compile(await readFile(new URL(path, import.meta.url), 'utf8'))
    .replace(/import \{ request \} from ['"]\.\/request['"];?/, 'const request = globalThis.__parallelResearchRequest;'))
}
const { getMedicalResearch } = await loadApi('../src/api/medical.ts')
const { getResearchRunState } = await loadApi('../src/api/session.ts')

function deferredAdapter() {
  const pending = []
  request.defaults.adapter = config => new Promise(resolve => {
    pending.push({ config, finish: () => resolve({
      config, status: 200, statusText: 'OK', headers: {},
      data: { success: true, state: {
        pipeline: 'graph', exists: true, run_status: 'failed', status: 'running',
        pending_nodes: ['research'], draft: { final_report: '已保存内容' },
      } },
    }) })
  })
  return pending
}

for (const [name, calls] of [
  ['页面资料读取和恢复状态读取并发时均成功', [getMedicalResearch, getResearchRunState]],
  ['恢复状态先请求时不会被页面资料读取取消', [getResearchRunState, getMedicalResearch]],
  ['自动刷新和手动重新读取并发时均成功', [getMedicalResearch, getMedicalResearch]],
]) {
  test(name, async () => {
    const pending = deferredAdapter()
    const outcomes = Promise.allSettled(calls.map(fn => fn('parallel-run')))
    await new Promise(resolve => setImmediate(resolve))
    pending.forEach(item => item.finish())
    const results = await outcomes
    assert.deepEqual(results.map(result => result.status), ['fulfilled', 'fulfilled'])
    assert.equal(pending.length, 2)
    for (let i = 0; i < results.length; i++) {
      const value = results[i].value
      assert.equal(calls[i] === getMedicalResearch ? value.final_report : value.data.state.draft.final_report, '已保存内容')
    }
  })
}

test('真实HTTP错误和网络错误继续抛出，不能伪装成读取成功', async () => {
  for (const status of [401, 403, 500, undefined]) {
    let error
    request.defaults.adapter = async config => {
      error = new axios.AxiosError('注入读取失败', status ? 'ERR_BAD_RESPONSE' : 'ERR_NETWORK', config,
        undefined, status ? { status, data: { detail: '读取被拒绝' }, config, headers: {} } : undefined)
      throw error
    }
    await assert.rejects(getMedicalResearch('failure-run'), e => e === error)
  }
})
