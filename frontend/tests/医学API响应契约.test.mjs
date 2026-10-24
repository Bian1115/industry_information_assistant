import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

const compile = source => ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText
const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const errorSource = compile(await readFile(new URL('../src/api/request/error.ts', import.meta.url), 'utf8'))
const pluginSource = compile(await readFile(new URL('../src/api/request/plugins/service.ts', import.meta.url), 'utf8'))
  .replace("'../error'", JSON.stringify(dataUrl(errorSource)))
const { servicePlugin } = await import(dataUrl(pluginSource))
let onSuccess
let onError
servicePlugin.install({ interceptors: { response: { use(success, failure) { onSuccess = success; onError = failure } } } })

test('医学领域状态不会被误判为通用API错误', async () => {
  for (const status of ['partial', 'completed_with_limits']) {
    const response = { status: 200, config: { serviceEnvelope: false }, data: { status, papers: [], run_status: 'completed' } }
    assert.equal(await onSuccess(response), response)
  }
})

test('旧接口仍检查服务信封失败状态', async () => {
  const response = { status: 200, config: {}, data: { status: 'error', message: '旧接口失败' } }
  await assert.rejects(onSuccess(response), /旧接口失败/)
})

test('医学接口关闭信封转换仍保留HTTP权限及版本错误', async () => {
  for (const status of [401, 403, 404, 409, 422]) {
    const error = { response: { status, config: { serviceEnvelope: false }, data: { detail: '请求被拒绝' } } }
    await assert.rejects(onError(error), value => value === error)
  }
})
