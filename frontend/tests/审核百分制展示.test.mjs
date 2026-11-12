import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

const compiled = ts.transpileModule(await readFile(new URL('../src/pages/chat/research-events.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
}).outputText
const { formatResearchReview } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)

test('展示三维百分制以及程序计算的通过结论', () => {
  const result = formatResearchReview({ quality_score: 61, verdict: 'pass', quality_dimension_scores: {
    readability: 60, comprehensiveness: 61, depth: 62,
  } })
  assert.equal(result.passed, true)
  for (const value of ['61/100', '可读性 60/100', '全面性 61/100', '深度 62/100']) {
    assert.ok(result.text.includes(value))
  }
})

test('高分但存在严重问题时，前端沿用服务端未通过结论', () => {
  const result = formatResearchReview({ quality_score: 95, verdict: 'major_issues' })
  assert.equal(result.passed, false)
  assert.ok(result.text.includes('需要修订'))
})

test('零分正确显示；缺少审核结论时不凭分数宣布通过', () => {
  assert.ok(formatResearchReview({ quality_score: 0, verdict: 'major_issues' }).text.includes('0/100'))
  assert.equal(formatResearchReview({ quality_score: 100 }).passed, false)
})
