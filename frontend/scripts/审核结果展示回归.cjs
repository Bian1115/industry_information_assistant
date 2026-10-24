/* 离线验证审核展示，使用现有 TypeScript 编译器与 Node 内置测试工具。 */
const assert = require('node:assert/strict')
const { test } = require('node:test')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')

const source = fs.readFileSync(path.join(__dirname, '../src/pages/chat/research-events.ts'), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } })
const context = { exports: {} }
vm.runInNewContext(compiled.outputText, context)
const { formatResearchReview } = context.exports

test('明确的返工结论优先于高分和 passed 字段', () => {
  for (const verdict of ['needs_revision', 'major_issues']) {
    const review = formatResearchReview({ quality_score: 9, verdict, passed: true })
    assert.equal(review.passed, false)
    assert.match(review.text, /需要修订/)
    assert.doesNotMatch(review.text, /审核通过/)
  }
  assert.equal(formatResearchReview({ quality_score: 9, passed: false }).passed, false)
})

test('报告三维评分与 URL 和论断对的计数分别展示', () => {
  const review = formatResearchReview({
    quality_score: 9, verdict: 'pass',
    quality_dimension_scores: {
      logical_coherence: 10, requirement_coverage: 10, freshness: 4,
      evidence_traceability: 1, factual_reliability: 1,
    },
    citation_audit: {
      traceability: { status: 'passed', citations_verified: 1, citations_total: 1 },
      support: { status: 'passed', supported: 3, citations_total: 3, unsupported: 0, unknown: 0 },
    },
  })
  assert.equal(review.passed, true)
  assert.match(review.text, /整体逻辑 10\/10；要求覆盖 10\/10；时效性 4\/10/)
  assert.match(review.text, /可追溯 1\/1/)
  assert.match(review.text, /支持 3\/3/)
  assert.match(review.text, /不代表全篇事实均已核验/)
  assert.doesNotMatch(review.text, /evidence_traceability|factual_reliability/)
})

test('缺失和关闭的引用核验不显示为通过或百分之百', () => {
  const missing = formatResearchReview({}).text
  assert.match(missing, /未评分/)
  assert.match(missing, /未核验；支持 —\/—/)
  const disabled = formatResearchReview({ citation_audit: {
    traceability: { status: 'no_citations', citations_total: 0, citations_verified: 0 },
    support: { status: 'disabled', supported: 0, citations_total: 0, unsupported: 0, unknown: 0 },
  } }).text
  assert.match(disabled, /没有可核验引用/)
  assert.match(disabled, /核验已关闭/)
  assert.doesNotMatch(missing + disabled, /100%|引用支持（论断—引用对）：通过/)
})

test('引用核验失败保留不支持和未确认数量', () => {
  const review = formatResearchReview({ quality_score: 4, verdict: 'major_issues', citation_audit: {
    support: { status: 'failed', citations_total: 3, supported: 1, unsupported: 1, unknown: 1 },
  } })
  assert.match(review.text, /未通过；支持 1\/3；不支持 1；未确认 1/)
  assert.equal(review.passed, false)
})
