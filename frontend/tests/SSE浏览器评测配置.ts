import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
const streams = new Map<string, Set<any>>()

// 只在隔离Vite评测实例转换源码；不改生产源文件，不写入正式构建。
export default defineConfig({
  root: path.resolve(import.meta.dirname, '..'),
  define: { 'import.meta.env.VITE_API_BASE': JSON.stringify('/api') },
  resolve: { alias: [{ find: /^@\//, replacement: '/src/' }] },
  server: { host: '127.0.0.1', port: 5187, strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:18765', configure(proxy) {
      // 让浏览器连接中断传播至真实上游TCP；避免开发代理保留已无消费者的流。
      proxy.on('proxyReq', (upstream, _req, response) => {
        response.on('close', () => { if (!response.writableFinished) upstream.destroy() })
      })
    } }, '/evaluation': 'http://127.0.0.1:18765' } },
  plugins: [{
    name: 'SSE评测埋点', enforce: 'pre',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url?.startsWith('/evaluation/cut-stream/')) {
          const runId = req.url.split('/').pop()!
          const active = streams.get(runId) || new Set()
          const count = active.size
          for (const response of active) response.socket?.destroy()
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ run_id: runId, closed_connections: count, epoch_ms: Date.now() }))
          return
        }
        if (req.url?.startsWith('/api/research/resume/')) {
          const runId = req.url.split('/').pop()!.split('?')[0]
          const active = streams.get(runId) || new Set()
          active.add(res); streams.set(runId, active)
          res.on('close', () => active.delete(res))
        }
        next()
      })
    },
    transform(code, id) {
      if (id.endsWith('/pages/chat/research-events.ts')) {
        code = code.replace('if (event) await onEvent(event)',
          'if (event) { (window as any).__SSEProbe?.receive(event); await onEvent(event) }')
        code = code.replace('return next\n}', '(window as any).__SSEProbe?.apply(event, next); return next\n}')
        code = code.replace("return { state, accepted: false, reason: 'duplicate' }",
          "(window as any).__SSEProbe?.rejected(event, 'duplicate'); return { state, accepted: false, reason: 'duplicate' }")
        code = code.replace("reason: 'gap',", "reason: 'gap',")
        code = code.replace('if (sameRun && event.seq > state.lastSeq + 1) {',
          "if (sameRun && event.seq > state.lastSeq + 1) { (window as any).__SSEProbe?.rejected(event, 'gap');")
        return { code, map: null }
      }
      if (id.endsWith('/pages/chat/index.tsx')) {
        code = code.replace('res = await api.session.deepsearch({',
          ';(window as any).__SSEProbe?.request(turnId, 0); res = await api.session.deepsearch({')
        code = code.replace('researchProtocolRef.current = advance.state\n          const json',
          'researchProtocolRef.current = advance.state; (window as any).__SSEProbe?.initialAccepted(protocolEvent, advance.state)\n          const json')
        code = code.replace('let projection = projectionFromRunState(runState)',
          ';(window as any).__SSEProbe?.reset(); let projection = projectionFromRunState(runState)')
        code = code.replace('setCurrentChatItem(assistant)',
          'setCurrentChatItem(assistant); (window as any).__SSEProbe?.projection(checkpoint, projection)')
        // 直接记录真实Axios恢复请求发起边界，重连也保留独立时间。
        code = code.replace('const streamRes = await api.session.resumeResearch(runId!, protocol.lastSeq)',
          '(window as any).__SSEProbe?.request(runId, protocol.lastSeq); const streamRes = await api.session.resumeResearch(runId!, protocol.lastSeq)')
        return { code, map: null }
      }
      if (id.endsWith('/pages/chat/component/graph-workbench.tsx')) {
        code = code.replace("const reportText = report || bundle?.final_report || ''",
          "const reportText = report || bundle?.final_report || ''; (window as any).__SSEProbe?.render({ status, report: reportText })")
        return { code, map: null }
      }
    },
  }, react()],
})
