/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

import * as api from '@/api'
import { isAxiosError, isCancel } from 'axios'
import type { ResearchCheckpoint } from '@/api/session'
import ComPageLayout from '@/components/page-layout'
import ComSender, { AttachmentInfo } from '@/components/sender'
import { ChatRole, ChatType } from '@/configs'
import { deviceActions, deviceState } from '@/store/device'
import { sessionState } from '@/store/session'
import { usePageTransport } from '@/utils'
import { useUnmount } from 'ahooks'
import { uniqueId } from 'lodash-es'
import { Select, message } from 'antd'
import MedicalWorkbench from './component/medical-workbench'
import GraphWorkbench from './component/graph-workbench'
import './component/research-start.scss'
import { defaultMedicalScope, getMedicalResearch, scopeFromBundle, type MedicalScope, type MedicalBundle } from '@/api/medical'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { proxy, useSnapshot } from 'valtio'
import ChatMessage from './component/chat-message'
import Drawer from './component/drawer'
import Source from './component/source'
import StepDetailPanel, { StepDetailData } from './component/step-detail-panel'
import ResearchDetail, { ResearchDetailData, ResearchStep } from './component/research-detail'
import styles from './index.module.scss'
import { createChatId, createChatIdText, transportToChatEnter } from './shared'
import { applyRecoveryToMessages } from './recovery-messages'
import {
  advanceResearchProtocol,
  consumeResearchSse,
  createRecoveryProjection,
  projectionFromRunState,
  flattenResearchEvent,
  formatResearchReview,
  initialResearchProtocolState,
  type ResearchProtocolState,
  parseResearchSseData,
  reduceRecoveryProjection,
  reduceResearchProtocol,
  type ResearchRecoveryProjection,
} from './research-events'

async function scrollToBottom() {
  await new Promise((resolve) => setTimeout(resolve))

  const threshold = 200
  const distanceToBottom =
    document.documentElement.scrollHeight -
    document.documentElement.scrollTop -
    document.documentElement.clientHeight

  if (distanceToBottom <= threshold) {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: 'smooth',
    })
  }
}

export default function Index() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [medicalBundle, setMedicalBundle] = useState<MedicalBundle | null>(null)
  const [medicalRunId, setMedicalRunId] = useState<string | null>(null)
  const [medicalStatus, setMedicalStatus] = useState('idle')
  const [medicalError, setMedicalError] = useState('')
  const [resolvedQueries, setResolvedQueries] = useState<Record<string, string>>({})
  const [recoveryNonce, setRecoveryNonce] = useState(0)
  const requestedResumeRunRef = useRef<string | null>(null)
  const researchStopRequestedRef = useRef(false)
  const scopeRef = useRef<MedicalScope>(defaultMedicalScope())
  const previousMedicalRunRef = useRef<string | undefined>(undefined)
  const activeMedicalRunRef = useRef<string | null>(null)
  const resultRequestRef = useRef(0)

  // 研究详情保存报告正文，首跑、恢复和结果读取共用同一份数据。
  const [researchSteps, setResearchSteps] = useState<ResearchStep[]>([])
  const researchStepsRef = useRef<ResearchStep[]>([])
  const [selectedResearchDetail, setSelectedResearchDetail] = useState<ResearchDetailData | null>(null)
  const researchDetailsRef = useRef<Map<string, ResearchDetailData>>(new Map())
  const [researchDataVersion, setResearchDataVersion] = useState(0)
  const resetResearchDetails = useCallback(() => {
    researchDetailsRef.current.clear()
    setSelectedResearchDetail(null)
    setResearchDataVersion(version => version + 1)
  }, [])
  const updateReportDetail = useCallback((update: string | ((current: string) => string)) => {
    const previous = researchDetailsRef.current.get('writing')
    const report = typeof update === 'function' ? update(previous?.streamingReport || '') : update
    if (previous?.streamingReport === report) return
    const detail: ResearchDetailData = {
      ...previous,
      stepId: 'writing',
      stepType: 'writing',
      title: previous?.title || '写作阶段',
      streamingReport: report,
    }
    researchDetailsRef.current.set('writing', detail)
    setSelectedResearchDetail(selected => selected?.stepId === 'writing' ? { ...detail } : selected)
    setResearchDataVersion(version => version + 1)
  }, [])

  const applyMedicalBundle = useCallback((bundle: MedicalBundle) => {
    setMedicalBundle(bundle)
    const scope = scopeFromBundle(bundle)
    if (scope) scopeRef.current = scope
    if ('final_report' in bundle) updateReportDetail(previous => {
      const saved = bundle.final_report || ''
      return bundle.run_status === 'running' && previous.length > saved.length ? previous : saved
    })
  }, [updateReportDetail])
  const loadMedicalRun = useCallback(async (runId: string, status = 'completed', quiet = false) => {
    const requestId = ++resultRequestRef.current
    if (activeMedicalRunRef.current !== runId) {
      setMedicalBundle(null)
      resetResearchDetails()
      setMedicalRunId(runId)
      setMedicalStatus(status)
    }
    activeMedicalRunRef.current = runId
    try {
      const bundle = await getMedicalResearch(runId)
      if (activeMedicalRunRef.current !== runId || requestId !== resultRequestRef.current) return
      const resolvedQuery = bundle.graph_state?.resolved_query
      if (resolvedQuery) setResolvedQueries(previous => previous[runId] === resolvedQuery ? previous : { ...previous, [runId]: resolvedQuery })
      applyMedicalBundle({ ...bundle, fetched_at: new Date().toISOString() })
      setMedicalRunId(runId)
      setMedicalStatus(bundle.run_status || status)
      setMedicalError(bundle.graph_state?.error || '')
    } catch (error) {
      // 主动取消不是读取故障；真正的 HTTP / 网络错误仍由下方提示。
      if (isCancel(error)) return
      if (!quiet && activeMedicalRunRef.current === runId && requestId === resultRequestRef.current) {
        const code = isAxiosError(error) ? error.response?.status : undefined
        setMedicalError(code === 404 ? '未找到本人的研究记录。' : code === 401 || code === 403 ? '登录已过期或无权读取此研究，请重新登录。' : '读取研究记录失败，请重试。')
      }
    }
  }, [applyMedicalBundle, resetResearchDetails])
  useEffect(() => {
    if (!medicalRunId || medicalBundle?.pipeline === 'legacy') return
    // 事件流驱动即时进度，定时读取补充检查点中的资料；终态再读取一次。
    if (medicalStatus !== 'running') {
      void loadMedicalRun(medicalRunId, medicalStatus, true)
      return
    }
    const timer = setInterval(() => { void loadMedicalRun(medicalRunId, medicalStatus, true) }, 5000)
    return () => clearInterval(timer)
  }, [medicalRunId, medicalStatus, medicalBundle?.pipeline, loadMedicalRun])
  const { data: ctx } = usePageTransport(transportToChatEnter)

  const [currentChatItem, setCurrentChatItem] = useState<API.ChatItem | null>(
    null,
  )

  // 步骤详情状态 (旧版)
  const [selectedStepDetail, setSelectedStepDetail] = useState<StepDetailData | null>(null)
  const stepDetailsRef = useRef<Map<string, StepDetailData>>(new Map())

  // 同步 researchSteps 到 ref
  useEffect(() => {
    researchStepsRef.current = researchSteps
  }, [researchSteps])

  // 附件状态管理
  const [attachments, setAttachments] = useState<AttachmentInfo[]>([])
  const attachmentPollingRef = useRef<NodeJS.Timeout | null>(null)

  const [chat] = useState(() => {
    return proxy({
      list: [] as API.ChatItem[],
    })
  })
  const { list } = useSnapshot(chat) as {
    list: API.ChatItem[]
  }

  const loading = useMemo(() => {
    return list.some((o) => o.loading)
  }, [list])
  const loadingRef = useRef(loading)
  loadingRef.current = loading
  useEffect(() => {
    deviceActions.setChatting(loading)
  }, [loading])
  useUnmount(() => {
    deviceActions.setChatting(false)
    // 清理轮询
    if (attachmentPollingRef.current) {
      clearInterval(attachmentPollingRef.current)
    }
  })

  // 用于取消请求的 ref
  const readerRef = useRef<ReadableStreamDefaultReader<any> | null>(null)
  const currentSessionIdRef = useRef<string | null>(null)
  const currentRunIdRef = useRef<string | null>(null)
  const researchProtocolRef = useRef(initialResearchProtocolState)

  // 停止生成
  const handleStop = useCallback(async () => {
    console.log('[handleStop] 用户点击停止按钮')
    setMedicalStatus('cancelled')
    researchStopRequestedRef.current = true

    // 取消读取流
    if (readerRef.current) {
      try {
        await readerRef.current.cancel()
        console.log('[handleStop] 读取流已取消')
      } catch (e) {
        console.error('[handleStop] 取消读取流失败:', e)
      }
      readerRef.current = null
    }

    // 调用后端取消 API
    if (currentSessionIdRef.current && currentRunIdRef.current) {
      try {
        await api.session.cancelResearch(
          currentSessionIdRef.current,
          currentRunIdRef.current,
        )
        console.log('[handleStop] 后端取消请求已发送')
      } catch (e) {
        console.error('[handleStop] 调用取消 API 失败:', e)
      }
    }

    // 停止当前聊天项的加载状态
    const loadingItem = chat.list.find(item => item.loading)
    if (loadingItem) {
      loadingItem.loading = false
      loadingItem.researchStatus = 'cancelled'
      if (!loadingItem.content) {
        loadingItem.content = '⏹️ 已停止生成'
      }
    }

    // 更新研究步骤状态
    setResearchSteps(prev => prev.map(s =>
      s.status === 'running' ? { ...s, status: 'completed' as const } : s
    ))
  }, [chat])

  // 轮询检查附件处理状态
  useEffect(() => {
    const pendingAttachments = attachments.filter(
      att => att.status === 'pending' || att.status === 'processing'
    )

    if (pendingAttachments.length > 0 && !attachmentPollingRef.current) {
      attachmentPollingRef.current = setInterval(async () => {
        for (const att of pendingAttachments) {
          try {
            const res = await api.session.getAttachment(att.id)
            if (res.data) {
              setAttachments(prev =>
                prev.map(a =>
                  a.id === att.id ? { ...a, status: res.data.status } : a
                )
              )
            }
          } catch (e) {
            console.error('Failed to check attachment status', e)
          }
        }
      }, 2000)
    } else if (pendingAttachments.length === 0 && attachmentPollingRef.current) {
      clearInterval(attachmentPollingRef.current)
      attachmentPollingRef.current = null
    }

    return () => {
      if (attachmentPollingRef.current) {
        clearInterval(attachmentPollingRef.current)
        attachmentPollingRef.current = null
      }
    }
  }, [attachments])

  // 上传附件
  const handleUploadAttachment = useCallback(async (file: File) => {
    if (!id) {
      message.error('请先创建会话')
      return null
    }

    // 添加临时附件
    const tempId = uniqueId('temp-attachment-')
    setAttachments(prev => [
      ...prev,
      { id: tempId, filename: file.name, status: 'uploading' }
    ])

    try {
      const res = await api.session.uploadAttachment(id, file)
      if (res.data) {
        // 替换临时附件为真实附件
        setAttachments(prev =>
          prev.map(a =>
            a.id === tempId
              ? { id: res.data.id, filename: res.data.filename, status: res.data.status }
              : a
          )
        )
        message.success(`附件 ${file.name} 上传成功`)
        return res.data
      }
    } catch (e: any) {
      message.error(`附件上传失败: ${e.message || '未知错误'}`)
      // 移除失败的附件
      setAttachments(prev => prev.filter(a => a.id !== tempId))
    }
    return null
  }, [id])

  // 移除附件
  const handleRemoveAttachment = useCallback(async (attachmentId: string) => {
    try {
      // 只有非临时 ID 才需要调用删除 API
      if (!attachmentId.startsWith('temp-')) {
        await api.session.deleteAttachment(attachmentId)
      }
      setAttachments(prev => prev.filter(a => a.id !== attachmentId))
    } catch (e) {
      console.error('Failed to delete attachment', e)
    }
  }, [])

  const sendChat = useCallback(
    async (
      target: API.ChatItem,
      message: string,
      turnId: string,
      attachmentIds?: string[],
    ) => {
      setCurrentChatItem(target)
      target.loading = true
      try {
        let res
        if (target.type === ChatType.Deepsearch) {
          target.runId = turnId
          target.researchStatus = 'running'
          researchStopRequestedRef.current = false
          setMedicalRunId(turnId)
          activeMedicalRunRef.current = turnId
          setMedicalStatus('running')
          setMedicalError('')
          resetResearchDetails()
          setMedicalBundle({ pipeline: 'graph' })
          // 唯一主链路：不再选择研究类型，检索范围由后端配置的指定网站决定。
          res = await api.session.deepsearch({
            query: message,
            session_id: id!,  // PostgreSQL 会话 ID
            turn_id: turnId,
          })
        } else {
          // 用户勾选了知识库就限定范围；未勾选则留空，由后端检索全部知识库
          const kbIds = deviceState.selectedKbIds as string[]
          const kbParams = kbIds.length > 0 ? { kb_ids: [...kbIds] } : {}
          const sources = deviceState.searchModes as string[]
          const sourceParams = {
            search_web: sources.includes('web'),
            search_knowledge: sources.includes('local'),
          }

          if (attachmentIds && attachmentIds.length > 0) {
            // 使用带附件的聊天接口
            res = await api.session.chatWithAttachments({
              session_id: id!,
              turn_id: turnId,
              question: message,
              attachment_ids: attachmentIds,
              ...sourceParams,
              ...kbParams,
            })
          } else {
            res = await api.session.chat({
              session_id: id!,
              turn_id: turnId,
              question: message,
              ...sourceParams,
              ...kbParams,
            })
          }
        }

        const reader = res.data.getReader()
        if (!reader) return
        if (currentSessionIdRef.current !== id) { await reader.cancel(); return }

        // 存储 reader 和 session ID 用于取消
        readerRef.current = reader
        currentSessionIdRef.current = id || null
        currentRunIdRef.current = turnId

        await read(reader)
        if (currentSessionIdRef.current === id && target.type === ChatType.Deepsearch && target.researchStatus === 'running') {
          target.researchStatus = 'failed'
          setMedicalStatus('failed')
          setMedicalError('连接已结束，但尚未收到研究完成事件。请刷新已保存的研究记录。')
        }

        // 清理 reader ref
        readerRef.current = null
      } catch (error) {
        if (currentSessionIdRef.current !== id) return
        target.researchStatus = 'failed'
        if (target.type === ChatType.Deepsearch) {
          setMedicalStatus('failed')
          setMedicalError(error instanceof Error ? error.message : '研究请求失败，请重试。')
        }
        target.content = target.content || '请求中断，请查看错误信息后重试。'
      } finally {
        target.loading = false
      }

      async function read(reader: ReadableStreamDefaultReader<any>) {
        if (target.type === ChatType.Deepsearch) {
          await consumeResearchSse(reader, async event => {
            await parseData(`data: ${JSON.stringify(event)}`)
            scrollToBottom()
          })
          target.loading = false
          return
        }
        let temp = ''
        const decoder = new TextDecoder('utf-8')
        while (true) {
          const { value, done } = await reader.read()
          temp += decoder.decode(value)

          while (true) {
            const index = temp.indexOf('\n')
            if (index === -1) break

            const slice = temp.slice(0, index)
            temp = temp.slice(index + 1)

            if (slice.startsWith('data: ')) {
              await parseData(slice)
              scrollToBottom()
            }
          }

          if (done) {
            console.debug('数据接受完毕', temp)
            target.loading = false
            break
          }
        }
      }

      async function parseData(slice: string) {
        if (currentSessionIdRef.current !== id) return
        try {
          const str = slice
            .trim()
            .replace(/^data\: /, '')
            .trim()
          if (str === '[DONE]') {
            return
          }

          const protocolEvent = parseResearchSseData(str)
          if (!protocolEvent) return
          const previousProtocol = researchProtocolRef.current
          const advance = advanceResearchProtocol(previousProtocol, protocolEvent)
          if (advance.reason === 'gap') {
            const missing = await api.session.getResearchEvents(
              protocolEvent.run_id,
              advance.gapAfterSeq || 0,
            )
            const recovered = missing.data.events || []
            if (!recovered.length) {
              throw new Error(`SSE事件缺口无法补齐: after_seq=${advance.gapAfterSeq || 0}`)
            }
            const firstRecovered = parseResearchSseData(JSON.stringify(recovered[0]))
            if (
              !firstRecovered
              || firstRecovered.seq !== (advance.gapAfterSeq || 0) + 1
            ) {
              throw new Error(`SSE事件账本存在永久缺口: after_seq=${advance.gapAfterSeq || 0}`)
            }
            for (const raw of recovered) {
              await parseData(`data: ${JSON.stringify(raw)}`)
            }
            return
          }
          if (!advance.accepted) return
          researchProtocolRef.current = advance.state
          const json = flattenResearchEvent(protocolEvent)
          if (json.type === 'phase_change') {
            json.type = 'phase'
            json.phase = json.content?.phase
            json.content = json.content?.message || json.phase
          }
          if (json.type === 'report_chunk') {
            const chunk = json.content?.chunk || ''
            updateReportDetail(previous => previous + chunk)
            target.content = (target.content || '') + chunk
          }
          if (json.type === 'literature_update') {
            const bundle = json.content || json.medical_research
            if (bundle && typeof bundle === 'object') applyMedicalBundle(bundle as MedicalBundle)
            setMedicalRunId(protocolEvent.run_id)
          }
          if (target.type === ChatType.Deepsearch) {
            // 辅助函数：从 V2 格式中提取实际内容
            const extractContent = (data: any): string => {
              if (typeof data === 'string') return data
              if (typeof data === 'object' && data !== null) {
                // V2 格式: content 是对象 { agent, content: "实际内容" }
                if (typeof data.content === 'string') return data.content
                // 如果 content 也是对象，尝试 JSON 格式化
                return JSON.stringify(data, null, 2)
              }
              return String(data || '')
            }

            // V2 研究开始事件
            if (json.type === 'research_start') {
              target.reactMode = true
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              target.reactSteps.push({
                step: 0,
                type: 'plan',
                content: `🔬 开始深度研究: ${json.query || ''}`,
                timestamp: Date.now(),
              })
              // 重置研究步骤
              console.log(`[前端] ⚠️ research_start: 清空 researchDetailsRef`)
              setResearchSteps([])
              resetResearchDetails()
              researchProtocolRef.current = reduceResearchProtocol(
                initialResearchProtocolState,
                protocolEvent,
              )
            }

            // V2 研究步骤事件 (新增)
            if (json.type === 'research_step') {
              const content = json.content || json
              const stepType = content.step_type as ResearchStep['type']

              // 转换 stats 从 snake_case 到 camelCase
              const rawStats = content.stats || {}
              const stats = {
                resultsCount: rawStats.results_count,
                chartsCount: rawStats.charts_count,
                entitiesCount: rawStats.entities_count,
                sectionsCount: rawStats.sections_count,
                wordCount: rawStats.word_count,
                questionsCount: rawStats.questions_count,
                sourcesCount: rawStats.sources_count,
                referencesCount: rawStats.references_count,
              }

              setResearchSteps(prev => {
                const existing = prev.find(s => s.type === stepType)
                let newSteps: ResearchStep[]
                if (existing) {
                  // 更新现有步骤 - 保持 id 为 stepType
                  newSteps = prev.map(s => s.type === stepType ? {
                    ...s,
                    status: content.status,
                    stats,
                  } : s)
                } else {
                  // 添加新步骤 - 使用 stepType 作为 id
                  newSteps = [...prev, {
                    id: stepType,  // 使用 stepType 作为 id，与 detail key 保持一致
                    type: stepType,
                    title: content.title || stepType,
                    subtitle: content.subtitle || '',
                    status: content.status || 'running',
                    stats,
                  }]
                }
                // 同步更新 ref，确保后续事件能立即访问
                researchStepsRef.current = newSteps
                return newSteps
              })

              // 初始化详情数据 - 使用 stepType 作为 key，确保一致性
              if (!researchDetailsRef.current.has(stepType)) {
                const newDetail: ResearchDetailData = {
                  stepId: stepType,  // 使用类型作为 ID
                  stepType,
                  title: content.title || stepType,
                  subtitle: content.subtitle,
                  searchResults: [],
                  charts: [],
                }
                researchDetailsRef.current.set(stepType, newDetail)
                console.log(`[前端] research_step: 创建 detail, key=${stepType}, detailsSize=${researchDetailsRef.current.size}`)
                // 自动选中新的步骤详情（特别是 searching/researching 步骤）
                if (stepType === 'searching' || content.status === 'running') {
                  setSelectedResearchDetail({ ...newDetail })
                }
              } else {
                console.log(`[前端] research_step: detail 已存在, key=${stepType}`)
              }
            }

            // V2 搜索结果事件 (详情面板用)
            if (json.type === 'search_results') {
              const content = json.content || json
              const results = content.results || []
              const isIncremental = content.isIncremental || false
              // 使用 stepType 作为 key 查找 detail
              const searchingType = researchStepsRef.current.find(s => s.type === 'searching') ? 'searching' : 'researching'
              const detail = researchDetailsRef.current.get(searchingType)
              console.log(`[前端] search_results: key=${searchingType}, detail=${detail ? '找到' : '未找到'}, results=${results.length}`)
              if (detail) {
                const newResults = results.map((r: any, i: number) => ({
                  id: r.id || `sr_${Date.now()}_${i}`,
                  title: r.title,
                  source: r.source,
                  date: r.date,
                  url: r.url,
                  snippet: r.snippet,
                }))
                // 增量模式：累加结果；否则替换
                if (isIncremental && detail.searchResults) {
                  detail.searchResults = [...detail.searchResults, ...newResults]
                } else {
                  detail.searchResults = newResults
                }
                // 更新步骤统计
                setResearchSteps(prev => prev.map(s =>
                  s.type === searchingType
                    ? { ...s, stats: { ...s.stats, resultsCount: detail.searchResults?.length || 0 } }
                    : s
                ))
                // 自动选中并触发聚合数据更新
                setSelectedResearchDetail({ ...detail })
                setResearchDataVersion(v => v + 1)
              }
            }

            // V2 知识图谱事件
            if (json.type === 'knowledge_graph') {
              const content = json.content || json
              const graph = content.graph || content
              // 优先存储到 analyzing，其次 researching/searching - 使用 stepType 作为 key
              const targetType = researchDetailsRef.current.has('analyzing') ? 'analyzing'
                : researchDetailsRef.current.has('researching') ? 'researching' : 'searching'
              const detail = researchDetailsRef.current.get(targetType)
              console.log(`[前端] knowledge_graph: key=${targetType}, detail=${detail ? '找到' : '未找到'}, nodes=${graph.nodes?.length || 0}, edges=${graph.edges?.length || 0}`)
              if (detail) {
                detail.knowledgeGraph = {
                  nodes: graph.nodes || [],
                  edges: graph.edges || [],
                  stats: content.stats || graph.stats,
                }
                setSelectedResearchDetail({ ...detail })
                setResearchDataVersion(v => v + 1)
                console.log(`[前端] knowledge_graph: ✅ 已存储到 detail[${targetType}]`)
              } else {
                console.warn(`[前端] knowledge_graph: ⚠️ 未找到 detail, 可用 keys:`, Array.from(researchDetailsRef.current.keys()))
              }
            }

            // V2 图表事件 (DataAnalyst 发送的 ECharts 图表)
            if (json.type === 'charts') {
              const content = json.content || json
              const charts = content.charts || []
              console.log(`[前端] 收到 charts 事件，图表数量: ${charts.length}`)

              // 使用 stepType 作为 key 查找 detail
              const detail = researchDetailsRef.current.get('analyzing')
              console.log(`[前端] 查找 analyzing detail: ${detail ? '找到' : '未找到'}`)
              if (detail) {
                detail.charts = charts
                // 更新步骤统计
                setResearchSteps(prev => prev.map(s =>
                  s.type === 'analyzing'
                    ? { ...s, stats: { ...s.stats, chartsCount: charts.length } }
                    : s
                ))
                setSelectedResearchDetail({ ...detail })
                setResearchDataVersion(v => v + 1)
                console.log(`[前端] ✅ charts 已存储到 detail，触发更新`)
              }
              // 同时保存到 target.charts 供报告使用
              if (!target.charts) {
                target.charts = []
              }
              target.charts.push(...charts)
              console.log(`[前端] target.charts 总数: ${target.charts.length}`)
            }

            // V2 阶段切换事件
            if (json.type === 'phase') {
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              const phaseLabels: Record<string, string> = {
                routing: '🔀 追问路由',
                user_revising: '📝 按要求修订',
                planning: '📋 规划阶段',
                researching: '🔍 搜索阶段',
                analyzing: '📊 分析阶段',
                writing: '✍️ 写作阶段',
                reviewing: '🔎 审核阶段',
                re_researching: '🔄 补充搜索',
                rewriting: '📝 重写阶段',
                revising: '📝 修订阶段',
              }
              target.reactSteps.push({
                step: target.reactSteps.length + 1,
                type: 'thought',
                content: `${phaseLabels[json.phase] || json.phase}: ${extractContent(json.content)}`,
                timestamp: Date.now(),
              })

              // 同时更新研究步骤条 - 映射 phase 到 step_type
              const phaseToStepType: Record<string, ResearchStep['type']> = {
                writing: 'writing',
                reviewing: 'reviewing',
                re_researching: 're_researching',
                rewriting: 'revising',
                routing: 'planning',
                user_revising: 'revising',
                revising: 'revising',
              }
              const stepType = phaseToStepType[json.phase]
              if (stepType) {
                setResearchSteps(prev => {
                  const existing = prev.find(s => s.type === stepType)
                  if (!existing) {
                    const newSteps = [...prev, {
                      id: stepType,  // 使用 stepType 作为 ID
                      type: stepType,
                      title: phaseLabels[json.phase] || json.phase,
                      subtitle: extractContent(json.content) || '',
                      status: 'running' as const,
                    }]
                    researchStepsRef.current = newSteps

                    // 同时初始化 researchDetail - 使用 stepType 作为 key
                    if (!researchDetailsRef.current.has(stepType)) {
                      const newDetail: ResearchDetailData = {
                        stepId: stepType,
                        stepType,
                        title: phaseLabels[json.phase] || json.phase,
                        subtitle: extractContent(json.content) || '',
                        searchResults: [],
                        charts: [],
                        streamingReport: '',
                      }
                      researchDetailsRef.current.set(stepType, newDetail)
                      // 对于 writing 步骤，自动选中以便显示过程报告
                      if (stepType === 'writing') {
                        setSelectedResearchDetail({ ...newDetail })
                      }
                    }

                    return newSteps
                  }
                  return prev
                })
              }
            }

            if (json.type === 'session_clarification') {
              target.type = ChatType.Normal
              target.content = String(json.question || json.content || '')
              target.researchStatus = 'clarification'
              target.runId = undefined
              target.loading = false
              setMedicalRunId('')
              setMedicalStatus('idle')
              setMedicalError('')
            }

            // V2 大纲事件
            if (json.type === 'outline') {
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              const outlineContent = json.content || json
              const outline = outlineContent.outline || []
              const questions = outlineContent.research_questions || []

              let content = '**研究大纲**\n\n'
              if (outline.length > 0) {
                content += outline.map((sec: any, i: number) =>
                  `${i + 1}. **${sec.title}**\n   ${sec.description || ''}`
                ).join('\n\n')
              }
              if (questions.length > 0) {
                content += '\n\n**核心问题**\n' + questions.map((q: string) => `• ${q}`).join('\n')
              }

              target.reactSteps.push({
                step: target.reactSteps.length + 1,
                type: 'plan',
                content,
                timestamp: Date.now(),
              })
            }

            // V2 研究完成事件
            if (json.type === 'research_complete') {
              target.researchStatus = 'completed'
              setMedicalStatus('completed')
              if (json.medical_research) applyMedicalBundle(json.medical_research)
              console.log('研究完成事件:', json)
              // 设置最终报告为内容
              if (json.final_report) {
                target.content = json.final_report
                console.log('设置报告内容，长度:', json.final_report.length)

                updateReportDetail(json.final_report)
                // 打印所有 detail 的状态
                console.log(`[前端] research_complete: 所有 detail keys:`, Array.from(researchDetailsRef.current.keys()))
                researchDetailsRef.current.forEach((d, k) => {
                  console.log(`[前端] detail[${k}]: searchResults=${d.searchResults?.length || 0}, charts=${d.charts?.length || 0}, hasGraph=${!!d.knowledgeGraph}, hasReport=${!!d.streamingReport}`)
                })
              }
              // 设置引用
              if (json.references && json.references.length > 0) {
                target.reference = json.references.map((ref: any, i: number) => ({
                  id: i + 1,
                  title: ref.title || ref.source_name || '来源',
                  link: ref.link || ref.url || ref.source_url || '',
                  content: ref.content || ref.summary || '',
                  source: ref.source_type === 'local' ? 'knowledge' : 'web',
                }))
              }

              // 标记所有研究步骤为完成
              setResearchSteps(prev => prev.map(s => ({ ...s, status: 'completed' as const })))
              // 确保触发重新计算
              console.log(`[前端] research_complete: ✅ 研究完成，强制触发 researchDataVersion 更新`)
              setResearchDataVersion(v => v + 1)
            }

            // 检测 ReAct 模式
            if (json.mode === 'react' || json.mode === 'optimized' || json.type === 'react_start') {
              target.reactMode = true
            }

            // 研究计划事件 (V1)
            if (json.type === 'plan' && json.understanding) {
              target.researchPlan = {
                understanding: json.understanding || '',
                strategy: json.strategy || '',
                subQueries: (json.sub_queries || []).map((sq: any) => ({
                  query: sq.query,
                  purpose: sq.purpose,
                  tool: sq.tool,
                })),
                expectedAspects: json.expected_aspects || [],
              }
              // 同时添加到 reactSteps 用于展示
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              target.reactSteps.push({
                step: 0,
                type: 'plan',
                content: `**研究计划**\n\n理解: ${json.understanding}\n\n策略: ${json.strategy}\n\n子查询:\n${(json.sub_queries || []).map((sq: any) => `• ${sq.query} (${sq.purpose})`).join('\n')}`,
                timestamp: Date.now(),
              })
            }

            // ReAct 事件处理 (兼容 V1 和 V2)
            if (json.type === 'thought') {
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              target.reactSteps.push({
                step: json.step || target.reactSteps.length + 1,
                type: 'thought',
                content: extractContent(json.content),
                timestamp: Date.now(),
              })
            } else if (json.type === 'action') {
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              // V2 格式的 action
              const actionContent = json.content || json
              const tool = actionContent.tool || json.tool
              const isParallel = tool === 'parallel_search'
              const queries = actionContent.queries || json.params?.queries || []
              const section = actionContent.section || ''

              let displayContent = ''
              if (isParallel) {
                displayContent = `并行搜索${section ? ` (${section})` : ''} ${queries.length} 个查询:\n${queries.map((q: string) => `• ${q}`).join('\n')}`
              } else {
                displayContent = `调用工具: ${tool}${section ? ` - ${section}` : ''}`
              }

              target.reactSteps.push({
                step: json.step || target.reactSteps.length + 1,
                type: 'action',
                content: displayContent,
                tool: tool,
                params: json.params || actionContent,
                queries: isParallel ? queries : undefined,
                timestamp: Date.now(),
              })
            } else if (json.type === 'observation') {
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              // V2 格式的 observation
              const obsContent = json.content || json
              let displayContent = ''

              if (typeof obsContent === 'object') {
                const parts = []
                if (obsContent.section) parts.push(`📑 ${obsContent.section}`)
                if (obsContent.facts_count) parts.push(`事实: ${obsContent.facts_count} 条`)
                if (obsContent.data_points_count) parts.push(`数据点: ${obsContent.data_points_count} 个`)
                if (obsContent.duplicates_removed) parts.push(`去重: ${obsContent.duplicates_removed} 条`)
                if (obsContent.insights && obsContent.insights.length > 0) {
                  parts.push(`洞察:\n${obsContent.insights.map((i: string) => `  • ${i}`).join('\n')}`)
                }
                if (obsContent.source_quality) parts.push(`来源质量: ${obsContent.source_quality}`)
                displayContent = parts.join('\n') || JSON.stringify(obsContent, null, 2)
              } else {
                displayContent = typeof json.result === 'string' ? json.result : JSON.stringify(json.result || obsContent)
              }

              const stepId = `obs_${Date.now()}_${target.reactSteps.length}`
              target.reactSteps.push({
                step: json.step || target.reactSteps.length + 1,
                type: 'observation',
                content: displayContent,
                tool: json.tool,
                queries: json.queries_executed,
                success: json.success !== false,
                timestamp: Date.now(),
                stepId, // 添加 stepId 用于关联详情
              })

              // 存储步骤详情用于右侧面板展示
              if (typeof obsContent === 'object') {
                const stepDetail: StepDetailData = {
                  stepId,
                  type: obsContent.agent || 'observation',
                  section: obsContent.section,
                  searchResults: obsContent.search_results,
                  extractedFacts: obsContent.extracted_facts,
                  dataPoints: obsContent.data_points,
                  insights: obsContent.insights,
                }
                stepDetailsRef.current.set(stepId, stepDetail)
                // 自动选中最新的步骤详情
                setSelectedStepDetail(stepDetail)
              }
            } else if (json.type === 'section_draft') {
              // V2 章节撰写完成事件
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              const content = json.content || json
              target.reactSteps.push({
                step: target.reactSteps.length + 1,
                type: 'observation',
                content: `✍️ 章节「${content.section_title || '未知'}」撰写完成\n字数: ${content.word_count || 0}\n要点: ${(content.key_points || []).join('、')}`,
                timestamp: Date.now(),
              })
            } else if (json.type === 'section_content') {
              // V2 章节内容事件 - 用于"过程报告"tab的流式显示
              const content = json.content || json
              const sectionContent = content.content || ''
              const sectionTitle = content.section_title || ''

              if (sectionContent) {
                console.log(`section_content 收到章节「${sectionTitle}」，长度:`, sectionContent.length)

                // 使用 stepType 作为 key 查找或创建 detail
                const writingType = 'writing'

                // 如果没有找到写作步骤，创建一个（兜底逻辑）
                if (!researchStepsRef.current.find(s => s.type === writingType)) {
                  console.log('section_content: 未找到写作步骤，创建兜底步骤')
                  const newStep: ResearchStep = {
                    id: writingType,  // 使用 type 作为 id
                    type: writingType,
                    title: '✍️ 写作阶段',
                    subtitle: '撰写研究报告',
                    status: 'running',
                  }
                  setResearchSteps(prev => {
                    const updated = [...prev, newStep]
                    researchStepsRef.current = updated
                    return updated
                  })
                }

                // 获取或创建详情 - 使用 stepType 作为 key
                let detail = researchDetailsRef.current.get(writingType)
                if (!detail) {
                  console.log(`section_content: 未找到详情，创建: ${writingType}`)
                  detail = {
                    stepId: writingType,
                    stepType: writingType,
                    title: '写作阶段',
                    streamingReport: '',
                    searchResults: [],
                    charts: [],
                    sections: [],  // 初始化 sections 数组
                  }
                  researchDetailsRef.current.set(writingType, detail)
                }

                // 添加章节到 sections 数组
                const sectionId = content.section_id || `section_${Date.now()}`
                if (!detail.sections) {
                  detail.sections = []
                }
                // 检查是否已存在，避免重复
                const existingIndex = detail.sections.findIndex(s => s.id === sectionId)
                if (existingIndex >= 0) {
                  detail.sections[existingIndex] = {
                    id: sectionId,
                    title: sectionTitle,
                    content: sectionContent,
                    wordCount: sectionContent.length,
                  }
                } else {
                  detail.sections.push({
                    id: sectionId,
                    title: sectionTitle,
                    content: sectionContent,
                    wordCount: sectionContent.length,
                  })
                }
                console.log(`section_content: 已添加章节「${sectionTitle}」到 sections，当前数量: ${detail.sections.length}`)

                // 从按章节 ID 更新后的集合生成正文，同章重传或修订不会重复追加。
                detail.streamingReport = detail.sections.map(section =>
                  `${section.title ? `## ${section.title}\n\n` : ''}${section.content}`,
                ).join('\n\n')
                setSelectedResearchDetail({ ...detail })
                setResearchDataVersion(v => v + 1)

                // 同时添加到 reactSteps
                if (!target.reactSteps) {
                  target.reactSteps = []
                }
                target.reactSteps.push({
                  step: target.reactSteps.length + 1,
                  type: 'observation',
                  content: `✍️ 章节「${sectionTitle}」已写入过程报告\n字数: ${sectionContent.length}\n要点: ${(content.key_points || []).slice(0, 2).join('、') || '无'}`,
                  timestamp: Date.now(),
                })
              }
            } else if (json.type === 'report_draft') {
              // V2 报告草稿完成事件
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              const eventContent = json.content || json
              const reportContent = typeof eventContent === 'string' ? eventContent : eventContent.content || ''

              target.reactSteps.push({
                step: target.reactSteps.length + 1,
                type: 'observation',
                content: `📝 研究报告撰写完成\n字数: ${eventContent.word_count || reportContent.length || 0}\n引用数: ${eventContent.references_count || 0}`,
                timestamp: Date.now(),
              })

              if (reportContent) {
                console.log('report_draft 收到报告内容，长度:', reportContent.length)
                updateReportDetail(reportContent)
                // 同时设置为聊天消息内容
                target.content = reportContent
              }

              // 标记写作步骤完成
              setResearchSteps(prev => prev.map(s =>
                s.type === 'writing' || s.type === 'generating'
                  ? { ...s, status: 'completed' as const }
                  : s
              ))
            } else if (json.type === 'review') {
              // V2 审核反馈事件
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              const content = json.content || json
              const review = formatResearchReview(content)
              target.reactSteps.push({
                step: target.reactSteps.length + 1,
                type: 'thought',
                content: review.text,
                timestamp: Date.now(),
              })

              // 更新审核步骤状态
              setResearchSteps(prev => prev.map(s =>
                s.type === 'reviewing'
                  ? { ...s, status: review.passed ? 'completed' as const : 'running' as const }
                  : s
              ))
            } else if (json.type === 'revision_complete') {
              // V2 修订完成事件
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              const content = json.content || json
              target.reactSteps.push({
                step: target.reactSteps.length + 1,
                type: 'observation',
                content: `📝 修订完成，共 ${content.changes_count || 0} 处修改`,
                timestamp: Date.now(),
              })
            } else if (json.type === 'error') {
              target.researchStatus = 'failed'
              setMedicalStatus('failed')
              setMedicalError(extractContent(json.content) || '研究执行失败')
              // V2 错误事件
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              target.reactSteps.push({
                step: target.reactSteps.length + 1,
                type: 'thought',
                content: `❌ 错误: ${extractContent(json.content)}`,
                timestamp: Date.now(),
              })
            } else if (json.type === 'research_cancelled') {
              target.researchStatus = 'cancelled'
              setMedicalStatus('cancelled')
              // 研究被取消事件
              console.log('[前端] 收到 research_cancelled 事件')
              if (!target.reactSteps) {
                target.reactSteps = []
              }
              target.reactSteps.push({
                step: target.reactSteps.length + 1,
                type: 'thought',
                content: `⏹️ 研究已被用户取消`,
                timestamp: Date.now(),
              })
              target.loading = false
              if (!target.content) {
                target.content = '⏹️ 研究已被用户取消'
              }
              // 标记所有研究步骤为完成
              setResearchSteps(prev => prev.map(s => ({ ...s, status: 'completed' as const })))
            } else if (json.type === 'chart') {
              // 解包 content（后端将数据包在 content 里）
              const content = json.content || json
              console.log(`[前端] 收到 chart 事件 (单个图表)`)
              console.log(`[前端] chart 内容: title=${content.title}, has_echarts=${!!content.echarts_option}, has_image=${!!(content.image || content.image_base64)}`)

              // 构建图表对象
              const chartObj = {
                id: uniqueId('chart_'),
                type: content.chart_type || 'generated',
                title: content.title || '数据图表',
                echarts_option: content.echarts_option,
                image_base64: content.image || content.image_base64,
                data: content.data,
              }

              // 存入 target.charts（供报告使用）
              if (!target.charts) {
                target.charts = []
              }
              target.charts.push(chartObj)
              console.log(`[前端] 图表已添加到 target.charts，总数: ${target.charts.length}`)

              // 同时存入 research detail（供可视化面板使用）- 使用 stepType 作为 key
              const detail = researchDetailsRef.current.get('analyzing')
              console.log(`[前端] 查找 analyzing detail: ${detail ? '找到' : '未找到'}`)
              if (detail) {
                if (!detail.charts) {
                  detail.charts = []
                }
                detail.charts.push(chartObj)
                setResearchSteps(prev => prev.map(s =>
                  s.type === 'analyzing'
                    ? { ...s, stats: { ...s.stats, chartsCount: detail.charts?.length || 0 } }
                    : s
                ))
                setSelectedResearchDetail({ ...detail })
                setResearchDataVersion(v => v + 1)
                console.log(`[前端] ✅ chart 已存储到 detail.charts，总数: ${detail.charts.length}`)
              } else {
                console.warn(`[前端] ⚠️ 未找到 analyzing detail，图表可能无法显示在可视化面板`)
              }
            } else if (json.type === 'stock_quote') {
              // 股票实时行情
              const content = json.content || json
              target.stockQuote = {
                code: content.code,
                name: content.name,
                price: content.price,
                change: content.change,
                change_percent: content.change_percent,
                high: content.high,
                low: content.low,
                volume: content.volume,
                turnover: content.turnover,
                open: content.open,
                prev_close: content.prev_close,
              }
            } else if (json.type === 'data_insight') {
              if (!target.insights) {
                target.insights = []
              }
              target.insights.push(...(json.insights || []))
            } else if (['status', 'search_results', 'thinking_step'].includes(json.type)) {
              // 兼容原有状态事件
              if (!target.thinks) {
                target.thinks = []
              }

              const lastThink = target.thinks[target.thinks.length - 1]

              if (lastThink?.type === json.type) {
                lastThink.results!.push({
                  id: uniqueId('think_result'),
                  content: json.subquery || json.content,
                  count: json.count,
                })
              } else {
                target.thinks.push({
                  id: uniqueId('think_result'),
                  type: json.type as 'status' | 'search_results',
                  results: [
                    {
                      id: uniqueId('think_result'),
                      content: json.subquery || json.content,
                      count: json.count,
                    },
                  ],
                })
              }
            } else if (json.type === 'search_result_item') {
              if (!target.search_results) {
                target.search_results = []
              }

              try {
                target.search_results.push({
                  ...json.result,
                  id: uniqueId('search-results'),
                  host: json.result?.url ? new URL(json.result.url).host : '',
                })
              } catch (e) {
                console.debug('Parse URL error', e)
              }
            } else if (json.type === 'thinking') {
              target.think = `${target.think || ''}${json.content || ''}`
            } else if (['answer', 'final_answer'].includes(json.type)) {
              target.content = `${target.content}${json.content || ''}`
            } else if (json.type === 'reference_materials') {
              target.reference = json.content?.map((o: any) => ({
                id: o.reference_id,
                title: o.name,
                link: o.url,
                content: o.summary,
                source: o.source === 'local' ? 'knowledge' : 'web',
              }))
            }
          } else {
            if (json?.content) {
              if (json.thinking) {
                target.think = `${target.think || ''}${json.content || ''}`
              } else {
                target.content = `${target.content || ''}${json.content || ''}`
              }
            }

            if (json?.documents?.length) {
              target.reference = json.documents
            }

            if (json?.image_results) {
              target.image_results = json.image_results
            }
          }
        } catch {
          console.debug('解析失败')
          console.debug(slice)
        }
      }
    },
    [id, applyMedicalBundle, resetResearchDetails, updateReportDetail],
  )

  const send = useCallback(
    async (message: string, attachmentIds?: string[]) => {
      if (loadingRef.current) return
      // 统一研究接口以 query + session_id 读取多轮上下文；重新打开图研究时
      // 没有旧医疗 protocol.disease，不能用该可选字段拦截追问。
      if (!message && (!attachmentIds || attachmentIds.length === 0)) return

      const turnId = crypto.randomUUID()
      chat.list.push({
        id: createChatId(),
        role: ChatRole.User,
        type: ChatType.Normal,
        content: message || '(附件问答)',
        runId: deviceState.executionMode === 'research' ? turnId : undefined,
      })

      chat.list.push({
        id: createChatId(),
        role: ChatRole.Assistant,
        type: deviceState.executionMode === 'research' ? ChatType.Deepsearch : ChatType.Normal,
        content: '',
      })
      scrollToBottom()

      const target = chat.list[chat.list.length - 1]

      await sendChat(target, message || '请分析附件内容', turnId, attachmentIds)

      // 发送后清空附件列表
      if (attachmentIds && attachmentIds.length > 0) {
        setAttachments([])
      }
    },
    [chat, sendChat],
  )
  const hasSentInitialMessage = useRef(false)
  const hasLoadedCheckpoint = useRef(false)
  const hasLoadedMessages = useRef(false)
  const previousIdRef = useRef<string | undefined>(undefined)

  // 当 session ID 变化时，重置加载状态
  useEffect(() => {
    if (id !== previousIdRef.current) {
      console.log('[会话切换] 从', previousIdRef.current, '切换到', id)
      previousIdRef.current = id
      if (readerRef.current) void readerRef.current.cancel().catch(() => {})
      readerRef.current = null
      currentSessionIdRef.current = id || null
      currentRunIdRef.current = null
      researchStopRequestedRef.current = false
      requestedResumeRunRef.current = null
      hasLoadedMessages.current = false
      hasLoadedCheckpoint.current = false
      hasSentInitialMessage.current = false
      // 清空消息列表和研究状态
      console.log(`[前端] ⚠️ 会话切换: 清空 researchDetailsRef`)
      chat.list.length = 0
      setResearchSteps([])
      researchStepsRef.current = []
      resetResearchDetails()
      setCurrentChatItem(null)
      setMedicalBundle(null)
      setMedicalRunId(null)
      activeMedicalRunRef.current = null
      setMedicalError('')
      setMedicalStatus('idle')
      scopeRef.current = defaultMedicalScope()
      setResolvedQueries({})
      previousMedicalRunRef.current = undefined
    }
  }, [id, chat, resetResearchDetails])

  // 加载会话历史消息
  useEffect(() => {
    if (!id || hasLoadedMessages.current) return
    if (ctx?.data?.sessionId === id) { hasLoadedMessages.current = true; return }

    // 辅助函数：将消息数组填充到 chat.list
    function populateMessages(messages: any[]) {
      if (currentSessionIdRef.current !== id) return
      chat.list.length = 0
      for (const msg of messages) {
        const chatItem: API.ChatItem = {
          id: createChatId(),
          role: msg.role === 'user' ? ChatRole.User : ChatRole.Assistant,
          type: msg.execution_mode === 'research' ? ChatType.Deepsearch : ChatType.Normal,
          content: msg.content || '',
          runId: msg.run_id || undefined,
          researchStatus: msg.research_status || undefined,
        }

        // 恢复助手消息的额外数据
        if (msg.role === 'assistant') {
          if (msg.thinking) {
            chatItem.think = msg.thinking
          }
          if (msg.references_data?.references) {
            chatItem.reference = msg.references_data.references as any
          }
        }

        chat.list.push(chatItem)
      }
      const latest = [...messages].reverse().find(msg => msg.run_id && msg.execution_mode === 'research')
      if (latest) void loadMedicalRun(latest.run_id, latest.research_status || 'completed', true)
      // 首次 effect 可能早于异步历史消息返回，此时没有 runId；消息就绪后补做恢复。
      if (latest && !hasLoadedCheckpoint.current) setRecoveryNonce(value => value + 1)
    }

    // 优先使用 store 中预加载的数据
    const cachedSession = sessionState.currentSession
    if (cachedSession && cachedSession.id === id && cachedSession.messages?.length > 0) {
      console.log('[加载消息] 使用预加载的数据:', cachedSession.messages.length, '条')
      hasLoadedMessages.current = true
      populateMessages(cachedSession.messages)
      return
    }

    // 否则从 API 加载
    async function loadSessionMessages() {
      try {
        console.log('[加载消息] 开始加载会话消息:', id)
        const res = await api.session.getSession(id!)
        const session = (res as any).data || res

        if (session && session.messages && session.messages.length > 0) {
          hasLoadedMessages.current = true
          console.log('[加载消息] 找到消息:', session.messages.length, '条')
          populateMessages(session.messages)
          console.log('[加载消息] 消息恢复完成')
        }
      } catch (e) {
        console.log('[加载消息] 加载失败或无消息:', e)
      }
    }

    loadSessionMessages()
  }, [id, chat])

  // 加载并恢复研究检查点状态
  useEffect(() => {
    if (!id || hasLoadedCheckpoint.current) return
    if (ctx?.data?.sessionId === id && !requestedResumeRunRef.current) { hasLoadedCheckpoint.current = true; return }

    let disposed = false
    let recoveryReader: ReadableStreamDefaultReader<Uint8Array> | null = null

    const normalizeStepType = (value: unknown): ResearchStep['type'] => {
      const raw = String(value || 'planning')
      if (raw === 'researching') return 'searching'
      if (raw === 'rewriting') return 'revising'
      if (raw === 'routing') return 'planning'
      if (raw === 'user_revising') return 'revising'
      if (raw === 'outline_review') return 'planning'
      const supported: ResearchStep['type'][] = [
        'planning', 'searching', 'analyzing', 'generating', 'writing',
        'reviewing', 're_researching', 'revising',
      ]
      return supported.includes(raw as ResearchStep['type'])
        ? raw as ResearchStep['type']
        : 'planning'
    }

    const applyProjection = (
      checkpoint: Partial<ResearchCheckpoint> & { run_id: string | null },
      projection: ResearchRecoveryProjection,
    ) => {
      if (disposed) return
      if (checkpoint.run_id && !projection.medical_research) {
        setMedicalRunId(checkpoint.run_id)
        activeMedicalRunRef.current = checkpoint.run_id
        setMedicalStatus(projection.status)
      }
      if (projection.medical_research) {
        applyMedicalBundle(projection.medical_research as MedicalBundle)
        setMedicalRunId(checkpoint.run_id || null)
        activeMedicalRunRef.current = checkpoint.run_id || null
        setMedicalStatus(projection.status)
      }

      let steps: ResearchStep[] = projection.research_steps.map(step => {
        const type = normalizeStepType(step.type || step.step_type)
        return {
          id: type,
          type,
          title: String(step.title || type),
          subtitle: String(step.subtitle || ''),
          status: projection.status === 'completed'
            ? 'completed'
            : (step.status === 'pending' || step.status === 'completed' ? step.status : 'running'),
          stats: step.stats,
        }
      }).filter((step, index, all) => all.findIndex(item => item.type === step.type) === index)

      if (steps.length === 0) {
        steps = (['planning', 'searching', 'analyzing', 'writing'] as ResearchStep['type'][])
          .map(type => ({
            id: type,
            type,
            title: type,
            subtitle: '',
            status: projection.status === 'completed' ? 'completed' : 'pending',
          }))
      }

      setResearchSteps(steps)
      researchStepsRef.current = steps

      const details = new Map<string, ResearchDetailData>()
      for (const step of steps) {
        details.set(step.type, {
          stepId: step.type,
          stepType: step.type,
          title: step.title,
          subtitle: step.subtitle,
          searchResults: [],
          charts: [],
        })
      }

      const searchDetail = details.get('searching') || details.get('re_researching')
      if (searchDetail) {
        searchDetail.searchResults = projection.search_results.map((result, index) => ({
          id: String(result.id || `replayed_${index}`),
          title: String(result.title || result.source_name || ''),
          source: String(result.source || result.source_type || 'web'),
          date: String(result.date || ''),
          url: String(result.url || result.source_url || ''),
          snippet: String(result.snippet || result.content || ''),
        }))
      }

      const analysisDetail = details.get('analyzing')
      if (analysisDetail) {
        if (projection.knowledge_graph) {
          analysisDetail.knowledgeGraph = projection.knowledge_graph as ResearchDetailData['knowledgeGraph']
        }
        analysisDetail.charts = projection.charts as ResearchDetailData['charts']
      }

      researchDetailsRef.current = details
      updateReportDetail(projection.streaming_report)
      const selected = steps.find(step => step.status === 'running')
        || (projection.streaming_report ? steps.find(step => step.type === 'writing') : undefined)
        || steps[steps.length - 1]
      if (selected) setSelectedResearchDetail({ ...details.get(selected.type)! })
      setResearchDataVersion(version => version + 1)

      if (checkpoint.run_id) {
        const assistant = applyRecoveryToMessages(
          chat.list, checkpoint.run_id, checkpoint.query || '', projection, createChatId,
        )
        setCurrentChatItem(assistant)
      }
    }

    async function loadCheckpoint() {
      try {
        // 恢复按 run 定位：图状态是新运行的权威业务底稿。
        const requestedRun = requestedResumeRunRef.current
        const targetRun = requestedRun
          || chat.list.filter(item => item.runId).slice(-1)[0]?.runId
          || null
        if (!targetRun) return
        console.log('[恢复状态] 开始读取图状态, run_id:', targetRun)
        const res = await api.session.getResearchRunState(targetRun)
        if (disposed) return
        const runState = res.data?.state
        if (!runState?.exists) {
          // 旧版本运行没有本图可用的检查点：只按历史记录只读展示，
          // 不重连 SSE，也不假装断点恢复。
          if (runState?.pipeline === 'graph') {
            hasLoadedCheckpoint.current = true
            requestedResumeRunRef.current = null
            await loadMedicalRun(targetRun)
            return
          }
          const legacy = await api.session.getLegacyResearchCheckpoint(id!, targetRun)
          if (disposed) return
          if (legacy.data?.checkpoint) {
            hasLoadedCheckpoint.current = true
            applyProjection(legacy.data.checkpoint, createRecoveryProjection(legacy.data.checkpoint))
          }
          requestedResumeRunRef.current = null
          return
        }

        {
          const checkpoint: Partial<ResearchCheckpoint> & { run_id: string } = {
            run_id: targetRun,
            query: runState.query || '',
            status: runState.pending_nodes.length > 0
              ? 'running'
              : runState.status === 'completed' ? 'completed' : 'paused',
            phase: runState.phase,
            final_report: runState.final_report,
          }
          console.log('[恢复状态] 图状态:', {
            phase: runState.phase,
            status: runState.status,
            pending: runState.pending_nodes,
            graphVersion: runState.graph_version,
          })
          {
            hasLoadedCheckpoint.current = true
            let projection = projectionFromRunState(runState)
            let runId: string | null = targetRun
            let protocol: ResearchProtocolState = {
              ...initialResearchProtocolState,
              runId,
              lastSeq: 0,
              status: projection.status,
            }

            // 1. Checkpoint 快照之后按页拉取事件；2. 严格按 seq 回放。
            if (runId) {
              let cursor = protocol.lastSeq
              let hasMore = true
              while (hasMore && !disposed) {
                const eventsRes = await api.session.getResearchEvents(runId, cursor)
                const page = eventsRes.data
                for (const raw of page.events || []) {
                  const event = parseResearchSseData(JSON.stringify(raw))
                  if (!event) continue
                  const advance = advanceResearchProtocol(protocol, event)
                  if (advance.reason === 'gap') {
                    throw new Error(`事件账本存在seq缺口: after_seq=${advance.gapAfterSeq || 0}`)
                  }
                  if (!advance.accepted) continue
                  protocol = advance.state
                  projection = reduceRecoveryProjection(projection, event)
                }
                cursor = Math.max(cursor, Number(page.last_seq || protocol.lastSeq))
                hasMore = Boolean(page.has_more)
              }
            }

            if (disposed) return
            if (!requestedRun && runState.run_status) {
              projection.status = projectionFromRunState(runState).status
              protocol.status = projection.status
            }
            if (requestedRun && projection.status !== 'completed') {
              projection.status = 'running'
              protocol.status = 'running'
              researchStopRequestedRef.current = false
              requestedResumeRunRef.current = null
            }
            researchProtocolRef.current = protocol
            currentSessionIdRef.current = id || null
            currentRunIdRef.current = runId
            applyProjection(checkpoint, projection)

            // 3. 用回放后的最新 seq 重连 SSE。SSE 再次中断时仍从最新游标重试。
            let retry = 0
            while (!disposed && Boolean(requestedRun) && !researchStopRequestedRef.current && projection.status === 'running') {
              let activeLease = false
              try {
                const streamRes = await api.session.resumeResearch(runId!, protocol.lastSeq)
                recoveryReader = streamRes.data.getReader()
                readerRef.current = recoveryReader
                await consumeResearchSse(recoveryReader, async event => {
                  if (disposed || currentSessionIdRef.current !== id) return
                  if (!runId) {
                    runId = event.run_id
                    currentRunIdRef.current = runId
                  }
                  const advance = advanceResearchProtocol(protocol, event)
                  if (advance.reason === 'gap') {
                    const eventsRes = await api.session.getResearchEvents(
                      event.run_id,
                      advance.gapAfterSeq || 0,
                    )
                    for (const raw of eventsRes.data.events || []) {
                      const recovered = parseResearchSseData(JSON.stringify(raw))
                      if (!recovered) continue
                      const recoveredAdvance = advanceResearchProtocol(protocol, recovered)
                      if (recoveredAdvance.reason === 'gap') {
                        throw new Error(
                          `恢复事件账本存在永久缺口: after_seq=${recoveredAdvance.gapAfterSeq || 0}`,
                        )
                      }
                      if (!recoveredAdvance.accepted) continue
                      protocol = recoveredAdvance.state
                      projection = reduceRecoveryProjection(projection, recovered)
                    }
                    researchProtocolRef.current = protocol
                    applyProjection(checkpoint, projection)
                    return
                  }
                  if (!advance.accepted) return
                  retry = 0
                  protocol = advance.state
                  researchProtocolRef.current = protocol
                  projection = reduceRecoveryProjection(projection, event)
                  applyProjection(checkpoint, projection)
                })
              } catch (error) {
                if (!disposed && !researchStopRequestedRef.current) {
                  const httpStatus = isAxiosError(error) ? error.response?.status : undefined
                  activeLease = httpStatus === 409
                  // 其他标签中的运行可能持有租约；持续读取它的新事件，不盲目重启。
                  try {
                    if (runId) {
                      const eventsRes = await api.session.getResearchEvents(runId, protocol.lastSeq)
                      for (const raw of eventsRes.data.events || []) {
                        const recovered = parseResearchSseData(JSON.stringify(raw))
                        if (!recovered) continue
                        const advance = advanceResearchProtocol(protocol, recovered)
                        if (advance.reason === 'gap') {
                          projection.status = 'failed'
                          setMedicalError('研究事件记录不连续，已停止自动恢复，请刷新后重试。')
                          break
                        }
                        if (!advance.accepted) continue
                        retry = 0
                        protocol = advance.state
                        projection = reduceRecoveryProjection(projection, recovered)
                      }
                    }
                    if (httpStatus === 400 && projection.status === 'running') {
                      const latest = await api.session.getResearchRunState(runId!)
                      const latestState = latest.data?.state
                      if (latestState?.exists && latestState.pending_nodes.length === 0) {
                        projection = projectionFromRunState(latestState)
                      }
                    }
                    researchProtocolRef.current = protocol
                    applyProjection(checkpoint, projection)
                  } catch (replayError) {
                    console.warn('[恢复状态] 读取追加事件失败:', replayError)
                  }
                  if (projection.status === 'running' && httpStatus && [400, 401, 403, 404, 422].includes(httpStatus)) {
                    projection.status = 'failed'
                    setMedicalError('无法继续此研究，请刷新记录或检查登录状态。')
                  }
                }
              } finally {
                if (readerRef.current === recoveryReader) readerRef.current = null
                recoveryReader = null
              }

              if (!disposed && !researchStopRequestedRef.current && projection.status === 'running') {
                if (!activeLease) retry += 1
                if (retry >= 6) {
                  projection.status = 'failed'
                  setMedicalError('研究连接多次中断，已暂停自动重连。可刷新结果或点击继续。')
                  break
                }
                await new Promise(resolve => setTimeout(resolve, Math.min(500 * 2 ** retry, 5000)))
              }
            }

            if (researchStopRequestedRef.current && projection.status === 'running') projection.status = 'cancelled'
            applyProjection(checkpoint, projection)
            console.log('[恢复状态] ✅ 图状态、事件回放和 SSE 重连完成:', {
              runId,
              lastSeq: protocol.lastSeq,
              status: projection.status,
            })
          }
        }
      } catch (e) {
        console.log('[恢复状态] 加载失败:', e)
        if (!disposed) {
          setMedicalError('读取或恢复研究失败，请刷新后重试。')
          setMedicalStatus('failed')
        }
      }
    }

    loadCheckpoint()

    return () => {
      disposed = true
      if (recoveryReader) void recoveryReader.cancel()
    }
    // chat 是组件生命周期内稳定的 proxy；chat.list 由 valtio 原地更新。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, chat, recoveryNonce])

  useEffect(() => {
    if (ctx?.data?.message && ctx.data.sessionId === id && !hasSentInitialMessage.current) {
      hasSentInitialMessage.current = true
      if (ctx.data.executionMode) deviceActions.setExecutionMode(ctx.data.executionMode)
      if (ctx.data.researchScope) {
        scopeRef.current = ctx.data.researchScope
        previousMedicalRunRef.current = ctx.data.previousRunId
        deviceActions.setExecutionMode('research')
      }
      void send(ctx.data.message, ctx.data.attachmentIds)
    }
  }, [ctx, send, id])

  useEffect(() => {
    const handleScroll = () => {
      const anchors: {
        id: string
        top: number
        item: API.ChatItem
      }[] = []

      chat.list
        .filter((o) => o.type === ChatType.Deepsearch)
        .forEach((item, index) => {
          const id = createChatIdText(item.id)
          const dom = document.getElementById(id)
          if (!dom) return

          const top = dom.offsetTop
          if (index === 0 || top < window.scrollY) {
            anchors.push({ id, top, item })
          }
        })

      if (anchors.length) {
        const current = anchors.reduce((prev, curr) =>
          curr.top > prev.top ? curr : prev,
        )

        setCurrentChatItem(current.item)
      }
    }

    window.addEventListener('scroll', handleScroll)

    return () => {
      window.removeEventListener('scroll', handleScroll)
    }
  }, [])

  // 处理步骤点击，切换显示详情 (旧版)
  const handleStepClick = useCallback((stepId: string) => {
    const detail = stepDetailsRef.current.get(stepId)
    if (detail) {
      setSelectedStepDetail(detail)
    }
  }, [])

  // 处理研究步骤点击 (新版)
  const handleResearchStepClick = useCallback((stepId: string) => {
    const detail = researchDetailsRef.current.get(stepId)
    if (detail) {
      setSelectedResearchDetail(detail)
    }
  }, [])

  // 判断是否在深度研究模式（只要是 Deepsearch 类型就启用宽布局）
  const isDeepResearchMode = currentChatItem?.type === ChatType.Deepsearch

  // 调试日志：跟踪 currentChatItem 变化
  useEffect(() => {
    console.log('[前端] currentChatItem 变化:', {
      hasItem: !!currentChatItem,
      type: currentChatItem?.type,
      isDeepsearch: currentChatItem?.type === ChatType.Deepsearch,
      ChatTypeDeepsearch: ChatType.Deepsearch,
    })
  }, [currentChatItem])

  // 聚合所有研究步骤的数据，用于在tab中显示完整信息
  const aggregatedResearchData = useMemo(() => {
    console.log(`[前端] ========== 计算 aggregatedResearchData ==========`)
    console.log(`[前端] isDeepResearchMode=${isDeepResearchMode}, detailsSize=${researchDetailsRef.current.size}, version=${researchDataVersion}`)
    console.log(`[前端] currentChatItem?.type=${currentChatItem?.type}, ChatType.Deepsearch=${ChatType.Deepsearch}`)
    console.log(`[前端] researchSteps=`, researchSteps.map(s => s.type))
    console.log(`[前端] detail keys=`, Array.from(researchDetailsRef.current.keys()))

    if (!isDeepResearchMode || researchDetailsRef.current.size === 0) {
      console.log(`[前端] ⚠️ 跳过聚合: isDeepResearchMode=${isDeepResearchMode}, size=${researchDetailsRef.current.size}`)
      return null
    }

    // 从所有步骤中收集数据
    let allSearchResults: ResearchDetailData['searchResults'] = []
    let knowledgeGraph: ResearchDetailData['knowledgeGraph'] = undefined
    let allCharts: ResearchDetailData['charts'] = []
    let streamingReport = ''
    let allSections: ResearchDetailData['sections'] = []

    researchDetailsRef.current.forEach((detail, stepId) => {
      console.log(`[前端] 聚合步骤 ${stepId}: searchResults=${detail.searchResults?.length || 0}, charts=${detail.charts?.length || 0}, hasGraph=${!!detail.knowledgeGraph}, hasReport=${!!detail.streamingReport}, sections=${detail.sections?.length || 0}`)

      // 收集搜索结果
      if (detail.searchResults && detail.searchResults.length > 0) {
        allSearchResults = [...allSearchResults!, ...detail.searchResults]
      }
      // 取最新的知识图谱
      if (detail.knowledgeGraph) {
        knowledgeGraph = detail.knowledgeGraph
      }
      // 收集图表
      if (detail.charts && detail.charts.length > 0) {
        allCharts = [...allCharts!, ...detail.charts]
      }
      // 取最新的流式报告
      if (detail.streamingReport) {
        streamingReport = detail.streamingReport
      }
      // 收集章节草稿
      if (detail.sections && detail.sections.length > 0) {
        allSections = [...allSections!, ...detail.sections]
      }
    })

    console.log(`[前端] 聚合结果: searchResults=${allSearchResults.length}, charts=${allCharts.length}, hasGraph=${!!knowledgeGraph}, hasReport=${!!streamingReport}, sections=${allSections.length}`)

    // 创建聚合的数据对象
    const aggregated: ResearchDetailData = {
      stepId: selectedResearchDetail?.stepId || 'aggregated',
      stepType: selectedResearchDetail?.stepType || 'aggregated',
      title: selectedResearchDetail?.title || '研究详情',
      subtitle: selectedResearchDetail?.subtitle,
      searchResults: allSearchResults,
      knowledgeGraph,
      charts: allCharts,
      streamingReport,
      sections: allSections,
    }

    return aggregated
  }, [isDeepResearchMode, selectedResearchDetail, researchSteps, researchDataVersion])  // researchSteps/version 变化时重新计算

  // 确定右侧面板显示内容
  const rightPanelContent = useMemo(() => {
    if (medicalRunId || medicalBundle) {
      const reportDetail = researchDetailsRef.current.get('writing')
      return <div style={{ overflowY: 'auto', height: '100%' }}>
        {Array.from(new Set(list.filter(item => item.runId).map(item => item.runId!))).length > 1 && <Select aria-label="历史研究版本" disabled={loading} value={medicalRunId} style={{ margin: '16px 24px 0', width: 280 }} onChange={runId => { const item = list.find(row => row.runId === runId); void loadMedicalRun(runId, item?.researchStatus || 'completed') }} options={Array.from(new Set(list.filter(item => item.runId).map(item => item.runId!))).map((runId, index) => ({ value: runId, label: `研究 ${index + 1} · ${runId.slice(0, 8)}` }))} />}
        {medicalBundle?.pipeline !== 'legacy' ? <GraphWorkbench bundle={medicalBundle} runId={medicalRunId} status={medicalStatus} detail={reportDetail} error={medicalError} title={scopeRef.current.disease} onRefresh={async () => { if (medicalRunId) await loadMedicalRun(medicalRunId, medicalStatus) }} onResume={() => { requestedResumeRunRef.current = medicalRunId; hasLoadedCheckpoint.current = false; setMedicalError(''); setRecoveryNonce(value => value + 1) }} /> : <MedicalWorkbench key={medicalRunId || 'pending'} bundle={medicalBundle} runId={medicalRunId} status={medicalStatus} detail={reportDetail} error={medicalError} onBundleChange={applyMedicalBundle} onRefresh={async () => { if (medicalRunId) await loadMedicalRun(medicalRunId, medicalStatus) }} onResume={() => { requestedResumeRunRef.current = medicalRunId; hasLoadedCheckpoint.current = false; setMedicalError(''); setRecoveryNonce(value => value + 1) }} onUpdate={() => navigate('/chat', { state: { scope: scopeRef.current, previousRunId: medicalRunId } })} />}
      </div>
    }
    // 新版: 深度研究模式，显示研究详情面板
    if (isDeepResearchMode) {
      return (
        <ResearchDetail
          data={aggregatedResearchData}
          steps={researchSteps}
          onStepClick={handleResearchStepClick}
          onClose={() => setSelectedResearchDetail(null)}
        />
      )
    }
    // 旧版: 如果当前在深度搜索模式且有步骤详情，显示旧的步骤详情面板
    if (currentChatItem?.type === ChatType.Deepsearch && (selectedStepDetail || currentChatItem?.reactSteps?.length)) {
      return <StepDetailPanel detail={selectedStepDetail} />
    }
    // 否则显示搜索来源
    if (currentChatItem?.search_results?.length) {
      return (
        <Drawer title="搜索来源">
          <Source list={currentChatItem.search_results} />
        </Drawer>
      )
    }
    return null
  }, [currentChatItem, selectedStepDetail, isDeepResearchMode, aggregatedResearchData, researchSteps, researchDataVersion, handleResearchStepClick, medicalRunId, medicalBundle, medicalStatus, medicalError, applyMedicalBundle, loadMedicalRun, navigate, list, loading])

  return (
    <ComPageLayout
      sender={
        <>
          <ComSender
            loading={loading}
            attachments={attachments}
            onSend={send}
            onStop={handleStop}
            onUploadAttachment={handleUploadAttachment}
            onRemoveAttachment={handleRemoveAttachment}
          />
        </>
      }
      right={rightPanelContent}
      wideRight={isDeepResearchMode || Boolean(medicalRunId)}
    >
      <div className="research-conversation-header"><strong>AI DeepResearch · 研究对话</strong><span>继续追问、补充范围或调整报告要求</span></div>
      <div className={styles['chat-page']}>
        <ChatMessage list={list} resolvedQueries={resolvedQueries} activeRunId={medicalRunId} onSend={send} onStepClick={handleStepClick} />
      </div>
    </ComPageLayout>
  )
}
