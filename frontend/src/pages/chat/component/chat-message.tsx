/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

import ComSpinner from '@/components/spin/spinner'
import { ChatRole } from '@/configs'
import classNames from 'classnames'
import styles from './chat-message.module.scss'
import { Result } from './result'
import { useEffect, useRef } from 'react'

function UserMessage(props: { item: API.ChatItem; resolvedQuery?: string }) {
  const { item, resolvedQuery } = props

  return (
    <div className={classNames(styles['chat-message-item--user'])} data-research-question={item.runId} title={resolvedQuery ? item.content : undefined}>
      {resolvedQuery || item.content}
    </div>
  )
}

function AssistantMessage(props: {
  item: API.ChatItem
  isEnd?: boolean
  onSend?: (text: string) => void
  onOpenCiations?: () => void
  onStepClick?: (stepId: string) => void
}) {
  const { item, isEnd, onSend, onStepClick } = props

  return (
    <div
      className={classNames(
        styles['chat-message-item'],
        styles['chat-message-item--assistant'],
      )}
    >
      <Result item={item} isEnd={isEnd} onSend={onSend} onStepClick={onStepClick} />

      {item.loading ? <ComSpinner /> : null}
    </div>
  )
}

export default function ChatMessage(props: {
  list: API.ChatItem[]
  resolvedQueries?: Record<string, string>
  activeRunId?: string | null
  onSend?: (text: string) => void
  onOpenCiations?: (item: API.ChatItem) => void
  onStepClick?: (stepId: string) => void
}) {
  const { list, onSend, onOpenCiations, onStepClick, resolvedQueries = {}, activeRunId } = props
  const containerRef = useRef<HTMLDivElement>(null)
  const focusedRunRef = useRef<string | null>(null)
  const activeQuery = activeRunId ? resolvedQueries[activeRunId] : undefined
  const hasActiveQuestion = list.some(item => item.role === ChatRole.User && item.runId === activeRunId)
  useEffect(() => {
    if (!activeRunId) { focusedRunRef.current = null; return }
    if (!activeQuery || !hasActiveQuestion || focusedRunRef.current === activeRunId) return
    const question = Array.from(containerRef.current?.querySelectorAll<HTMLElement>('[data-research-question]') || [])
      .find(element => element.dataset.researchQuestion === activeRunId)
    if (question) {
      question.scrollIntoView({ block: 'start' })
      focusedRunRef.current = activeRunId
    }
  }, [activeRunId, activeQuery, hasActiveQuestion])

  return (
    <div className={styles['chat-message']} ref={containerRef}>
      {list.map((item, index) => {
        if (item.role === ChatRole.User) {
          return <UserMessage key={item.id} item={item} resolvedQuery={item.runId ? resolvedQueries[item.runId] : undefined} />
        }

        return (
          <AssistantMessage
            key={item.id}
            item={item}
            isEnd={list.length - 1 === index}
            onSend={onSend}
            onOpenCiations={() => onOpenCiations?.(item)}
            onStepClick={onStepClick}
          />
        )
      })}
    </div>
  )
}
