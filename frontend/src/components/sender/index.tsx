/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

import IconFile from '@/assets/component/file.svg'
import IconSend from '@/assets/component/send.svg'
import { deviceActions, deviceState, ExecutionMode, SearchMode } from '@/store/device'
import { knowledgeActions, knowledgeState } from '@/store/knowledge'
import { Button, Input, Space, Dropdown, Checkbox, Tag, Segmented } from 'antd'
import { FileOutlined, LoadingOutlined, SyncOutlined, CheckCircleOutlined, SearchOutlined, DownOutlined, DatabaseOutlined } from '@ant-design/icons'
import classNames from 'classnames'
import { PropsWithChildren, useState, useRef, useEffect } from 'react'
import { useSnapshot } from 'valtio'
import { Attachment } from '@/api/session'
import './index.scss'

export interface AttachmentInfo {
  id: string
  filename: string
  status: 'uploading' | 'pending' | 'processing' | 'completed' | 'failed'
  file?: File
}

export default function ComSender(
  props: PropsWithChildren<{
    className?: string
    loading?: boolean
    attachments?: AttachmentInfo[]
    onSend?: (value: string, attachmentIds?: string[]) => void | Promise<void>
    onStop?: () => void | Promise<void>
    onContract?: () => void
    onUploadAttachment?: (file: File) => Promise<Attachment | null>
    onRemoveAttachment?: (id: string) => void
  }>,
) {
  const {
    className,
    onSend,
    onStop,
    onContract,
    loading,
    attachments = [],
    onUploadAttachment,
    onRemoveAttachment,
    ...rest
  } = props
  const [value, setValue] = useState('')
  const device = useSnapshot(deviceState)
  const knowledge = useSnapshot(knowledgeState)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 拉取知识库列表供选择器使用；失败不阻断输入框（未登录时接口会 401）
  useEffect(() => {
    knowledgeActions.fetchKnowledgeBases().catch(() => {})
  }, [])

  // valtio 的 useSnapshot 返回深度只读结构，这里按只读类型消费即可
  const knowledgeBases: readonly { id: string; name: string }[] = knowledge.knowledgeBases
  const selectedKbIds: readonly string[] = device.selectedKbIds
  // 勾选过但知识库已被删除的 ID 不计入，避免按钮显示的数量对不上
  const validSelectedCount = selectedKbIds.filter(id =>
    knowledgeBases.some(kb => kb.id === id),
  ).length
  const kbButtonLabel =
    validSelectedCount === 0
      ? '知识库：全部'
      : validSelectedCount === 1
        ? knowledgeBases.find(kb => kb.id === selectedKbIds.find(id => knowledgeBases.some(k => k.id === id)))?.name || '知识库'
        : `知识库：${validSelectedCount} 个`

  async function send() {
    if (loading) return
    if (!value && attachments.length === 0) return

    // 过滤出已完成的附件
    const completedAttachmentIds = attachments
      .filter(att => att.status === 'completed')
      .map(att => att.id)

    setValue('')
    await onSend?.(value, completedAttachmentIds.length > 0 ? completedAttachmentIds : undefined)
  }

  function handlePressEnter(e: any) {
    if (e.key === 'Enter') {
      if (e.shiftKey) {
        return
      } else {
        e.preventDefault()
        send()
      }
    }
  }

  function handleFileClick() {
    fileInputRef.current?.click()
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files
    if (!files || files.length === 0) return

    const file = files[0]
    if (onUploadAttachment) {
      await onUploadAttachment(file)
    }

    // 清空 input 以便可以重复选择同一文件
    e.target.value = ''
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'uploading':
      case 'pending':
      case 'processing':
        return 'processing'
      case 'completed':
        return 'success'
      case 'failed':
        return 'error'
      default:
        return 'default'
    }
  }

  function getStatusText(status: string) {
    switch (status) {
      case 'uploading':
        return '上传中'
      case 'pending':
        return '等待处理'
      case 'processing':
        return '处理中'
      case 'completed':
        return '就绪'
      case 'failed':
        return '失败'
      default:
        return ''
    }
  }

  return (
    <div className={classNames('com-sender', className)} {...rest}>
      {/* 附件列表 */}
      {attachments.length > 0 && (
        <div className="com-sender__attachments">
          {attachments.map((att) => (
            <Tag
              key={att.id}
              color={getStatusColor(att.status)}
              closable
              onClose={() => onRemoveAttachment?.(att.id)}
              className="com-sender__attachment-tag"
            >
              {att.status === 'uploading' ? (
                <LoadingOutlined style={{ marginRight: 4 }} />
              ) : att.status === 'pending' || att.status === 'processing' ? (
                <SyncOutlined spin style={{ marginRight: 4 }} />
              ) : att.status === 'completed' ? (
                <CheckCircleOutlined style={{ marginRight: 4 }} />
              ) : (
                <FileOutlined style={{ marginRight: 4 }} />
              )}
              <span className="attachment-filename">{att.filename}</span>
              <span className="attachment-status">({getStatusText(att.status)})</span>
            </Tag>
          ))}
        </div>
      )}

      <Input.TextArea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="按 Enter 发送，Shift + Enter 换行"
        autoSize={{ minRows: 2 }}
        autoFocus
        onPressEnter={handlePressEnter}
      />

      <div className="com-sender__actions">
        <Space className="com-sender__actions-left" size={12}>
          <input
            ref={fileInputRef}
            type="file"
            style={{ display: 'none' }}
            onChange={handleFileChange}
            accept=".pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.jpg,.jpeg,.png,.bmp"
          />
          <Button variant="text" color="default" onClick={handleFileClick} disabled={device.executionMode === 'research'} title="附件可在快速问答或资料知识库中使用">
            <img src={IconFile} />
            附件
          </Button>

          <Segmented
            value={device.executionMode}
            options={[
              { label: '快速问答', value: 'chat' },
              { label: '深度研究', value: 'research' },
            ]}
            onChange={value => deviceActions.setExecutionMode(value as ExecutionMode)}
          />

          {device.executionMode === 'chat' ? <><Dropdown
            trigger={['click']}
            dropdownRender={() => (
              <div className="com-sender__search-dropdown">
                <Checkbox
                  checked={(device.searchModes as SearchMode[]).includes('web')}
                  onChange={() => deviceActions.toggleSearchMode('web')}
                >
                  网络
                </Checkbox>
                <Checkbox
                  checked={(device.searchModes as SearchMode[]).includes('local')}
                  onChange={() => deviceActions.toggleSearchMode('local')}
                >
                  本地知识库
                </Checkbox>
              </div>
            )}
          >
            <Button
              type={(device.searchModes as SearchMode[]).length > 0 ? 'primary' : 'default'}
              ghost={(device.searchModes as SearchMode[]).length > 0}
              icon={<SearchOutlined />}
            >
              {(device.searchModes as SearchMode[]).length === 0 ? '资料来源：无' :
               (device.searchModes as SearchMode[]).length === 2 ? '资料来源：网络+本地' :
               (device.searchModes as SearchMode[]).includes('web') ? '资料来源：网络' : '资料来源：本地'}
              <DownOutlined style={{ fontSize: 10, marginLeft: 4 }} />
            </Button>
          </Dropdown>

          {/* 知识库选择：不勾选即检索全部，勾选后只查选中的几个 */}
          {knowledgeBases.length > 0 && (
            <Dropdown
              trigger={['click']}
              dropdownRender={() => (
                <div className="com-sender__search-dropdown">
                  <div className="com-sender__kb-hint">不勾选则检索全部知识库</div>
                  {knowledgeBases.map(kb => (
                    <Checkbox
                      key={kb.id}
                      checked={selectedKbIds.includes(kb.id)}
                      onChange={() => deviceActions.toggleKb(kb.id)}
                    >
                      {kb.name}
                    </Checkbox>
                  ))}
                  {validSelectedCount > 0 && (
                    <Button
                      type="link"
                      size="small"
                      style={{ paddingLeft: 0 }}
                      onClick={() => deviceActions.clearSelectedKbs()}
                    >
                      清除选择
                    </Button>
                  )}
                </div>
              )}
            >
              <Button
                type={validSelectedCount > 0 ? 'primary' : 'default'}
                ghost={validSelectedCount > 0}
                icon={<DatabaseOutlined />}
              >
                {kbButtonLabel}
                <DownOutlined style={{ fontSize: 10, marginLeft: 4 }} />
              </Button>
            </Dropdown>
          )}
          </> : <Tag color="cyan">网络检索与来源分析</Tag>}
        </Space>

        <Space className="com-sender__actions-right" size={12}>
          {loading ? (
            <Button
              color="default"
              variant="filled"
              shape="circle"
              onClick={onStop}
              className="com-sender__stop-btn"
              title="停止生成"
            >
              <span style={{
                display: 'inline-block',
                width: 12,
                height: 12,
                backgroundColor: '#ff4d4f',
                borderRadius: 2,
              }} />
            </Button>
          ) : (
            <Button
              color="default"
              variant="text"
              shape="circle"
              onClick={send}
            >
              <img src={IconSend} />
            </Button>
          )}
        </Space>
      </div>
    </div>
  )
}
