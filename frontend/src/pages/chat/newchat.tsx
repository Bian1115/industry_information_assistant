import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRightOutlined, CompassOutlined, FileTextOutlined, SearchOutlined } from '@ant-design/icons'
import { Button, Form, Input, message } from 'antd'
import * as api from '@/api'
import { deviceActions } from '@/store/device'
import { setPageTransport } from '@/utils'
import { transportToChatEnter } from './shared'
import './component/research-start.scss'

const examples = [
  { label: '技术研究', question: '调研长上下文模型与检索增强生成的研究进展，比较它们的适用场景、成本与主要局限，并引用关键论文和公开评测。' },
  { label: '行业洞察', question: '分析储能行业的主要技术路线与商业化进展，比较成本、应用场景和发展瓶颈，并说明关键数据的来源和时间。' },
  { label: '方案比较', question: '比较开源大模型的部署方案，从推理性能、硬件需求、成本和维护复杂度分析取舍，给出不同使用场景下的建议。' },
]

export default function NewChat() {
  const navigate = useNavigate()
  const [form] = Form.useForm<{ question: string }>()
  const [submitting, setSubmitting] = useState(false)
  const inFlight = useRef(false)
  async function start({ question }: { question: string }) {
    if (inFlight.current) return
    inFlight.current = true
    setSubmitting(true)
    try {
      const title = question.trim().replace(/\s+/g, ' ').slice(0, 60)
      const { data } = await api.session.createSession({ title, session_type: 'deepsearch' })
      deviceActions.setExecutionMode('research')
      deviceActions.setSearchModes(['web'])
      setPageTransport(transportToChatEnter, { data: { sessionId: data.id, message: question, executionMode: 'research' } })
      navigate(`/chat/${data.id}`)
    } catch { message.error('创建研究失败，请重试') }
    finally { inFlight.current = false; setSubmitting(false) }
  }
  return <main className="research-start">
    <div className="research-start__eyebrow"><CompassOutlined /> AI DEEPRESEARCH</div>
    <h1>把复杂问题，研究清楚。</h1>
    <p className="research-start__intro">从研究规划、资料检索到分析与写作，协作完成一份可以追溯来源的研究报告。</p>
    <Form form={form} layout="vertical" onFinish={start} requiredMark={false} className="research-start__composer">
      <Form.Item name="question" label="你想深入研究什么？" rules={[{ required: true, whitespace: true, message: '请输入研究问题' }]}>
        <Input.TextArea autoSize={{ minRows: 5, maxRows: 12 }} maxLength={20000} disabled={submitting}
          placeholder="描述你的研究问题。可以说明研究背景、需要比较的对象、时间范围和交付要求。"
          onKeyDown={event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.nativeEvent.isComposing) { event.preventDefault(); form.submit() } }} />
      </Form.Item>
      <div className="research-start__composer-footer"><span><SearchOutlined /> 网络深度研究<span className="research-start__shortcut"> · Ctrl / ⌘ + Enter 开始</span></span><Button type="primary" htmlType="submit" size="large" loading={submitting} icon={<ArrowRightOutlined />} iconPosition="end">开始研究</Button></div>
    </Form>
    <div className="research-start__examples" aria-label="研究问题示例"><span>试试这些方向</span>{examples.map(example => <Button key={example.label} disabled={submitting} onClick={() => { form.setFieldValue('question', example.question); form.getFieldInstance('question')?.focus() }}>{example.label}<ArrowRightOutlined /></Button>)}</div>
    <div className="research-start__steps">
      {[{ icon: <CompassOutlined />, title: '拆解问题', detail: '明确研究目标，规划需要回答的关键问题。' },
        { icon: <SearchOutlined />, title: '查找与分析', detail: '检索资料、读取来源，比较证据和不同观点。' },
        { icon: <FileTextOutlined />, title: '交付研究报告', detail: '整合结论与引用，保留资料及研究过程。' }].map((step, index) => <article key={step.title}><span className="research-start__step-icon">{step.icon}</span><span className="research-start__step-number">0{index + 1}</span><h2>{step.title}</h2><p>{step.detail}</p></article>)}
    </div>
    <p className="research-start__footer">支持技术、商业、行业与学术研究。研究完成后，可以继续追问、调整报告要求，或导出报告与研究记录。</p>
  </main>
}
