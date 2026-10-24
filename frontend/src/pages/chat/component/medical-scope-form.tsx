import { Button, Checkbox, Collapse, Form, Input, InputNumber, Select, Space } from 'antd'
import { defaultMedicalScope, focusLabels, type MedicalScope } from '@/api/medical'
import './medical-workbench.scss'

interface Props {
  initialValue?: MedicalScope
  initialQuestion?: string
  loading?: boolean
  onSubmit: (scope: MedicalScope, question: string, previousRunId?: string) => void | Promise<void>
  submitLabel?: string
  previousRunId?: string
}

export default function MedicalScopeForm({ initialValue, initialQuestion, loading, onSubmit, submitLabel = '开始文献研究', previousRunId }: Props) {
  const [form] = Form.useForm()
  return <Form form={form} layout="vertical" initialValues={{ ...defaultMedicalScope(), ...initialValue, question: initialQuestion || '', previous_run_id: previousRunId }}
    onFinish={({ question, previous_run_id, ...scope }) => onSubmit({ ...scope, disease: scope.disease.trim(), start_date: scope.start_date || null }, question || `梳理${scope.disease}的${focusLabels[scope.focus as keyof typeof focusLabels]}，比较主要研究、局限与尚未解决的问题。`, previous_run_id || undefined)}>
    <div className="medical-form-grid">
      <Form.Item label="疾病 / 研究主题" name="disease" rules={[{ required: true, whitespace: true, message: '请输入疾病名称' }]}><Input placeholder="例如：非小细胞肺癌 / non-small cell lung cancer" maxLength={200} /></Form.Item>
      <Form.Item label="研究方向" name="focus"><Select options={Object.entries(focusLabels).map(([value, label]) => ({ value, label }))} /></Form.Item>
      <Form.Item label="起始日期（留空不限）" name="start_date" rules={[{ validator: (_, value) => !value || value >= '1800-01-01' ? Promise.resolve() : Promise.reject(new Error('日期不能早于1800年')) }]}><Input type="date" min="1800-01-01" /></Form.Item>
      <Form.Item label="检索截止日期" name="end_date" dependencies={['start_date']} rules={[{ required: true, message: '请选择截止日期' }, { validator: (_, value) => !value || ((!form.getFieldValue('start_date') || value >= form.getFieldValue('start_date')) && value <= defaultMedicalScope().end_date && value >= '1800-01-01') ? Promise.resolve() : Promise.reject(new Error('截止日期需介于起始日期（不早于1800年）与今天之间')) }]}><Input type="date" /></Form.Item>
    </div>
    <Form.Item label="具体希望了解什么" name="question"><Input.TextArea placeholder="例如：近三年的治疗进展、关键临床研究及证据仍不足的方向" rows={2} /></Form.Item>
    <Collapse ghost items={[{ key: 'scope', forceRender: true, label: '细化人群、检索来源与纳排条件', children: <>
      <Form.Item label="英文名与同义词（回车添加）" name="synonyms" extra="中文疾病名建议补充英文检索词。"><Select mode="tags" tokenSeparators={[';']} placeholder="例如：NSCLC" /></Form.Item>
      <Form.Item label="研究人群" name="population"><Input maxLength={1000} placeholder="例如：成人、晚期、既往未接受治疗" /></Form.Item>
      <Form.Item label="检索来源" name="sources" rules={[{ required: true, type: 'array', min: 1, message: '至少选择一个来源' }]}><Checkbox.Group options={[{ label: 'PubMed', value: 'pubmed' }, { label: 'Europe PMC', value: 'europe_pmc' }, { label: 'ClinicalTrials.gov', value: 'clinical_trials' }]} /></Form.Item>
      <Form.Item name="include_preprints" valuePropName="checked"><Checkbox>纳入预印本（单独标记，尚未经过同行评议）</Checkbox></Form.Item>
      <Form.Item label="纳入条件（回车添加）" name="inclusion_criteria"><Select mode="tags" placeholder="例如：随机对照研究" /></Form.Item>
      <Form.Item label="排除条件（回车添加）" name="exclusion_criteria"><Select mode="tags" placeholder="例如：动物实验" /></Form.Item>
      <Form.Item label="文献语言（留空不限）" name="languages"><Select mode="multiple" options={[{ label: '英语', value: 'eng' }, { label: '中文', value: 'chi' }]} /></Form.Item>
      <Space wrap align="start"><Form.Item label="最多检索记录" name="max_results" rules={[{ required: true, type: 'number', min: 1, max: 200, message: '请输入1—200' }]}><InputNumber min={1} max={200} /></Form.Item><Form.Item label="最多获取全文" name="max_full_texts" rules={[{ required: true, type: 'number', min: 0, max: 40, message: '请输入0—40' }]}><InputNumber min={0} max={40} /></Form.Item></Space>
      <Form.Item label="与历史研究比较（可选）" name="previous_run_id" rules={[{ pattern: /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/, message: '请输入有效的历史研究编号' }]} extra="可从历史研究的检索记录页复制研究编号。"><Input placeholder="历史研究编号" /></Form.Item>
    </> }]} />
    <Button type="primary" htmlType="submit" size="large" loading={loading}>{submitLabel}</Button>
  </Form>
}
