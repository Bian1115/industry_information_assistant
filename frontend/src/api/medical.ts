import { request } from './request'
import type { ResearchRunState } from './session'

export type ResearchFocus = 'overview' | 'treatment' | 'diagnosis' | 'mechanism' | 'epidemiology'
export interface MedicalScope {
  disease: string
  synonyms: string[]
  focus: ResearchFocus
  population: string
  start_date: string | null
  end_date: string
  include_preprints: boolean
  sources: string[]
  max_results: number
  max_full_texts: number
  inclusion_criteria: string[]
  exclusion_criteria: string[]
  languages: string[]
}
export interface Paper {
  paper_id: string
  title: string
  authors?: string[]
  journal?: string
  doi?: string
  pmid?: string
  pmcid?: string
  publication_date?: string
  abstract?: string
  publication_types?: string[]
  source?: string
  url?: string
  content_level?: string
  access_status?: string
  sections?: Array<{ section_id?: string; id?: string; title?: string; text?: string; content?: string }>
  is_preprint?: boolean
  retraction_status?: string
  study_ids?: string[]
}
export interface ScreeningRecord {
  paper_id: string
  decision: string
  reason?: string
  reasons?: string[]
  reviewer?: string
}
export interface Evidence {
  evidence_id: string
  paper_id: string
  study_id?: string | null
  study_design?: string | null
  population?: string | null
  sample_size?: number | string | null
  intervention?: string | null
  comparator?: string | null
  outcome?: string | null
  effect?: string | null
  confidence_interval?: string | null
  follow_up?: string | null
  safety?: string | null
  limitations?: string | string[] | null
  quote?: string | null
  section_id?: string | null
  verification_status?: string | null
}
export interface MedicalBundle {
  fetched_at?: string
  pipeline?: 'graph' | 'legacy'
  graph_state?: ResearchRunState
  agent_steps?: Array<{ type: string; title: string; agent: string; status: string }>
  status?: string
  report_version?: number
  run_status?: string
  protocol?: Record<string, unknown>
  papers?: Paper[]
  screening?: ScreeningRecord[]
  evidence?: Evidence[]
  search_log?: Array<Record<string, unknown>>
  stats?: Record<string, unknown>
  limitations?: string[]
  review?: Record<string, unknown>
  changes?: Record<string, unknown>
  final_report?: string
}
export const focusLabels: Record<ResearchFocus, string> = {
  overview: '研究进展综述', treatment: '治疗进展', diagnosis: '诊断进展', mechanism: '发病机制', epidemiology: '流行病学',
}
export function defaultMedicalScope(): MedicalScope {
  const now = new Date()
  const localDay = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10)
  return { disease: '', synonyms: [], focus: 'overview', population: '', start_date: null, end_date: localDay, include_preprints: false, sources: ['pubmed', 'europe_pmc'], max_results: 60, max_full_texts: 12, inclusion_criteria: [], exclusion_criteria: [], languages: [] }
}
export function scopeFromBundle(bundle: MedicalBundle): MedicalScope | null {
  const protocol = bundle.protocol || {}
  const candidate = (protocol.scope || protocol.research_scope || protocol) as Partial<MedicalScope>
  return typeof candidate.disease === 'string' ? { ...defaultMedicalScope(), ...candidate } : null
}
export async function getMedicalResearch(runId: string): Promise<MedicalBundle> {
  // 新运行读取图状态；只有确实没有新运行时才尝试历史医疗快照。
  try {
    const response = await request.get<{ success: boolean; state: ResearchRunState }>(
      `/research/runs/${encodeURIComponent(runId)}/state`,
      // 资料展示、状态恢复和轮询可同时读取同一 run，不能相互取消。
      // 页面通过请求序号丢弃过期结果，不依赖全局重复请求取消。
      { loading: false, errorToast: false, serviceEnvelope: false, cancelRepeat: false },
    )
    const state = response.data.state
    if (state.pipeline !== 'legacy') {
      return { pipeline: 'graph', graph_state: state,
        run_status: state.run_status || state.status,
        final_report: state.draft?.final_report || state.final_report || '' }
    }
  } catch (error) {
    if ((error as { response?: { status?: number } }).response?.status !== 404) throw error
  }
  const response = await request.get(`/research/legacy/medical/${encodeURIComponent(runId)}`, { loading: false, errorToast: false, serviceEnvelope: false, cancelRepeat: false })
  const data = response.data
  return { ...(data.medical_research ? { ...data.medical_research, final_report: data.final_report } : data), pipeline: 'legacy' }
}
export async function getMedicalPaper(runId: string, paperId: string): Promise<Paper> {
  const response = await request.get<Paper>(`/research/legacy/medical/${encodeURIComponent(runId)}/papers/${encodeURIComponent(paperId)}`, { loading: false, errorToast: false, serviceEnvelope: false })
  return response.data
}
// 人工纳排与证据纠正入口已随独立医疗编排下线：历史记录只读，
// 不再提供 reviewPaper / correctEvidence 写入接口。
export type ExportFormat = 'markdown' | 'csv' | 'bibtex' | 'ris' | 'json' | 'html' | 'docx'
export async function exportMedicalResearch(runId: string, format: ExportFormat, disease: string) {
  const response = await request.get<Blob>(`/research/legacy/medical/${encodeURIComponent(runId)}/export`, {
    params: { format }, responseType: 'blob', loading: false, serviceEnvelope: false,
  })
  const extension = { markdown: 'md', csv: 'csv', bibtex: 'bib', ris: 'ris', json: 'json', html: 'html', docx: 'docx' }[format]
  const url = URL.createObjectURL(response.data)
  const link = document.createElement('a')
  link.href = url
  link.download = `${(disease || '疾病').replace(/[\\/:*?"<>|]/g, '_')}-研究进展-${runId.slice(0, 8)}.${extension}`
  document.body.append(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
