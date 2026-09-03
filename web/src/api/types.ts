/**
 * Типы контракта разметки действий по видео, schema_version 1.0.
 * Источник истины — contracts.md. Незнакомые поля игнорируем, а не падаем.
 */

export type JobStatus =
  | 'queued'
  | 'running'
  | 'done'
  | 'done_with_errors'
  | 'failed'
  | 'cancelled'

export type JobStage = 'decode' | 'proposals' | 'recognize' | 'keyframe' | 'validate'

export const JOB_STAGES: JobStage[] = [
  'decode',
  'proposals',
  'recognize',
  'keyframe',
  'validate',
]

export const STAGE_LABEL: Record<JobStage, string> = {
  decode: 'Декодирование',
  proposals: 'Поиск границ',
  recognize: 'Распознавание',
  keyframe: 'Ключевые кадры',
  validate: 'Валидация',
}

export const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'В очереди',
  running: 'Обработка',
  done: 'Готово',
  done_with_errors: 'Готово с ошибками',
  failed: 'Ошибка',
  cancelled: 'Отменено',
}

export interface Job {
  job_id: string
  status: JobStatus
  stage: JobStage | null
  progress: number
  created_at: string
  started_at: string | null
  finished_at: string | null
  error: ApiError['error'] | null
}

export interface ApiError {
  error: {
    code: string
    message: string
    details?: Record<string, unknown>
  }
}

export interface VideoMeta {
  duration_ms: number
  fps: number
  width: number
  height: number
}

export interface FieldValue {
  value: string
  confidence: number
}

export interface PredictionSegment {
  id: string
  start_ms: number
  end_ms: number
  boundary_confidence: number
  action: FieldValue
  object: FieldValue
  /** null допустим: стадия keyframe могла упасть — сегмент есть, кадр не выбран. */
  keyframe_ms: number | null
  keyframe_confidence: number | null
}

export interface PredictionError {
  stage: JobStage
  code: string
  message: string
  segment_ids: string[]
}

export interface Prediction {
  schema_version: string
  prediction_id: string
  job_id: string
  model_version: string
  vocab_version: string
  created_at: string
  video: VideoMeta
  segments: PredictionSegment[]
  stats: {
    latency_ms: number
    cost_usd: number
    frames_decoded: number
  }
  errors: PredictionError[]
}

export type SegmentOrigin = 'model' | 'human'

export interface ReviewSegment {
  id: string
  origin: SegmentOrigin
  start_ms: number
  end_ms: number
  action: string
  object: string
  keyframe_ms: number | null
}

export interface Review {
  schema_version: string
  prediction_id: string
  reviewer: string
  submitted_at: string
  segments: ReviewSegment[]
  time_spent_ms: number
}

export interface VocabAction {
  id: string
  label_ru: string
  color: string
}

export interface VocabObject {
  id: string
  label_ru: string
}

export interface Vocabulary {
  version: string
  actions: VocabAction[]
  objects: VocabObject[]
}

export interface CreateJobOptions {
  vocab_version?: string
  max_segments?: number
}
