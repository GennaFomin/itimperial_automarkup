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
  /** null — пайплайн такой величины не измеряет; см. capabilities у прогноза. */
  confidence: number | null
}

export interface PredictionSegment {
  id: string
  start_ms: number
  end_ms: number
  /** null: разрез — это выброс в статистике, а не вероятность. */
  boundary_confidence: number | null
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
    latency_ms: number | null
    /** Секунды моделей, ставка за час и сумма по ней. Валюта задаётся ставкой. */
    cost: Record<string, number>
    stages_ms: Record<string, number>
  }
  errors: PredictionError[]
  /**
   * Что пайплайн умеет на самом деле. Половина уверенностей приходит null —
   * без этого блока это читалось бы как недоделка, а не как свойство пайплайна.
   */
  capabilities: Capabilities
}

export interface Capabilities {
  boundary_confidence: boolean
  object_confidence: boolean
  keyframe_confidence: boolean
  /** "pair" — число оценивает связку «действие+объект», а не глагол отдельно. */
  action_confidence: 'pair' | boolean
  /** "midpoint" — ключевой кадр взят серединой отрезка, а не выбран моделью. */
  keyframe_source: 'midpoint' | 'selected'
  open_vocabulary: boolean
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
  /** Какие сегменты человек подтвердил глазами. Расширение контракта. */
  verified_ids?: string[]
  /** scratch — замер разметки с нуля, он не заменяет настоящую правку. */
  mode?: 'review' | 'scratch'
}

export interface ReviewResult {
  review_id: string
  saved_at: string
  problems: string[]
  /** Временный идентификатор клиента → присвоенный сервером. */
  id_map: Record<string, string>
}

/** Строка списка задач: сервер отдаёт всё нужное карточке одним запросом. */
export interface JobSummary extends Job {
  filename: string
  duration_ms: number
  warnings: string[]
  reviewed: boolean
}

/** Требования к ролику приходят с сервера, чтобы не дублировать их числами в UI. */
export interface Limits {
  max_duration_ms: number
  min_height: number
  allowed_extensions: string[]
  job_timeout_ms: number
}

export interface Stats {
  videos: number
  median_sec: number
  total_sec: number
  scratch: { videos: number; median_sec: number; total_sec: number }
  /** Во сколько раз правка быстрее разметки с нуля. Цель кейса — 3. */
  speedup?: number
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
  name?: string
  /** Список — подсказка, а не ограничение: модель отвечает своими словами. */
  open?: boolean
  /** Какие объекты допустимы для действия. Есть не у всех словарей. */
  pairs?: Record<string, string[]> | null
}

export interface CreateJobOptions {
  vocab_version?: string
  max_segments?: number
}
