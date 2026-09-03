/**
 * Моковый бэкенд: полный жизненный цикл job в памяти вкладки.
 * Живёт ровно по контракту из contracts.md, чтобы замена на реальный
 * бэкенд сводилась к установке VITE_API_BASE.
 */
import predictionOk from '../fixtures/prediction_ok.json'
import predictionWithErrors from '../fixtures/prediction_with_errors.json'
import predictionEmpty from '../fixtures/prediction_empty.json'
import vocabFixture from '../fixtures/vocab.json'
import type {
  Job,
  JobStage,
  Prediction,
  Review,
  Vocabulary,
  CreateJobOptions,
} from './types'
import { JOB_STAGES } from './types'

/** Сценарий, который отыграет мок для конкретной джобы. */
export type MockScenario = 'ok' | 'with_errors' | 'empty' | 'failed' | 'slow'

const PREDICTIONS: Record<string, Prediction> = {
  ok: predictionOk as Prediction,
  with_errors: predictionWithErrors as Prediction,
  empty: predictionEmpty as Prediction,
}

/** Сколько мок «обрабатывает» ролик до статуса done. */
const RUN_MS: Record<MockScenario, number> = {
  ok: 9_000,
  with_errors: 11_000,
  empty: 6_000,
  failed: 7_000,
  slow: 11 * 60_000,
}

interface MockJob {
  job_id: string
  scenario: MockScenario
  created_at: number
  cancelled: boolean
  durationMs: number | null
}

/**
 * Реестр джоб переживает перезагрузку вкладки.
 *
 * Список задач хранится в localStorage, и без этого после F5 карточки остались
 * бы, а открыть их было бы нельзя: мок отвечал бы JOB_NOT_FOUND. Реальному
 * бэкенду это не нужно — там состояние и так на сервере.
 */
const JOBS_KEY = 'automarkup.mock.jobs.v1'

const jobs = new Map<string, MockJob>(loadJobs())
const reviews = new Map<string, Review>()

function loadJobs(): [string, MockJob][] {
  try {
    const raw = localStorage.getItem(JOBS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as [string, MockJob][]) : []
  } catch {
    return []
  }
}

function persistJobs() {
  try {
    localStorage.setItem(JOBS_KEY, JSON.stringify([...jobs.entries()]))
  } catch {
    // Приватный режим: джобы просто не переживут перезагрузку.
  }
}

let counter = 0
const newId = (prefix: string) =>
  `${prefix}_${(Date.now() % 0xffffff).toString(16)}${(counter++).toString(16)}`

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export function createJob(
  scenario: MockScenario,
  durationMs: number | null,
  _options?: CreateJobOptions,
): MockJob {
  const job: MockJob = {
    job_id: newId('j'),
    scenario,
    created_at: Date.now(),
    cancelled: false,
    durationMs,
  }
  jobs.set(job.job_id, job)
  persistJobs()
  return job
}

export async function getJob(jobId: string): Promise<Job> {
  await sleep(120)
  const job = jobs.get(jobId)
  if (!job) throw new MockApiError('JOB_NOT_FOUND', 'Задача не найдена', 404)

  const created = new Date(job.created_at).toISOString()
  if (job.cancelled) {
    return {
      job_id: job.job_id,
      status: 'cancelled',
      stage: null,
      progress: 1,
      created_at: created,
      started_at: created,
      finished_at: new Date().toISOString(),
      error: null,
    }
  }

  const elapsed = Date.now() - job.created_at
  const total = RUN_MS[job.scenario]
  const raw = Math.min(elapsed / total, 1)
  // progress монотонно не убывает и не долетает до 1 раньше времени.
  const progress = Math.min(0.99, Math.round(raw * 100) / 100)
  const stage: JobStage = JOB_STAGES[Math.min(JOB_STAGES.length - 1, Math.floor(raw * JOB_STAGES.length))]

  if (raw < 1) {
    return {
      job_id: job.job_id,
      status: elapsed < 900 ? 'queued' : 'running',
      stage: elapsed < 900 ? null : stage,
      progress,
      created_at: created,
      started_at: elapsed < 900 ? null : new Date(job.created_at + 900).toISOString(),
      finished_at: null,
      error: null,
    }
  }

  const finished = new Date(job.created_at + total).toISOString()
  if (job.scenario === 'failed') {
    return {
      job_id: job.job_id,
      status: 'failed',
      stage: 'recognize',
      progress: 0.41,
      created_at: created,
      started_at: created,
      finished_at: finished,
      error: {
        code: 'DECODE_FAILED',
        message: 'Не удалось декодировать поток видео',
        details: { at_ms: 12400 },
      },
    }
  }

  return {
    job_id: job.job_id,
    status: job.scenario === 'with_errors' ? 'done_with_errors' : 'done',
    stage: 'validate',
    progress: 1,
    created_at: created,
    started_at: created,
    finished_at: finished,
    error: null,
  }
}

export async function getPrediction(jobId: string): Promise<Prediction> {
  await sleep(180)
  const job = jobs.get(jobId)
  if (!job) throw new MockApiError('JOB_NOT_FOUND', 'Задача не найдена', 404)
  const state = await getJob(jobId)
  if (state.status !== 'done' && state.status !== 'done_with_errors') {
    throw new MockApiError('NOT_READY', 'Прогноз ещё не готов', 409)
  }
  const base = PREDICTIONS[job.scenario === 'failed' || job.scenario === 'slow' ? 'ok' : job.scenario]
  return rescale(base, job.job_id, job.durationMs)
}

export async function saveReview(jobId: string, review: Review): Promise<{ review_id: string; saved_at: string }> {
  await sleep(220)
  reviews.set(jobId, review)
  return { review_id: newId('r'), saved_at: new Date().toISOString() }
}

export async function getVocab(): Promise<Vocabulary> {
  await sleep(80)
  return vocabFixture as Vocabulary
}

export async function cancelJob(jobId: string): Promise<void> {
  await sleep(120)
  const job = jobs.get(jobId)
  if (job) {
    job.cancelled = true
    persistJobs()
  }
}

/**
 * Фикстура снята с ролика фиксированной длины. Если пользователь загрузил
 * своё видео — растягиваем сегменты пропорционально, чтобы разметка легла
 * на реальный таймлайн, а не торчала за его пределы.
 */
function rescale(base: Prediction, jobId: string, durationMs: number | null): Prediction {
  const src = base.video.duration_ms
  if (!durationMs || durationMs <= 0 || Math.abs(durationMs - src) < 500) {
    return { ...base, job_id: jobId }
  }
  const k = durationMs / src
  const at = (ms: number) => Math.max(0, Math.min(durationMs, Math.round(ms * k)))
  return {
    ...base,
    job_id: jobId,
    video: { ...base.video, duration_ms: durationMs },
    segments: base.segments.map((s) => ({
      ...s,
      start_ms: at(s.start_ms),
      end_ms: at(s.end_ms),
      keyframe_ms: s.keyframe_ms === null ? null : at(s.keyframe_ms),
    })),
  }
}

export class MockApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
  ) {
    super(message)
    this.name = 'MockApiError'
  }
}
