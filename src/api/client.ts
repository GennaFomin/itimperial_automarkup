/**
 * Единая точка доступа к бэкенду.
 *
 * Пока VITE_API_BASE не задан — работаем против мока с фикстурами
 * (контракт §8: UI не ждёт ни ML, ни бэкенд). Как только переменная появится,
 * те же функции пойдут в реальные эндпоинты, а типы и вызовы не изменятся.
 */
import * as mock from './mock'
import type { MockScenario } from './mock'
import type { Job, Prediction, Review, Vocabulary, CreateJobOptions } from './types'

const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '')
export const USING_MOCK = !BASE

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
    public details?: Record<string, unknown>,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...init?.headers,
    },
  })
  if (!res.ok) {
    let code = 'INTERNAL'
    let message = `Запрос завершился с кодом ${res.status}`
    let details: Record<string, unknown> | undefined
    try {
      const body = await res.json()
      if (body?.error) {
        code = body.error.code ?? code
        message = body.error.message ?? message
        details = body.error.details
      }
    } catch {
      // Тело не JSON — оставляем текст по умолчанию.
    }
    throw new ApiError(code, message, res.status, details)
  }
  return (await res.json()) as T
}

export interface CreateJobInput {
  file?: File
  videoUrl?: string
  options?: CreateJobOptions
  /** Только для мока: какой сценарий отыграть. Реальный бэкенд поле игнорирует. */
  scenario?: MockScenario
  /** Длительность загруженного ролика, чтобы мок растянул фикстуру под него. */
  durationMs?: number | null
}

export async function createJob(input: CreateJobInput): Promise<{ job_id: string }> {
  if (USING_MOCK) {
    const job = mock.createJob(input.scenario ?? 'ok', input.durationMs ?? null, input.options)
    return { job_id: job.job_id }
  }
  if (input.file) {
    const form = new FormData()
    form.append('file', input.file)
    if (input.options) form.append('options', JSON.stringify(input.options))
    return request('/api/v1/jobs', { method: 'POST', body: form })
  }
  return request('/api/v1/jobs', {
    method: 'POST',
    body: JSON.stringify({ video_url: input.videoUrl, options: input.options }),
  })
}

export async function getJob(jobId: string): Promise<Job> {
  if (USING_MOCK) return toApiError(() => mock.getJob(jobId))
  return request(`/api/v1/jobs/${jobId}`)
}

export async function getPrediction(jobId: string): Promise<Prediction> {
  if (USING_MOCK) return toApiError(() => mock.getPrediction(jobId))
  return request(`/api/v1/jobs/${jobId}/prediction`)
}

export async function saveReview(
  jobId: string,
  review: Review,
): Promise<{ review_id: string; saved_at: string }> {
  if (USING_MOCK) return mock.saveReview(jobId, review)
  return request(`/api/v1/jobs/${jobId}/review`, {
    method: 'POST',
    body: JSON.stringify(review),
  })
}

export async function cancelJob(jobId: string): Promise<void> {
  if (USING_MOCK) return mock.cancelJob(jobId)
  await request(`/api/v1/jobs/${jobId}/cancel`, { method: 'POST' })
}

export async function getVocab(): Promise<Vocabulary> {
  if (USING_MOCK) return mock.getVocab()
  return request('/api/v1/vocab')
}

async function toApiError<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn()
  } catch (e) {
    if (e instanceof mock.MockApiError) throw new ApiError(e.code, e.message, e.status)
    throw e
  }
}

/**
 * Поллинг статуса по контракту §2: 2 с, после первой минуты бэкофф до 10 с.
 * Возвращает функцию отмены — вызывающий обязан дёрнуть её при размонтировании.
 */
export function pollJob(
  jobId: string,
  onUpdate: (job: Job) => void,
  onError: (error: ApiError) => void,
): () => void {
  let stopped = false
  let timer: ReturnType<typeof setTimeout> | undefined
  const startedAt = Date.now()

  const tick = async () => {
    if (stopped) return
    try {
      const job = await getJob(jobId)
      if (stopped) return
      onUpdate(job)
      if (job.status === 'queued' || job.status === 'running') schedule()
    } catch (e) {
      if (stopped) return
      onError(e instanceof ApiError ? e : new ApiError('INTERNAL', String(e), 0))
    }
  }

  const schedule = () => {
    const elapsed = Date.now() - startedAt
    timer = setTimeout(tick, elapsed < 60_000 ? 2000 : 10_000)
  }

  void tick()
  return () => {
    stopped = true
    if (timer) clearTimeout(timer)
  }
}
