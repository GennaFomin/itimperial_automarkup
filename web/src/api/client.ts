/**
 * Единая точка доступа к бэкенду.
 *
 * По умолчанию UI говорит с настоящим пайплайном. Базовый адрес пуст, то есть
 * запросы идут на тот же origin: в разработке их проксирует Vite, в сборке SPA
 * отдаёт сам FastAPI. Ни CORS, ни preflight в этой схеме не участвуют.
 *
 * Мок с фикстурами включается ЯВНО, флагом VITE_API_MOCK=1. Он не запасной путь,
 * а инструмент: на нём держатся обязательные сценарии контракта §8 — пустой
 * результат, деградировавший прогон, упавшая задача, обработка дольше десяти
 * минут, — которые живой бэкенд по требованию не воспроизводит.
 *
 * Именно поэтому мок не может быть значением по умолчанию: с ним пустая
 * переменная окружения в докере молча показала бы фикстуры вместо результатов
 * модели, и подмену никто бы не заметил.
 */
import type {
  Job,
  JobSummary,
  Limits,
  Prediction,
  Review,
  ReviewResult,
  Stats,
  Vocabulary,
} from './types'

const BASE = ((import.meta.env.VITE_API_BASE as string | undefined) ?? '').replace(/\/$/, '')
export const USING_MOCK = import.meta.env.VITE_API_MOCK === '1'

const V1 = '/api/v1'

/** Мок грузится только когда включён — иначе фикстуры попали бы в сборку. */
type MockModule = typeof import('./mock')
let mockModule: Promise<MockModule> | null = null
const useMock = (): Promise<MockModule> => {
  mockModule ??= import('./mock')
  return mockModule
}

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
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...init?.headers,
      },
    })
  } catch (cause) {
    // Сеть или сервер недоступны: без своего кода это выглядело бы как ошибка
    // контракта, хотя запрос вообще не дошёл.
    throw new ApiError('NETWORK', 'Бэкенд недоступен', 0, { cause: String(cause) })
  }

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
      } else if (body?.detail) {
        // Ответ внутреннего API: у него другой конверт.
        message = typeof body.detail === 'string' ? body.detail : message
      }
    } catch {
      // Тело не JSON — оставляем текст по умолчанию.
    }
    throw new ApiError(code, message, res.status, details)
  }
  return (await res.json()) as T
}

export interface CreateJobInput {
  file: File
  /** Только для мока: какой сценарий отыграть. Реальный бэкенд поле игнорирует. */
  scenario?: string
  /** Длительность загруженного ролика, чтобы мок растянул фикстуру под него. */
  durationMs?: number | null
}

export async function createJob(input: CreateJobInput): Promise<{ job_id: string }> {
  if (USING_MOCK) {
    const mock = await useMock()
    const job = mock.createJob((input.scenario as never) ?? 'ok', input.durationMs ?? null)
    return { job_id: job.job_id }
  }
  const form = new FormData()
  form.append('file', input.file)
  return request(`${V1}/jobs`, { method: 'POST', body: form })
}

export async function listJobs(): Promise<JobSummary[]> {
  if (USING_MOCK) return (await useMock()).listJobs()
  return request(`${V1}/jobs`)
}

export async function getJob(jobId: string): Promise<Job> {
  if (USING_MOCK) return toApiError(async () => (await useMock()).getJob(jobId))
  return request(`${V1}/jobs/${jobId}`)
}

/** Прогноз модели: неизменяем, служит базой для диффа правок. */
export async function getPrediction(jobId: string): Promise<Prediction> {
  if (USING_MOCK) return toApiError(async () => (await useMock()).getPrediction(jobId))
  return request(`${V1}/jobs/${jobId}/prediction`)
}

/**
 * Актуальная разметка: правка человека, если она есть, иначе прогноз.
 *
 * Открывать редактор на прогнозе нельзя: он показал бы модель поверх уже
 * сохранённой работы, а следующее сохранение стёрло бы её.
 */
export async function getAnnotation(jobId: string): Promise<Prediction> {
  if (USING_MOCK) return toApiError(async () => (await useMock()).getPrediction(jobId))
  return request(`${V1}/jobs/${jobId}/annotation?source=current`)
}

export async function saveReview(jobId: string, review: Review): Promise<ReviewResult> {
  if (USING_MOCK) return (await useMock()).saveReview(jobId, review)
  return request(`${V1}/jobs/${jobId}/review`, {
    method: 'POST',
    body: JSON.stringify(review),
  })
}

export async function cancelJob(jobId: string): Promise<void> {
  if (USING_MOCK) return (await useMock()).cancelJob(jobId)
  await request(`${V1}/jobs/${jobId}/cancel`, { method: 'POST' })
}

/** Удалить задачу целиком: запись, видео, кадры. Идемпотентно, как и отмена. */
export async function deleteJob(jobId: string): Promise<void> {
  if (USING_MOCK) return (await useMock()).deleteJob(jobId)
  await request(`${V1}/jobs/${jobId}`, { method: 'DELETE' })
}

export async function getVocab(): Promise<Vocabulary> {
  if (USING_MOCK) return (await useMock()).getVocab()
  return request(`${V1}/vocab`)
}

export async function getLimits(): Promise<Limits> {
  if (USING_MOCK) return (await useMock()).getLimits()
  return request(`${V1}/limits`)
}

export async function getStats(): Promise<Stats> {
  if (USING_MOCK) return (await useMock()).getStats()
  return request(`${V1}/stats`)
}

/**
 * Активные секунды работы человека — единственный источник времени для метрики
 * «в три раза быстрее». Шлётся дельтами по ходу работы: вкладку закрывают, и
 * замер, накопленный только в памяти, пропал бы целиком.
 */
export async function reportActivity(
  jobId: string,
  mode: 'review' | 'scratch',
  seconds: number,
): Promise<void> {
  if (USING_MOCK || seconds < 1) return
  try {
    await fetch(`${BASE}${V1}/jobs/${jobId}/activity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, seconds }),
      // Отправка на закрытии вкладки должна пережить сам документ.
      keepalive: true,
    })
  } catch {
    // Потеря замера не повод ломать работу человека.
  }
}

/** Видео с сервера: переживает перезагрузку вкладки, в отличие от blob-URL. */
export const mediaUrl = (jobId: string) => `${BASE}${V1}/jobs/${jobId}/media`

/** Кадр на заданной миллисекунде. Бэкенд кэширует его на диске. */
export const frameUrl = (jobId: string, ms: number) =>
  `${BASE}${V1}/jobs/${jobId}/frame?ms=${Math.max(0, Math.round(ms))}`

export const exportUrl = (jobId: string, format: 'json' | 'csv', source: 'review' | 'prediction') =>
  `${BASE}${V1}/jobs/${jobId}/export?format=${format}&source=${source}`

async function toApiError<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn()
  } catch (e) {
    const mock = await useMock()
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
