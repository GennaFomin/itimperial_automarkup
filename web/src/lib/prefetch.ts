/**
 * Прогрев: ролик и ключевые кадры готовых задач качаются заранее, пока человек ещё
 * на списке. Кадры оседают в HTTP-кэше браузера (сервер отдаёт их с `immutable`),
 * а ролик держим в памяти вкладки как blob: плеер запрашивает видео по кускам, и
 * такие запросы Chrome в HTTP-кэше не ищет — по узкому каналу это те же секунды.
 */
import { frameUrl, getPrediction, mediaUrl } from '../api/client'

const warmed = new Set<string>()
const queue: string[] = []
let running = 0
const PARALLEL = 2
const mediaBlobs = new Map<string, string>()
const mediaLoading = new Map<string, Promise<string | null>>()

export function warm(jobIds: string[]) {
  for (const id of jobIds) {
    if (!warmed.has(id) && !queue.includes(id)) queue.push(id)
  }
  void pump()
}

/** Локальная копия ролика, если она уже скачана; иначе null — играть с сервера. */
export function cachedMediaUrl(jobId: string): string | null {
  return mediaBlobs.get(jobId) ?? null
}

/** Скачать ролик в память вкладки; повторные вызовы возвращают ту же копию. */
export function cacheMedia(jobId: string): Promise<string | null> {
  const ready = mediaBlobs.get(jobId)
  if (ready) return Promise.resolve(ready)
  const pending = mediaLoading.get(jobId)
  if (pending) return pending
  const task = fetch(mediaUrl(jobId))
    .then(async (response) => {
      if (!response.ok) return null
      const url = URL.createObjectURL(await response.blob())
      mediaBlobs.set(jobId, url)
      return url
    })
    .catch(() => null)
    .finally(() => mediaLoading.delete(jobId))
  mediaLoading.set(jobId, task)
  return task
}

async function pump() {
  while (running < PARALLEL && queue.length) {
    const id = queue.shift()!
    warmed.add(id)
    running++
    void warmOne(id).finally(() => {
      running--
      void pump()
    })
  }
}

async function warmOne(id: string) {
  try {
    const prediction = await getPrediction(id)
    // Сначала кадры: они маленькие и нужны первыми, ролик — следом.
    for (const seg of prediction.segments) {
      if (seg.keyframe_ms !== null) await fetch(frameUrl(id, seg.keyframe_ms)).catch(() => undefined)
    }
    await cacheMedia(id)
  } catch {
    warmed.delete(id) // не получилось — попробуем в следующий раз
  }
}
