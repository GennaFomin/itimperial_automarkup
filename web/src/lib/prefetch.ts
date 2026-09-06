/**
 * Прогрев кэша браузера: ролик и ключевые кадры готовых задач качаются заранее,
 * пока человек ещё на списке. Сервер отдаёт их с `immutable`, поэтому при открытии
 * задачи браузер берёт всё из кэша и не ходит по узкому каналу заново.
 */
import { frameUrl, getPrediction, mediaUrl } from '../api/client'

const warmed = new Set<string>()
const queue: string[] = []
let running = 0
const PARALLEL = 2

export function warm(jobIds: string[]) {
  for (const id of jobIds) {
    if (!warmed.has(id) && !queue.includes(id)) queue.push(id)
  }
  void pump()
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
    await fetch(mediaUrl(id)).catch(() => undefined)
  } catch {
    warmed.delete(id) // не получилось — попробуем в следующий раз
  }
}
