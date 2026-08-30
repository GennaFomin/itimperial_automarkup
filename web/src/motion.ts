/** Кандидаты в границы шагов из сигнала движения.
 *
 * Локальные минимумы движения — это моменты, где рука почти остановилась: смена
 * захвата, конец подведения детали. Пока пайплайн не отдаёт свои события, редактор
 * притягивает границы сюда, и это уже заметно точнее, чем ставить их на глаз.
 */
export function motionMinima(motion: number[], duration: number): number[] {
  if (motion.length < 5 || duration <= 0) return []
  const step = duration / motion.length
  const smoothed = smooth(motion, 3)
  const candidates: number[] = []

  for (let i = 2; i < smoothed.length - 2; i += 1) {
    const value = smoothed[i]
    const isMinimum =
      value <= smoothed[i - 1] &&
      value <= smoothed[i - 2] &&
      value <= smoothed[i + 1] &&
      value <= smoothed[i + 2]
    if (isMinimum) candidates.push((i + 0.5) * step)
  }
  return dedupe(candidates, step * 2)
}

export function snap(time: number, candidates: number[], tolerance: number): number {
  let best = time
  let distance = tolerance
  for (const candidate of candidates) {
    const delta = Math.abs(candidate - time)
    if (delta < distance) {
      distance = delta
      best = candidate
    }
  }
  return best
}

function smooth(values: number[], radius: number): number[] {
  return values.map((_, index) => {
    const from = Math.max(0, index - radius)
    const to = Math.min(values.length, index + radius + 1)
    let sum = 0
    for (let i = from; i < to; i += 1) sum += values[i]
    return sum / (to - from)
  })
}

function dedupe(times: number[], minGap: number): number[] {
  const result: number[] = []
  for (const time of times) {
    if (!result.length || time - result[result.length - 1] > minGap) result.push(time)
  }
  return result
}
