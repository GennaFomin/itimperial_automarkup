import type { Annotation, Step } from './types'

/** Минимальная длительность шага. Короче — это уже не действие, а дрожание границы. */
export const MIN_STEP = 0.4

const round3 = (value: number) => Math.round(value * 1000) / 1000

/** Правка пользователя перестаёт быть автоматической разметкой — это видно в экспорте.
 *  Правка одновременно означает проверку: если шаг меняли, на него точно смотрели. */
const touched = (step: Step): Step => ({
  ...step,
  source: step.source === 'auto' ? 'edited' : step.source,
  verified: true,
})

export const sortSteps = (steps: Step[]): Step[] =>
  [...steps].sort((a, b) => a.start_sec - b.start_sec)

export const withSteps = (annotation: Annotation, steps: Step[]): Annotation => ({
  ...annotation,
  steps: sortSteps(steps),
})

const clampKeyframe = (step: Step): Step => {
  if (step.keyframe_sec === null) return step
  const inside = Math.min(Math.max(step.keyframe_sec, step.start_sec), step.end_sec)
  return inside === step.keyframe_sec ? step : { ...step, keyframe_sec: round3(inside) }
}

export const stepAt = (steps: Step[], time: number): Step | null =>
  sortSteps(steps).find((s) => time >= s.start_sec && time < s.end_sec) ?? null

/** Сдвиг одного края шага.
 *
 * Раньше края двигались парами, потому что шаги считались сплошным покрытием. Это было
 * неверно: между действиями бывают паузы, и край должен двигаться сам по себе, упираясь
 * только в соседний шаг.
 */
export function moveEdge(
  steps: Step[],
  id: number,
  edge: 'start' | 'end',
  time: number,
  duration: number,
): Step[] | null {
  const ordered = sortSteps(steps)
  const index = ordered.findIndex((s) => s.id === id)
  if (index < 0) return null

  const step = ordered[index]
  const previous = ordered[index - 1]
  const next = ordered[index + 1]

  if (edge === 'start') {
    const lower = previous ? previous.end_sec : 0
    const upper = step.end_sec - MIN_STEP
    if (upper < lower) return null
    const at = round3(Math.min(Math.max(time, lower), upper))
    if (at === step.start_sec) return null
    ordered[index] = clampKeyframe(touched({ ...step, start_sec: at }))
  } else {
    const lower = step.start_sec + MIN_STEP
    const upper = next ? next.start_sec : duration
    if (upper < lower) return null
    const at = round3(Math.min(Math.max(time, lower), upper))
    if (at === step.end_sec) return null
    ordered[index] = clampKeyframe(touched({ ...step, end_sec: at }))
  }
  return ordered
}

/** Новый шаг в пропуске вокруг указанного времени: модель шаг пропустила, человек добавляет. */
export function addStep(
  steps: Step[],
  time: number,
  duration: number,
  action: string,
): Step[] | null {
  const ordered = sortSteps(steps)
  const previous = [...ordered].reverse().find((s) => s.end_sec <= time)
  const next = ordered.find((s) => s.start_sec >= time)

  const lower = previous ? previous.end_sec : 0
  const upper = next ? next.start_sec : duration
  if (upper - lower < MIN_STEP) return null
  if (ordered.some((s) => time > s.start_sec && time < s.end_sec)) return null

  const start = round3(Math.max(lower, Math.min(time, upper - MIN_STEP)))
  const end = round3(Math.min(upper, start + Math.max(MIN_STEP, 2)))
  const nextId = ordered.length ? Math.max(...ordered.map((s) => s.id)) + 1 : 0

  return [
    ...ordered,
    {
      id: nextId,
      level: 'coarse',
      parent_id: null,
      start_sec: start,
      end_sec: end,
      action,
      object: null,
      keyframe_sec: round3((start + end) / 2),
      confidence: null,
      source: 'manual',
      verified: true,
    },
  ]
}

export function splitStep(steps: Step[], id: number, time: number): Step[] | null {
  const ordered = sortSteps(steps)
  const index = ordered.findIndex((s) => s.id === id)
  if (index < 0) return null

  const step = ordered[index]
  if (time - step.start_sec < MIN_STEP || step.end_sec - time < MIN_STEP) return null

  const at = round3(time)
  const nextId = Math.max(...ordered.map((s) => s.id)) + 1
  const first = clampKeyframe(touched({ ...step, end_sec: at }))
  const second = touched({
    ...step,
    id: nextId,
    start_sec: at,
    keyframe_sec: round3((at + step.end_sec) / 2),
  })
  ordered.splice(index, 1, first, second)
  return ordered
}

export function mergeWithNext(steps: Step[], id: number): Step[] | null {
  const ordered = sortSteps(steps)
  const index = ordered.findIndex((s) => s.id === id)
  if (index < 0 || index === ordered.length - 1) return null

  const merged = touched({ ...ordered[index], end_sec: ordered[index + 1].end_sec })
  ordered.splice(index, 2, clampKeyframe(merged))
  return ordered
}

/** Удаление шага. На его месте остаётся пропуск — это нормальное состояние разметки. */
export function deleteStep(steps: Step[], id: number): Step[] | null {
  const ordered = sortSteps(steps)
  const index = ordered.findIndex((s) => s.id === id)
  if (index < 0) return null
  ordered.splice(index, 1)
  return ordered
}

export function updateStep(steps: Step[], id: number, patch: Partial<Step>): Step[] {
  return steps.map((step) => (step.id === id ? clampKeyframe(touched({ ...step, ...patch })) : step))
}

export const isUncertain = (step: Step): boolean =>
  step.source === 'auto' && step.confidence !== null && step.confidence < 0.5

export const setVerified = (steps: Step[], id: number, verified: boolean): Step[] =>
  steps.map((step) => (step.id === id ? { ...step, verified } : step))

/** Подтвердить всё непроверенное разом — этим заканчивается разбор ролика. */
export const verifyAll = (steps: Step[]): Step[] =>
  steps.map((step) => (step.verified ? step : { ...step, verified: true }))

/** Следующий непроверенный шаг после текущего, по кругу. Основа быстрого разбора. */
export function nextUnverified(steps: Step[], afterId: number | null): Step | null {
  const ordered = sortSteps(steps)
  const start = ordered.findIndex((step) => step.id === afterId)
  for (let offset = 1; offset <= ordered.length; offset += 1) {
    const candidate = ordered[(start + offset + ordered.length) % ordered.length]
    if (!candidate.verified) return candidate
  }
  return null
}

export const verifiedCount = (steps: Step[]): number => steps.filter((s) => s.verified).length
