import type { Annotation, Step } from './types'

/** Минимальная длительность шага. Короче — это уже не действие, а дрожание границы. */
export const MIN_STEP = 0.4

const round3 = (value: number) => Math.round(value * 1000) / 1000

/** Правка пользователя перестаёт быть автоматической разметкой — это видно в экспорте. */
const touched = (step: Step): Step => ({
  ...step,
  source: step.source === 'auto' ? 'edited' : step.source,
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

/** Сдвиг границы между шагом `leftId` и следующим. Соседи всегда остаются встык. */
export function moveBoundary(steps: Step[], leftId: number, time: number): Step[] | null {
  const ordered = sortSteps(steps)
  const index = ordered.findIndex((s) => s.id === leftId)
  if (index < 0 || index === ordered.length - 1) return null

  const left = ordered[index]
  const right = ordered[index + 1]
  const lower = left.start_sec + MIN_STEP
  const upper = right.end_sec - MIN_STEP
  if (upper < lower) return null

  const at = round3(Math.min(Math.max(time, lower), upper))
  if (at === left.end_sec) return null

  ordered[index] = clampKeyframe(touched({ ...left, end_sec: at }))
  ordered[index + 1] = clampKeyframe(touched({ ...right, start_sec: at }))
  return ordered
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

/** Удаление шага: интервал забирает сосед, чтобы на таймлайне не появилось дыры. */
export function deleteStep(steps: Step[], id: number): Step[] | null {
  const ordered = sortSteps(steps)
  if (ordered.length < 2) return null
  const index = ordered.findIndex((s) => s.id === id)
  if (index < 0) return null

  const removed = ordered[index]
  if (index > 0) {
    ordered[index - 1] = touched({ ...ordered[index - 1], end_sec: removed.end_sec })
  } else {
    ordered[index + 1] = touched({ ...ordered[index + 1], start_sec: removed.start_sec })
  }
  ordered.splice(index, 1)
  return ordered.map(clampKeyframe)
}

export function updateStep(steps: Step[], id: number, patch: Partial<Step>): Step[] {
  return steps.map((step) => (step.id === id ? clampKeyframe(touched({ ...step, ...patch })) : step))
}

export const isUncertain = (step: Step): boolean =>
  step.source === 'auto' && step.confidence !== null && step.confidence < 0.5
