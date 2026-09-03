/**
 * Чистые операции над дорожкой сегментов.
 *
 * Инварианты, которые держит каждая функция (контракт §1, §3, §9-6):
 *   - сегменты отсортированы по start_ms и не пересекаются;
 *   - дыры разрешены — это idle-участки, заполнять их не нужно;
 *   - start_ms < end_ms, обе границы внутри [0, duration_ms];
 *   - start_ms <= keyframe_ms <= end_ms, либо keyframe_ms === null.
 *
 * Функции ничего не мутируют и возвращают новый массив: на этом построены
 * undo/redo в editorStore.
 */
import type { PredictionSegment, ReviewSegment, SegmentOrigin } from '../api/types'
import { clamp } from './time'

/** Сегмент в состоянии редактора: review-поля + то, что осталось от прогноза. */
export interface EditableSegment {
  id: string
  origin: SegmentOrigin
  start_ms: number
  end_ms: number
  action: string
  object: string
  keyframe_ms: number | null
  /** Confidence из прогноза; у созданных руками сегментов null. */
  boundary_confidence: number | null
  action_confidence: number | null
  object_confidence: number | null
  /** Правил ли человек этот сегмент — подсвечиваем в UI и считаем в дифф. */
  edited: boolean
  /**
   * Человек посмотрел на сегмент и подтвердил его. Отличается от `edited`:
   * правка означает, что сегмент меняли, проверка — что на него смотрели.
   *
   * Любая правка ставит отметку заодно: чтобы тронуть сегмент, его пришлось
   * увидеть. Обратное неверно — подтвердить можно и не меняя ничего, и именно
   * это отличает просмотренную разметку от просто принятой на веру.
   */
  verified: boolean
}

/** Короче этого сегмент не делаем: ниже уже неразличимо на таймлайне. */
export const MIN_SEGMENT_MS = 120

export function fromPrediction(segments: PredictionSegment[]): EditableSegment[] {
  return segments.map((s) => ({
    id: s.id,
    origin: 'model' as const,
    start_ms: s.start_ms,
    end_ms: s.end_ms,
    action: s.action.value,
    object: s.object.value,
    keyframe_ms: s.keyframe_ms,
    boundary_confidence: s.boundary_confidence,
    action_confidence: s.action.confidence,
    object_confidence: s.object.confidence,
    edited: false,
    verified: false,
  }))
}

export function toReviewSegments(segments: EditableSegment[]): ReviewSegment[] {
  return sortSegments(segments).map((s) => ({
    id: s.id,
    origin: s.origin,
    start_ms: s.start_ms,
    end_ms: s.end_ms,
    action: s.action,
    object: s.object,
    keyframe_ms: s.keyframe_ms,
  }))
}

export const sortSegments = (segments: EditableSegment[]): EditableSegment[] =>
  [...segments].sort((a, b) => a.start_ms - b.start_ms || a.end_ms - b.end_ms)

export function segmentAt(segments: EditableSegment[], ms: number): EditableSegment | null {
  return segments.find((s) => ms >= s.start_ms && ms < s.end_ms) ?? null
}

export function indexOfId(segments: EditableSegment[], id: string): number {
  return segments.findIndex((s) => s.id === id)
}

let idCounter = 0
export const nextSegmentId = () => `seg_new_${Date.now().toString(36)}${(idCounter++).toString(36)}`

/** Keyframe по умолчанию — середина сегмента: лучше, чем случайная граница. */
const midpoint = (start: number, end: number) => Math.round((start + end) / 2)

/** Держим keyframe внутри новых границ, не теряя его, если он и так внутри. */
function keepKeyframe(kf: number | null, start: number, end: number): number | null {
  if (kf === null) return null
  return clamp(kf, start, end)
}

/**
 * Разрезать сегмент в точке `ms` на два. Классы наследуются обеими половинами —
 * дальше человек меняет класс той половине, которой нужно.
 */
export function splitSegment(
  segments: EditableSegment[],
  segmentId: string,
  ms: number,
): { segments: EditableSegment[]; newIds: string[] } {
  const i = indexOfId(segments, segmentId)
  if (i === -1) return { segments, newIds: [] }
  const seg = segments[i]
  const at = Math.round(ms)
  if (at - seg.start_ms < MIN_SEGMENT_MS || seg.end_ms - at < MIN_SEGMENT_MS) {
    return { segments, newIds: [] }
  }

  const left: EditableSegment = {
    ...seg,
    end_ms: at,
    keyframe_ms: keepKeyframe(seg.keyframe_ms, seg.start_ms, at),
    edited: true,
    verified: true,
  }
  const right: EditableSegment = {
    ...seg,
    id: nextSegmentId(),
    origin: 'human',
    start_ms: at,
    keyframe_ms: midpoint(at, seg.end_ms),
    edited: true,
    verified: true,
  }
  const next = [...segments]
  next.splice(i, 1, left, right)
  return { segments: next, newIds: [left.id, right.id] }
}

/**
 * Вырезать кусок `[fromMs, toMs)` из середины дорожки и назначить ему свой класс.
 *
 * Главный инструмент правки: модель склеила два шага в один, человек выделяет
 * середину и даёт ей верное действие. Диапазон может накрывать несколько
 * сегментов и дыры — всё перекрытое подрезается, целиком накрытое удаляется,
 * на месте выреза появляется один новый сегмент.
 */
export function carveOut(
  segments: EditableSegment[],
  fromMs: number,
  toMs: number,
  action: string,
  object: string,
): { segments: EditableSegment[]; newId: string | null } {
  const start = Math.round(Math.min(fromMs, toMs))
  const end = Math.round(Math.max(fromMs, toMs))
  if (end - start < MIN_SEGMENT_MS) return { segments, newId: null }

  const out: EditableSegment[] = []
  for (const seg of segments) {
    // Не пересекается с вырезом — оставляем как есть.
    if (seg.end_ms <= start || seg.start_ms >= end) {
      out.push(seg)
      continue
    }
    // Вырез целиком внутри сегмента — остаются хвост слева и хвост справа.
    if (seg.start_ms < start && seg.end_ms > end) {
      if (start - seg.start_ms >= MIN_SEGMENT_MS) {
        out.push({
          ...seg,
          end_ms: start,
          keyframe_ms: keepKeyframe(seg.keyframe_ms, seg.start_ms, start),
          edited: true,
          verified: true,
        })
      }
      if (seg.end_ms - end >= MIN_SEGMENT_MS) {
        out.push({
          ...seg,
          id: nextSegmentId(),
          origin: 'human',
          start_ms: end,
          keyframe_ms: midpoint(end, seg.end_ms),
          edited: true,
          verified: true,
        })
      }
      continue
    }
    // Подрезаем хвост слева.
    if (seg.start_ms < start && start - seg.start_ms >= MIN_SEGMENT_MS) {
      out.push({
        ...seg,
        end_ms: start,
        keyframe_ms: keepKeyframe(seg.keyframe_ms, seg.start_ms, start),
        edited: true,
        verified: true,
      })
      continue
    }
    // Подрезаем хвост справа.
    if (seg.end_ms > end && seg.end_ms - end >= MIN_SEGMENT_MS) {
      out.push({
        ...seg,
        start_ms: end,
        keyframe_ms: keepKeyframe(seg.keyframe_ms, end, seg.end_ms),
        edited: true,
        verified: true,
      })
      continue
    }
    // Накрыт целиком (или огрызок короче минимума) — удаляем.
  }

  const carved: EditableSegment = {
    id: nextSegmentId(),
    origin: 'human',
    start_ms: start,
    end_ms: end,
    action,
    object,
    keyframe_ms: midpoint(start, end),
    boundary_confidence: null,
    action_confidence: null,
    object_confidence: null,
    edited: true,
    verified: true,
  }
  out.push(carved)
  return { segments: sortSegments(out), newId: carved.id }
}

/**
 * Подвинуть одну границу сегмента. Соседей не двигаем: по контракту дыры
 * разрешены, поэтому граница просто упирается в соседа, а не толкает его.
 */
export function moveBoundary(
  segments: EditableSegment[],
  segmentId: string,
  edge: 'start' | 'end',
  ms: number,
  durationMs: number,
): EditableSegment[] {
  const sorted = sortSegments(segments)
  const i = indexOfId(sorted, segmentId)
  if (i === -1) return segments
  const seg = sorted[i]
  const prev = sorted[i - 1]
  const next = sorted[i + 1]

  let start = seg.start_ms
  let end = seg.end_ms
  if (edge === 'start') {
    const lo = prev ? prev.end_ms : 0
    start = clamp(Math.round(ms), lo, seg.end_ms - MIN_SEGMENT_MS)
  } else {
    const hi = next ? next.start_ms : durationMs
    end = clamp(Math.round(ms), seg.start_ms + MIN_SEGMENT_MS, hi)
  }

  const updated: EditableSegment = {
    ...seg,
    start_ms: start,
    end_ms: end,
    keyframe_ms: keepKeyframe(seg.keyframe_ms, start, end),
    edited: true,
    verified: true,
  }
  const out = [...sorted]
  out[i] = updated
  return out
}

/** Сдвинуть сегмент целиком, не меняя длину и не наезжая на соседей. */
export function moveSegment(
  segments: EditableSegment[],
  segmentId: string,
  newStartMs: number,
  durationMs: number,
): EditableSegment[] {
  const sorted = sortSegments(segments)
  const i = indexOfId(sorted, segmentId)
  if (i === -1) return segments
  const seg = sorted[i]
  const len = seg.end_ms - seg.start_ms
  const lo = sorted[i - 1] ? sorted[i - 1].end_ms : 0
  const hi = (sorted[i + 1] ? sorted[i + 1].start_ms : durationMs) - len
  if (hi < lo) return segments

  const start = clamp(Math.round(newStartMs), lo, hi)
  const shift = start - seg.start_ms
  const out = [...sorted]
  out[i] = {
    ...seg,
    start_ms: start,
    end_ms: start + len,
    keyframe_ms: seg.keyframe_ms === null ? null : seg.keyframe_ms + shift,
    edited: true,
    verified: true,
  }
  return out
}

export function updateSegment(
  segments: EditableSegment[],
  segmentId: string,
  patch: Partial<Pick<EditableSegment, 'action' | 'object' | 'keyframe_ms'>>,
): EditableSegment[] {
  return segments.map((s) => {
    if (s.id !== segmentId) return s
    const merged = { ...s, ...patch, edited: true }
    merged.keyframe_ms = keepKeyframe(merged.keyframe_ms, merged.start_ms, merged.end_ms)
    return merged
  })
}

/**
 * Отметить сегмент проверенным, ничего в нём не меняя.
 *
 * Отдельная операция, а не часть updateSegment: подтверждение — это утверждение
 * о работе человека, а не о содержании разметки, и в дифф правок оно не идёт.
 */
export function setVerified(
  segments: EditableSegment[],
  segmentId: string,
  value: boolean,
): EditableSegment[] {
  return segments.map((s) => (s.id === segmentId ? { ...s, verified: value } : s))
}

/** Подтвердить всё непроверенное разом — когда просмотр уже сделан глазами. */
export function verifyAll(segments: EditableSegment[]): EditableSegment[] {
  if (segments.every((s) => s.verified)) return segments
  return segments.map((s) => (s.verified ? s : { ...s, verified: true }))
}

export function verifiedCount(segments: EditableSegment[]): number {
  return segments.reduce((count, s) => count + (s.verified ? 1 : 0), 0)
}

/** Следующий непроверенный по кругу — основной маршрут обхода разметки. */
export function nextUnverified(
  segments: EditableSegment[],
  afterId: string | null,
): EditableSegment | null {
  const sorted = sortSegments(segments)
  if (!sorted.length) return null
  const from = afterId ? sorted.findIndex((s) => s.id === afterId) + 1 : 0
  for (let i = 0; i < sorted.length; i++) {
    const candidate = sorted[(from + i) % sorted.length]
    if (!candidate.verified) return candidate
  }
  return null
}

export function deleteSegment(segments: EditableSegment[], segmentId: string): EditableSegment[] {
  return segments.filter((s) => s.id !== segmentId)
}

/**
 * Создать сегмент в свободном промежутке. Возвращает null, если места нет —
 * вызывающий код показывает подсказку вместо молчаливого no-op.
 */
export function createSegment(
  segments: EditableSegment[],
  fromMs: number,
  toMs: number,
  action: string,
  object: string,
  durationMs: number,
): { segments: EditableSegment[]; newId: string | null } {
  const start = clamp(Math.round(Math.min(fromMs, toMs)), 0, durationMs)
  const end = clamp(Math.round(Math.max(fromMs, toMs)), 0, durationMs)
  if (end - start < MIN_SEGMENT_MS) return { segments, newId: null }
  if (segments.some((s) => start < s.end_ms && end > s.start_ms)) {
    return { segments, newId: null }
  }
  const created: EditableSegment = {
    id: nextSegmentId(),
    origin: 'human',
    start_ms: start,
    end_ms: end,
    action,
    object,
    keyframe_ms: midpoint(start, end),
    boundary_confidence: null,
    action_confidence: null,
    object_confidence: null,
    edited: true,
    verified: true,
  }
  return { segments: sortSegments([...segments, created]), newId: created.id }
}

/** Слить сегмент со следующим: дыра между ними, если была, поглощается. */
export function mergeWithNext(segments: EditableSegment[], segmentId: string): EditableSegment[] {
  const sorted = sortSegments(segments)
  const i = indexOfId(sorted, segmentId)
  if (i === -1 || i === sorted.length - 1) return segments
  const a = sorted[i]
  const b = sorted[i + 1]
  const merged: EditableSegment = {
    ...a,
    end_ms: b.end_ms,
    keyframe_ms: a.keyframe_ms ?? b.keyframe_ms,
    edited: true,
    verified: true,
  }
  const out = [...sorted]
  out.splice(i, 2, merged)
  return out
}

/** Ближайший «магнит» для прилипания: границы сегментов и keyframe. */
export function snapPoints(segments: EditableSegment[], durationMs: number): number[] {
  const pts = new Set<number>([0, durationMs])
  for (const s of segments) {
    pts.add(s.start_ms)
    pts.add(s.end_ms)
    if (s.keyframe_ms !== null) pts.add(s.keyframe_ms)
  }
  return [...pts].sort((a, b) => a - b)
}

export function snap(value: number, points: number[], toleranceMs: number): number {
  let best = value
  let bestDist = toleranceMs
  for (const p of points) {
    const d = Math.abs(p - value)
    if (d <= bestDist) {
      best = p
      bestDist = d
    }
  }
  return Math.round(best)
}

/** Проблемы, из-за которых review нельзя отправлять (контракт §9-6). */
export interface ValidationIssue {
  segmentId: string | null
  message: string
}

export function validate(segments: EditableSegment[], durationMs: number): ValidationIssue[] {
  const issues: ValidationIssue[] = []
  const sorted = sortSegments(segments)
  const seen = new Set<string>()

  for (let i = 0; i < sorted.length; i++) {
    const s = sorted[i]
    if (seen.has(s.id)) issues.push({ segmentId: s.id, message: `Дубль id ${s.id}` })
    seen.add(s.id)
    if (s.start_ms >= s.end_ms) {
      issues.push({ segmentId: s.id, message: 'Нулевая или отрицательная длина' })
    }
    if (s.start_ms < 0 || s.end_ms > durationMs) {
      issues.push({ segmentId: s.id, message: 'Границы выходят за длительность видео' })
    }
    if (s.keyframe_ms !== null && (s.keyframe_ms < s.start_ms || s.keyframe_ms > s.end_ms)) {
      issues.push({ segmentId: s.id, message: 'Ключевой кадр вне сегмента' })
    }
    const next = sorted[i + 1]
    if (next && next.start_ms < s.end_ms) {
      issues.push({ segmentId: s.id, message: `Пересекается со следующим сегментом` })
    }
  }
  return issues
}

/** Дифф с прогнозом — телеметрия правок (контракт §4). */
export interface ReviewDiff {
  boundaries_edited: number
  actions_changed: number
  objects_changed: number
  keyframes_moved: number
  segments_added: number
  segments_deleted: number
  segments_untouched: number
}

export function diffAgainstPrediction(
  original: PredictionSegment[],
  current: EditableSegment[],
): ReviewDiff {
  const byId = new Map(original.map((s) => [s.id, s]))
  const currentIds = new Set(current.map((s) => s.id))
  const diff: ReviewDiff = {
    boundaries_edited: 0,
    actions_changed: 0,
    objects_changed: 0,
    keyframes_moved: 0,
    segments_added: 0,
    segments_deleted: 0,
    segments_untouched: 0,
  }

  for (const s of current) {
    const src = byId.get(s.id)
    if (!src) {
      diff.segments_added += 1
      continue
    }
    let touched = false
    if (src.start_ms !== s.start_ms || src.end_ms !== s.end_ms) {
      diff.boundaries_edited += 1
      touched = true
    }
    if (src.action.value !== s.action) {
      diff.actions_changed += 1
      touched = true
    }
    if (src.object.value !== s.object) {
      diff.objects_changed += 1
      touched = true
    }
    if (src.keyframe_ms !== s.keyframe_ms) {
      diff.keyframes_moved += 1
      touched = true
    }
    if (!touched) diff.segments_untouched += 1
  }

  for (const s of original) {
    if (!currentIds.has(s.id)) diff.segments_deleted += 1
  }
  return diff
}

/** Доля видео, покрытая сегментами. Идёт в шапку редактора. */
export function coverage(segments: EditableSegment[], durationMs: number): number {
  if (durationMs <= 0) return 0
  const covered = segments.reduce((acc, s) => acc + (s.end_ms - s.start_ms), 0)
  return clamp(covered / durationMs, 0, 1)
}
