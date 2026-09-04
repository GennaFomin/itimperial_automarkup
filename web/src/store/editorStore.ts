/**
 * Состояние экрана разметки: дорожка сегментов, выделение, вьюпорт таймлайна,
 * история правок.
 *
 * История хранит снимки массива сегментов. Операции из lib/segments чистые и
 * возвращают новый массив, поэтому снимок — это просто ссылка, а не копия.
 */
import { create } from 'zustand'
import type { Prediction, Vocabulary } from '../api/types'
import {
  type EditableSegment,
  carveOut,
  createSegment,
  deleteSegment,
  fromPrediction,
  mergeWithNext,
  moveBoundary,
  moveSegment,
  nextUnverified,
  setVerified,
  splitSegment,
  updateSegment,
  verifyAll,
} from '../lib/segments'
import { clamp } from '../lib/time'

const HISTORY_LIMIT = 100

/** Что делает перетаскивание по пустому месту таймлайна. */
export type TimelineTool = 'select' | 'carve'

interface EditorState {
  prediction: Prediction | null
  vocab: Vocabulary | null
  segments: EditableSegment[]
  past: EditableSegment[][]
  future: EditableSegment[][]

  selectedId: string | null
  playheadMs: number
  durationMs: number

  /** Окно таймлайна в миллисекундах — им управляет зум и панорама. */
  viewStartMs: number
  viewEndMs: number

  tool: TimelineTool
  /** Активное выделение диапазона для выреза: [from, to] в мс. */
  rangeSelection: [number, number] | null
  snapEnabled: boolean

  /**
   * Замер разметки с нуля: прогноз спрятан, дорожка пуста, время идёт в
   * отдельный счётчик. Это знаменатель KPI «в три раза быстрее» — без него
   * отношение не с чем сравнивать.
   */
  mode: 'review' | 'scratch'

  load: (prediction: Prediction, vocab: Vocabulary, mode?: 'review' | 'scratch') => void
  reset: () => void

  select: (id: string | null) => void
  setPlayhead: (ms: number) => void
  setTool: (tool: TimelineTool) => void
  toggleSnap: () => void
  setRangeSelection: (range: [number, number] | null) => void

  setView: (startMs: number, endMs: number) => void
  zoomAt: (anchorMs: number, factor: number) => void
  zoomToFit: () => void
  zoomToSegment: (id: string) => void
  panBy: (deltaMs: number) => void

  applySplit: (segmentId: string, ms: number) => void
  applyCarve: (fromMs: number, toMs: number, action: string, object: string) => void
  applyCreate: (fromMs: number, toMs: number, action: string, object: string) => void
  applyBoundary: (segmentId: string, edge: 'start' | 'end', ms: number) => void
  applyMove: (segmentId: string, startMs: number) => void
  applyUpdate: (
    segmentId: string,
    patch: Partial<Pick<EditableSegment, 'action' | 'object' | 'keyframe_ms'>>,
  ) => void
  applyDelete: (segmentId: string) => void
  applyMerge: (segmentId: string) => void
  applyVerify: (segmentId: string, value: boolean) => void
  applyVerifyAll: () => void
  /** Перейти к следующему непроверенному сегменту; null — все проверены. */
  gotoNextUnverified: () => EditableSegment | null

  undo: () => void
  redo: () => void
}

/** Меньше этого окно не сжимаем: иначе теряется контекст и ломается арифметика. */
const MIN_VIEW_MS = 400

const initial = {
  prediction: null,
  vocab: null,
  segments: [] as EditableSegment[],
  past: [] as EditableSegment[][],
  future: [] as EditableSegment[][],
  selectedId: null,
  playheadMs: 0,
  durationMs: 0,
  viewStartMs: 0,
  viewEndMs: 0,
  tool: 'select' as TimelineTool,
  rangeSelection: null,
  snapEnabled: true,
  mode: 'review' as 'review' | 'scratch',
}

export const useEditorStore = create<EditorState>((set, get) => {
  /** Записать новую дорожку в историю. Пустой результат операции не пишем. */
  const commit = (next: EditableSegment[], selectId?: string | null) =>
    set((s) => {
      if (next === s.segments) return s
      return {
        segments: next,
        past: [...s.past, s.segments].slice(-HISTORY_LIMIT),
        future: [],
        selectedId: selectId === undefined ? s.selectedId : selectId,
      }
    })

  return {
    ...initial,

    load: (prediction, vocab, mode = 'review') => {
      const duration = prediction.video.duration_ms
      set({
        ...initial,
        prediction,
        vocab,
        // В режиме замера прогноз остаётся в сторе как база сравнения, но на
        // дорожку не попадает: человек размечает ролик с чистого листа.
        segments: mode === 'scratch' ? [] : fromPrediction(prediction.segments),
        durationMs: duration,
        viewStartMs: 0,
        viewEndMs: duration,
        mode,
      })
    },

    reset: () => set({ ...initial }),

    select: (id) => set({ selectedId: id }),

    setPlayhead: (ms) => set((s) => ({ playheadMs: clamp(Math.round(ms), 0, s.durationMs) })),

    setTool: (tool) => set({ tool, rangeSelection: null }),

    toggleSnap: () => set((s) => ({ snapEnabled: !s.snapEnabled })),

    setRangeSelection: (rangeSelection) => set({ rangeSelection }),

    setView: (startMs, endMs) =>
      set((s) => {
        const duration = s.durationMs || 1
        let a = Math.max(0, Math.round(startMs))
        let b = Math.min(duration, Math.round(endMs))
        if (b - a < MIN_VIEW_MS) {
          const mid = (a + b) / 2
          a = clamp(Math.round(mid - MIN_VIEW_MS / 2), 0, duration - MIN_VIEW_MS)
          b = a + MIN_VIEW_MS
        }
        return { viewStartMs: a, viewEndMs: b }
      }),

    /**
     * Зум вокруг точки: `anchorMs` остаётся на том же месте экрана.
     * Это то, что делает колесо над таймлайном предсказуемым.
     */
    zoomAt: (anchorMs, factor) => {
      const { viewStartMs, viewEndMs, durationMs } = get()
      const span = viewEndMs - viewStartMs
      const nextSpan = clamp(span * factor, MIN_VIEW_MS, durationMs)
      const ratio = span === 0 ? 0.5 : (anchorMs - viewStartMs) / span
      let start = anchorMs - nextSpan * ratio
      start = clamp(start, 0, Math.max(0, durationMs - nextSpan))
      get().setView(start, start + nextSpan)
    },

    zoomToFit: () => {
      const { durationMs } = get()
      set({ viewStartMs: 0, viewEndMs: durationMs })
    },

    zoomToSegment: (id) => {
      const { segments, durationMs } = get()
      const seg = segments.find((s) => s.id === id)
      if (!seg) return
      // Поля по 40% длины сегмента с каждой стороны: видно и сам шаг, и стык с соседями.
      const pad = Math.max(400, (seg.end_ms - seg.start_ms) * 0.4)
      get().setView(Math.max(0, seg.start_ms - pad), Math.min(durationMs, seg.end_ms + pad))
    },

    panBy: (deltaMs) => {
      const { viewStartMs, viewEndMs, durationMs } = get()
      const span = viewEndMs - viewStartMs
      const start = clamp(viewStartMs + deltaMs, 0, Math.max(0, durationMs - span))
      set({ viewStartMs: start, viewEndMs: start + span })
    },

    applySplit: (segmentId, ms) => {
      const res = splitSegment(get().segments, segmentId, ms)
      // Выделяем правую половину: разрез обычно делают, чтобы переклассифицировать её.
      if (res.newIds.length) commit(res.segments, res.newIds[1])
    },

    applyCarve: (fromMs, toMs, action, object) => {
      const res = carveOut(get().segments, fromMs, toMs, action, object)
      if (res.newId) commit(res.segments, res.newId)
      set({ rangeSelection: null })
    },

    applyCreate: (fromMs, toMs, action, object) => {
      const { segments, durationMs } = get()
      const res = createSegment(segments, fromMs, toMs, action, object, durationMs)
      if (res.newId) commit(res.segments, res.newId)
      set({ rangeSelection: null })
    },

    applyBoundary: (segmentId, edge, ms) => {
      const { segments, durationMs } = get()
      commit(moveBoundary(segments, segmentId, edge, ms, durationMs))
    },

    applyMove: (segmentId, startMs) => {
      const { segments, durationMs } = get()
      commit(moveSegment(segments, segmentId, startMs, durationMs))
    },

    applyUpdate: (segmentId, patch) => commit(updateSegment(get().segments, segmentId, patch)),

    applyDelete: (segmentId) => {
      commit(deleteSegment(get().segments, segmentId), null)
    },

    applyMerge: (segmentId) => commit(mergeWithNext(get().segments, segmentId)),

    applyVerify: (segmentId, value) => commit(setVerified(get().segments, segmentId, value)),

    applyVerifyAll: () => commit(verifyAll(get().segments)),

    gotoNextUnverified: () => {
      const { segments, selectedId } = get()
      const next = nextUnverified(segments, selectedId)
      if (next) {
        set({ selectedId: next.id })
        get().zoomToSegment(next.id)
      }
      return next
    },

    undo: () =>
      set((s) => {
        if (!s.past.length) return s
        const prev = s.past[s.past.length - 1]
        return {
          segments: prev,
          past: s.past.slice(0, -1),
          future: [s.segments, ...s.future].slice(0, HISTORY_LIMIT),
          selectedId: prev.some((x) => x.id === s.selectedId) ? s.selectedId : null,
        }
      }),

    redo: () =>
      set((s) => {
        if (!s.future.length) return s
        const next = s.future[0]
        return {
          segments: next,
          past: [...s.past, s.segments].slice(-HISTORY_LIMIT),
          future: s.future.slice(1),
          selectedId: next.some((x) => x.id === s.selectedId) ? s.selectedId : null,
        }
      }),
  }
})
