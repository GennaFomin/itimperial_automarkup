import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import type { EditableSegment } from '../lib/segments'
import { MIN_SEGMENT_MS, snap, snapPoints } from '../lib/segments'
import { clamp, formatShort, tickStepMs } from '../lib/time'
import type { VocabAction, VocabObject } from '../api/types'
import { RangePicker } from './RangePicker'
import { frequentValues } from '../lib/vocab'
import './Timeline.css'

/** Уверенность ниже этого порога подсвечиваем штриховкой — место вероятной ошибки. */
const LOW_CONFIDENCE = 0.6

/** Прилипание границ: в миллисекундах это зависит от текущего масштаба. */
const SNAP_PX = 7

interface Props {
  actions: VocabAction[]
  objects: VocabObject[]
  /** Перемотать видео — таймлайн не знает про плеер напрямую. */
  onSeek: (ms: number) => void
}

type Drag =
  | { kind: 'scrub' }
  | { kind: 'pan'; startX: number; startViewMs: number }
  | { kind: 'boundary'; id: string; edge: 'start' | 'end' }
  | { kind: 'move'; id: string; grabOffsetMs: number }
  | { kind: 'keyframe'; id: string }
  | { kind: 'range'; anchorMs: number }
  | { kind: 'mini-window'; startX: number; startViewMs: number }

export function Timeline({ actions, objects, onSeek }: Props) {
  const segments = useEditorStore((s) => s.segments)
  const vocabPairs = useEditorStore((s) => s.vocab?.pairs)
  const openVocabulary = useEditorStore((s) => s.vocab?.open ?? false)
  const durationMs = useEditorStore((s) => s.durationMs)
  const viewStartMs = useEditorStore((s) => s.viewStartMs)
  const viewEndMs = useEditorStore((s) => s.viewEndMs)
  const playheadMs = useEditorStore((s) => s.playheadMs)
  const selectedId = useEditorStore((s) => s.selectedId)
  const tool = useEditorStore((s) => s.tool)
  const rangeSelection = useEditorStore((s) => s.rangeSelection)
  const snapEnabled = useEditorStore((s) => s.snapEnabled)

  const select = useEditorStore((s) => s.select)
  const setTool = useEditorStore((s) => s.setTool)
  const toggleSnap = useEditorStore((s) => s.toggleSnap)
  const setRangeSelection = useEditorStore((s) => s.setRangeSelection)
  const setView = useEditorStore((s) => s.setView)
  const zoomAt = useEditorStore((s) => s.zoomAt)
  const zoomToFit = useEditorStore((s) => s.zoomToFit)
  const applyBoundary = useEditorStore((s) => s.applyBoundary)
  const applyMove = useEditorStore((s) => s.applyMove)
  const applyUpdate = useEditorStore((s) => s.applyUpdate)
  const applyCarve = useEditorStore((s) => s.applyCarve)
  const applyCreate = useEditorStore((s) => s.applyCreate)

  const trackRef = useRef<HTMLDivElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const miniRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<Drag | null>(null)
  const [width, setWidth] = useState(0)
  /** Показывать ли выбор класса для только что выделенного диапазона. */
  const [pickerOpen, setPickerOpen] = useState(false)

  useLayoutEffect(() => {
    const el = trackRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    ro.observe(el)
    setWidth(el.getBoundingClientRect().width)
    return () => ro.disconnect()
  }, [])

  const span = Math.max(1, viewEndMs - viewStartMs)
  const msPerPx = span / Math.max(1, width)

  const msToPx = useCallback((ms: number) => ((ms - viewStartMs) / span) * width, [viewStartMs, span, width])
  const pxToMs = useCallback((px: number) => viewStartMs + (px / Math.max(1, width)) * span, [viewStartMs, span, width])

  const localX = (e: { clientX: number }, el: HTMLElement | null) =>
    e.clientX - (el?.getBoundingClientRect().left ?? 0)

  const points = useMemo(() => snapPoints(segments, durationMs), [segments, durationMs])
  const maybeSnap = useCallback(
    (ms: number) => (snapEnabled ? snap(ms, points, SNAP_PX * msPerPx) : Math.round(ms)),
    [snapEnabled, points, msPerPx],
  )

  // Рисуем только то, что попало в окно: на длинном ролике сегментов сотни.
  const visible = useMemo(
    () => segments.filter((s) => s.end_ms >= viewStartMs - span * 0.1 && s.start_ms <= viewEndMs + span * 0.1),
    [segments, viewStartMs, viewEndMs, span],
  )

  const colorOf = useCallback(
    (actionId: string) => actions.find((a) => a.id === actionId)?.color ?? '#9AA3AD',
    [actions],
  )
  const labelOf = useCallback(
    (actionId: string) => actions.find((a) => a.id === actionId)?.label_ru ?? actionId,
    [actions],
  )
  const objectLabelOf = useCallback(
    (objectId: string) => objects.find((o) => o.id === objectId)?.label_ru ?? objectId,
    [objects],
  )

  const ticks = useMemo(() => {
    if (width <= 0) return []
    const step = tickStepMs(msPerPx)
    const first = Math.ceil(viewStartMs / step) * step
    const out: { ms: number; x: number }[] = []
    for (let ms = first; ms <= viewEndMs; ms += step) out.push({ ms, x: msToPx(ms) })
    return out
  }, [width, msPerPx, viewStartMs, viewEndMs, msToPx])

  /* ---------------- Перетаскивания ---------------- */

  const beginDrag = (e: React.PointerEvent, drag: Drag) => {
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    dragRef.current = drag
  }

  const onTrackPointerDown = (e: React.PointerEvent) => {
    if (e.button === 1 || e.button === 2) {
      e.preventDefault()
      beginDrag(e, { kind: 'pan', startX: e.clientX, startViewMs: viewStartMs })
      return
    }
    if (e.button !== 0) return
    const ms = maybeSnap(pxToMs(localX(e, trackRef.current)))
    if (tool === 'carve') {
      setPickerOpen(false)
      setRangeSelection([ms, ms])
      beginDrag(e, { kind: 'range', anchorMs: ms })
      return
    }
    // Пустое место инструментом «Выбор»: ставим playhead, снимаем выделение.
    select(null)
    setRangeSelection(null)
    setPickerOpen(false)
    onSeek(clamp(ms, 0, durationMs))
    beginDrag(e, { kind: 'scrub' })
  }

  const onRulerPointerDown = (e: React.PointerEvent) => {
    if (e.button === 1) {
      e.preventDefault()
      beginDrag(e, { kind: 'pan', startX: e.clientX, startViewMs: viewStartMs })
      return
    }
    onSeek(clamp(pxToMs(localX(e, e.currentTarget as HTMLElement)), 0, durationMs))
    beginDrag(e, { kind: 'scrub' })
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current
    if (!drag) return
    const x = localX(e, trackRef.current)
    const ms = pxToMs(x)

    switch (drag.kind) {
      case 'scrub':
        onSeek(clamp(ms, 0, durationMs))
        break
      case 'pan':
        setView(
          drag.startViewMs - (e.clientX - drag.startX) * msPerPx,
          drag.startViewMs - (e.clientX - drag.startX) * msPerPx + span,
        )
        break
      case 'boundary':
        applyBoundary(drag.id, drag.edge, maybeSnap(ms))
        break
      case 'move':
        applyMove(drag.id, maybeSnap(ms - drag.grabOffsetMs))
        break
      case 'keyframe':
        applyUpdate(drag.id, { keyframe_ms: Math.round(ms) })
        break
      case 'range':
        setRangeSelection([Math.min(drag.anchorMs, maybeSnap(ms)), Math.max(drag.anchorMs, maybeSnap(ms))])
        break
      case 'mini-window': {
        const miniWidth = miniRef.current?.getBoundingClientRect().width ?? 1
        const deltaMs = ((e.clientX - drag.startX) / miniWidth) * durationMs
        const start = clamp(drag.startViewMs + deltaMs, 0, Math.max(0, durationMs - span))
        setView(start, start + span)
        break
      }
    }
  }

  const onPointerUp = (e: React.PointerEvent) => {
    const drag = dragRef.current
    dragRef.current = null
    try {
      ;(e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
    } catch {
      // Захват мог не установиться — не мешает завершению жеста.
    }
    if (drag?.kind === 'range') {
      const sel = useEditorStore.getState().rangeSelection
      if (sel && sel[1] - sel[0] >= MIN_SEGMENT_MS) setPickerOpen(true)
      else setRangeSelection(null)
    }
  }

  // Колесо обрабатываем нативным слушателем: React вешает wheel пассивно, и
  // preventDefault из синтетического обработчика не сработал бы — страница
  // прокручивалась бы вместо масштабирования.
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const state = useEditorStore.getState()
      const currentSpan = state.viewEndMs - state.viewStartMs
      const trackWidth = trackRef.current?.getBoundingClientRect().width ?? 1
      const perPx = currentSpan / Math.max(1, trackWidth)
      if (e.shiftKey) {
        state.panBy(e.deltaY * perPx * 1.4)
        return
      }
      const anchorMs = state.viewStartMs + (localX(e, trackRef.current) / Math.max(1, trackWidth)) * currentSpan
      state.zoomAt(anchorMs, e.deltaY > 0 ? 1.18 : 1 / 1.18)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  /* ---------------- Зум-слайдер ---------------- */

  // Ползунок линеен по логарифму: иначе весь полезный диапазон жмётся в край.
  const zoomPos = useMemo(() => {
    if (durationMs <= 0) return 0
    const maxLog = Math.log(durationMs / 400)
    if (maxLog <= 0) return 0
    return clamp(1 - Math.log(span / 400) / maxLog, 0, 1)
  }, [span, durationMs])

  const onZoomSlider = (value: number) => {
    const maxLog = Math.log(Math.max(1.01, durationMs / 400))
    const nextSpan = 400 * Math.exp((1 - value) * maxLog)
    // Держим playhead на месте, если он в кадре, иначе центрируем окно.
    const anchor = playheadMs >= viewStartMs && playheadMs <= viewEndMs ? playheadMs : viewStartMs + span / 2
    const ratio = (anchor - viewStartMs) / span
    setView(anchor - nextSpan * ratio, anchor - nextSpan * ratio + nextSpan)
  }

  const onMiniPointerDown = (e: React.PointerEvent) => {
    const el = miniRef.current
    if (!el) return
    const ratio = localX(e, el) / el.getBoundingClientRect().width
    const center = ratio * durationMs
    const start = clamp(center - span / 2, 0, Math.max(0, durationMs - span))
    setView(start, start + span)
    beginDrag(e, { kind: 'mini-window', startX: e.clientX, startViewMs: start })
  }

  const rangePx = rangeSelection
    ? { left: msToPx(rangeSelection[0]), width: Math.max(2, msToPx(rangeSelection[1]) - msToPx(rangeSelection[0])) }
    : null

  return (
    <div className="tl">
      <div className="tl__bar">
        <div className="tl__tools">
          <button
            className={`tl__tool${tool === 'select' ? ' tl__tool--active' : ''}`}
            onClick={() => setTool('select')}
            title="Выбор и правка границ (V)"
          >
            ▚ Выбор
          </button>
          <button
            className={`tl__tool${tool === 'carve' ? ' tl__tool--active' : ''}`}
            onClick={() => setTool('carve')}
            title="Вырезать диапазон и задать ему класс (C)"
          >
            ✂ Вырезать
          </button>
        </div>

        <label className="tl__check">
          <input type="checkbox" checked={snapEnabled} onChange={toggleSnap} />
          Прилипание
        </label>

        <span className="tl__zoom-label mono">
          окно {formatShort(span)} · {msPerPx.toFixed(1)} мс/px
        </span>

        <div className="tl__zoom">
          <button className="btn btn--ghost btn--sm" onClick={() => zoomAt(playheadMs, 1.4)} title="Отдалить (−)">
            −
          </button>
          <input
            className="tl__range"
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={zoomPos}
            onChange={(e) => onZoomSlider(Number(e.target.value))}
            aria-label="Масштаб таймлайна"
          />
          <button className="btn btn--ghost btn--sm" onClick={() => zoomAt(playheadMs, 1 / 1.4)} title="Приблизить (+)">
            +
          </button>
          <button className="btn btn--sm" onClick={zoomToFit} title="Показать целиком (0)">
            Весь ролик
          </button>
        </div>
      </div>

      {/* Обзорная полоса: где мы находимся внутри всего ролика. */}
      <div className="tl__mini" ref={miniRef} onPointerDown={onMiniPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}>
        {segments.map((s) => (
          <div
            key={s.id}
            className="tl__mini-seg"
            style={{
              left: `${(s.start_ms / Math.max(1, durationMs)) * 100}%`,
              width: `${Math.max(0.15, ((s.end_ms - s.start_ms) / Math.max(1, durationMs)) * 100)}%`,
              background: colorOf(s.action),
            }}
          />
        ))}
        <div
          className="tl__mini-window"
          style={{
            left: `${(viewStartMs / Math.max(1, durationMs)) * 100}%`,
            width: `${Math.max(0.5, (span / Math.max(1, durationMs)) * 100)}%`,
          }}
        />
        <div className="tl__mini-playhead" style={{ left: `${(playheadMs / Math.max(1, durationMs)) * 100}%` }} />
      </div>

      <div
        className="tl__ruler"
        onPointerDown={onRulerPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        {ticks.map((t) => (
          <div key={t.ms} className="tl__tick" style={{ left: t.x }}>
            <span className="tl__tick-label">{formatShort(t.ms)}</span>
          </div>
        ))}
      </div>

      <div className="tl__track-wrap" ref={wrapRef}>
        <div
          ref={trackRef}
          className={`tl__track${tool === 'carve' ? ' tl__track--carve' : ''}`}
          onPointerDown={onTrackPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onContextMenu={(e) => e.preventDefault()}
        >
          {visible.map((seg) => (
            <SegmentView
              key={seg.id}
              seg={seg}
              left={msToPx(seg.start_ms)}
              width={Math.max(2, msToPx(seg.end_ms) - msToPx(seg.start_ms))}
              keyframeX={seg.keyframe_ms === null ? null : msToPx(seg.keyframe_ms)}
              color={colorOf(seg.action)}
              actionLabel={labelOf(seg.action)}
              objectLabel={objectLabelOf(seg.object)}
              selected={seg.id === selectedId}
              carveMode={tool === 'carve'}
              onSelect={() => {
                if (tool === 'carve') return
                select(seg.id)
                setRangeSelection(null)
                setPickerOpen(false)
              }}
              // В режиме выреза сегмент не перехватывает жест: событие всплывает
              // к дорожке, и та выделяет диапазон прямо поверх существующих шагов.
              onBoundaryDown={(e, edge) => {
                if (tool === 'carve') return
                e.stopPropagation()
                select(seg.id)
                beginDrag(e, { kind: 'boundary', id: seg.id, edge })
              }}
              onBodyDown={(e) => {
                if (tool === 'carve') return
                e.stopPropagation()
                select(seg.id)
                setPickerOpen(false)
                onSeek(clamp(pxToMs(localX(e, trackRef.current)), 0, durationMs))
                beginDrag(e, { kind: 'move', id: seg.id, grabOffsetMs: pxToMs(localX(e, trackRef.current)) - seg.start_ms })
              }}
              onKeyframeDown={(e) => {
                if (tool === 'carve') return
                e.stopPropagation()
                select(seg.id)
                beginDrag(e, { kind: 'keyframe', id: seg.id })
              }}
            />
          ))}

          {rangePx && (
            <div className="tl__range-sel" style={{ left: rangePx.left, width: rangePx.width }}>
              {rangeSelection && (
                <span className="tl__range-badge">
                  {formatShort(rangeSelection[1] - rangeSelection[0])}
                </span>
              )}
            </div>
          )}

          <div className="tl__playhead" style={{ left: msToPx(playheadMs) }} />
        </div>

        {pickerOpen && rangeSelection && rangePx && (
          <RangePicker
            actions={actions}
            objects={objects}
            pairs={vocabPairs}
            allowFree={openVocabulary}
            frequentActions={frequentValues(segments, 'action')}
            frequentObjects={frequentValues(segments, 'object')}
            // Держим попап в пределах дорожки, чтобы он не уезжал за край окна.
            x={clamp(rangePx.left + rangePx.width / 2, 168, Math.max(168, width - 168))}
            range={rangeSelection}
            mode={
              segments.some((s) => rangeSelection[0] < s.end_ms && rangeSelection[1] > s.start_ms)
                ? 'carve'
                : 'create'
            }
            onApply={(actionId, objectId, mode) => {
              if (mode === 'carve') applyCarve(rangeSelection[0], rangeSelection[1], actionId, objectId)
              else applyCreate(rangeSelection[0], rangeSelection[1], actionId, objectId)
              setPickerOpen(false)
            }}
            onCancel={() => {
              setPickerOpen(false)
              setRangeSelection(null)
            }}
          />
        )}
      </div>

      <div className="tl__hint">
        <span>
          <kbd>колесо</kbd> масштаб · <kbd>Shift</kbd>+<kbd>колесо</kbd> прокрутка
        </span>
        <span>
          <kbd>ПКМ</kbd> тянуть — панорама
        </span>
        <span>
          <kbd>S</kbd> разрезать в позиции курсора
        </span>
        <span>
          <kbd>C</kbd> инструмент выреза
        </span>
      </div>
    </div>
  )
}

interface SegmentViewProps {
  seg: EditableSegment
  left: number
  width: number
  keyframeX: number | null
  color: string
  actionLabel: string
  objectLabel: string
  selected: boolean
  /** Активен инструмент выреза — сегмент отдаёт жесты дорожке. */
  carveMode: boolean
  onSelect: () => void
  onBoundaryDown: (e: React.PointerEvent, edge: 'start' | 'end') => void
  onBodyDown: (e: React.PointerEvent) => void
  onKeyframeDown: (e: React.PointerEvent) => void
}

function SegmentView({
  seg,
  left,
  width,
  keyframeX,
  color,
  actionLabel,
  objectLabel,
  selected,
  carveMode,
  onSelect,
  onBoundaryDown,
  onBodyDown,
  onKeyframeDown,
}: SegmentViewProps) {
  const lowConfidence =
    seg.action_confidence !== null && seg.action_confidence < LOW_CONFIDENCE && !seg.edited
  const classes = [
    'seg',
    selected ? 'seg--selected' : '',
    seg.edited ? 'seg--edited' : '',
    seg.origin === 'human' ? 'seg--human' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div
      className={classes}
      style={{ left, width, background: color, cursor: carveMode ? 'crosshair' : undefined }}
      onPointerDown={onBodyDown}
      onClick={onSelect}
      title={`${actionLabel} · ${objectLabel}`}
    >
      {lowConfidence && <div className="seg__lowconf" />}
      {width > 46 && (
        <div className={`seg__label${isLight(color) ? '' : ' seg__label--light'}`}>
          <span className="seg__action">{actionLabel}</span>
          {width > 90 && <span className="seg__object">{objectLabel}</span>}
        </div>
      )}
      {keyframeX !== null && width > 14 && (
        <div
          className="seg__kf"
          style={{ left: keyframeX - left }}
          onPointerDown={onKeyframeDown}
          title="Ключевой кадр — потяните, чтобы сдвинуть"
        />
      )}
      <div className="seg__handle seg__handle--start" onPointerDown={(e) => onBoundaryDown(e, 'start')} />
      <div className="seg__handle seg__handle--end" onPointerDown={(e) => onBoundaryDown(e, 'end')} />
    </div>
  )
}

/** Светлая ли заливка — от этого зависит цвет подписи внутри сегмента. */
function isLight(hex: string): boolean {
  const m = hex.replace('#', '')
  if (m.length < 6) return true
  const r = parseInt(m.slice(0, 2), 16)
  const g = parseInt(m.slice(2, 4), 16)
  const b = parseInt(m.slice(4, 6), 16)
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.6
}
