import { useEffect, useMemo, useRef, useState } from 'react'
import { FrameExtractor } from '../lib/frames'
import { useEditorStore } from '../store/editorStore'
import { formatShort } from '../lib/time'
import type { VocabAction } from '../api/types'

interface Props {
  videoSrc: string | null
  actions: VocabAction[]
  onPick: (segmentId: string, keyframeMs: number) => void
}

/**
 * Ключевые кадры всех сегментов сеткой. Клик по кадру — прыжок на него в
 * плеере и на таймлайне: это основной способ быстро пробежать разметку
 * глазами, не проматывая ролик.
 */
export function KeyframePanel({ videoSrc, actions, onPick }: Props) {
  const segments = useEditorStore((s) => s.segments)
  const selectedId = useEditorStore((s) => s.selectedId)
  const [frames, setFrames] = useState<Record<string, string>>({})
  const extractorRef = useRef<FrameExtractor | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!videoSrc) {
      setFrames({})
      return
    }
    const ex = new FrameExtractor(videoSrc, (key, url) =>
      setFrames((prev) => ({ ...prev, [key]: url })),
    )
    extractorRef.current = ex
    return () => {
      ex.dispose()
      extractorRef.current = null
      setFrames({})
    }
  }, [videoSrc])

  // Ключ кадра включает время: сдвинули keyframe — картинка пересчитается.
  const items = useMemo(
    () =>
      segments.map((s) => ({
        seg: s,
        key: s.keyframe_ms === null ? null : `${s.id}@${s.keyframe_ms}`,
      })),
    [segments],
  )

  useEffect(() => {
    const ex = extractorRef.current
    if (!ex) return
    for (const item of items) {
      if (item.key && item.seg.keyframe_ms !== null) {
        ex.request({ key: item.key, timeMs: item.seg.keyframe_ms })
      }
    }
  }, [items])

  // Выделенный сегмент подтягиваем в зону видимости панели.
  useEffect(() => {
    if (!selectedId) return
    const el = containerRef.current?.querySelector(`[data-seg="${CSS.escape(selectedId)}"]`)
    el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selectedId])

  if (segments.length === 0) {
    return (
      <div className="empty">
        <div className="empty__title">Сегментов нет</div>
        <p>
          Модель не нашла ни одного шага. Это валидный результат — разметьте ролик вручную
          инструментом «Вырезать».
        </p>
      </div>
    )
  }

  return (
    <div className="kf-grid" ref={containerRef}>
      {items.map(({ seg, key }) => {
        const action = actions.find((a) => a.id === seg.action)
        const url = key ? frames[key] : undefined
        return (
          <button
            key={seg.id}
            data-seg={seg.id}
            className={`kf${seg.id === selectedId ? ' kf--on' : ''}`}
            onClick={() => onPick(seg.id, seg.keyframe_ms ?? seg.start_ms)}
          >
            <div className="kf__thumb">
              {url ? (
                <img src={url} alt="" loading="lazy" />
              ) : seg.keyframe_ms === null ? (
                <span className="kf__placeholder">кадр не выбран</span>
              ) : videoSrc ? (
                <span className="kf__spinner" />
              ) : (
                <span className="kf__placeholder">нет видео</span>
              )}
              {seg.keyframe_ms === null && <span className="kf__warn">нет kf</span>}
              <span className="kf__stripe" style={{ background: action?.color ?? '#9AA3AD' }} />
            </div>
            <div className="kf__meta">
              <span className="kf__action">{action?.label_ru ?? seg.action}</span>
              <span className="kf__time">
                {seg.keyframe_ms === null
                  ? `${formatShort(seg.start_ms)}–${formatShort(seg.end_ms)}`
                  : formatShort(seg.keyframe_ms)}
              </span>
            </div>
          </button>
        )
      })}
    </div>
  )
}
