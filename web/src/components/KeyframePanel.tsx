import { useEffect, useRef } from 'react'
import { useEditorStore } from '../store/editorStore'
import { formatShort } from '../lib/time'
import type { VocabAction } from '../api/types'

interface Props {
  /**
   * Адрес кадра по времени. Кадры приходят с сервера, где уже лежит их дисковый
   * кэш: браузеру остаётся обычная картинка, а вкладке не нужен второй декодер
   * видео и очередь перемоток ради каждой миниатюры.
   */
  frameUrl: ((ms: number) => string) | null
  actions: VocabAction[]
  onPick: (segmentId: string, keyframeMs: number) => void
}

/**
 * Ключевые кадры всех сегментов сеткой. Клик по кадру — прыжок на него в
 * плеере и на таймлайне: это основной способ быстро пробежать разметку
 * глазами, не проматывая ролик.
 */
export function KeyframePanel({ frameUrl, actions, onPick }: Props) {
  const segments = useEditorStore((s) => s.segments)
  const selectedId = useEditorStore((s) => s.selectedId)
  const containerRef = useRef<HTMLDivElement>(null)

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
      {segments.map((seg) => {
        const action = actions.find((a) => a.id === seg.action)
        // Адрес включает время кадра, поэтому сдвиг keyframe сам обновляет
        // картинку и попадает в кэш браузера как отдельный ресурс.
        const url = frameUrl && seg.keyframe_ms !== null ? frameUrl(seg.keyframe_ms) : undefined
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
              ) : (
                <span className="kf__placeholder">нет видео</span>
              )}
              {seg.keyframe_ms === null && <span className="kf__warn">нет kf</span>}
              <span className="kf__stripe" style={{ background: action?.color ?? '#9AA3AD' }} />
            </div>
            <div className="kf__meta">
              <span className="kf__action">
                {seg.verified && <span className="kf__check">✓</span>}
                {action?.label_ru ?? seg.action}
              </span>
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
