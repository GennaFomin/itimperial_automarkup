import { useEffect, useMemo, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { formatPrecise, parseTimecode } from '../lib/time'
import { frequentValues, objectsForAction } from '../lib/vocab'
import { LabelPicker } from './LabelPicker'
import type { VocabAction, VocabObject } from '../api/types'

interface Props {
  actions: VocabAction[]
  objects: VocabObject[]
  onSeek: (ms: number) => void
}

/** Насколько двигают границу стрелками — один клик по «нюджу». */
const NUDGE_MS = 100

/**
 * Инспектор выделенного сегмента.
 *
 * Принцип из кейса: граница, действие, объект и keyframe правятся независимо.
 * Никакой единой кнопки «переделать всё» здесь нет.
 */
export function SegmentInspector({ actions, objects, onSeek }: Props) {
  const segments = useEditorStore((s) => s.segments)
  const selectedId = useEditorStore((s) => s.selectedId)
  const playheadMs = useEditorStore((s) => s.playheadMs)
  const applyUpdate = useEditorStore((s) => s.applyUpdate)
  const applyBoundary = useEditorStore((s) => s.applyBoundary)
  const applyDelete = useEditorStore((s) => s.applyDelete)
  const applyMerge = useEditorStore((s) => s.applyMerge)
  const applyVerify = useEditorStore((s) => s.applyVerify)
  const applySplit = useEditorStore((s) => s.applySplit)
  const zoomToSegment = useEditorStore((s) => s.zoomToSegment)
  const openVocabulary = useEditorStore((s) => s.vocab?.open ?? false)
  const pairs = useEditorStore((s) => s.vocab?.pairs)

  const seg = segments.find((s) => s.id === selectedId) ?? null
  // Частые значения ролика — то, что реально нужно под рукой при десятках классов.
  const frequentActions = useMemo(() => frequentValues(segments, 'action'), [segments])
  const frequentObjects = useMemo(() => frequentValues(segments, 'object'), [segments])
  const allowedObjects = useMemo(
    () => (seg ? objectsForAction(objects, pairs, seg.action, [seg.object, 'unknown']) : objects),
    [objects, pairs, seg],
  )
  const actionOption = seg ? actions.find((a) => a.id === seg.action) : undefined
  const objectOption = seg ? objects.find((o) => o.id === seg.object) : undefined
  const index = seg ? segments.findIndex((s) => s.id === seg.id) : -1
  const isLast = index === segments.length - 1

  if (!seg) {
    return (
      <div className="empty">
        <div className="empty__title">Сегмент не выбран</div>
        <p>Кликните по сегменту на таймлайне или по ключевому кадру, чтобы открыть его свойства.</p>
      </div>
    )
  }

  const playheadInside = playheadMs > seg.start_ms && playheadMs < seg.end_ms

  return (
    <div className="insp">
      <div className="insp__head">
        <div>
          <div className="insp__id">{seg.id}</div>
          <div className="insp__title">
            Сегмент {index + 1} из {segments.length}
          </div>
        </div>
        <span className="chip">{seg.origin === 'human' ? 'вручную' : 'модель'}</span>
      </div>

      {/* Проверка — утверждение о работе человека, а не о содержании разметки,
          поэтому это отдельное действие, а не следствие правки. */}
      <label className={`insp__verify${seg.verified ? ' insp__verify--on' : ''}`}>
        <input
          type="checkbox"
          checked={seg.verified}
          onChange={(e) => applyVerify(seg.id, e.target.checked)}
        />
        <span>{seg.verified ? 'Проверено' : 'Отметить проверенным'}</span>
        <kbd>Y</kbd>
      </label>

      {seg.keyframe_ms === null && (
        <span className="insp__flag">⚠ Ключевой кадр не посчитан моделью</span>
      )}

      <div>
        <div className="insp__section-label">
          Действие
          {!openVocabulary && actionOption?.unknown && (
            <span className="insp__oov" title="Такого действия нет в словаре задачи">вне словаря</span>
          )}
        </div>
        <LabelPicker
          key={`a-${seg.id}`}
          options={actions}
          value={seg.action}
          onChange={(id) => applyUpdate(seg.id, { action: id })}
          allowFree={openVocabulary}
          frequent={frequentActions}
          hotkeys
          placeholder={openVocabulary ? 'любое действие, например: повернул' : 'найти действие'}
        />
        <Confidence value={seg.action_confidence} edited={seg.edited} />
      </div>

      <div>
        <div className="insp__section-label">
          Объект
          {!openVocabulary && objectOption?.unknown && (
            <span className="insp__oov" title="Такого объекта нет в словаре задачи">вне словаря</span>
          )}
        </div>
        <LabelPicker
          key={`o-${seg.id}`}
          options={allowedObjects}
          value={seg.object}
          onChange={(id) => applyUpdate(seg.id, { object: id })}
          allowFree={openVocabulary}
          frequent={frequentObjects}
          placeholder={openVocabulary ? 'любой объект, например: бутылка' : 'найти объект'}
        />
        <Confidence value={seg.object_confidence} edited={seg.edited} />
      </div>

      <div>
        <div className="insp__section-label">Границы</div>
        <div className="times">
          <TimeField
            label="Начало"
            value={seg.start_ms}
            onCommit={(ms) => applyBoundary(seg.id, 'start', ms)}
            onGoto={() => onSeek(seg.start_ms)}
          />
          <TimeField
            label="Конец"
            value={seg.end_ms}
            onCommit={(ms) => applyBoundary(seg.id, 'end', ms)}
            onGoto={() => onSeek(seg.end_ms)}
          />
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
          <button className="btn btn--sm" onClick={() => applyBoundary(seg.id, 'start', playheadMs)}>
            Начало = курсор
          </button>
          <button className="btn btn--sm" onClick={() => applyBoundary(seg.id, 'end', playheadMs)}>
            Конец = курсор
          </button>
        </div>
        <Confidence value={seg.boundary_confidence} edited={seg.edited} label="границы" />
      </div>

      <div>
        <div className="insp__section-label">Ключевой кадр</div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
          <span className="mono" style={{ fontSize: 12.5 }}>
            {seg.keyframe_ms === null ? '—' : formatPrecise(seg.keyframe_ms)}
          </span>
          <button
            className="btn btn--sm"
            onClick={() => applyUpdate(seg.id, { keyframe_ms: playheadMs })}
            disabled={!playheadInside}
            title={playheadInside ? undefined : 'Курсор должен быть внутри сегмента'}
          >
            Взять текущий кадр
          </button>
          {seg.keyframe_ms !== null && (
            <button className="btn btn--sm" onClick={() => onSeek(seg.keyframe_ms!)}>
              Перейти
            </button>
          )}
        </div>
      </div>

      <div>
        <div className="insp__section-label">Операции</div>
        <div className="insp__actions">
          <button
            className="btn btn--sm"
            onClick={() => applySplit(seg.id, playheadMs)}
            disabled={!playheadInside}
            title="Разрезать по курсору (S)"
          >
            ✂ Разрезать
          </button>
          <button className="btn btn--sm" onClick={() => applyMerge(seg.id)} disabled={isLast}>
            ⇥ Слить со следующим
          </button>
          <button className="btn btn--sm" onClick={() => zoomToSegment(seg.id)}>
            ⤢ Приблизить
          </button>
          <button className="btn btn--sm btn--danger" onClick={() => applyDelete(seg.id)}>
            ✕ Удалить
          </button>
        </div>
      </div>

      <p className="insp__note">
        Длина {formatPrecise(seg.end_ms - seg.start_ms)}.{' '}
        {seg.edited ? 'Сегмент отредактирован — confidence модели больше не показывается.' : ''}
      </p>
    </div>
  )
}

function Confidence({
  value,
  edited,
  label,
}: {
  value: number | null
  edited: boolean
  label?: string
}) {
  if (value === null || edited) return null
  const color = value < 0.5 ? 'var(--danger)' : value < 0.7 ? 'var(--warn)' : 'var(--ok)'
  return (
    <div className="conf" title={`Уверенность модели${label ? ` в ${label}` : ''}`}>
      <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>conf</span>
      <span className="conf__bar">
        <span className="conf__fill" style={{ width: `${value * 100}%`, background: color }} />
      </span>
      <span className="conf__val">{value.toFixed(2)}</span>
    </div>
  )
}

function TimeField({
  label,
  value,
  onCommit,
  onGoto,
}: {
  label: string
  value: number
  onCommit: (ms: number) => void
  onGoto: () => void
}) {
  const [text, setText] = useState(() => formatPrecise(value))
  const [focused, setFocused] = useState(false)

  // Пока поле в фокусе — не перетираем то, что печатает пользователь.
  useEffect(() => {
    if (!focused) setText(formatPrecise(value))
  }, [value, focused])

  const commit = () => {
    const ms = parseTimecode(text)
    if (ms === null) setText(formatPrecise(value))
    else onCommit(ms)
  }

  return (
    <div className="time-field">
      <span className="time-field__label">{label}</span>
      <div className="time-field__row">
        <input
          className="input mono"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => {
            setFocused(false)
            commit()
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
            if (e.key === 'ArrowUp') {
              e.preventDefault()
              onCommit(value + NUDGE_MS)
            }
            if (e.key === 'ArrowDown') {
              e.preventDefault()
              onCommit(value - NUDGE_MS)
            }
          }}
        />
        <button className="btn btn--sm nudge" onClick={onGoto} title="Перейти к этой точке">
          →
        </button>
      </div>
    </div>
  )
}
